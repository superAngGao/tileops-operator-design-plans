# 外部 Backend 的参数语义与 Callable 缓存契约

本文从 [TileOps #1976](https://github.com/tile-ai/TileOPs/pull/1976) 的 resolved external builder params 出发，整理两个相关但彼此独立的问题：

1. **公共算子参数由谁解析**：TileOps Op，还是各外部 backend；
2. **外部 callable 如何缓存**：TileOps 的 cache key 是否覆盖外部 backend 的 kernel dispatch 条件。

#1976 直接处理第一个问题，但讨论也暴露了第二个更深层的接口约束。

## 一、起因：同一组公共参数被重复解析

部分 manifest 参数在 Op 实例化时已经确定，例如显式传入的 `is_causal=True`。另一些参数表达的是依赖实际输入的默认策略，例如：

- `sm_scale=None` 表示使用 `D**-0.5`；
- `dtype=None` 表示输出 dtype 跟随输入；
- `rotary_dim=None` 表示使用完整的 head dimension `D`。

这里的 `None` 不是缺少配置，而是需要看到实际输入后才能解析的公共算子语义。

假设用户构造：

```python
op = GroupedQueryAttentionDenseFwdOp(
    sm_scale=None,
    dtype=None,
    rotary_dim=None,
)
```

本次输入的 `D=64`、`q.dtype=torch.float16`，最终语义应当是：

```python
{
    "sm_scale": 0.125,
    "dtype": torch.float16,
    "rotary_dim": 64,
}
```

在原有路径中，TileOps 的 BUILTIN 实现可以根据真实输入得到这些值；外部 builder 收到的却仍是 Op 实例保存的原始 manifest 参数：

```python
{
    "sm_scale": None,
    "dtype": None,
    "rotary_dim": None,
}
```

外部 backend 并非无法构造 kernel。它同时持有输入的 `TensorSpec`，可以再次执行：

```python
sm_scale = q_spec.shape[-1] ** -0.5
dtype = q_spec.dtype
rotary_dim = q_spec.shape[-1]
```

这些计算本身非常便宜。主要问题不是算力开销，而是 TileOps 和每个外部 backend 都要实现相同的默认规则，从而带来：

- 公共语义重复维护；
- TileOps 修改规则后，第三方 backend 未必同步；
- 不同 target 可能逐渐产生不同的算子行为；
- raw value、sentinel 和最终值之间的接口含义不稳定。

## 二、问题一：公共算子参数由谁解析

### 方案 A：Op 统一解析并传递 resolved params

#1976 为 `get_or_build_kernel` 增加可选的 `params`：

```python
dim = q.shape[-1]
params = self._manifest_params()
params.update(
    sm_scale=self.sm_scale if self.sm_scale is not None else dim**-0.5,
    dtype=self.dtype if self.dtype is not None else q.dtype,
    rotary_dim=self.rotary_dim if self.rotary_dim is not None else dim,
)

kernel = self.get_or_build_kernel(
    "gqa_dense",
    inputs,
    params=params,
)
```

外部路径在 cache miss 时调用：

```python
kernel = external_builder(*tensor_specs, **params)
```

职责边界变成：

```text
Op
  -> 验证输入
  -> 解析公共默认语义
  -> 规范化输入
  -> 传递 resolved params

External backend
  -> 判断是否支持
  -> 选择并构造具体 kernel
```

优点是公共语义只由 Op 实现一次，切换 target 原则上不会改变算子定义。

代价是需要与第三方 backend 建立明确契约：

- builder 收到的是 raw params 还是 resolved params；
- `params` 是完整参数表还是局部覆盖；
- 每个字段在传入 builder 时允许哪些类型和 sentinel；
- manifest 新增参数时如何处理版本兼容。

外部 builder 原本就接收 `**manifest_params`，因此方案 A 不一定改变函数形态；真正变化的是参数值的语义契约。依赖 `None` 自行解析的旧 backend 也需要适配。

### 方案 B：选择 target 后，由各实现自行解析

另一种方案是先决定 target，再分别处理：

```text
选择 target
  ├─ BUILTIN：TileOps 解析 params，再 dispatch/build
  └─ External：传递 raw params，由外部 backend 解析并 dispatch/build
```

这种设计把外部 backend 看作更完整的 Op provider，而不只是 concrete kernel provider。它可以自行解释 `None`，也可以按自己的实现方式组合参数解析和 kernel dispatch。

优点是第三方 backend 自主性更强，不必适配 TileOps 定义的 resolved-param 传递时机。

代价是公共默认语义分散到各 backend。要保证 target-independent correctness，外部 backend 仍然必须理解并遵守 manifest 对 raw params 的定义，并通过一致性测试证明其行为与 TileOps 公共契约相同。

### 两种方案的区别

| | 方案 A：Op 传 resolved params | 方案 B：backend 处理 raw params |
| --- | --- | --- |
| 公共默认语义的实现位置 | TileOps Op | BUILTIN 与各外部 backend |
| 外部 backend 的角色 | kernel provider | 更完整的 Op provider |
| 第三方 backend 自主性 | 较低 | 较高 |
| 重复语义实现 | 少 | 多 |
| 主要接口契约 | resolved value 的含义和版本 | raw value/sentinel 的含义和版本 |
| target 行为一致性 | 更容易集中保证 | 依赖各 backend 的 conformance |

两种方案在技术上都成立。选择哪一个，实质上是在决定 TileOps Op 与第三方 backend 的职责边界。

## 三、更深层的问题：外部 callable 的缓存等价类

无论选择方案 A 还是 B，只要 TileOps 负责缓存外部 backend 返回的 concrete callable，就必须确保 cache key 覆盖外部 dispatch 所依赖的事实。

### 当前外部缓存记录什么

当前外部路径近似缓存：

```python
self._kernel_roles[kernel_name][external_signature] = external_callable
```

`external_signature` 由以下信息组成：

```python
(
    device,
    (input_0.dtype, input_0.shape),
    (input_1.dtype, input_1.shape),
    ...,
)
```

缺失的 optional input 会保留 `None` 槽位。此外还有几个隐含作用域：

- cache 属于单个 Op 实例，因此不变的构造参数不必重复放进 key；
- cache 按 kernel role/name 分开；
- 一个 Op 实例只绑定一个 target。

BUILTIN 使用调用点传入的 `key`；外部路径忽略这个 BUILTIN key，统一使用上述输入 signature。

在 cache miss 时：

```python
external_callable = external_builder(*tensor_specs, **params)
entries[external_signature] = external_callable
```

相同 signature 再次调用时，TileOps 直接返回缓存的 callable，external builder 不会再次执行。

### cache key 必须覆盖什么

正确性约束可以表述为：

> 如果两个调用命中同一个 TileOps external cache key，那么第一次缓存的 callable 必须也能正确处理第二个调用。

外部 backend 用于 dispatch 的每个 item，至少需要满足以下一个条件：

1. 已包含在 TileOps external cache key 中；
2. 在当前 Op 实例、target 和 role 的 cache 作用域内恒定；
3. 由缓存的 callable 在运行时继续处理。

例如：

- `is_causal` 通常固定在 Op 实例上，即使不显式进入 key 也安全；
- `D` 已包含在输入 shape 中，根据 `D` dispatch 也安全；
- 如果某个 backend item 在相同 device/dtype/shape 下仍会变化，并且 cached callable 不能处理这种变化，就不安全。

### key 未覆盖 backend dispatch 时会发生什么

```text
第一次调用
  TileOps external key = K
  backend dispatch item = A
  builder 返回 kernel_A
  TileOps 缓存 K -> kernel_A

第二次调用
  TileOps external key 仍然是 K
  backend dispatch item = B
  TileOps 命中缓存，builder 不再运行
  仍然调用 kernel_A
```

如果 `kernel_A` 不能正确处理 B，就会错误复用 kernel。如果 `kernel_A` 和 `kernel_B` 都正确，只是后者性能更好，那么结果仍然正确，但第一次选择会固定后续性能。

因此 key 粒度和 backend dispatch 粒度之间的关系是：

| 关系 | 结果 |
| --- | --- |
| TileOps key 比 backend dispatch 更细 | 正确，但可能重复构建 |
| 两者等价 | 理想情况 |
| TileOps key 比 backend dispatch 更粗 | 可能错误复用，或固定次优实现 |

## 四、参数契约与缓存契约彼此独立

方案 A 解决的是公共参数的所有权，但不会自动解决 external cache key：

```text
Op 传入 resolved params
≠
这些 params 已经包含在 external cache key 中
```

#1976 明确不把 resolved params 放入 external cache key，因此要求它们是 Op 实例配置与输入 device/dtype/shape signature 的确定性函数。同一个 cache 等价类不能产生不同的 resolved params。

方案 B 同样不会自动解决缓存问题。即使 external backend 自己解释 raw params，只要它的 concrete-kernel dispatch 使用了 TileOps key 未覆盖、且会在相同 signature 下变化的 item，TileOps 仍可能提前命中旧 callable，使 backend 失去重新 dispatch 的机会。

所以这里实际存在两个需要分别决策的接口：

| 接口问题 | 核心决策 |
| --- | --- |
| 参数语义契约 | Op 传 resolved params，还是 backend 处理 raw params |
| specialization cache 契约 | TileOps 固定 external key，backend 参与 key，还是缓存 backend dispatcher |

## 五、外部缓存契约的可选方向

### 方向 1：TileOps 固定 external signature

维持当前设计，并明确要求第三方 backend：

> builder 只能根据当前 signature 和 Op 实例内不变配置构造 callable；返回的 callable 必须服务该 signature 表示的整个等价类。

该方案最简单，但限制了第三方 backend 的 dispatch 自由度。

### 方向 2：允许 backend 提供 specialization-key hook

backend 注册 builder 时同时提供一个确定性的 key 函数：

```python
register_kernel_builder(
    op="GroupedQueryAttentionDenseFwdOp",
    target="external",
    specialization_key=external_gqa_key,
    build_kernel=build_external_gqa,
)
```

TileOps 将 backend key 与基础 signature 组合后缓存 concrete callable。这样 backend 可以表达自己的 specialization 边界，同时保留单层 Op cache。

key hook 必须便宜、可哈希、确定，并且不能依赖需要读取 tensor 内容或触发设备同步的动态值。

### 方向 3：缓存 backend runtime dispatcher

外部 builder 返回一个通用 callable，由它在每次执行时继续 dispatch：

```python
def external_callable(*real_inputs):
    kernel = backend_runtime_dispatch(real_inputs)
    return kernel(*real_inputs)
```

这给第三方 backend 最大自主性，也能处理 tensor-value-dependent dispatch；代价是可能引入第二层 cache、额外热路径开销，并削弱 TileOps 对 concrete kernel 的统一枚举和 autotune 管理。

## 六、需要明确的决策

后续设计需要分别回答：

1. 外部 backend 是 concrete kernel provider，还是更完整的 Op provider；
2. builder 接收 raw params 还是 resolved params；
3. resolved params 是否以及如何参与 external cache key；
4. 外部 backend 能否扩展 specialization key；
5. 若 backend 根据 tensor 内容 dispatch，是否接受 runtime dispatcher 或第二层 cache；
6. 如何测试不同 target 的公共语义一致性，以及同一 cache key 下 callable 的有效范围。

在这些问题明确前，不能仅通过“是否传递 params”判断接口是否稳定。参数解析和 callable 缓存是两份独立契约，二者都必须与第三方 backend 的职责边界一致。
