# 外部 Backend 的参数与缓存契约

本文从 [TileOps #1976](https://github.com/tile-ai/TileOPs/pull/1976) 出发，整理两个相关但独立的问题：

1. 公共算子参数由 Op 还是外部 backend 解析；
2. TileOps 对外部 callable 的 cache key 是否覆盖 backend 的 kernel dispatch 条件。

## 一、起因

有些 manifest 参数在实例化时已经确定；另一些 `None` 表达依赖实际输入的默认策略，例如：

```python
sm_scale=None    # D**-0.5
dtype=None       # 跟随输入 dtype
rotary_dim=None  # 使用完整 head dimension D
```

原有路径中，BUILTIN 可以根据真实输入解析这些参数，外部 builder 收到的仍可能是原始 `None`，因此需要根据 `TensorSpec` 再解析一次。

计算本身很便宜。主要风险是 TileOps 和每个第三方 backend 重复实现公共语义，可能导致规则不同步、target 行为不一致，以及 raw value、sentinel 和最终值之间的契约不稳定。

## 二、两种参数方案

### 方案 A：Op 传递 resolved params

#1976 允许 Op 在看到真实输入后解析参数，再传给 external builder：

```python
params = self._manifest_params()
params.update(
    sm_scale=self.sm_scale if self.sm_scale is not None else D**-0.5,
    dtype=self.dtype if self.dtype is not None else q.dtype,
    rotary_dim=self.rotary_dim if self.rotary_dim is not None else D,
)

self.get_or_build_kernel(..., params=params)
```

职责边界是：

```text
Op：验证输入并解析公共语义
Backend：选择并构造具体 kernel
```

优点是公共语义集中在 Op，切换 target 原则上不改变算子定义。代价是需要与第三方 backend 约定收到的是 resolved params、参数是否完整，以及 manifest 演进时如何兼容。

外部 builder 原本就接收 `**manifest_params`，所以变化主要不是函数形态，而是参数值的语义。

### 方案 B：选择 target 后分别解析

```text
选择 target
  ├─ BUILTIN：TileOps 解析 params
  └─ External：backend 接收 raw params 并自行解析
```

这种设计给予第三方 backend 更大的自主性，也把它视为更完整的 Op provider。代价是公共默认语义分散到各 backend，需要它们分别维护并通过一致性测试证明符合 manifest 契约。

| | 方案 A | 方案 B |
| --- | --- | --- |
| 参数形式 | resolved params | raw params / sentinel |
| 公共语义实现位置 | TileOps Op | BUILTIN 与各 backend |
| backend 自主性 | 较低 | 较高 |
| 语义一致性 | 更容易集中保证 | 依赖各 backend conformance |

两种方案都可行，实质上是在决定外部 backend 是 concrete kernel provider，还是更完整的 Op provider。

## 三、更深层的缓存隐患

当前外部路径缓存的是 external builder 返回的 callable：

```python
self._kernel_roles[kernel_name][external_signature] = external_callable
```

`external_signature` 主要包含：

```python
(
    device,
    (input_0.dtype, input_0.shape),
    (input_1.dtype, input_1.shape),
    ...,
)
```

cache 还隐含限定在单个 Op 实例、target 和 kernel role 内。相同 signature 再次调用时，TileOps 会直接复用 callable，external builder 不会重新 dispatch。

因此必须满足：

> 如果两个调用命中同一个 TileOps external cache key，第一次缓存的 callable 必须也能正确处理第二个调用。

外部 backend 用于 dispatch 的每个 item，至少需要满足一个条件：

1. 已包含在 TileOps external cache key 中；
2. 在当前 cache 作用域内恒定；
3. 由缓存的 callable 在运行时继续处理。

否则可能发生：

```text
第一次：key=K，backend item=A，缓存 kernel_A
第二次：key=K，backend item=B，仍然复用 kernel_A
```

如果 `kernel_A` 不支持 B，这是正确性错误；如果它能处理 B，但 `kernel_B` 更快，则是性能选择被第一次缓存固定。

TileOps key 比 backend dispatch 更细通常安全，只会重复构建；TileOps key 更粗则可能错误复用或固定次优实现。

## 四、两个问题彼此独立

方案 A 解决参数语义所有权，但不会自动解决 cache key。#1976 不把 resolved params 放入 external key，因此要求它们是 Op 实例配置与输入 signature 的确定性函数。

方案 B 同样不会自动解决缓存问题。即使 backend 自己解析 raw params，只要其 dispatch 使用了 TileOps key 未覆盖、并且会在相同 signature 下变化的 item，TileOps 仍可能提前命中旧 callable。

所以需要分别决定：

| 问题 | 决策 |
| --- | --- |
| 参数语义契约 | Op 传 resolved params，还是 backend 处理 raw params |
| specialization cache 契约 | TileOps 固定 key、backend 参与 key，还是缓存 backend runtime dispatcher |

## 五、结论

#1976 只处理参数语义契约，没有自动处理第三方 backend 的 specialization cache 契约。

无论选择方案 A 还是 B，只要 TileOps 缓存 external backend 返回的 concrete callable，就必须明确该 cache key 表示的等价类。若希望第三方 backend 拥有更自由的 dispatch，后续还需要在以下方向中选择：限制 backend 只能使用现有 signature、允许 backend 提供 specialization-key hook，或者缓存 backend 自己的 runtime dispatcher。
