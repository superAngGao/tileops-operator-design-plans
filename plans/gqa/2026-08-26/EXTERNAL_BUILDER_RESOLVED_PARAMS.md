# 外部 Backend 构造参数解析

本文简要说明 [TileOps #1976](https://github.com/tile-ai/TileOPs/pull/1976) 增加 `get_or_build_kernel(..., params=...)` 前后的区别。

## 背景

外部 backend 的 builder 在构建 kernel 时会收到两类信息：

- 输入 tensor 的 `device`、`dtype` 和 `shape`，以 `TensorSpec` 传递；
- Op manifest 中声明的构造参数，以关键字参数传递。

部分构造参数在 Op 实例化时已经确定，例如显式指定的 `is_causal=True`。另一些参数允许使用依赖实际输入的默认策略，例如：

- `sm_scale=None` 表示使用 `D**-0.5`；
- `dtype=None` 表示输出 dtype 跟随输入；
- `rotary_dim=None` 表示使用完整的 head dimension `D`。

这里的 `None` 不是缺少配置，而是需要等到 Op 看到实际输入后才能解析的策略。

## 没有 resolved params

原来的外部构建路径固定传递 Op 实例中保存的 manifest 参数：

```python
kernel = builder(*specs, **self._manifest_params())
```

例如，用户构造：

```python
op = GroupedQueryAttentionDenseFwdOp(
    sm_scale=None,
    dtype=None,
    rotary_dim=None,
)
```

当输入的 `D=64`、`q.dtype=torch.float16` 时，外部 backend 仍会收到：

```python
{
    "sm_scale": None,
    "dtype": None,
    "rotary_dim": None,
}
```

backend 并非无法构造 kernel：它同时持有输入的 `TensorSpec`，因此可以自行解析这些值。问题是每个 backend 都必须重复实现 TileOps 的公共 Op 语义：

```python
sm_scale = q_spec.shape[-1] ** -0.5
dtype = q_spec.dtype
rotary_dim = q_spec.shape[-1]
```

不同 backend 的解析逻辑可能逐渐产生差异。

## 有 resolved params

新的路径允许调用点在看到真实输入后解析参数，并通过 `params` 传给外部 builder：

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

外部 backend 最终收到具体值：

```python
{
    "sm_scale": 0.125,
    "dtype": torch.float16,
    "rotary_dim": 64,
}
```

传递链如下：

```text
真实 Tensor
  -> Op 解析输入相关的公共语义
  -> get_or_build_kernel(..., params=resolved_params)
  -> external builder(*TensorSpecs, **resolved_params)
  -> concrete kernel
```

## 区别总结

| | 没有 resolved params | 有 resolved params |
| --- | --- | --- |
| Op 传递的参数 | 构造时保存的原始值 | 结合本次输入解析后的最终值 |
| backend 能否构造 kernel | 可以 | 可以 |
| `None` 等默认策略由谁解释 | 每个 backend 分别解释 | Op 统一解释 |
| 公共语义归属 | 分散在 Op 和各 backend | 保留在 Op 层 |
| 现有固定参数 Op | 正常工作 | 行为不变，不传 `params` 即可 |
| TileOps 内置实现 | 使用原有 `key + build` 路径 | 行为不变，忽略 `params` |

`params` 是完整替换而不是局部 merge，因此调用点通常先读取 `_manifest_params()`，再覆盖需要按输入解析的字段。

resolved params 不进入外部 kernel cache key。它们必须是 **Op 实例配置与输入 device/dtype/shape signature 的确定性函数**；同一 Op 实例和同一输入签名不得产生不同参数。
