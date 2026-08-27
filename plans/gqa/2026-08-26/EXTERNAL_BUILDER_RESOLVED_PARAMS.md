# 外部 Backend 的参数契约

本文从 [TileOps #1976](https://github.com/tile-ai/TileOPs/pull/1976) 出发，讨论公共算子参数应由 TileOps Op 统一解析，还是由各外部 backend 自行解析。

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
