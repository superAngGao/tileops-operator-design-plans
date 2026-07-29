# Nightly CUDA Events fallback 事实与原因归因（2026-07-29）

## 结论

TileOps 最新成功 Nightly（run
[#187](https://github.com/tile-ai/TileOPs/actions/runs/30417463015)，commit
[`db4494c`](https://github.com/tile-ai/TileOPs/commit/db4494c59149b4270278680f2c5761c62c360877)）
的 `profile_run.log` 中共有 **146 条 benchmark result 被标记为
`timing = cuda-events`**，其中 **68 条属于 GQA**。

这里的“146 次”以 benchmark report 的结果行为统计单位：每一行代表一次
`bench_kernel` 测量结果在 CUPTI 投影校验失败后，改用 CUDA Events 重新测量。
它不是 CUDA Event API 调用次数，也不是日志中出现 “fallback” 单词的次数。

这批数据不应直接与正常的纯 kernel CUPTI 历史数据混合做 regression 判断。
当前实现的 warning 指出 CUDA Events 路径可能带入约 50–60 μs/call 的
launch overhead，并使小于 10 μs 的 fast kernel 延迟膨胀约 6–7 倍；该量级
是代码中的风险提示，不是本报告从 artifact 独立测得的数值。

## 证据范围

本结论使用以下证据：

1. Nightly run #187 的 `tileops_benchmark_30417463015` artifact。
2. artifact 中 `tileops_benchmarks.log` 末尾拼接的
   `profile_run.log summary`。
3. Nightly 对应 commit `db4494c` 的
   [`benchmarks/benchmark_base.py`](https://github.com/tile-ai/TileOPs/blob/db4494c59149b4270278680f2c5761c62c360877/benchmarks/benchmark_base.py)。

主 GitHub Actions job log 中只出现两次 `fallback`，但它们是
`bench_gated_deltanet_prefill.py` 的 workload 名称，不是 timing fallback。
真正的 timing backend 记录在 artifact 的 report 表格中。

## 事实分布

### GQA：68 条

| Op | `cuda-events` 条数 |
| --- | ---: |
| `GroupedQueryAttentionFwdOp` | 15 |
| `GroupedQueryAttentionBwdOp` | 16 |
| `GroupedQueryAttentionPrefillFwdOp` | 16 |
| `GroupedQueryAttentionPrefillVarlenFwdOp` | 3 |
| `GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp` | 1 |
| `GroupedQueryAttentionDecodePagedWithKVCacheFwdOp` | 2 |
| `GroupedQueryAttentionSlidingWindowFwdOp` | 15 |
| **GQA 小计** | **68** |

### 非 GQA：78 条

| Op / report section | `cuda-events` 条数 |
| --- | ---: |
| `grouped_gemm_3wg_baselines` | 34 |
| `GemmFp8Op` | 26 |
| `GatedDeltaNetBwdOp` | 6 |
| `DeltaNetBwdOp` | 6 |
| `GLABwdOp` | 5 |
| `Conv3dFwdOp` | 1 |
| **非 GQA 小计** | **78** |
| **Nightly 总计** | **146** |

fallback 跨越 GQA、GEMM、线性 attention 和 Conv3d，不符合“某一个 GQA
kernel 自身错误”的分布特征。它更像共用 profiling/timing 基础设施的问题。

## 直接原因：CUPTI annotation window 投影不完整

Nightly 对应版本的 `bench_kernel` 使用 `torch.profiler`/Kineto 获取 CUPTI
device events，并用 `record_function("tileops_bench_kernel")` 标记每个 timed
repeat。随后 `_sum_kernel_time_us`：

1. 从 Kineto events 中收集投影到 CUDA timeline 的 user annotation windows；
2. 只累计落在这些 window 内的 CUDA kernel duration；
3. 返回累计 kernel time 和投影出的 window 数 `n_regions`。

每个 trial 默认执行 `n_repeat = 50`。实现要求：

```text
n_regions == n_repeat
```

只要不相等，就抛出 `_CuptiProjectionError`，清空 CUPTI trial results，并将
`_bench_meta.timing` 设为 `cuda-events` 后重新计时。因此，**可以从代码和
`timing = cuda-events` 标记确定的直接原因，是 Kineto trace 没有为每个
repeat 提供完整的 projected annotation window**。

这条校验是合理的保护：如果继续使用不完整的 CUPTI trace，平均 latency
会因为分子只包含部分 kernel、分母仍使用完整 repeat 数而被低估。

## 上游原因归因：runner 上的 profiler projection 不稳定

目前最符合证据的上游归因是：**Nightly runner 上
`torch.profiler`/Kineto/CUPTI 的 annotation-to-device projection
出现不稳定或不完整**。

支持这一归因的事实：

- fallback 集中出现于共用 `bench_kernel` 的多个无关算子族；
- fallback 的唯一捕获异常是 `_CuptiProjectionError`，CUDA runtime error
  和 OOM 不会被该逻辑吞掉；
- 同一 report 中并非所有结果都标成 `cuda-events`，说明 CUPTI 并非全局
  完全不可用，而是部分测量的 window-count invariant 被破坏；
- runner 环境为 H200、Torch `2.10.0+cu129`、CUDA 12.9、driver
  `575.57.08`。这些是复现条件，不足以单独证明某一版本组件是根因。

但现有 artifact **没有保留每次失败的 `n_regions/n_repeat` 和 captured
CUDA kernel count**。原因是 mismatch 详情只写到 logger；pytest 对通过
用例的 captured logging 不会进入当前上传的主日志/report。因此现在无法
进一步区分：

- annotation window 完全未投影（例如 `0/50`）；
- 偶发丢失少量 window（例如 `49/50`）；
- 特定 callable/多 kernel 结构导致稳定投影缺失；
- runner 长时间运行后 profiler 状态退化。

上述四项都应视为待验证假设，而不是已经确定的根因。

## 数据影响

CUDA Events fallback 测量的是 start/end event 之间的 elapsed time，不再是
CUPTI trace 中筛出的纯 kernel duration。当前实现自己的 warning 将两者差异
描述为约 50–60 μs/call 的 launch overhead；这个量级仍需用建议的最小复现
矩阵独立验证，不应仅凭 warning 当作 runner 的实测常数。

影响最大的是微秒级 kernel：

- latency 会系统性偏高；
- TFLOPS、bandwidth 会系统性偏低；
- 与 CUPTI 历史值比较可能产生假 regression；
- tileops 与 baseline 若走不同 timing backend，ratio 也不可比。

因此 `timing` 字段应作为 benchmark 数据有效性的一部分，而不是普通说明列。

## 建议

### P0：让 Nightly 默认拒绝 fallback 数据

在正式 Nightly 设置：

```bash
TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=0
```

一旦 CUPTI 投影不完整就让对应 benchmark 失败，避免把不可比数据发布到
perf history。开发机或诊断 run 仍可显式设为 `1`。

### P0：持久化 mismatch 诊断

将以下字段写进 JUnit properties、benchmark result 或独立 JSON，而不是只写
pytest logger：

- `timing_backend`
- `cupti_projected_regions`
- `cupti_expected_regions`
- `cupti_cuda_kernels_captured`
- workload/op/baseline tag
- trial index

这样下一次 Nightly 可以直接按 failure signature 聚类，而不是只看到
`cuda-events` 结果。

### P1：报告层隔离不可比数据

- nightly report 对 `timing != cupti` 显示醒目的 invalid/uncomparable 状态；
- 不把 fallback latency 写入历史 best/regression baseline；
- 汇总 fallback 总数和按 op/backend 的分布；
- 若 fallback 数大于 0，让 publish job 至少发出强 warning。

### P1：最小复现矩阵

从本次 fallback 中各选一例 GQA、Grouped GEMM、FP8 GEMM 和 DeltaNet，
固定 workload，在同一 runner 进程中重复执行并记录 `n_regions`：

1. 单独运行；
2. 按 Nightly 文件顺序运行；
3. 新 Python 进程与长驻 Python 进程对比；
4. `n_repeat = 1/10/50` 对比；
5. 同版本环境在另一台 H200 runner 对比。

该矩阵可以区分 workload 相关、进程寿命相关和 runner 相关问题。

## 可复核的统计口径

下载 run #187 的 benchmark artifact 后，对拼接了
`profile_run.log summary` 的日志执行：

```bash
rg -c '\| cuda-events \|' tileops_benchmarks.log
```

结果为：

```text
146
```

按最近的 `## <section>` 标题聚合即可得到上表。统计应只匹配表格单元格
`| cuda-events |`，避免把 workload 名中的 `fallback` 或说明文字计入。
