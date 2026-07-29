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

## 为什么误差量化与历史污染评估是关键证据

fallback 条数只能回答“问题覆盖多广”，不能回答下面两个直接影响工程决策的
问题：

1. fallback 带来的偏差是否越过 nightly 10% regression threshold；
2. 历史 best、regression、improvement 和 baseline ratio 中有多少结论不可用。

如果偏差只有 measurement noise 量级，可以保留结果并扩大阈值；如果偏差达到
数倍，则不能靠改阈值解决，必须阻止不同 timing backend 的数据进入同一比较
集合。当前证据属于后一种情况。

### 已确认的历史污染

对比以下两个成功 Nightly：

| Run | Commit | `cuda-events` report rows |
| --- | --- | ---: |
| [#183](https://github.com/tile-ai/TileOPs/actions/runs/30170991562) | `7892d02` | 2,835 |
| [#187](https://github.com/tile-ai/TileOPs/actions/runs/30417463015) | `db4494c` | 146 |

run #183 的 fallback latency 已写入后续使用的 `perf_history.json`。但 history
中的每条 config 只保留 latency/throughput 等数值，**没有保留
`timing_backend`**。因此下游无法区分 CUPTI 与 CUDA Events，仍会把两者直接
用于 14-day best 和 regression 计算。

run #187 的正式 `nightly_report.md` 已表现出这种污染：

- 报告了 **602 项 improvement**；其中大量 80%–97% 的“提升”是旧的
  CUDA Events 高延迟与新的 CUPTI 低延迟之间的 backend 切换；
- 报告了 **38 项 regression**；逐项回查当前 report 后，其中 **34 项当前值
  标记为 `cuda-events`**，因此至少 34 项不能作为可信的 kernel regression；
- 38 项 regression 中剩余 4 项没有 `cuda-events` 标记，仍需按正常性能变化
  调查。

将 run #183 与 run #187 的 perf history 按相同 op + config 名称配对，可得到
886 对记录。旧值/新值的描述性统计为：

| 指标 | 结果 |
| --- | ---: |
| 配对记录 | 886 |
| 旧值/新值中位数 | 2.13x |
| 旧值/新值平均数 | 4.03x |
| 大于等于 2x | 468 |
| 大于等于 5x | 180 |
| 最大值 | 38.65x |

这不是严格的 timing-backend A/B，因为两个 run 的 commit 不同；它不能把每一
对差异都归因于 CUDA Events。但结合 run #183 的 2,835 个
`cuda-events` 标记，它足以证明当前 history/report 已把 backend 变化大规模
解释成 kernel 性能变化。

L2Norm 的相同 workload 给出了容易复核的例子：

| Workload | run #183 CUDA Events | run #187 CUPTI | 旧值/新值 |
| --- | ---: | ---: | ---: |
| hidden-state fp16 | 0.0665 ms | 0.0098 ms | 6.79x |
| hidden-state bf16 | 0.0658 ms | 0.0092 ms | 7.15x |
| long-seq bf16 | 0.0646 ms | 0.0056 ms | 11.54x |

这三对仍包含跨 commit 的混杂因素，因此是历史污染实例，不替代同 commit
受控实验。不过差异已经远超 10% regression threshold。

### 当前能确定与不能确定的边界

可以确定：

- CUDA Events 数据已经进入 perf history；
- history 丢失 timing backend，无法在下游自动隔离；
- backend 改变已经制造大量不可解释为 kernel change 的 improvement；
- 最新 38 项 regression 中至少 34 项使用了不可比的当前 fallback 值。

不能仅靠历史 run 确定：

- 886 对记录中每一对的纯 backend bias；
- bias 与真实 kernel duration 的精确函数关系；
- single-kernel 与 multi-kernel callable 的 bias 是否相同；
- 哪些历史 best 是 CUPTI，哪些历史 best 是 CUDA Events。

## 如何收集可归因的误差数据

需要两条互补的数据线：同 commit 受控 A/B 用于量化 measurement bias；历史
artifact audit 用于量化已污染的决策。

### 1. 同 commit、同进程、同 workload 的受控 A/B

固定 Nightly commit、runner image、H200、clocks、输入、warmup、L2 flush、
`n_repeat` 和 `n_trials`。对每个 workload 在同一 Python 进程内交替执行：

```text
CUPTI -> forced CUDA Events -> CUDA Events -> CUPTI
```

forced CUDA Events 只应改变 timing backend，不改变 callable、输入池或 cache
protocol。每个 workload 至少做 5 个成对 trial，并保存原始 trial values，
而不只保存 median。

采样矩阵应按真实 kernel duration 分层：

| 层 | CUPTI latency | 目的 |
| --- | ---: | --- |
| ultra-fast | < 5 μs | 观察固定开销主导区 |
| fast | 5–20 μs | 覆盖 decode、norm、elementwise |
| medium | 20–100 μs | 覆盖短 GQA/GEMM |
| long | 100 μs–1 ms | 观察相对误差衰减 |
| very long | > 1 ms | 验证 bias 是否趋近可忽略 |

每层同时选 single-kernel 与 multi-kernel callable。至少保留：

- op、config、commit、image、GPU/driver/clocks；
- timing backend；
- CUPTI `n_regions/n_repeat`、captured kernel count；
- 每个 trial 的 raw latency；
- kernel count per logical call；
- absolute bias（μs）、relative ratio、CV。

输出应报告 absolute bias 和 ratio 关于 CUPTI latency 的分布，而不是用一个
全局 “6–7x” 常数概括所有 workload。

### 2. 历史 artifact audit

在 artifact 过期前下载保留期内每次 Nightly 的：

- `tileops_benchmark_<run_id>/tileops_benchmarks.log`
- `tileops_perf_history/perf_history.json`
- `tileops_op_test_<run_id>/nightly_report.md`

从 benchmark report 恢复每个 op/config/backend 的 timing backend，与 history
记录 join。然后重新计算三份报告：

```text
raw report:
  保持当前行为，作为对照

cupti-only report:
  current 与 historical best 都只允许 timing=cupti

same-backend report:
  只比较 timing backend 相同的 current/history
```

比较三份报告的 regression、improvement、baseline alert 集合，输出：

- 被 invalidated 的 regression/improvement 数；
- 因 backend 缺失而无法判定的记录数；
- 每个 op family 的污染比例；
- 最早/最近一个可信 CUPTI best；
- 需要从 history quarantine 的 run/config。

### 3. 数据模型必须先补 timing provenance

后续每条 perf history record 至少需要：

```json
{
  "latency_ms": 0.0098,
  "timing_backend": "cupti",
  "timing_status": "valid",
  "fallback_reason": null
}
```

在完成历史 backfill 前，`timing_backend` 缺失的旧记录应标记为
`unknown`，不能默认当作 CUPTI。

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
