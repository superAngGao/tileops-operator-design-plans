# GQA Decode AKO-Style 算子调优计划

日期：2026-06-09

## 目标

本计划用于组织 GQA decode kernel 的持续调优工作。它借鉴 AKO 的核心思想：
不要把 kernel 优化做成一次性的临时实验，而是建立一个可重复、可审计、可积累的
闭环系统。

当前我们要解决的问题是：

```text
给定 GQA decode 参数空间，找到不同参数区间下的最佳算法族和 dispatch 策略。
```

因此，调优工作需要同时维护三类信息：

```text
1. 参数空间：B, S, Hq, Hkv, group, D, dtype, KV layout 等。
2. 候选方向：split-KV, block_N, backend, WS, combine, small-M path 等。
3. 实验轨迹：每轮假设、配置、结果、结论、下一步。
```

## AKO 给我们的启发

AKO 的价值不在于预设所有优化方向，而在于提供一套实验制度：

```text
baseline
profile / benchmark
选择方向
修改或生成 variant
重新 benchmark
记录 iteration
归档 variant
复盘 traps
进入下一轮
```

对应到 TileOps GQA decode，我们不直接照搬 AKO4X 的多 agent campaign，而是先做
一个轻量版本：

```text
明确方向表
统一 benchmark adapter
统一 JSONL result schema
维护 ITERATIONS.md
维护 TRAPS.md
维护 variants archive
定期生成实验报告
```

这套机制的作用是避免三类常见问题：

```text
1. 只记住单点最快结果，忘记参数区间。
2. 多个 backend / split / kernel variant 的结果不可比较。
3. kernel 改动、benchmark artifact、unsupported case 混在一起，无法复盘。
```

## 本 case 中 AKO 相比 autotune 的实际收益

本阶段要把 AKO 的收益口径限制在当前 case 内，不把排雷、归因或 correctness audit
包装成 kernel 本体收益。

当前结论：

```text
仅就 dense contiguous GQA decode 这个 case 而言：

AKO 带来了比常规单 kernel autotune 更高的实际收益。

原因不是 AKO 生成了一个全面强于 upstream 的 non-WS kernel，
而是它搜索和沉淀的是 dispatch / policy 空间：
    backend choice
    num_split choice
    WS vs non-WS boundary
    scenario-specific rule
```

与常规 autotune 的区别：

```text
常规单 kernel autotune:
    通常在已经选定的 kernel family 内搜索局部 knobs，
    例如 block_N、num_stages、threads、num_split 等。

本次 AKO-style 调优:
    把 workload scenario、backend、split、WS variant 和 upstream comparison
    放到同一张 policy surface 中比较，
    目标是选择正确策略，而不是只把单个 kernel tune 到局部最优。
```

当前 case 的实测收益主要来自 Qwen3.5 group=8 的 split policy：

```text
upstream/default policy: split=16
AKO-style policy:       split=32

Qwen3.5 group=8, compared with upstream/default split=16:
    S=4K:    1.19x
    S=8K:    1.17x
    S=16K:   1.33x
    S=32K:   1.57x
    S=64K:   1.77x
    S=128K:  1.91x
```

WS 的收益也被限定在明确区间：

```text
Qwen3.5 group=8:
    S=4K:   tileops_ws split=32 最优，约比 non-WS split=32 快 9%
    S>=8K:  tileops_split split=32 更优

Llama4 group=5:
    S=4K-128K: tileops_split split=16 更合适
    WS:        当前不默认启用
```

因此，分享时应使用下面这个表述：

```text
AKO 在本 case 中没有直接产出一个全面优于 upstream 的 GQA decode kernel；
它带来的收益是比常规单 kernel autotune 更高的 dispatch policy 收益。

这说明当前优化机会主要位于 policy / dispatch 层，
而不是 non-WS kernel 内部局部参数层。
```

## 实验目录结构

建议建立独立实验目录：

```text
experiments/ws_kernel_evolution/gqa_decode_policy/
  README.md
  WORKLOADS.md
  DIRECTIONS.md
  ITERATIONS.md
  TRAPS.md
  AUDIT.md
  scripts/
    benchmark_adapter.py
    run_microbench.py
    summarize_jsonl.py
    compare_variants.py
  results/
    raw/
    summary/
  reports/
    gqa_decode_microbenchmark_report.md
  variants/
    v001_upstream_split/
    v002_fp32_workspace/
    v003_ws_producer_consumer/
```

短期内可以不一次性实现所有文件，但命名和记录方式应按这个结构推进。

## Workloads

第一阶段只覆盖 small-batch dense contiguous decode：

```text
single-token decode
fp16
D = 128
dense KV
contiguous KV
layout = NHD
```

主 workloads：

```text
llama4_g5_hkv8:
    Hq = 40
    Hkv = 8
    group = 5

qwen35_g8_hkv2:
    Hq = 16
    Hkv = 2
    group = 8
```

主 sweep 参数：

```text
B in {1, 2}
S in {4K, 8K, 16K, 32K, 64K, 128K}
num_split in {1, 2, 4, 8, 12, 15, 16, 24, 32}
block_N in {64, 128, 256}
backend in {tileops_split, tileops_ws, fa3, flashinfer_tc}
```

## Directions

方向表需要显式维护。AKO 的方向更多来自 prompt、archive、profile 和 history；
我们这个问题有明确领域结构，因此应该把方向预先分层。

### Level 0. 不改 kernel 的参数 sweep

目标：先建立 latency surface。

```text
S sweep
num_split sweep
block_N sweep
backend comparison
WS vs non-WS
FA3 num_splits comparison
FlashInfer TC comparison
```

产出：

```text
best num_split vs S
best backend vs scenario
TileOps vs FA3 crossover
TileOps vs FlashInfer crossover
WS vs non-WS crossover
unsupported cases
```

### Level 1. 局部 kernel variant

目标：围绕已有 split-KV / WS kernel 做低风险改动。

```text
fp32 partial_lse / partial_o workspace
combine kernel optimization
block_N-specific specialization
num_split-specific specialization
WS pipeline tuning
producer / consumer balance
threads / num_stages tuning
```

每个 variant 必须记录：

```text
variant id
changed files
hypothesis
expected improvement region
correctness status
benchmark result
decision: keep / reject / defer
```

### Level 2. 算法族变化

目标：探索新的 algorithm family。

```text
small-M decode path
MMA path
WGMMA path
group-aware query tiling
variable chunk planner
optimized attention-state composition
```

这些方向需要更强的建模依据，不能只因为某个单点慢就直接开写。

### Level 3. 后续扩展

暂缓到 dense contiguous dispatch model 跑通以后：

```text
paged KV
varlen request
load-balanced scheduling
shared-prefix
sliding-window / softcap / RoPE variants
serving-level scheduler integration
```

## Benchmark Adapter Contract

microbenchmark 应该逐步收敛成 adapter contract，而不是一个临时脚本。

建议接口：

```text
list_workloads()
list_backends()
make_inputs(workload)
make_backend(workload, backend_config)
run(workload, backend_config)
profile(workload, backend_config)
check(workload, backend_config)
pack_result(...)
```

每条 result 使用统一 schema：

```text
type: result / environment / skip / error
status: ok / unsupported / failed / invalid
scenario
B
S
Hq
Hkv
group
D
dtype
layout
backend
block_N
num_split
chunk_len
scheduler
use_ws
latency_ms
tflops
bandwidth_tbs
max_diff
skip_reason
error_log
git_commit
variant_id
environment
```

关键要求：

```text
1. Unsupported case 不能中断 sweep，必须记录 skip_reason。
2. FA3 必须显式传 num_splits，避免 reference 不可比。
3. FlashInfer group 不支持时记录 unsupported，而不是混入 failed。
4. 每个 backend 使用同一组输入和同一套 timing protocol。
5. correctness check 可以抽样，但必须可打开。
```

## Iteration Log

每一轮实验都要维护 `ITERATIONS.md`。建议模板：

```text
## Iteration N: 标题

Date:
Variant:
Direction:

Hypothesis:
    为什么做这一轮。

Config:
    workload / backend / split / block_N / S / B / timing protocol。

Result:
    关键表格或 JSONL 路径。

Interpretation:
    结果说明了什么，是否支持 hypothesis。

Decision:
    keep / reject / defer / rerun。

Next:
    下一轮要做什么。
```

原则：

```text
1. 跑完一轮先记录，再进入下一轮。
2. 结果不理想也要记录，失败方向是重要资产。
3. 不把单次异常点直接写成结论，需要 rerun 或标记 uncertainty。
```

## Variant Archive

每个 kernel 方向进入 `variants/`：

```text
variants/v003_ws_producer_consumer/
  README.md
  PATCH_SUMMARY.md
  correctness.md
  benchmark_summary.md
  status.md
```

`status.md` 使用固定状态：

```text
active
kept
rejected
deferred
superseded
```

variant README 至少记录：

```text
目的
修改点
预期收益区间
实测收益区间
失败原因或限制
后续方向
```

## Traps

维护 `TRAPS.md`，记录所有会误导结论的问题。

当前已知 traps：

```text
FA3 不显式传 num_splits 会导致对比不公平。
FlashInfer TC single decode 不支持 group_size=5。
TileLang JIT 在当前机器需要 TILELANG_CLEANUP_TEMP_FILES=1。
不同 timing protocol 的结果不能混在同一张 dispatch 表里。
WS correctness 通过不代表性能会赢 non-WS。
单点最快不能外推到整个 S / group 区间。
```

## Audit

每次形成结论前做一次 audit：

```text
1. 是否同一 GPU / 同一 Docker / 同一 timing protocol？
2. 是否记录 git commit 和 variant id？
3. FA3 / FlashInfer 是否使用正确 interface？
4. unsupported case 是否被正确标记？
5. correctness 是否覆盖代表点？
6. 是否有 rerun 或误差范围？
7. 结论是否只覆盖已测参数区间？
```

如果 audit 不通过，结论只能写为 observation，不能写为 dispatch rule。

## 第一阶段执行计划

### Step 1. 修好 microbenchmark smoke

当前要先确保：

```text
TileOps split 可以输出 JSONL
TileOps WS 可以输出 JSONL
FA3 显式 num_splits
FlashInfer unsupported group=5 记录 skip_reason
Qwen3.5 group=8 FlashInfer 可正常测
```

### Step 2. 跑 Level 0 sweep

先跑：

```text
scenario in {llama4_g5_hkv8, qwen35_g8_hkv2}
B = 1
S in {4K, 8K, 16K, 32K, 64K, 128K}
num_split in {1, 2, 4, 8, 12, 15, 16, 24, 32}
block_N = 128
backend in {tileops_split, tileops_ws, fa3, flashinfer_tc}
```

再决定是否扩展：

```text
B = 2
block_N in {64, 256}
```

### Step 3. 总结第一张 dispatch 表

输出：

```text
scenario
S range
best backend
best num_split
WS decision
reference comparison
confidence
```

### Step 4. 决定 Level 1 variant

根据 Level 0 数据选择下一轮 kernel 改动，而不是提前假设。

候选：

```text
combine optimization
block_N-specific specialization
WS pipeline tuning
small-M path prototype
```

## 成功标准

第一阶段成功不以“某个 kernel 打败所有 baseline”为标准，而以以下结果为标准：

```text
1. 有可复现的 microbenchmark harness。
2. 有覆盖 Llama4 / Qwen3.5 的 JSONL 数据。
3. 有明确 unsupported / failed / ok 状态记录。
4. 有第一版 dispatch 表。
5. 有至少一轮从数据出发选择 kernel variant 的闭环记录。
```

这时我们才进入下一阶段 kernel 实现，而不是继续靠单点直觉调参。
