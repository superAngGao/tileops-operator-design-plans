# GQA Decode Kernel 参数建模与 Dispatch 策略计划

日期：2026-06-09

## 目标摘要

GQA decode kernel 的性能调优本质上是一个参数驱动的算法选择问题。不同模型、
不同 batch 形态、不同 KV 长度和不同 GQA group 会改变 kernel 的最佳算法形态：
有些区间适合单 CTA decode，有些区间需要 split-KV；有些区间 Tensor Core path
有优势，有些区间 small-M path 或更轻量的调度方式更合适。

因此，本阶段目标不是证明某一个 kernel 在所有场景中最优，而是建立一套可测量、
可解释、可落地的 dispatch 策略：

```text
给定 GQA decode 参数 -> 选择合适算法族 -> 选择对应 kernel / split / chunk 策略
```

第一阶段先聚焦小 batch、dense KV、contiguous KV 的 decode 场景，覆盖 Llama 4
和 Qwen3.5 两类代表性 workload，得到初步 dispatch policy。后续再把方法推广到
varlen、paged KV、shared-prefix 和 serving scheduler 等更复杂场景。

## 问题定义

GQA decode 的复杂性首先来自输入参数空间，而不是某个具体实现细节。需要建模的
核心参数包括：

```text
B: batch size
S: KV sequence length
Hq: query heads
Hkv: KV heads
group = Hq / Hkv
D: head dimension
dtype: fp16 / bf16 / fp8
KV organization: dense / paged
KV continuity: contiguous / varlen
KV layout: NHD / HND
attention semantics: full / sliding-window / softcap 等
```

这些参数会导致算法多样性。例如：

```text
S 较短:
    launch overhead、tile overhead、workspace overhead 更容易主导
    no-split 或轻量 small-M path 可能更合适

S 较长:
    KV 主循环成本变大
    split-KV / chunk parallelism 可能更重要

group 较小:
    每个 KV head 对应的有效 query rows 较少
    large-M Tensor Core tile 未必是最佳选择

group 增大:
    Tensor Core / MMA / WGMMA 的利用率可能提高
    但 Hkv、CTA 数量、combine 成本也会改变整体收益

dense contiguous KV:
    可以先用规则网格和固定 split 建模

varlen / paged KV:
    需要 runtime plan、chunk table 和 load-balanced scheduling
```

这里要特别区分两类概念：

```text
参数变量:
    B, S, Hq, Hkv, group, D, dtype, KV format, layout, continuity

策略变量:
    backend choice, tile shape, split/chunk plan, CTA mapping,
    workspace format, combine method, scheduling mode, WS usage
```

前者是 workload 本身，决定问题形态；后者是我们为了适配不同参数而选择的算法和
实现策略。dispatch policy 要做的是从参数变量映射到策略变量。

## 第一阶段范围

第一阶段先限制在可控、可复现实验范围内：

```text
small batch
single-token decode
dense KV
contiguous KV
fp16
D = 128
Hopper GPU
```

选择这个范围有三个原因：

```text
1. 它贴近 Llama 4 / Qwen3.5 decode 的核心热点。
2. 它排除了 paged/varlen/serving scheduler 的额外干扰，便于先看清算法本身。
3. 它仍然包含足够多的策略变化：backend、split、chunk、WS、combine 都会变化。
```

本阶段覆盖两个主 workload：

```text
Llama 4:
    Hq = 40
    Hkv = 8
    group = 5
    D = 128

Qwen3.5-35B-A3B:
    Hq = 16
    Hkv = 2
    group = 8
    D = 128
```

`group=16` 暂时不作为第一阶段主 sweep。它可以作为后续扩展，用来覆盖更大的
Qwen3.5 MoE 变体。

## 策略空间

在第一阶段 dense contiguous decode 中，需要比较的策略包括：

```text
algorithm family:
    no-split decode
    split-KV decode
    fixed chunk decode

backend:
    CUDA-core / non-TC path
    MMA path
    WGMMA path
    FlashInfer TC reference
    FA3 reference

tiling:
    effective M / group-aware query tile
    block_N / KV tile
    num_warps / threads
    num_stages / buffer depth

split/chunk:
    num_split
    chunk_len
    equal split
    variable chunk plan

workspace and composition:
    partial_o dtype
    partial_lse dtype
    separate combine kernel
    fused or optimized combine candidate

scheduling:
    static grid
    fixed chunk table
    load-balanced plan candidate

WS:
    non-WS baseline
    producer/consumer WS kernel
```

这些策略不是复杂性的根因，而是候选解。建模的任务是判断它们分别适合哪些参数
区间。

## 方法

### 1. 定义参数空间

第一阶段参数空间：

```text
B in {1, 2}
S in {4K, 8K, 16K, 32K, 64K, 128K}
D = 128
dtype = fp16
KV = dense contiguous
layout = NHD
scenario in {llama4_g5_hkv8, qwen35_g8_hkv2}
```

候选策略参数：

```text
backend in {tileops_split, tileops_ws, fa3, flashinfer_tc}
num_split in {1, 2, 4, 8, 12, 15, 16, 24, 32}
block_N in {64, 128, 256} where supported
scheduler in {static_grid}
```

### 2. 建 microbenchmark sweep

benchmark 输出统一记录为 CSV/JSONL，至少包括：

```text
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
max_diff
reference_backend
environment metadata
```

输出目标不是只找单个最优点，而是得到曲线：

```text
latency vs S
latency vs num_split
latency vs chunk_len
latency vs group / scenario
WS vs non-WS crossover
FlashInfer / FA3 / TileOps crossover
combine cost vs num_split
```

### 3. 建解释性模型

第一版模型不追求精确解析预测，而是解释主要趋势和 crossover：

```text
T_total = T_main + T_combine + T_overhead
```

其中：

```text
T_main:
    KV 主循环成本，受 S、chunk_len、backend、tile shape 影响

T_combine:
    split partial states 的合并成本，受 Hq、D、num_split、workspace dtype 影响

T_overhead:
    launch、workspace 初始化、barrier、scheduler、WS 同步等固定或半固定成本
```

对于 split/chunk：

```text
chunk_len ~= S / num_split
num_tasks ~= B * Hkv * num_split
waves ~= ceil(num_tasks / SM_count)
tail_penalty ~= f(task_time_variance, scheduler)
```

对于 group-aware backend 选择：

```text
effective_M ~= group
M_utilization ~= effective_M / backend_M_tile
```

这些量不一定直接决定最终性能，但可以解释不同策略的优势来源：

```text
small S:
    overhead 和 tile mismatch 更重要

large S:
    chunk parallelism 和 memory/Tensor Core throughput 更重要

small group:
    small-M path 或更细粒度 tile 可能更有优势

larger group:
    MMA/WGMMA 的利用率可能改善，但还要看 Hkv 和 split 后任务数
```

### 4. 找 dispatch 边界

通过 sweep 和模型得到以下边界：

```text
no-split vs split-KV
TileOps split vs FlashInfer TC
TileOps split vs FA3
WS vs non-WS
num_split 最优区间
block_N 最优区间
```

第一版 dispatch policy 可以是 piecewise rules：

```text
if scenario == llama4_g5_hkv8:
    choose policy_from_llama4_model(S, B, num_split_candidates)
elif scenario == qwen35_g8_hkv2:
    choose policy_from_qwen35_model(S, B, num_split_candidates)
else:
    choose conservative baseline
```

更细的规则必须来自实测数据，而不是预设判断。

## 当前 AKO-style 调优结论

本阶段要明确区分两类收益：

```text
kernel 本体收益:
    某个 non-WS / WS kernel implementation 本身比 upstream 更快。

dispatch policy 收益:
    在不同 scenario / S / split / backend / WS 条件下，
    选择更合适的执行路径。
```

截至当前实验，AKO-style 方法没有证明本地 non-WS kernel 本体全面强于 upstream；
本地 non-WS 在 fp16 partial fast path 下与 upstream-style split path 基本对齐。

但仅就当前 dense contiguous GQA decode case 而言，AKO-style 方法带来了比常规单
kernel autotune 更高的实际收益。原因是主要优化机会不在单个 kernel 内部 knob，
而在 dispatch / policy 空间。

具体收益：

```text
Qwen3.5 group=8:
    upstream/default split=16
    AKO-style policy 选择 split=32

Compared with upstream/default split=16:
    S=4K:    1.19x
    S=8K:    1.17x
    S=16K:   1.33x
    S=32K:   1.57x
    S=64K:   1.77x
    S=128K:  1.91x
```

WS 的有效区间也被收窄为：

```text
Qwen3.5 group=8:
    S=4K:   tileops_ws split=32
    S>=8K:  tileops_split split=32

Llama4 group=5:
    S=4K-128K: tileops_split split=16
    WS:        当前不默认启用
```

因此，当前结论应表述为：

```text
AKO-style 调优在本 case 中胜过常规单 kernel autotune，
不是因为它带来了更强的 non-WS kernel，
而是因为它发现并固化了更好的 dispatch policy。
```

## FlashInfer 论文对本阶段的启发

FlashInfer 对本阶段最有价值的不是某个单独 kernel，而是参数化 dispatch 和 runtime
planning 的方法。

可迁移到当前阶段的内容：

```text
1. group-aware query tile policy
2. backend selection
3. variable chunk / split planning
4. attention-state composition: partial output + logsumexp
5. measurement-driven dispatch modeling
```

需要记录但暂不作为第一阶段实现目标的内容：

```text
6. load-balanced scheduling for varlen / paged / mixed requests
7. KV-cache format composition
8. attention variant JIT
```

其中 load-balanced scheduling 很重要，但它主要解决 varlen、paged KV、mixed request
下的任务不均衡问题。dense contiguous B1/B2 阶段可以先用 `static_grid` 和 fixed
chunk plan 建模，同时在 benchmark schema 中保留 `scheduler` 字段，方便后续扩展。

## 当前观测

已经观察到的 Llama 4 `Hq=40, Hkv=8, group=5` 现象：

```text
S=4K:  FlashInfer TC 快于 TileOps split/WS
S=8K:  FlashInfer TC 接近 FA3，并快于 TileOps split/WS
S=32K: TileOps split-KV 快于 FlashInfer TC
S=64K: TileOps split-KV 明显快于 FlashInfer TC
S=128K: 待补测
```

当前 S64K 数据：

```text
FlashInfer TC: 0.1284 ms
FA3 split15:   0.1059 ms
TileOps split: 0.0814 ms
TileOps WS:    0.0856 ms
```

其他观测：

```text
FlashInfer non-TC single decode 不支持 group_size=5。
当前 TileOps WS correctness 已通过，但大多数场景还没有赢 non-WS。
```

已经观察到的 Qwen3.5 型 `Hq=16, Hkv=2, group=8` 现象：

```text
S=4K:  TileOps WS 略快于 non-WS
S=8K:  TileOps WS 略快于 non-WS
S=32K: TileOps WS 略慢于 non-WS
```

当前 group=8 / Hkv=2 数据：

```text
S=4K:  TileOps split 0.0128 ms, TileOps WS 0.0125 ms
S=8K:  TileOps split 0.0186 ms, TileOps WS 0.0184 ms
S=32K: TileOps split 0.0352 ms, TileOps WS 0.0359 ms
```

这些数据说明：

```text
1. split-KV 在长 S 区间有明显价值。
2. WS 目前还没有形成稳定优势。
3. group=5 与 group=8 的 crossover 可能不同。
4. 需要补齐 S=16K/64K/128K、num_split、block_N 的系统曲线。
```

## 工作计划

### Step 1. 统一 benchmark harness

实现或整理 dense contiguous GQA decode benchmark，保证 TileOps、FA3、FlashInfer
在同一参数、同一 reference、同一 GPU 上测量。

### Step 2. 补齐参数 sweep

对两个主 scenario 执行 sweep：

```text
scenario in {llama4_g5_hkv8, qwen35_g8_hkv2}
B in {1, 2}
S in {4K, 8K, 16K, 32K, 64K, 128K}
num_split in {1, 2, 4, 8, 12, 15, 16, 24, 32}
block_N in {64, 128, 256}
backend in {tileops_split, tileops_ws, fa3, flashinfer_tc}
```

### Step 3. 分析模型与 crossover

分析：

```text
best num_split vs S / scenario
best backend vs S / scenario
WS vs non-WS crossover
FlashInfer / FA3 / TileOps crossover
combine cost 是否成为瓶颈
```

### Step 4. 形成第一版 dispatch policy

把 sweep 结果整理成规则表：

```text
scenario
S range
B range
selected backend
selected num_split
selected block_N
use_ws
expected latency
reference comparison
```

这张表作为后续 TileOps kernel dispatch 实现的依据。

### Step 5. 决定下一轮 kernel 实现

根据 policy surface 决定下一轮优先实现：

```text
small-M / group-aware decode path
improved split planner
optimized combine
WS 只在有明确 crossover 的区间启用
load-balanced scheduler for varlen/paged later
KV-cache format composition later
attention variant JIT later
```

## 当前非目标

第一阶段暂不处理：

```text
完整 paged KV support
varlen mixed-request scheduling
shared-prefix format composition
RoPE/logits-softcap/sliding-window variant explosion
serving-level scheduler integration
```

这些方向都重要，但应在 small-batch dense contiguous dispatch 模型跑通后再展开。

## 分享报告提纲

题目候选：

```text
AKO-style Kernel Dispatch Tuning for GQA Decode
```

核心信息：

```text
1. 这不是一个“生成更强 kernel”的故事。
2. 这是一个“找到更强 dispatch policy”的故事。
3. 在当前 case 中，收益主要来自 Qwen3.5 group=8 的 split=32 policy。
4. WS 不是通用替代路径，只在 Qwen S=4K split=32 上进入默认候选。
5. 后续 kernel 优化要基于 policy surface 选择方向，而不是靠单点直觉。
```

建议结构：

```text
1. Problem:
    GQA decode 的性能取决于 workload 参数和策略选择，不是单个 kernel knob。

2. Method:
    用 AKO-style loop 建立 workload / backend / split / WS 的可审计 sweep。

3. Result:
    Llama4 -> split=16
    Qwen3.5 -> split=32
    Qwen3.5 S=4K -> WS split=32

4. AKO vs autotune:
    常规 autotune 优化单 kernel 局部参数；
    AKO-style 调优找 dispatch boundary；
    本 case 的收益主要在后者。

5. Lessons:
    不要把 policy gain 误读成 kernel gain；
    不要把 WS 当作全局 replacement；
    先建立 policy surface，再决定下一轮 kernel variant。

6. Next:
    实现第一版 dispatch rule；
    整理最小 correctness test；
    评估 combine / WS pipeline / small-M path 等下一轮 variant。
```
