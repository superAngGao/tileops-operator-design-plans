# TileOps Benchmark Timing 技术报告：NV SOL 对齐与改进路线

更新日期：2026-08-03

## 摘要

TileOps 最初希望 benchmark 基础设施对齐 NVIDIA SOL-ExecBench：固定 warmup/repeat/trial、L2 flush、输入地址扰动、锁频和稳定报告口径。实际落地时，TileOps 采用了 `torch.profiler` / Kineto 暴露的 CUPTI CUDA activity，并通过 `record_function` projection 判断每个 repeat 的 GPU 计时窗口。

重新核对 NV SOL 代码后，需要纠正一个历史混淆：NV SOL 当前默认 timing methodology 是 **native CUPTI activity timing**。当前 TileOps 的主要问题集中在 Kineto projection 依赖：当 projected annotation window 数不是 `n_repeat` 时，benchmark fallback 到 CUDA events。CUDA events 对 fast kernels 会带入 launch overhead，使 nightly latency 和 roofline 数据不可直接与 CUPTI kernel-only history 混合比较。

本轮实验进一步确认：当两条路径都测到 single-kernel CUPTI activity 时，Kineto 和 SOL native 的 latency 基本吻合；主要差异来自 Kineto 大量 fallback 到 CUDA events。全量 benchmark 与同 case 四路计时对比数据见 [GPU benchmark 方法调研笔记 §8](archive/benchmark-methods-survey-note-cn.md#8-tileops-四路计时实验事实)。

当前决策是：**以 SOL native CUPTI 作为 TileOps benchmark 主路径；使用 native CUPTI discovery 识别 kernel name/count；single-kernel op 使用 CUPTI kernel duration；multi-kernel op 先保守 fallback 到 CUDA events**。这样避免继续依赖 Kineto projection，同时避免把 multi-kernel duration sum 误当成 operator latency。

## 1. NV SOL 的具体流程

参考实现：

- NVIDIA SOL-ExecBench timing code: https://github.com/NVIDIA/SOL-ExecBench/blob/main/src/sol_execbench/core/bench/timing.py
- CUPTI utility code: https://github.com/NVIDIA/SOL-ExecBench/blob/main/src/sol_execbench/core/bench/cupti_utils.py

NV SOL 的默认 timing backend 是 `methodology="cupti"`。它使用 Python CUPTI binding 开启 GPU activity collection，主要采集：

```text
CONCURRENT_KERNEL
MEMCPY
MEMSET
```

这三类 activity 覆盖了 benchmark 中最常见的 GPU work：kernel launch、显式/隐式 memcpy，以及 memset/clear cache 相关操作。SOL 后续会从这些 activity 中识别用户调用对应的序列，并用这个序列进行正式计时归因。

### 1.1 Iteration 准备

每个 iteration 前，NV SOL 会执行 `prepare_iteration()`：

```text
setup args
reset persisting L2 cache
clear L2 cache buffer
optional torch.cuda.synchronize()
```

这一步把“准备输入”和“准备硬件状态”收在同一个入口里。`setup args` 允许每次 iteration 使用新参数或新地址；reset persisting L2 cache 和 clear L2 cache buffer 让每次测量尽量从一致的 cache 状态开始；可选 synchronize 用来在需要时把准备阶段的 GPU work drain 掉，避免它和后续用户调用混在一起。

### 1.2 Warmup

正式测量前，NV SOL 会先执行 warmup：

```text
torch.cuda.synchronize()
repeat warmup:
  prepare_iteration()
  runner(args)
torch.cuda.synchronize()
```

warmup 的目的不是计时，而是让 JIT、autotune、library initialization 和冷启动状态尽量退出正式测量窗口。

### 1.3 Discovery iteration

正式计时前，NV SOL 先单独跑一次 discovery：

```text
prepare_iteration()
collect CUPTI activities:
  runner(args)
  torch.cuda.synchronize()
```

然后从这一次用户调用中得到 expected activity sequence：

```text
expected_kernel_names
expected_kernel_counts
```

这一步的目的是建立“用户调用应该长什么样”的 GPU activity 指纹。后续正式计时不会盲目使用一个 window 内的全部 activity，而是尝试从 window 内选出和 discovery 一致的 activity sequence。这样可以把 setup/cache/helper noise 和被测用户调用区分开。

### 1.4 Timed iterations

正式计时时，NV SOL 在同一个 CUPTI collection window 中跑 `rep` 次。每次 logical iteration 的 CPU timestamp window 用 CUPTI 自己的 timestamp API 标记：

```text
start_cpu = cupti.get_timestamp()
runner(args)
torch.cuda.synchronize()
end_cpu = cupti.get_timestamp()
```

这里的 `start_cpu` / `end_cpu` 使用 CUPTI timestamp，和 CUPTI activity 使用同一套时间基准。`end_cpu` 放在 `torch.cuda.synchronize()` 之后，目的是确保本次 call 发出的 CUDA work 已经完成，窗口内可以包含本次 logical iteration 的完整 GPU activity。

### 1.5 Attribution 与 latency

CUPTI collection 结束后，NV SOL 会把所有 GPU activity 按 start/end/correlation id 排序。对每个 iteration：

```text
1. 取 start 落在 [start_cpu, end_cpu] 的 activity
2. 在这个 window 内查找 discovery 得到的 expected sequence
3. 要求 activity counts 与 discovery counts 一致
4. latency = max(activity.end) - min(activity.start)
```

这里有两个关键设计点：

1. **用 sequence 做归因**：正式计时只接受和 discovery 一致的 activity sequence；如果找不到预期序列，说明本次测量窗口存在无法解释的噪声或 dispatch 变化，应显式失败。
2. **用 GPU span 表示 latency**：latency 是这组 activity 的 `max(end) - min(start)`，不是把 kernel duration 简单相加。对于 multi-kernel logical call，这更接近 GPU 侧 operator latency；对于 single-kernel logical call，它退化为单个 kernel duration。

## 2. TileOps Benchmark 的发展脉络

### 2.1 对齐 SOL 的初始目标

TileOps 的 benchmark overhaul 由以下 issue / PR 推进：

| 链接 | 作用 |
| --- | --- |
| [tile-ai/TileOPs#689](https://github.com/tile-ai/TileOPs/issues/689) | benchmark infrastructure overhaul umbrella |
| [tile-ai/TileOPs#692](https://github.com/tile-ai/TileOPs/issues/692) | 引入 `bench_kernel()`，固定 warmup/repeat/trial |
| [tile-ai/TileOPs#697](https://github.com/tile-ai/TileOPs/pull/697) | 合入 SOL-style benchmark protocol |

#697 对齐了 SOL 的外层 protocol：

```text
10 warmup
50 timed repeats
3 trials
L2 flush
input clone / address diversity
locked clocks
median trial result
```

但 #697 没有采用 NV SOL 当前代码中的 native CUPTI sequence attribution。它选择的是 PyTorch profiler/Kineto 路径：

```text
torch.profiler.profile(activities=[CUDA])
  -> kineto_results.events()
  -> sum CUDA kernel duration
```

### 2.2 为什么使用 Kineto

当时使用 Kineto 的主要考虑是低成本集成：

```text
Python benchmark closure
  -> torch.profiler.profile
  -> raw Kineto C++ events
  -> CUDA kernel name / duration
```

优点是：

- 可以直接嵌入现有 pytest benchmark；
- 不需要维护 native CUPTI buffer callback / flush / finalize 逻辑；
- 可以通过 `record_function` 给每个 repeat 建立看似自然的窗口；
- 更快：Kineto 路径主要省掉了 SOL native 里的 per-repeat attribution / validation 工作，包括 timestamp window slicing、sequence matching 和 count validation；详细对比见 [调研笔记 §8.1](archive/benchmark-methods-survey-note-cn.md#81-kineto-与-sol-native-的流程差异)。
- 绕过 `key_averages()` 后，raw Kineto event iteration 解析开销显著下降。

相关记录：

| 链接 | 内容 |
| --- | --- |
| [tile-ai/TileOPs#690](https://github.com/tile-ai/TileOPs/issues/690) | 绕过 `key_averages()`，直接遍历 raw C++ Kineto events |
| [tile-ai/TileOPs#691](https://github.com/tile-ai/TileOPs/issues/691) | 说明大 `n_repeat` 下 CUPTI event count / backend behavior drift，推动固定 repeat |

### 2.3 CUDA event fallback 的来源

需要明确：main 上的 CUDA event fallback 不是未合入的 #1576 带来的，而是 #697 原始实现就已经带入。

#697 合入时的逻辑是：

```python
try:
    run torch.profiler CUPTI path
except RuntimeError:
    pass

if not trial_means:
    run CUDA event fallback
```

当时 docstring 写的是：

```text
Falls back to CUDA events if CUPTI is unavailable.
```

所以早期 fallback 的定位是工程兜底：当 `torch.profiler` / CUPTI 不可用、抛 `RuntimeError` 或没有产出 trial results 时，让 benchmark 继续跑完。它不是为 projection mismatch 设计的精确策略。

### 2.4 后续暴露的问题

合入后，Kineto 路径陆续暴露出几类 benchmark infrastructure 问题：

| 链接 | 问题 |
| --- | --- |
| [tile-ai/TileOPs#693](https://github.com/tile-ai/TileOPs/issues/693) | session-scoped `warmup_cupti` fixture 和 `bench_kernel()` 内部 profiler 冲突 |
| [tile-ai/TileOPs#713](https://github.com/tile-ai/TileOPs/pull/713) | 删除 shared `cupti_session`；nested profiler 曾导致小 kernel 约 7x overhead |
| [tile-ai/TileOPs#1723](https://github.com/tile-ai/TileOPs/issues/1723) / [#1725](https://github.com/tile-ai/TileOPs/pull/1725) | L2 flush kernel 可能被 Kineto projection 纳入 timed window；通过 flush 后同步隔离 |
| [tile-ai/TileOPs#1745](https://github.com/tile-ai/TileOPs/pull/1745) | benchmark failure / fallback 行为开始显式记录，避免静默降级 |
| [tile-ai/TileOPs#1792](https://github.com/tile-ai/TileOPs/pull/1792) | 为 CUPTI projection failure 增加 fallback control 和诊断 |

这些问题的共同点是：它们不是具体 kernel 算法错误，而是 PyTorch/Kineto profiler lifecycle、projection window 和 benchmark 计时语义之间的耦合。

## 3. TileOps 当前形式

当前 TileOps `bench_kernel()` 的 CUPTI 路径可简化为：

```python
for i in range(n_repeat):
    cache.zero_()
    torch.cuda.synchronize()
    with torch.profiler.record_function("tileops_bench_kernel"):
        _run(i)
    torch.cuda.synchronize()
```

随后 `_sum_kernel_time_us()` 从 Kineto results 中读取：

```text
1. projected user annotation windows
2. CUDA kernel events
3. 落在 projected windows 内的 kernel duration
4. projected window count: n_regions
```

当前 correctness gate 是：

```text
n_regions == n_repeat
```

如果不满足，就抛出 `_CuptiProjectionError`，并根据 fallback policy 走 CUDA events。

### 3.1 三种窗口

当前实现实际混合了三种窗口：

| 窗口 | 来源 | 用途 | 同步保证 |
| --- | --- | --- | --- |
| CPU annotation window | `record_function` | 标记一次 logical repeat | CPU scope 完整 |
| GPU projected window | Kineto projection | 过滤 CUDA activity | 没有公开 projection-ready semaphore |
| CUDA activity event | CUPTI/Kineto | 记录 kernel start/end/duration/name | kernel event 本身完整 |

`torch.cuda.synchronize()` 只能保证 CUDA work drain 完，不能保证 Kineto 已经把每个 CPU annotation scope 稳定投影成 GPU timeline 上的 window。

### 3.2 当前 fallback 的实际语义

当前 fallback 到 CUDA events 后，测到的是：

```text
CUDA event elapsed time around _run(i)
```

这会包含 launch/runtime path 的影响。对于较大的 op，这个影响可能不显著；对于 `<10us` 级别的 fast kernel，几十微秒 launch overhead 会主导结果。Nightly 中因此可能出现：

```text
CUPTI kernel-only latency: 约 0.009 ms
CUDA event latency:       约 0.067 ms
```

这类数据不能和 CUPTI kernel-only history 直接混合做 regression / best / roofline 判断。

## 4. 当前具体问题

### 4.1 没有真正对齐 NV SOL 的 attribution 机制

TileOps 对齐了 fixed-repeat protocol，但没有对齐 NV SOL 的 native CUPTI activity sequence attribution。当前 TileOps 仍依赖 Kineto projection：

```text
CPU annotation -> projected GPU window -> count regions
```

NV SOL 则是：

```text
native CUPTI activity -> discovery sequence -> timestamp window -> sequence match
```

这是本轮问题的结构性差异。

### 4.2 Projection mismatch 会把可用 CUPTI trace 判为失败

当前失败形态包括：

```text
49/50 projected windows
48/50 projected windows
0/50 projected windows
```

对于 single-kernel op，`49/50` 可能仍然包含 49 个完整 kernel activity sample；但旧 gate 使用 `n_repeat` 作为唯一 denominator，不能安全接受 partial sample。

对于 multi-kernel op，单纯 kernel count 更不能推回 logical call count；kernel duration sum 也不等于 operator latency。

### 4.3 Fallback 从可用性兜底变成数据质量风险

#697 里的 fallback 初衷是 CUPTI 不可用时继续跑 benchmark。现在它被 projection mismatch 大量触发后，实际变成：

```text
profiling attribution 不稳定
  -> backend 切换
  -> latency 口径切换
  -> history / regression 混入不可比数据
```

这不是一个普通 fallback，而是 benchmark provenance 变化。

### 4.4 Perf history 缺少 timing provenance

Nightly artifact 能看到 `timing = cuda-events`，但历史比较文件没有完整保存 timing backend / fallback reason / CUPTI mismatch diagnostics。结果是：

```text
CUPTI kernel-only latency
CUDA event latency
unknown backend latency
```

可能进入同一条 history 和 regression 比较路径。

### 4.5 NCU 不是最终生产路径的自然选择

NCU 可以通过 NVTX range 对动态 kernel dispatch 做外部 profiling，而且不依赖 Kineto projection。但作为 nightly benchmark backend，它有明显成本：

- 每个 case 需要外部 profiler process；
- 总 wall time 和工程复杂度更高；
- 与常规 pytest result/report 集成成本更大；
- 更适合作为 diagnostic backend，而不是默认生产 timing backend。

## 5. 改进策略

### 5.1 短期：修正数据治理

无论后端如何演进，先把 benchmark provenance 补齐：

```text
timing_backend
fallback_reason
n_regions / n_repeat
captured_cuda_kernel_count
business_kernel_names
sampled_call_count
```

这些字段应该进入 benchmark record、JUnit artifact、profile log 和 perf history。Regression / best / improvement 判断必须只比较可比 backend；旧数据缺失 backend 时标为 `unknown`。

### 5.2 三种候选方案

当前有三种可选 benchmark timing 路线：

| 方案 | single-kernel op | multi-kernel op | 优点 | 风险 |
| --- | --- | --- | --- | --- |
| A. 完全照搬 SOL | native CUPTI kernel duration | native CUPTI activity sequence span | 最完整对齐 NV SOL；不依赖 Kineto projection；可以统一处理 single/multi kernel | 需要确认 SOL-style multi-kernel span 在 TileOps 动态 dispatch、helper kernel、外部 baseline 下都稳定 |
| B. single-kernel Kineto + multi-kernel CUDA event | Kineto/CUPTI kernel duration | CUDA event operator latency | **更快**；改动较小；延续现有 TileOps profiler infrastructure | 仍依赖 Kineto projection/window；如果 multi-kernel 判定也依赖 Kineto，可能把真实 multi-kernel 误判成 single-kernel |
| C. single-kernel SOL native + multi-kernel CUDA event | native CUPTI kernel duration | CUDA event operator latency | 绕开 Kineto projection；保留 fast single-kernel pure kernel time；避免 multi-kernel duration sum 误当 op latency | 比 Kineto 路径慢一些；需要维护 native CUPTI binding 和 discovery 逻辑 |

调研实验显示，方案 B 的速度优势是明确存在的，但方案 C 的额外成本没有大到不可接受；同时 single-kernel 的 Kineto/CUPTI 与 SOL native CUPTI latency 基本吻合。具体数据见 [调研笔记 §8](archive/benchmark-methods-survey-note-cn.md#8-tileops-四路计时实验事实)。

但方案 B 的根本风险也仍然存在：如果 single/multi-kernel 判定来自 Kineto projected window，它可能漏看 activity。第一轮中 `torch-sdpa` backward 就出现了 Kineto 侧 `activity_count=150`、SOL native 侧 `activity_count=600` 的差异，说明 Kineto window 内看到的 kernel names/counts 不一定等于 op 的真实 dispatch 结构。

### 5.3 当前决策：native CUPTI 主路径 + multi-kernel fallback

本轮决策不再把 Kineto projection 作为主要修补方向，而是吸收 NV SOL 的 native CUPTI discovery / activity attribution 机制：

```text
1. warmup 后执行 native CUPTI discovery
2. 从 discovery activity 中识别 business kernel names/counts
3. 如果 logical call 是 single-kernel：
     measurement 使用 CUPTI kernel duration / sampled count
4. 如果 logical call 是 multi-kernel：
     fallback 到 CUDA events，报告 operator-level latency
5. 如果 measurement activity 与 discovery identity/count 不一致：
     显式 fallback 或失败，并记录 diagnostics
```

这样保留了 SOL native 对真实 CUPTI activity 的直接访问，绕开 Kineto annotation projection；同时避免把 multi-kernel 的 duration sum 误当成 op latency。该决策采用方案 C：**single-kernel SOL native + multi-kernel CUDA event**。

相关探索：

- [tile-ai/TileOPs#1797](https://github.com/tile-ai/TileOPs/pull/1797)
- [tile-ai/TileOPs#1796](https://github.com/tile-ai/TileOPs/issues/1796)
- [GPU benchmark 方法调研笔记 §8](archive/benchmark-methods-survey-note-cn.md#8-tileops-四路计时实验事实)

调研数据支持这个选择：

- single-kernel 时，SOL native CUPTI 与 Kineto/CUPTI 的 latency 基本一致；
- fast single-kernel 如果 fallback 到 CUDA event / CPU wall，会被固定开销显著放大；
- multi-kernel 时，kernel duration sum 与 operator latency 不是同一口径，CUDA event / CPU wall 更接近 operator span；
- SOL native 比 Kineto 路径慢一些，但全量成本差异在可接受范围内。

### 5.4 中期：实现 TileOps native CUPTI backend

更正统的方向是实现一个 TileOps 原生 CUPTI backend，对齐 NV SOL 的 attribution 机制：

```text
1. warmup outside profiler
2. discovery iteration:
     collect native CUPTI activity
     derive expected activity sequence
3. timed iterations:
     start_cpu = cupti.get_timestamp()
     _run(i)
     torch.cuda.synchronize()
     end_cpu = cupti.get_timestamp()
4. postprocess:
     select expected sequence inside each timestamp window
     validate counts
     single-kernel: latency = kernel duration
     multi-kernel: fallback to CUDA events
5. report:
     backend = native-cupti
     sequence / counts / diagnostics persisted
```

这个方案先服务 TileOps 当前 nightly 口径：

- single-kernel fast op：继续使用 pure kernel time，避免 launch overhead；
- multi-kernel op：使用 CUDA event 报 operator-level latency；
- noisy helper kernels：通过 discovery identity 和过滤规则排除；
- dynamic dispatch：如果 discovery 与 measurement 不一致，不能静默报 CUPTI latency。

NV SOL 对 multi-kernel logical call 使用 activity span `max(end)-min(start)` 是一个更完整的方向；TileOps 当前先选择 multi-kernel fallback，是为了避免在没有充分验证各类 dynamic dispatch / helper kernel / baseline library 行为前，把 kernel-sum 或错误 sequence 当作稳定 nightly latency。

### 5.5 NCU 定位为诊断后端

NCU 仍然有价值，但更适合定位为：

```text
diagnostic backend
golden cross-check
small targeted reproduction
review artifact for suspicious workloads
```

不建议把它作为 nightly 默认路径，除非后续证明 native CUPTI binding 在 runner 中不可维护。

### 5.6 推荐落地顺序

建议分三阶段推进：

| 阶段 | 目标 | 结果 |
| --- | --- | --- |
| P0 | 持久化 timing provenance | 先停止不可比数据静默污染 history |
| P1 | Prototype native CUPTI SOL-style backend | 用 discovery/name/count 识别 single-kernel 与 multi-kernel |
| P2 | single-kernel CUPTI + multi-kernel CUDA event fallback | 替换 Kineto projection 依赖，保持 latency 语义保守 |
| P3 | NCU diagnostic runner | 用于疑难 case cross-check，不作为默认 nightly backend |

## 6. 结论

TileOps 当前 benchmark 问题不是“CUPTI 是否可用”这么简单，而是 **CUPTI activity attribution 机制没有对齐 NV SOL**。

当前实现的核心风险是：

```text
把 logical repeat correctness 绑在 Kineto projection count 上。
```

短期应补齐 timing provenance；主路径应转向 native CUPTI activity discovery，避免继续依赖 `record_function` projection 的稳定性。

最终推荐方向：

```text
production timing:
  native CUPTI SOL-style discovery
  single-kernel CUPTI duration
  multi-kernel CUDA event fallback

diagnostic:
  NCU / NVTX targeted runner
```

## 7. 参考资料

- NVIDIA SOL-ExecBench timing implementation: https://github.com/NVIDIA/SOL-ExecBench/blob/main/src/sol_execbench/core/bench/timing.py
- NVIDIA SOL-ExecBench CUPTI utilities: https://github.com/NVIDIA/SOL-ExecBench/blob/main/src/sol_execbench/core/bench/cupti_utils.py
- [GPU benchmark 方法调研笔记](archive/benchmark-methods-survey-note-cn.md)
