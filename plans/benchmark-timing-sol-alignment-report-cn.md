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

这三类 activity 覆盖了 benchmark 中最常见的 GPU work：kernel execution、显式/隐式 memcpy，以及 memset/clear cache 相关操作。SOL 后续会从这些 activity 中识别用户调用对应的序列，并用这个序列进行正式计时归因。

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

然后从这一次用户调用中得到 expected activity sequence，其中包括 kernel name/count 等信息：

```text
expected_activity_sequence
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
| CUDA activity event | CUPTI/Kineto | 记录 kernel start/end/duration/name | 一旦被捕获，event 自身有完整 start/end；但不保证 collection 覆盖每个已 launch kernel |

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

### 4.1 Kineto timing path 不稳定

Kineto 的 CUPTI timing 原理和 NV SOL native CUPTI 并不矛盾：二者正常工作时，single-kernel op 测到的都是 CUDA kernel activity duration。本轮全量对比和同 case 四路实验都显示，当 Kineto 和 SOL native 都成功走到 CUPTI kernel activity 口径时，latency 基本吻合；数据见 [调研笔记 §8.2](archive/benchmark-methods-survey-note-cn.md#82-全量-benchmark-对比) 和 [§8.3](archive/benchmark-methods-survey-note-cn.md#83-同-case-四路计时对比)。

问题出在当前 TileOps 的 Kineto 写法依赖 `record_function` projection 作为 repeat/window gate：

```text
CPU annotation -> projected GPU window -> count regions
```

这个 projection 路径没有公开 ready semaphore 或细粒度诊断接口。我们用 Kineto trace 和 native CUPTI probe 做了 targeted 对照：在 Kineto partial trace 中，缺失 repeat 的 CPU annotation、CUDA launch API 和 CUDA event elapsed time 都存在，但 Kineto result 中没有对应 correlation id 的 GPU business kernel activity；另一次 native CUPTI probe 在同一 case 上可以用 CPU timestamp window 和 external correlation id 把前几轮 GPU kernel activity 正常归属到对应 repeat。这个对照不支持“GPU activity 整体后移几轮”的解释，更支持问题出在 Kineto profiler/projection 路径的 session 头部 activity 覆盖或归因上。具体实验见 [调研笔记 §8.4](archive/benchmark-methods-survey-note-cn.md#84-kineto-projection-不稳定性定位实验)。

因此，当前故障链路是：

```text
Kineto projection / activity 归因出现 partial trace
  -> projected window count != n_repeat
  -> CUPTI path 被判失败
  -> fallback 到 CUDA event
  -> 小 kernel latency 被 event gap / launch / synchronize 影响放大
```

Nightly 中可见的典型形态是：

```text
49/50 projected windows
48/50 projected windows
0/50 projected windows
```

这不是具体 operator kernel 的性能问题，而是 benchmark infrastructure 的稳定性问题。

### 4.2 Artifact 缺少 timing provenance

Nightly artifact 能看到 `timing = cuda-events`，但历史比较文件没有完整保存 timing backend / fallback reason / CUPTI mismatch diagnostics。结果是：

```text
CUPTI kernel-only latency
CUDA event latency
unknown backend latency
```

可能进入同一条 history 和 regression 比较路径。

这是独立的数据治理问题。无论最终 timing backend 选择哪条路径，导出的 artifact 都应该清楚记录本次 latency 使用的计时标准，避免把不可比数据静默混入 history / regression / roofline 判断。

## 5. 改进策略

本轮决策基于调研笔记中的几轮实验：

- [§8.1](archive/benchmark-methods-survey-note-cn.md#81-kineto-与-sol-native-的流程差异)：Kineto 与 SOL native 的归因流程差异；
- [§8.2](archive/benchmark-methods-survey-note-cn.md#82-全量-benchmark-对比)：全量 benchmark wall time 与 matched latency 对比；
- [§8.3](archive/benchmark-methods-survey-note-cn.md#83-同-case-四路计时对比)：SOL native、Kineto、CUDA event、CPU wall 的同 case 差异；
- [§8.4](archive/benchmark-methods-survey-note-cn.md#84-kineto-projection-不稳定性定位实验)：Kineto projection partial trace 的 targeted probe。

### 5.1 三种候选方案

当前有三种可选 benchmark timing 路线：

| 方案 | single-kernel op | multi-kernel op | 优点 | 风险 |
| --- | --- | --- | --- | --- |
| A. 完全照搬 SOL | native CUPTI kernel duration | native CUPTI activity sequence span | 最完整对齐 NV SOL；不依赖 Kineto projection；可以统一处理 single/multi kernel | 需要确认 SOL-style multi-kernel span 在 TileOps 动态 dispatch、helper kernel、外部 baseline 下都稳定 |
| B. single-kernel Kineto + multi-kernel CUDA event | Kineto/CUPTI kernel duration | CUDA event operator latency | **更快**；改动较小；延续现有 TileOps profiler infrastructure | 仍依赖 Kineto projection/window；如果 multi-kernel 判定也依赖 Kineto，可能把实际 multi-kernel 误判成 single-kernel |
| C. single-kernel SOL native + multi-kernel CUDA event | native CUPTI kernel duration | CUDA event operator latency | 绕开 Kineto projection；保留 fast single-kernel pure kernel time；避免 multi-kernel duration sum 误当 op latency | 比 Kineto 路径慢一些；需要维护 native CUPTI binding 和 discovery 逻辑 |

NCU / NVTX runner 仍然适合作为 diagnostic backend 或 review artifact，用于疑难 case cross-check；但它不是当前 nightly production timing 的自然主路径。

调研实验显示，方案 B 的速度优势是明确存在的，但方案 C 的额外成本没有大到不可接受；同时 single-kernel 的 Kineto/CUPTI 与 SOL native CUPTI latency 基本吻合。具体数据见 [调研笔记 §8](archive/benchmark-methods-survey-note-cn.md#8-tileops-四路计时实验事实)。

方案 B 的主要风险仍然存在：它继续依赖 Kineto projection，而 targeted probe 显示这条路径会出现 session 头部 partial trace。第一轮中 `torch-sdpa` backward 也出现了 Kineto 侧 `activity_count=150`、SOL native 侧 `activity_count=600` 的差异，说明 Kineto window 内看到的 kernel names/counts 不一定等于 op 的完整 dispatch 结构。

### 5.2 当前决策与实现伪代码

本轮决策不再把 Kineto projection 作为主要修补方向，而是吸收 NV SOL 的 native CUPTI discovery / activity attribution 机制。该决策采用方案 C：**single-kernel SOL native + multi-kernel CUDA event**。

```text
profile(fn, args):
  warmup(fn, args)

  discovery = collect_native_cupti_once(fn, args)
  business = filter_business_kernels(discovery.kernels)
  kernel_names = unique_names(business)
  kernels_per_call = len(business)

  if kernels_per_call == 1 and len(kernel_names) == 1:
      timing = collect_native_cupti_repeats(fn, args, n_repeat, n_trials)
      samples = filter_business_kernels(timing.kernels, name=kernel_names[0])
      validate(samples.count > 0)
      validate(unique_names(samples) == kernel_names)
      latency = sum(samples.duration) / samples.count
      backend = "native-cupti-single-kernel"
  else:
      latency = measure_cuda_event_span(fn, args, n_repeat, n_trials)
      backend = "cuda-event-multi-kernel"

  record timing backend, kernel names/counts, sampled count, fallback reason
  return latency
```

这样保留了 SOL native 对 native CUPTI activity 的直接访问，绕开 Kineto annotation projection；同时避免把 multi-kernel 的 duration sum 误当成 op latency。

相关探索：

- [tile-ai/TileOps#1797](https://github.com/tile-ai/TileOPs/pull/1797)
- [tile-ai/TileOps#1796](https://github.com/tile-ai/TileOPs/issues/1796)
- [GPU benchmark 方法调研笔记 §8](archive/benchmark-methods-survey-note-cn.md#8-tileops-四路计时实验事实)

这个选择的依据是：

- single-kernel 时，SOL native CUPTI 与 Kineto/CUPTI 的 latency 基本一致；
- fast single-kernel 如果 fallback 到 CUDA event / CPU wall，会被固定开销显著放大；
- multi-kernel 时，kernel duration sum 与 operator latency 不是同一口径，CUDA event / CPU wall 更接近 operator span；
- SOL native 比 Kineto 路径慢一些，但全量成本差异在可接受范围内。

### 5.3 Artifact 数据治理

无论后端如何演进，都要把 benchmark provenance 补齐：

```text
timing_backend
fallback_reason
business_kernel_names
business_kernel_count_per_call
captured_cuda_kernel_count
sampled_call_count
cupti_activity_count
timing_diagnostics
```

这些字段应该进入 benchmark record、JUnit artifact、profile log 和 perf history。Regression / best / improvement 判断必须只比较可比 backend；旧数据缺失 backend 时标为 `unknown`。

## 6. 结论

TileOps 当前 benchmark 问题不是 “CUPTI 是否可用”，而是当前 Kineto projection 写法会出现非确定性的 partial trace，并把 latency 口径切到 CUDA event。

```text
正常时：
  Kineto/CUPTI single-kernel duration ~= SOL native CUPTI single-kernel duration

异常时：
  Kineto projection partial
    -> fallback CUDA event
    -> small kernel latency 被固定开销放大
```

主路径应转向 native CUPTI discovery，避免继续把 nightly timing gate 绑在 `record_function` projection 上。最终推荐方向是方案 C：

| op 类型 | timing backend | latency 语义 |
| --- | --- | --- |
| single-kernel | native CUPTI | kernel-only duration |
| multi-kernel | CUDA event | operator-level stream span |

同时，artifact 必须完整记录 timing provenance，保证 history / regression / roofline 只比较同口径数据。

## 7. 参考资料

- NVIDIA SOL-ExecBench timing implementation: https://github.com/NVIDIA/SOL-ExecBench/blob/main/src/sol_execbench/core/bench/timing.py
- NVIDIA SOL-ExecBench CUPTI utilities: https://github.com/NVIDIA/SOL-ExecBench/blob/main/src/sol_execbench/core/bench/cupti_utils.py
- [GPU benchmark 方法调研笔记](archive/benchmark-methods-survey-note-cn.md)
