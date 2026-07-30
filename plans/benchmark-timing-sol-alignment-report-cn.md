# TileOps Benchmark Timing 技术报告：NV SOL 对齐与改进路线

更新日期：2026-07-30

## 摘要

TileOps 最初希望 benchmark 基础设施对齐 NVIDIA SOL-ExecBench：固定 warmup/repeat/trial、L2 flush、输入地址扰动、锁频和稳定报告口径。实际落地时，TileOps 采用了 `torch.profiler` / Kineto 暴露的 CUPTI CUDA activity，并通过 `record_function` projection 判断每个 repeat 的 GPU 计时窗口。

重新核对 NV SOL 代码后，需要纠正一个历史混淆：NV SOL 当前默认 timing methodology 是 **native CUPTI activity timing**，不是 PyTorch/Kineto projection，也不是 NCU。TileOps 和 NV SOL 的核心差异不是“是否使用 CUPTI”，而是：

```text
NV SOL:
  native CUPTI activity
  + discovery sequence
  + per-iteration CUPTI timestamp window
  + sequence matching

TileOps current:
  torch.profiler / Kineto CUDA activity
  + CPU record_function
  + Kineto projected GPU annotation window
  + regions == n_repeat gate
```

当前问题集中在 TileOps 对 Kineto projection 的依赖：当 projected annotation window 数不是 `n_repeat` 时，benchmark fallback 到 CUDA events。CUDA events 对 fast kernels 会带入 launch overhead，使 nightly latency 和 roofline 数据不可直接与 CUPTI kernel-only history 混合比较。

## 1. NV SOL 的具体流程

参考实现：

- NVIDIA SOL-ExecBench timing code: https://github.com/NVIDIA/SOL-ExecBench/blob/main/src/sol_execbench/core/bench/timing.py
- CUPTI utility code: https://github.com/NVIDIA/SOL-ExecBench/blob/main/src/sol_execbench/core/bench/cupti_utils.py

NV SOL 的默认 timing backend 是 `methodology="cupti"`。它使用 Python CUPTI binding 直接开启 CUPTI activity collection，主要采集：

```text
CONCURRENT_KERNEL
MEMCPY
MEMSET
```

它不依赖 `torch.profiler.record_function` 投影，也不要求每个 Python annotation 在 GPU timeline 上变成一个 projected region。

### 1.1 Warmup 与准备

每个 iteration 前，NV SOL 会执行 `prepare_iteration()`：

```text
setup args
reset persisting L2 cache
clear L2 cache buffer
optional torch.cuda.synchronize()
```

随后先做 warmup：

```text
torch.cuda.synchronize()
repeat warmup:
  prepare_iteration()
  runner(args)
torch.cuda.synchronize()
```

warmup 的目的不是计时，而是让 JIT、autotune、library initialization 和冷启动状态尽量退出正式测量窗口。

### 1.2 Discovery iteration

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

这一步很关键：NV SOL 不是简单数 kernel 个数，也不是把所有 CUPTI activity duration 直接求和；它先识别“这一次 logical call 应该对应什么 GPU activity 序列”。

### 1.3 Timed iterations

正式计时时，NV SOL 在同一个 CUPTI collection window 中跑 `rep` 次。每次 logical iteration 的 CPU timestamp window 用 CUPTI 自己的 timestamp API 标记：

```text
start_cpu = cupti.get_timestamp()
runner(args)
torch.cuda.synchronize()
end_cpu = cupti.get_timestamp()
```

这里 `end_cpu` 放在 `torch.cuda.synchronize()` 之后，所以本次 call 发出的 CUDA work 会完整落在这个 CPU timestamp window 内。

### 1.4 Attribution 与 latency

CUPTI collection 结束后，NV SOL 会把所有 GPU activity 按 start/end/correlation id 排序。对每个 iteration：

```text
1. 取 start 落在 [start_cpu, end_cpu] 的 activity
2. 在这个 window 内查找 discovery 得到的 expected sequence
3. 要求 activity counts 与 discovery counts 一致
4. latency = max(activity.end) - min(activity.start)
```

这带来两个重要性质：

1. **支持 multi-kernel logical call**：latency 是这一组 activity 的 GPU span，不是 kernel duration sum，也不是 kernel count denominator。
2. **不依赖 Kineto projection**：logical call boundary 来自 CUPTI timestamp window + sequence matching，而不是 CPU annotation projected region。

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

#689 / #692 的文本中曾把 SOL timing 写成 CUDA events，并把 TileOps 的 CUPTI kernel-only time 描述为 “improvement over SOL”。这是当时文档中的历史表述；重新核对 NV SOL 当前代码后，不能再把这句话当成 NV SOL 真实实现的描述。

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

### 5.2 短期：减少错误 fallback，但不扩大错误 CUPTI 使用范围

在继续使用 Kineto 的阶段，可以采用保守修补：

```text
single-kernel callable:
  如果 business kernel name 唯一，且 sampled count 在合理范围内，
  可以用 CUPTI total kernel duration / sampled count。

multi-kernel callable:
  不用 kernel duration sum 伪装 operator latency；
  fallback 到 CUDA events，或等待 native CUPTI sequence timing。
```

这类修补能缓解 `49/50` 对 single-kernel fast op 的误 fallback，但不能解决 multi-kernel logical call attribution。

相关探索：

- [tile-ai/TileOPs#1797](https://github.com/tile-ai/TileOPs/pull/1797)
- [tile-ai/TileOPs#1796](https://github.com/tile-ai/TileOPs/issues/1796)

### 5.3 中期：实现 NV SOL-style native CUPTI timing

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
     latency = max(end) - min(start)
5. report:
     backend = native-cupti
     sequence / counts / diagnostics persisted
```

这个方案可以统一 single-kernel 和 multi-kernel logical call：

- single-kernel：span 等于 kernel duration；
- multi-kernel：span 表示 GPU-side operator latency；
- noisy helper kernels：通过 discovery sequence 和 sequence matching 过滤；
- dynamic dispatch：只要 discovery sequence 与 timed iterations 一致，就可以处理；如果 sequence 不稳定，应显式 fail 或 fallback。

### 5.4 NCU 定位为诊断后端

NCU 仍然有价值，但更适合定位为：

```text
diagnostic backend
golden cross-check
small targeted reproduction
review artifact for suspicious workloads
```

不建议把它作为 nightly 默认路径，除非后续证明 native CUPTI binding 在 runner 中不可维护。

### 5.5 推荐落地顺序

建议分三阶段推进：

| 阶段 | 目标 | 结果 |
| --- | --- | --- |
| P0 | 持久化 timing provenance | 先停止不可比数据静默污染 history |
| P1 | Kineto single-kernel conservative fix | 缓解 fast single-kernel fallback，不碰 multi-kernel 语义 |
| P2 | Prototype native CUPTI SOL-style backend | 对齐 NV SOL attribution，逐步替换 Kineto projection |
| P3 | NCU diagnostic runner | 用于疑难 case cross-check，不作为默认 nightly backend |

## 6. 结论

TileOps 当前 benchmark 问题不是“CUPTI 是否可用”这么简单，而是 **CUPTI activity attribution 机制没有对齐 NV SOL**。

当前实现的核心风险是：

```text
把 logical repeat correctness 绑在 Kineto projection count 上。
```

短期可以用 single-kernel classification 减少错误 fallback，并补齐 timing provenance；但长期如果要真正对齐 NV SOL，应实现 native CUPTI activity sequence timing，避免继续依赖 `record_function` projection 的稳定性。

最终推荐方向：

```text
production timing:
  native CUPTI SOL-style sequence timing

temporary bridge:
  Kineto single-kernel sampling + multi-kernel CUDA event fallback

diagnostic:
  NCU / NVTX targeted runner
```
