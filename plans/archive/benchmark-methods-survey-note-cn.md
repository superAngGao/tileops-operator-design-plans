# GPU Benchmark 方法调研笔记：NV SOL / PyTorch / Triton / FA3

更新日期：2026-07-31

这份笔记是 TileOps benchmark timing 讨论中的背景材料，不是最终决策文档。它整理 GPU kernel 连续调度、初始化与计时边界，以及 NV SOL、PyTorch、Triton、FlashAttention / FA3 等常见 benchmark 方法的差异。

## 1. GPU kernel 调度与计时基础

CUDA kernel launch 对 CPU 是异步的。CPU 发出 launch 后通常很快返回，GPU work 被排入 CUDA stream；同一 stream 内的 work 按提交顺序执行，不同 stream 之间可能并发或通过 event / dependency 同步。因此 benchmark 必须明确区分：

```text
CPU 提交窗口：Python/C++ 从开始调用到返回，包含 launch/runtime overhead
GPU 执行窗口：GPU stream 上真正执行 kernel/memcpy/memset 的时间段
逻辑 op 窗口：一次用户 callable _run(i) 希望代表的一次 operator 调用
```

一个 logical op 可能对应一个 kernel，也可能对应多个 kernel、memset、memcpy 或内部 library dispatch。单 kernel 时，operator latency、kernel duration、GPU span 往往接近；多 kernel 时，`sum(kernel durations)`、`max(end)-min(start)`、CUDA event span 和 CPU wall-time 是不同口径。

## 2. 计时前需要移出的开销

GPU benchmark 前通常需要让以下开销尽量退出正式计时窗口：

```text
CUDA context 初始化
module/kernel load
JIT 编译或 TileLang/Triton autotune
library handle / algorithm cache 初始化
cudaMalloc / allocator 扩容
输入输出 tensor 分配
cache / persisting L2 状态准备
```

常见手段是 warmup、预分配、输入 clone/address rotation、L2 flush、必要的 `torch.cuda.synchronize()`，以及在可控环境中锁 GPU clocks。

## 3. 常见 start/stop 机制

| 机制 | start/stop 信号 | 测量口径 | 优点 | 风险 |
| --- | --- | --- | --- | --- |
| CPU wall timer | CPU `timer()` 前后包住 callable，通常前后同步 | host 视角 wall-time | 简单，适合 end-to-end | 包含 launch、Python/runtime、sync overhead |
| CUDA events | 在 stream 中 enqueue start/end event | GPU stream span | 低成本，适合 operator wall latency | 对 fast single-kernel 会受 launch/queue 行为和多 kernel gap 影响 |
| CUPTI activity | CUPTI 记录 kernel/memcpy/memset start/end | GPU activity duration/span | 可以得到 kernel-only 或 activity sequence | 需要可靠 attribution 和 buffer 管理 |
| PyTorch/Kineto profiler | Kineto 收集 CUPTI activity，并可投影 CPU annotation | profiler event timeline | 接入简单，能从 Python 使用 | annotation projection 没有公开 ready semaphore |
| NCU/Nsight | 外部 profiler 采集 kernel/profile metric | profiler report | 诊断能力强 | 启动成本高，不适合默认 nightly 全量路径 |

这里的关键不是“哪个时间更真实”，而是 benchmark history 必须保持同一口径。

## 4. NV SOL

NV SOL 的默认 timing backend 是 native CUPTI activity timing。它使用 Python CUPTI binding 开启 GPU activity collection，主要采集：

```text
CONCURRENT_KERNEL
MEMCPY
MEMSET
```

SOL 的关键流程是：

```text
1. warmup
2. discovery iteration:
     collect native CUPTI activity
     derive expected activity sequence
3. timed iterations:
     start_cpu = cupti.get_timestamp()
     runner(args)
     torch.cuda.synchronize()
     end_cpu = cupti.get_timestamp()
4. attribution:
     select expected sequence inside timestamp window
     validate activity counts
     latency = max(activity.end) - min(activity.start)
```

SOL 的特点是用 discovery sequence 做归因，并用 activity span 表示一次 logical call 的 GPU 侧 latency。对于 single-kernel call，span 退化为 kernel duration；对于 multi-kernel call，它比简单相加 kernel duration 更接近 operator GPU span。

## 5. PyTorch `torch.utils.benchmark`

PyTorch `Timer` 基于 Python `timeit.Timer`，但增加了 PyTorch runtime 相关处理：warmup、线程数控制、accelerator 同步和 replicate/median 统计。CUDA 场景下，默认 timer 会在读取 host 时间前同步 accelerator。

典型流程是：

```text
setup outside stmt
warmup
start host timer, with accelerator synchronize
repeat stmt N 次
stop host timer, with accelerator synchronize
report Measurement
```

这个方法适合比较 PyTorch statement / module 的 end-to-end 执行时间；它不区分一个 statement 内部到底发了几个 kernel，也不是 kernel-only timing。FlashAttention 早期 benchmark helper 的 `benchmark_forward` / `benchmark_backward` 也主要包了一层 `torch.utils.benchmark.Timer`。

## 6. Triton `do_bench`

Triton `do_bench` 使用 CUDA event 计时。它会先运行一次函数并同步，再用 event 粗估 runtime，根据 `warmup` / `rep` 的毫秒预算计算 warmup 次数和 repeat 次数。正式测量时，每次 repeat 清 L2 cache、记录 start event、执行 `fn()`、记录 end event，最后统一同步并统计 event elapsed time。

典型流程是：

```text
fn()
synchronize()
estimate runtime with CUDA events
n_warmup = warmup_ms / estimate
n_repeat = rep_ms / estimate
warmup loop
for each repeat:
  clear L2 cache
  start_event.record()
  fn()
  end_event.record()
synchronize()
summarize event elapsed times
```

它的结果是 stream 上 start/end event 之间的 GPU span。对于 single-kernel callable，这通常接近 kernel duration；对于 multi-kernel callable，它更接近 operator GPU span，而不是每个 kernel 的 pure duration。

## 7. FlashAttention / FA3 benchmark

FlashAttention 仓库中同时存在几类 benchmark 写法：

- `flash_attn.utils.benchmark` 使用 `torch.utils.benchmark.Timer` 包装 forward/backward/combined；
- Hopper/FA3 的 `benchmark_attn.py` 中，forward timing 使用 `triton.testing.do_bench`；
- `benchmark_mla_decode.py` 使用 `do_bench` 或 `do_bench_cudagraph`，并显式建议锁 GPU clocks；
- 部分脚本在不同 case 之间 `sleep(1)`，避免上一个 benchmark 的功耗/温度状态影响下一项；
- decode/MLA 场景会预先计算 scheduler metadata，把可复用的准备工作移出被测 callable。

这类算子库 benchmark 的核心目标通常是 operator-level latency 和 throughput comparison，而不是 nightly history 中严格的 kernel-only attribution。它们适合比较 FA2/FA3/FlashInfer/cuDNN/PyTorch 等实现的端到端 op 延迟。

## 8. 参考资料

- NVIDIA SOL-ExecBench timing implementation: https://github.com/NVIDIA/SOL-ExecBench/blob/main/src/sol_execbench/core/bench/timing.py
- NVIDIA SOL-ExecBench CUPTI utilities: https://github.com/NVIDIA/SOL-ExecBench/blob/main/src/sol_execbench/core/bench/cupti_utils.py
- PyTorch `torch.utils.benchmark.Timer`: https://github.com/pytorch/pytorch/blob/main/torch/utils/benchmark/utils/timer.py
- PyTorch benchmark README: https://github.com/pytorch/pytorch/blob/main/torch/utils/benchmark/README.md
- Triton `triton.testing.do_bench`: https://github.com/triton-lang/triton/blob/main/python/triton/testing.py
- FlashAttention benchmark helpers: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/utils/benchmark.py
- FlashAttention Hopper / FA3 benchmark script: https://github.com/Dao-AILab/flash-attention/blob/main/hopper/benchmark_attn.py
