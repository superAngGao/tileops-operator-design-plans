# GPU Benchmark 方法调研笔记：NV SOL / PyTorch / Triton / FA3

更新日期：2026-07-31

这份笔记是 TileOps benchmark timing 讨论中的背景材料，不是最终决策文档。它整理 GPU kernel 连续调度、初始化与计时边界，以及 NV SOL、PyTorch、Triton、FlashAttention / FA3 等常见 benchmark 方法的差异。

## 1. GPU kernel 调度与计时基础

### 1.1 Kernel 在 GPU 上如何启动

CUDA kernel launch 对 CPU 是异步的。CPU 发出 launch 后通常很快返回；CUDA runtime / driver 会把 kernel function、grid/block 形状、参数地址、stream dependency 等信息提交到 GPU 可见的工作队列。GPU 侧调度器随后从队列中取出 work，把 thread blocks 分派到 SM 上执行。

一次 kernel 真正开始跑之前，通常已经需要满足几类前提。它们不是严格按下面顺序逐项发生；更准确地说，它们分布在 host runtime、CUDA driver、CUDA context、stream queue 和 GPU 执行侧。对 benchmark 来说，重要的是知道哪些可能落在第一次调用或 warmup 中，哪些才属于正式 kernel 的 GPU activity。

| 前提 | 主要发生位置 | 典型发生时间 | 是否严格排在 kernel launch 前 |
| --- | --- | --- | --- |
| kernel code 已经编译 | NVCC/AOT build、JIT compiler、Triton/TileLang compiler | 安装/构建时，或第一次调用/autotune 时 | 是。没有可执行 device code 就不能 launch |
| kernel code / module 已加载到当前 CUDA context | CUDA runtime / driver module management | 第一次调用该 kernel、显式 module load，或 lazy loading 触发时 | 逻辑上在该 kernel 执行前；可能由第一次 launch 隐式触发 |
| kernel 参数和输入/输出 tensor 地址已经确定 | host 侧 Python/C++ runtime、CUDA launch API | 每次 launch 时 | 是。launch command 需要携带参数值、指针、grid/block、shared memory、stream 等信息 |
| CUDA stream 里排在它前面的依赖已经完成 | GPU work queue / stream scheduler | GPU 执行前，由 stream 顺序和 event dependency 决定 | 是。它决定 kernel 何时能开始执行，但不一定阻塞 CPU launch 返回 |
| 必要的 library / handle 初始化已经发生 | host runtime、library runtime，例如 cuBLAS/cuDNN/FlashAttention wrapper | 通常在第一次调用、warmup 或显式 setup 阶段 | 对使用该 library 的 callable 是前置条件；不一定是每个底层 kernel 的前置步骤 |
| GPU 可以访问对应显存页和页表映射 | CUDA memory manager、GPU MMU、Unified Memory/page migration 机制 | `cudaMalloc`/allocator 分配后；managed memory 可能在首次 GPU 访问时迁移或 fault | 普通 device memory 通常已满足；managed/oversubscription 场景可能在 kernel 执行中触发 page fault |

这些动作不都发生在被测 kernel 的 GPU duration 里。比如 JIT 编译、module load、allocator 扩容、library handle 初始化通常发生在第一次调用或 warmup 阶段；CPU launch path 和 driver submit 通常在 host 侧；kernel activity duration 一般从 GPU 上 kernel 开始执行算起。

因此它们更像一组依赖关系，而不是每次 launch 都重复执行的线性步骤。steady-state benchmark 关心的是：正式计时前这些一次性或偶发性动作是否已经被 warmup/setup 吸收；正式窗口里留下的是 CPU submit、stream 排队、GPU kernel/memcpy/memset activity，还是纯 kernel activity。

先只看 **一个 kernel 被调用一次**。下面这张图的时间轴是从上到下；左侧是 CPU thread 发起 CUDA API，右侧是 CUDA driver / stream / GPU 侧逐步处理这些 work。图里同时标出了四类容易混淆的 timing marker：

```text
CUPTI Runtime/Driver API activity：每个 CUDA API 调用自己的 host-side start/end
SOL-style cupti.get_timestamp()：CPU 侧手动取 timestamp，形成 activity attribution window
CUDA event timestamp：event record 在 stream 上执行时产生的 start/end
CUPTI activity timestamp：kernel 在 GPU 上真正开始/结束执行的 start/end
```

图中的颜色在全文保持一致：

```mermaid
flowchart LR
    C["CUPTI signal"]:::cupti
    E["CUDA event signal"]:::event
    W["CPU wall timer signal"]:::cpuwall
    O["ordinary work / wait / gap"]:::work

    classDef cupti fill:#DBEAFE,stroke:#2563EB,color:#111827
    classDef event fill:#FEF3C7,stroke:#D97706,color:#111827
    classDef cpuwall fill:#DCFCE7,stroke:#16A34A,color:#111827
    classDef work fill:#F3F4F6,stroke:#6B7280,color:#111827
```

这是一个对比图：实际 benchmark backend 通常只选择其中一种或两种机制，不一定同时记录 CUDA events、SOL attribution timestamps 和 CUPTI API activities。尤其要注意，`cupti.get_timestamp()` 包住的是 CPU attribution window，不是 `cudaLaunchKernel` 这条 Runtime API 的 CUPTI activity；普通异步 launch 的 API end 通常早于 kernel 在 GPU 上执行完成。

```mermaid
flowchart TB
    subgraph CPU["CPU thread"]
        T0["SOL attribution t0<br/>cuptiGetTimestamp()"]:::cupti
        A0["CUPTI API activity<br/>cudaEventRecord(start)<br/>host start/end"]:::cupti
        A1["CUPTI API activity<br/>cudaLaunchKernel<br/>host start/end"]:::cupti
        A2["CUPTI API activity<br/>cudaEventRecord(end)<br/>host start/end"]:::cupti
        A3["CUPTI API activity<br/>cudaEventSynchronize(end)<br/>host start/end"]:::cupti
        T1["SOL attribution t1<br/>cuptiGetTimestamp()"]:::cupti
    end

    subgraph GPU["Driver / CUDA stream / GPU"]
        Q0["start-event command<br/>enters stream path"]:::work
        E0["CUDA event_start timestamp"]:::event
        G0["pre-kernel stream gap<br/>stream dependency resolution<br/>launch command processing<br/>dispatch latency"]:::work
        K0["CUPTI kernel.start"]:::cupti
        X0["kernel execution<br/>CTAs dispatched to SMs in waves<br/>resident blocks use SM resources"]:::work
        K1["CUPTI kernel.end"]:::cupti
        G1["post-kernel stream gap<br/>kernel completion propagation<br/>stream advances to end event"]:::work
        E1["CUDA event_end timestamp"]:::event
        D0["end_event complete"]:::work
    end

    T0 --> A0 --> A1 --> A2 --> A3 --> T1
    A0 -. enqueue .-> Q0
    A1 -. enqueue .-> G0
    A2 -. enqueue .-> G1
    Q0 --> E0 --> G0 --> K0 --> X0 --> K1 --> G1 --> E1 --> D0
    D0 -. unblocks .-> A3

    classDef cupti fill:#DBEAFE,stroke:#2563EB,color:#111827
    classDef event fill:#FEF3C7,stroke:#D97706,color:#111827
    classDef cpuwall fill:#DCFCE7,stroke:#16A34A,color:#111827
    classDef work fill:#F3F4F6,stroke:#6B7280,color:#111827
```

读这张图时要注意四点：

- `cudaEventRecord()` 的 CPU 调用只是在 stream 里放一个 event command；真正的 CUDA event timestamp 是 event command 在 GPU stream 上执行时产生的。
- CUPTI Runtime/Driver API activity 是 host 侧 API 调用的 start/end。`cudaLaunchKernel` 的 API end 只表示 CPU launch 调用返回，不表示 GPU kernel 已经结束。
- CUPTI kernel activity 的 `kernel_start/kernel_end` 对应 GPU 上 kernel activity 的开始和结束。single-kernel 的 pure kernel time 通常就是 `kernel_end - kernel_start`。
- SOL-style 的 `cupti.get_timestamp()` 在 CPU 侧取 timestamp，用来形成 attribution window `[t0, t1]`；它本身不是 kernel duration，而是帮助从 CUPTI activity buffer 中挑出属于这次 logical call 的 activity。

CUDA event span 比 CUPTI single-kernel duration 多包了两个 stream gap：

```text
CUDA event span =
  pre-kernel stream gap
  + CUPTI kernel duration
  + post-kernel stream gap

CUPTI kernel duration =
  kernel_end - kernel_start
```

这两个 gap 里通常不是业务计算，而是 stream event command、kernel command 和 event command 之间的设备侧过渡时间。`pre-kernel stream gap` 可能包含 stream dependency resolution、launch command processing、kernel 被 GPU 前端接受到首批 CTA 开始在 SM 上执行之间的 dispatch latency；`post-kernel stream gap` 可能包含 kernel completion 状态向 stream 推进，以及 stream 执行到 end-event record command 并写入 event timestamp 的时间。对于几十或几百微秒的大 kernel，它们占比很小；对于几微秒的 fast kernel，它们会变成可见的偏差和抖动来源。

这里的 `CTA dispatch / resident block resources` 也不是说 GPU 会在 kernel_start 前为整个 grid 一次性预分配寄存器和 shared memory。更准确地说，grid/block shape、dynamic shared memory、kernel metadata 等已经随 launch command 提交；kernel 执行期间，thread blocks 分批被调度到 SM，成为 resident block 时才占用该 SM 上的 registers、shared memory、thread slots 等资源。

这张图表达的是一个 steady-state launch 的常见逻辑顺序，不表示每次 launch 都会重新执行所有 setup 动作，也不承诺 driver 内部一定按图中每一行串行发生。比如 module load、JIT/autotune、allocator 扩容和 library handle 初始化通常应被 warmup 吸收；正式计时要么测 CPU host wall-time，要么测 CUDA event span，要么测 CUPTI activity，取决于 benchmark 选择的 start/stop 机制。

Cache 状态也需要单独看。GPU 不需要在启动 kernel 前把业务数据“预装”进 cache，kernel 会在执行时按访存指令自然填充 L1/L2。但 benchmark 需要决定 cache 初始状态是否受控：

```text
warm cache：多次运行同一输入，可能复用 L2 / library cache 状态
cold-ish cache：每次运行前清 L2 cache buffer，减少上一轮数据残留
stable cache policy：固定 warmup、flush、同步和输入地址扰动
```

所以 cache 不是 kernel 启动的必需准备项，但它是 benchmark 可重复性和数据口径的一部分。SOL / Triton / TileOps 这类 microbenchmark 通常会用 warmup 和 L2 flush 来控制这个状态。

### 1.2 为什么要区分不同计时窗口

CPU 发出 launch 后，GPU work 被排入 CUDA stream；同一 stream 内的 work 按提交顺序执行，不同 stream 之间可能并发或通过 event / dependency 同步。因此 benchmark 必须明确区分：

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

把第 3 步展开看，SOL timed iteration 的时序更像下面这样。左侧的 `start_cpu/end_cpu` 是 `cuptiGetTimestamp()` 取到的 CPU-side attribution window；右侧的 `kernel start/end` 是 CUPTI activity 记录到的 GPU activity 时间。SOL 最终不是用 `end_cpu - start_cpu` 当 latency，而是在这个 attribution window 里选择符合 discovery sequence 的 GPU activities。

```mermaid
flowchart TB
    subgraph CPU["CPU"]
        S0["start_cpu<br/>cuptiGetTimestamp()"]:::cupti
        R0["runner(args)<br/>Python / PyTorch dispatch"]:::work
        L0["cudaLaunchKernel(...)<br/>kernel command enqueued"]:::work
        L1["cudaLaunchKernel(...)<br/>another kernel command enqueued"]:::work
        SY0["torch.cuda.synchronize()<br/>CPU blocks or polls"]:::work
        S1["end_cpu<br/>cuptiGetTimestamp()"]:::cupti
    end

    subgraph GPU["GPU"]
        K0["CUPTI kernel 1 start"]:::cupti
        X0["kernel 1 execution"]:::work
        K1["CUPTI kernel 1 end"]:::cupti
        K2["CUPTI kernel 2 start"]:::cupti
        X1["kernel 2 execution"]:::work
        K3["CUPTI kernel 2 end"]:::cupti
        D0["all related GPU work complete"]:::work
    end

    S0 --> R0 --> L0 --> L1 --> SY0 --> S1
    L0 -. queued work .-> K0 --> X0 --> K1 --> K2 --> X1 --> K3 --> D0
    L1 -. queued work .-> K2
    D0 -. unblocks .-> SY0

    classDef cupti fill:#DBEAFE,stroke:#2563EB,color:#111827
    classDef event fill:#FEF3C7,stroke:#D97706,color:#111827
    classDef cpuwall fill:#DCFCE7,stroke:#16A34A,color:#111827
    classDef work fill:#F3F4F6,stroke:#6B7280,color:#111827
```

因此 SOL 的 `start_cpu/end_cpu` 更像“归因边界”，不是最终的 GPU latency 本身。只要 CUPTI activity 里的 kernel / memcpy / memset 完整落在这个边界内，SOL 就可以用 activity 的 `min(start)` 到 `max(end)` 得到一次 logical call 的 GPU span。

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

展开成时序图，大致如下。这里的 `start_host/stop_host` 是 CPU wall timer，不是 CUDA event，也不是 CUPTI kernel activity timestamp；前后的 synchronize 用来保证 GPU 上已有 work 不污染本轮计时，以及本轮提交的 GPU work 在 stop 前全部完成。

```mermaid
flowchart TB
    subgraph CPU["CPU"]
        P0["setup / warmup<br/>prepare stmt environment"]:::work
        P1["accelerator synchronize<br/>previous GPU work drained"]:::work
        H0["start_host<br/>CPU wall timer"]:::cpuwall
        R0["repeat stmt N times<br/>Python / C++ statement dispatch"]:::work
        L0["cudaLaunchKernel(...)<br/>kernel command enqueued"]:::work
        L1["cudaLaunchKernel(...)<br/>kernel command enqueued"]:::work
        SY0["accelerator synchronize<br/>CPU blocks or polls"]:::work
        H1["stop_host<br/>CPU wall timer"]:::cpuwall
    end

    subgraph GPU["GPU"]
        D0["previous GPU work drained"]:::work
        K0["kernel 1 start"]:::work
        X0["kernel 1 execution"]:::work
        K1["kernel 1 end"]:::work
        K2["kernel 2 start"]:::work
        X1["kernel 2 execution"]:::work
        K3["kernel 2 end"]:::work
        D1["submitted GPU work complete"]:::work
    end

    P0 --> P1 --> H0 --> R0 --> L0 --> L1 --> SY0 --> H1
    P1 -. waits for .-> D0
    L0 -. queued work .-> K0 --> X0 --> K1 --> K2 --> X1 --> K3 --> D1
    L1 -. queued work .-> K2
    D1 -. unblocks .-> SY0

    classDef cupti fill:#DBEAFE,stroke:#2563EB,color:#111827
    classDef event fill:#FEF3C7,stroke:#D97706,color:#111827
    classDef cpuwall fill:#DCFCE7,stroke:#16A34A,color:#111827
    classDef work fill:#F3F4F6,stroke:#6B7280,color:#111827
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
