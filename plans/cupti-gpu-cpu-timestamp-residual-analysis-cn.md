# TileOps CUPTI GPU/CPU 时间戳残差分析

更新日期：2026-08-07

关联实现：[tile-ai/TileOPs#1838](https://github.com/tile-ai/TileOPs/pull/1838)

## 摘要

TileOps 在将 benchmark 默认计时路径迁移到 native CUPTI activity attribution
时观察到一个看似违反 CUDA 同步语义的现象：即使 CPU 在
`torch.cuda.synchronize()` 返回之后才记录 attribution window 的结束时间，
部分 CUPTI kernel activity 的转换后 `end` 仍会落在 CPU window 的 `end`
之后。不同机器、进程和运行轮次上的残差并不固定；已有观测约为
8--16 微秒，并且 begin/end 两侧都可能越界。

本轮调查得到的核心结论是：

> `cudaDeviceSynchronize()` 保证执行顺序，不保证分别读取的 GPU PTIMER 与
> CPU 时钟在数值上天然一致。CUPTI 需要用 GPU/CPU correlation anchor 做
> 分段线性转换。NVIDIA 580.82.07 驱动的 TSC correlation 路径先读 GPU
> timer、后读 CPU TSC，没有 CPU-before/GPU/CPU-after bracket，也没有
> midpoint。可变的串行读取延迟会污染 anchor 的截距和斜率；当 CUPTI 对
> segment 做插值或外插时，就会形成方向和大小均不固定的微秒级残差。

这套模型同时解释了：

- synchronize 之后仍出现转换后的 kernel `end` 越界；
- begin 和 end 两侧都可能越界；
- 同一测试重跑时可能通过，也可能失败；
- 不同驱动、机器、GPU 和系统负载下表现不同；
- CPU/TSC 锚点本身只有亚微秒不确定度，却仍不能解释完整的 10--16 微秒
  GPU/CPU 残差。

当前 TileOps PR #1838 采用 `24/24/96 us`，分别表示 begin tolerance、end
tolerance 和 repeat guard。这是基于现有多机观测做出的工程折中，不是
NVIDIA 承诺的理论上界。

## 1. 问题背景

PR #1838 的正式计时流程可简化为：

```text
prepare iteration
torch.cuda.synchronize()

cpu_begin = cuptiGetTimestamp()
launch logical op
torch.cuda.synchronize()
cpu_end = cuptiGetTimestamp()
```

随后 CUPTI 在 activity post-processing 阶段，把 GPU 上记录的 kernel
`start/end` 转换到 CPU timestamp domain，并使用 CPU window 做 repeat
attribution。

从 CUDA 执行顺序看，目标 kernel 必须在第二次 synchronize 返回前完成：

```text
kernel end
    < synchronize returns
    < cpu_end
```

但实验中出现过：

```text
converted kernel end > cpu_end
```

这不表示 synchronize 失效。前者是事后经过 correlation/interpolation
转换的 GPU timestamp，后者是 CPU 直接读取的 timestamp。两者只有在转换
完全准确时才会严格保持原来的数值顺序。

## 2. 观测事实

### 2.1 原始残差

在 H200、CUDA/CUPTI 12.9、驱动 580.82.07 的运行中观察到：

- 另一台机器上的部分 case 需要约 8 微秒 end tolerance；
- 一个 GEMM 的转换后 `end` 超过 CPU window end 约 9.0--9.3 微秒；
- 已经同步完成的 prepare `Fill`，转换后 `end` 超过后续 CPU begin 约
  9.8 微秒；
- 一个 SELU case 的转换后 `start` 落在 CPU begin 之前约 16.4 微秒。

残差具有随机性。相同代码在另一轮或另一台机器上可能完全没有越界。

### 2.2 当前驱动上的对照

在 H200、CUDA/CUPTI 12.9、驱动 595.71.05 上运行最小 probe 时，当前一轮
没有复现旧机器的 10--16 微秒越界：

- normalized activity 的 kernel end 相对 CPU window end 中位数约
  `-2.38 us`，即仍在 window 内；
- 八张 H200 的中位数约为 `-2.12` 到 `-2.45 us`；
- 当前 targeted GEMM 试验的最大 end margin 约为 `-3.6 us`。

这不是对旧观测的否定。后续源码和动态探针显示，correlation anchor 的
质量依赖具体驱动路径、调度状态和采样时机；当前运行只是恰好取得了较稳定
的 anchors。

## 3. CUPTI 的两阶段时间转换

[CUPTI Activity API](https://docs.nvidia.com/cupti/12.9/api/group__CUPTI__ACTIVITY__API.html)
说明，kernel、memcpy 和 memset 等活动的时间戳直接记录在 GPU 上，并在
post-processing 阶段通过线性插值转换到 CPU 时间域。公开文档没有说明：

- GPU/CPU correlation anchor 如何采样；
- anchor 多久刷新一次；
- 是否使用 bracket 和 midpoint；
- 插值以外是否会使用 segment 外插；
- 转换误差的最大微秒上界。

对 CUDA 12.9 `libcupti.so.12` 的 disassembly、callback perturbation 和动态
拦截表明，完整路径是：

```text
GPU PTIMER/globaltimer
    -> CPU TSC
    -> CLOCK_REALTIME 或用户 timestamp callback
```

因此必须把两个误差层分开分析。

### 3.1 CPU TSC 到 CPU timestamp

此前 probe 中被称为 GPU samples 的一组值实际来自 `RDTSCP`。CUDA 12.9
CUPTI 的采样顺序是：

```text
R0, C0, R1, C1, R2, C2, R3, C3, R4, C4, R5
```

其中：

- `R_i` 是 CPU TSC；
- `C_i` 是 `cuptiGetTimestamp()` 默认使用的 `CLOCK_REALTIME`，或注册的
  timestamp callback 返回值。

CUPTI 计算五个 bracket：

```text
d_i = R_(i+1) - R_i
```

选择最窄的 `j`，并构造：

```text
tsc_anchor = R_j + d_j / 2
cpu_anchor = C_j
```

这是一种 minimum-round-trip/midpoint 策略。实测最窄 bracket 的不确定度
约为亚微秒到 1 微秒量级。因此，这一层不能解释完整的 10--16 微秒残差。

### 3.2 GPU PTIMER 到 CPU TSC

CUPTI 通过 NVIDIA RM control command
`NV2080_CTRL_CMD_TIMER_GET_GPU_CPU_TIME_CORRELATION_INFO`（`0x20800406`）
取得 GPU/CPU correlation。

在 NVIDIA 开源驱动 580.82.07 的 TSC clock-source 路径中，
[`tmrGetGpuAndCpuTimestampPair_OSTIMER`](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.82.07/src/nvidia/src/kernel/gpu/timer/timer_ostimer.c)
实现为：

```c
*pGpuTime = tmrGetTimeEx_HAL(pGpu, pTmr, NULL);
*pCpuTime = portUtilExReadTimestampCounter();
```

其物理顺序是：

```text
time ------------------------------------------------------------>

             tG                                  tC
              |                                   |
              +-- read GPU timer -----------------+-- read TSC
              |<------------- delta -------------->|

returned pair = (GPU(tG), TSC(tC))
```

驱动把这两个值作为一个 correlation pair 返回，但没有测量 `delta`。如果
两次读取之间出现锁竞争、MMIO 延迟或 CPU preemption，这个 pair 仍会被
接受。

## 4. 串行 anchor 如何产生残差

设真实 GPU 到 CPU 的局部映射为：

```text
C(G) = a * G + b
```

由于先读 GPU、后读 TSC，第 `i` 个观测 anchor 带有串行延迟 `e_i`：

```text
C_i(observed) = a * G_i + b + e_i
```

用两个被污染的 anchor 拟合直线后，任意 GPU 时间 `G` 上的转换误差为：

```text
error(G) = e_0 + (e_1 - e_0) * (G - G_0) / (G_1 - G_0)
```

这个公式说明：

1. 如果 `e_0 == e_1`，主要表现为固定时间平移；kernel end 可以在数值上
   晚于 synchronize 后记录的 CPU end。
2. 如果 `e_0 != e_1`，差值会污染拟合斜率，而不只是截距。
3. 在两个 anchor 之间，误差在两者之间变化。
4. 在 anchor 区间之外做外插时，误差可以超过两个 anchor 的误差，甚至
   改变符号。

因此，同一套机制可以同时产生正向 end 越界和负向 begin 越界。

## 5. 为什么 PLATFORM_API 更稳健

同一个 RM control 的相邻 `PLATFORM_API` 分支没有使用单次串行 pair。
[驱动源码](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.82.07/src/nvidia/src/kernel/gpu/subdevice/subdevice_ctrl_timer_kernel.c#L346-L403)
明确说明，串行读取 CPU/GPU 时间容易受到 preemption 影响，因此采用：

```text
C0 G0 C1 G1 C2 G2 C3
```

每个 GPU sample 都被两个 CPU sample 包围：

```text
G0 in [C0, C1]
G1 in [C1, C2]
G2 in [C2, C3]
```

驱动选择最窄的 CPU bracket，并返回其中点：

```text
j = argmin(C_(i+1) - C_i)
gpu_anchor = G_j
cpu_anchor = (C_j + C_(j+1)) / 2
```

这样有两个重要性质：

- anchor 不确定度可以由最窄 bracket 的一半估计；
- 如果某组读取中间发生 preemption，其 bracket 会明显变宽，通常不会被
  选中。

该路径还处理 GPU PTIMER 高低 32 位寄存器无法原子读取的问题：读取前后
高位并检查是否发生 wrap，只有高位一致时才接受中间的低位样本。

相较之下，TSC 路径没有 CPU-before sample，因此既不能估计不确定度，也
不能识别被 preempt 的 sample。

## 6. 动态实验

### 6.1 Raw activity 对照

CUDA 12.9 导出了未写入公开 header 的
`cuptiActivityEnableRawTimestamps(uint8_t)`。仅将它作为调查 probe 使用，
可以取得未经 CPU normalization 的 activity timestamp。

将 raw activity 与 kernel 内 `%globaltimer` 对比：

```text
activity start - %globaltimer start = -0.256 us
activity end   - %globaltimer end   = +0.416 us
```

这说明 activity capture 和 GPU 侧 timestamp 本身是亚微秒级一致的。主要
残差发生在后续 GPU 到 CPU 的 normalization，而不是 kernel activity 丢失
或 synchronize 失效。

### 6.2 时钟漂移与 anchor 刷新

一个 14.19 秒的受控实验同时记录 `%globaltimer` 和 host realtime，观察到
原始 GPU/CPU offset 变化约 48.6 微秒：

```text
48.6 us / 14.19 s = 3.42 us/s ~= 3.42 ppm
```

callback probe 观察到 CUPTI 后台约每四秒刷新 calibration：

```text
4000.36 ms
4000.38 ms
4000.44 ms
```

因此一段四秒区间内，两个原始时钟的相对 offset 约变化：

```text
3.42 us/s * 4 s = 13.7 us
```

漂移本身不是转换误差：精确 anchors 可以拟合出正确斜率。但它说明 CUPTI
必须定期重新拟合斜率；此时 `e_1-e_0` 会把串行 anchor 的可变延迟变成
斜率误差，并在 segment 外插时放大。

### 6.3 RM control 调用包络

在驱动 595.71.05 的 H200 上拦截 correlation RM control：

- 常见耗时为 44k--48k TSC cycles；
- invariant TSC 频率约 2.1 GHz；
- 对应外层调用包络约 21--23 微秒；
- 观察到约 46 微秒的 outlier。

这是整个 control call 的包络，不是 GPU-read-to-TSC-read 的精确间隔，不能
把 21--23 微秒直接当作 anchor bias。但它证明该路径中存在足以容纳
10--16 微秒可变偏移的时间范围。

### 6.4 强制 PLATFORM_API 实验

在驱动 595.71.05 的 `libcuda.so.1` 中，将内部 correlation request 的
`cpuClkId` 从 TSC 强制改为 `PLATFORM_API` 后，驱动确实返回了 bracketed
GPU/platform pair。但 CUPTI 仍按 TSC 语义解释第二个值：

```text
device %globaltimer duration: 10.144 us
normal TSC activity duration: 10.881 us
forced PLATFORM_API duration:  5.120 us
```

activity 的绝对 timestamp 也完全离开 CPU window 时间域。这是因为
`PLATFORM_API` 返回 platform performance-counter 的单位和 epoch，而 CUPTI
继续对它应用 TSC 到 CPU timestamp 的转换。

进一步把 CUPTI 第二阶段人为改成近似恒等映射，可以让 duration 恢复到约
10.240 微秒，但 activity 与 `CLOCK_MONOTONIC_RAW` 仍有约 1370 秒的 epoch
差，需要再增加一层私有校准。

因此可以得出：

- `PLATFORM_API` 的 bracket 机制适合作为根因验证；
- CUPTI 12.9 没有公开 API 选择该 clock source；
- 直接强制切换不能用于 TileOps；
- 自行实现需要私有 RM control、undocumented raw timestamps 和自定义时钟
  转换，不适合作为可维护的 benchmark 基础设施。

## 7. 为什么不同机器表现不同

残差不是固定的硬件常数，而是以下因素共同决定的：

| 因素 | 可能影响 |
| --- | --- |
| 驱动版本和 RM/GSP 路径 | 改变 GPU timer correlation 的实现和延迟 |
| CPU 调度、抢占和系统负载 | 改变两个串行读取之间的 `e_i` |
| NUMA/socket 和 CPU TSC 环境 | 改变 host correlation 路径与稳定性 |
| GPU/CPU 相对漂移 | 决定每个 calibration segment 需要修正的斜率 |
| anchor 刷新时机 | 决定采样是否恰好遇到 preemption/lock contention |
| kernel 在 segment 中的位置 | 决定使用插值还是外插，以及误差放大程度 |

因此可能出现：

```text
机器 A：anchor delay 稳定
  -> 主要表现为较小固定偏移

机器 B：相邻 anchor delay 差异大
  -> 拟合斜率被污染
  -> begin/end 两侧出现 10--16 us 残差

机器 C：本轮恰好取得高质量 anchors
  -> 残差只有 3--4 us
  -> 无法复现旧机器问题
```

同一台机器不同进程和不同运行时间也可能不同，因为 CUPTI 后台刷新 anchor
时的调度状态不同。这解释了全量 benchmark 某轮失败、立即重跑却通过的
随机性。

## 8. 对 TileOps attribution window 的影响

### 8.1 Window tolerance 的语义

Window tolerance 是 GPU/CPU timestamp conversion 的 attribution 容差，
不是对 CUDA 执行顺序的放宽。

容差太小会漏掉目标 kernel：

```text
converted target end > cpu_end + tolerance
```

容差太大则会捕获真实存在的相邻工作。本轮 `32/32/96 us` 全量实验中，
begin tolerance 把 prepare `Fill` 捕获进来：

```text
expected = [GEMM]
actual   = [Fill, GEMM]
```

`Fill` 的转换后 start 位于 begin 前约 31.9--32.0 微秒，因此 32 微秒 begin
tolerance 已接近误捕边界。

### 8.2 Repeat guard 为什么不能隔开当前 prepare

96 微秒 repeat guard 位于：

```text
previous repeat end
guard
current prepare
current begin
```

它可以防止前一轮扩大的 end window 触及下一轮 prepare，但不会在当前
prepare 和当前 begin 之间创建间隔。因此增大 guard 不能解决 widened begin
window 捕获当前 `Fill` 的问题。

如果未来必须继续扩大 begin tolerance，应增加独立的 post-prepare spacing，
而不是继续增大 repeat guard。

### 8.3 为什么独立 discovery 通常看起来更稳定

Discovery iteration 并没有使用另一套 GPU 时钟，它同样会受到上述
normalization residual 影响。它通常更容易成功，主要来自归因条件不同：

- discovery 只运行一个隔离的 logical call；
- prepare 已同步，前后没有密集排列的 timed repeats；
- discovery 的任务是建立 expected activity sequence，而不是拿既定 sequence
  对 widened window 做 exact match；
- 即使整组时间戳有几微秒平移，目标 kernel 通常仍有足够大的 start margin
  落在孤立窗口内。

正式 timing 则要在多个相邻 repeat 中切分 activity，并要求每一轮匹配同一个
expected sequence。此时同样大小的 conversion residual 可能把 prepare 前缀
纳入当前窗口，或把目标尾部推出窗口，因而更容易暴露问题。

所以“计时前那一次总是抓得准”是当前 workload 和隔离布局下的经验现象，
不是 discovery 使用了更准确的 timestamp source，也不是 CUPTI 的正式保证。

### 8.4 当前工程选择

PR #1838 当前默认值为：

```text
begin tolerance = 24 us
end tolerance   = 24 us
repeat guard    = 96 us
```

选择 24 微秒的原因是：

- 覆盖已经观察到的 8--16 微秒 conversion residual；
- 保留一定经验余量；
- 避免达到已知 `Fill` start 的 31.9--32.0 微秒误捕边界。

它不是永久安全边界。驱动、CUDA、机器或负载变化后，应重新统计：

```text
converted activity start - cpu_begin
converted activity end   - cpu_end
prepare activity 到 begin 的距离
相邻 repeat activity 到 widened window 的距离
```

## 9. 证据边界

本文有意区分事实、实现特定发现和机制推导。

### 9.1 已直接验证

- raw CUPTI activity 与 `%globaltimer` 亚微秒级一致；
- CUDA 12.9 CUPTI 使用 GPU/PTIMER -> TSC -> CPU timestamp 两阶段转换；
- CUPTI 的 TSC/CPU timestamp anchor 使用五次 bracket 并选最窄样本；
- 驱动 580.82.07 的 GPU/TSC pair 是先 GPU、后 TSC 的串行读取；
- 相邻 `PLATFORM_API` 路径明确用 bracket 缓解 preemption；
- CUPTI 后台 calibration 在当前实验中约四秒刷新；
- 当前机器的 GPU/CPU raw offset drift 约 3.42 ppm；
- 强制切换 `PLATFORM_API` 会因 CUPTI 时钟语义不匹配产生错误结果。

### 9.2 机制推导

由串行 anchor bias、实测 drift、分段线性转换和 observed residual，可以得到
一个能够解释现有全部现象的模型：不同 anchor 的可变 bias 污染截距和
斜率，插值/外插产生随机、双向的微秒级 residual。

我们没有直接取得 CUPTI 内部每个最终被选中 GPU/TSC anchor 的真实物理
采样时刻，因此不能把某一次 9.3 或 16.4 微秒残差逐纳秒分解为 RM、
preemption、drift 和 extrapolation 各自贡献多少。本文结论是机制解释和
量级闭环，不是对闭源 CUPTI 内部状态的完整形式化证明。

## 10. 建议

### 10.1 TileOps

1. 保留当前 `24/24/96 us` 默认值和环境变量覆盖能力。
2. attribution 继续 fail-closed：missing、unexpected 或 dropped activity
   不能静默产生正式 latency。
3. JUnit/nightly/history 保存 tolerance、guard、timing backend 和 protocol
   version。
4. 不同 timing protocol 的历史结果不得直接比较。
5. 在不同驱动/CUDA 组合上持续收集 begin/end residual 分布。
6. 如果未来扩大 begin tolerance，单独引入 post-prepare spacing。

### 10.2 NVIDIA/CUPTI

适合向 NVIDIA 提交最小复现并询问：

1. CUPTI 是否可以公开选择 bracketed GPU/CPU correlation source；
2. TSC 路径能否改为 `TSC-before/GPU/TSC-after`，重复采样后选择最窄
   bracket 和 midpoint；
3. CUPTI 能否公开 calibration refresh、segment 和 residual bound；
4. 是否可以提供受支持的 raw GPU activity timestamp 与 correlation API。

在 NVIDIA 提供正式支持前，不建议把私有 RM control 或 undocumented CUPTI
symbol 带入 TileOps production benchmark。

## 11. 参考资料

- [CUPTI 12.9 Activity API](https://docs.nvidia.com/cupti/12.9/api/group__CUPTI__ACTIVITY__API.html)
- [NVIDIA 580.82.07 GPU/CPU correlation control](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.82.07/src/nvidia/src/kernel/gpu/subdevice/subdevice_ctrl_timer_kernel.c)
- [NVIDIA 580.82.07 TSC pair implementation](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.82.07/src/nvidia/src/kernel/gpu/timer/timer_ostimer.c)
- [TileOps PR #1838: Align benchmarks with native CUPTI](https://github.com/tile-ai/TileOPs/pull/1838)
- [TileOps Benchmark Timing SOL 对齐技术报告](benchmark-timing-sol-alignment-report-cn.md)
- [TileOps Benchmark 计时后端决策](benchmark-timing-backend-decision-cn.md)
