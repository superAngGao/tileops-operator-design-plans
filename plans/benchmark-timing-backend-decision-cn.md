# TileOps Benchmark 计时后端决策：NCU vs CUPTI + CUDA Event

## 1. 计划定位

这份文档用于讨论 TileOps benchmark 基础设施的计时后端选择。它不是实现 PR 的完整设计文档，也不冻结 benchmark CLI；目标是把当前两个候选方案的实现方式、风险边界和取舍讲清楚，帮助后续 PR review 收敛。

当前需要二选一的方案是：

```text
方案 A: NCU benchmark backend
  使用 Nsight Compute 作为外部 profiler，通过 NVTX range 抓取每次 benchmark call 内的 CUDA kernel duration。

方案 B: CUPTI single-kernel sampling + multi-kernel CUDA event fallback
  保留现有 torch.profiler / Kineto / CUPTI 路径；只对可证明 single-kernel 的 op 使用 CUPTI sampled kernel duration；
  multi-kernel 或无法证明 single-kernel 的 op 走 CUDA event。
```

这两个方案都试图解决同一个问题：现有 CUPTI benchmark 依赖 Kineto 将 CPU `record_function` window 投影到 GPU timeline。实际 nightly 中，projection count 经常不是 `50/50`，导致大量 case fallback 到 CUDA event，而 CUDA event 对短 kernel 会包含明显 launch overhead。

### 1.1 关联 PR / Issue

| 链接 | 关系 |
| --- | --- |
| [tile-ai/TileOPs#1797](https://github.com/tile-ai/TileOPs/pull/1797) | 未合并的实现探索：CUPTI single-kernel sampling + multi-kernel CUDA event fallback |
| [tile-ai/TileOPs#1796](https://github.com/tile-ai/TileOPs/issues/1796) | follow-up issue：跟踪 benchmark CUPTI fallback / projection mismatch 问题 |
| [tile-ai/TileOPs#1795](https://github.com/tile-ai/TileOPs/pull/1795) | 未合并的早期分析 PR：通过 nightly 结果展示 fallback 的普遍性 |
| [Nightly run 30170991562](https://github.com/tile-ai/TileOPs/actions/runs/30170991562) | 触发本轮分析的 nightly benchmark 报告 |
| [superAngGao/tileops-operator-design-plans#1](https://github.com/superAngGao/tileops-operator-design-plans/pull/1) | 补充事实分析：区分可证明的直接触发、上游根因假设和 history 污染 |

### 1.2 实施状态与证据边界

截至 2026-07-29，方案 A 和方案 B 都还没有合入 TileOps `main`。本文讨论的是后续 benchmark timing backend 的设计取舍，不是当前 Nightly 已部署的行为。当前 `main` 仍使用旧规则：只要 projected annotation window 数不等于 `n_repeat`，整次 measurement 就 fallback 到 CUDA Events。

最新补充分析显示，Nightly run #187 仍有 146 条 benchmark report rows 使用 `cuda-events`，其中 GQA 68 条、非 GQA 78 条；run #183 曾有 2,835 条 `cuda-events` rows。由于 `perf_history.json` 没有保存 timing backend，fallback latency 已经和 CUPTI latency 混入同一条历史比较路径。

因此，本文后面的推荐方案还需要一个数据治理前提：timing backend 是 benchmark record 的 provenance，不只是说明字段。无论最终采用 NCU 方案还是 CUPTI sampling 方案，benchmark history 和 regression report 都必须能区分 CUPTI、CUDA Events、NCU diagnostic 和 unknown backend。

## 2. 现有问题

当前 CUPTI 路径的核心流程是：

```python
cache.zero_()
torch.cuda.synchronize()
with torch.profiler.record_function("tileops_bench_kernel"):
    _run(i)
torch.cuda.synchronize()
```

随后代码读取 Kineto events：

```text
CPU record_function
  -> Kineto projection
  -> GPU timeline 上的 annotation window
  -> 统计 window 内 CUDA kernel duration
```

这里有三个不同的窗口：

| 窗口 | 来源 | 用途 | 风险 |
| --- | --- | --- | --- |
| CPU call window | Python `record_function` | 描述一次 logical benchmark call | CPU 侧 window 不等于 GPU 侧完成信号 |
| GPU projected window | Kineto projection | 用来过滤哪些 CUDA kernel 属于本次 call | projection count 可能不是 `n_repeat` |
| CUDA activity event | CUPTI / Kineto CUDA event | 记录单个 kernel 的 start / duration / name | event 是 kernel 粒度，不是 logical call 粒度 |

`torch.cuda.synchronize()` 只能保证 CUDA work drain 完，不能保证 Kineto 的 CPU annotation projection 已经 ready。因此会出现这样的情况：

```text
n_repeat = 50
projected windows = 49
CUDA kernel events = 49 或 50 或其他值
```

旧实现要求 `projected windows == n_repeat`。只要 projection 不是 `50/50`，即使 CUPTI activity 中已经有可用 kernel duration，也会 fallback 到 CUDA event。

从现有 artifact 可以严格证明的直接触发是 `n_regions != n_repeat`。但如果日志没有持久化每次失败的 `n_regions/n_repeat`、captured CUDA kernel count 和 workload/tag，就不能进一步把上游根因写死为某个具体 Kineto、CUPTI、runner 或 callable 组件的问题。

## 3. 方案 A：NCU Benchmark Backend

### 3.1 实现方法

NCU 方案把 profiler 放到 benchmark 进程外部。benchmark 代码只负责标记测量区间：

```python
cache.zero_()
torch.cuda.synchronize()
torch.cuda.nvtx.range_push("tileops_bench_kernel:<call>:<trial>:<iter>")
try:
    _run(i)
    torch.cuda.synchronize()
finally:
    torch.cuda.nvtx.range_pop()
```

外部 runner 使用 Nsight Compute：

```bash
ncu \
  --set none \
  --metrics gpu__time_duration.sum \
  --csv \
  --page raw \
  --target-processes all \
  --nvtx \
  --nvtx-include 'regex:tileops_bench_kernel:.*/' \
  python -m pytest -q <single-nodeid>
```

runner 解析 raw CSV：

```text
NVTX range: tileops_bench_kernel:<call>:<trial>:<iter>
kernel rows under this range
duration metric: gpu__time_duration.sum
```

对每个 logical iteration：

```text
iteration_latency = sum(duration of all kernels inside this NVTX range)
case_latency = average or median over repeat iterations
```

这里不需要提前知道 kernel name。dispatch 到不同 kernel、kernel name 不稳定，都不影响 matching；只要 kernel 落在 NVTX range 内，就会被归属到对应 logical iteration。

### 3.2 优点

| 优点 | 说明 |
| --- | --- |
| 不依赖 Kineto projection | 不需要 CPU `record_function` 投影到 GPU timeline，绕开 `49/50` projection mismatch |
| 不依赖 kernel name | dynamic dispatch、template name 变化、不同 backend kernel name 变化都不会影响归属 |
| logical iteration 边界清晰 | 每次 benchmark iteration 有独立 NVTX range，可按 range 判断是否捕获到对应 kernel |
| 诊断信息丰富 | raw CSV 可保留 kernel name、duration、launch config 等字段，适合 debug |
| 可作为准官方 profiler 对照 | Nsight Compute 是 NVIDIA 工具链，对外解释成本低 |

### 3.3 缺点

| 缺点 | 说明 |
| --- | --- |
| 开销很大 | NCU 每个 kernel 采 metric 会慢很多；即使 repeat 只设为 3，跑完整 nightly 也可能显著变慢 |
| runner 复杂度高 | 需要外部 launcher、单 case 执行、NVTX filter、CSV 解析、sidecar metadata join |
| CI 集成成本高 | 需要 runner 镜像中稳定提供 `ncu`，并处理权限、lock、输出体积、timeout |
| 对 multi-kernel latency 语义仍需定义 | NCU raw duration sum 是 kernel time sum，不一定等于 op wall-clock latency；如果多个 kernel overlap，sum 会大于端到端 latency |
| 报告格式需要改造 | 现有 benchmark / JUnit / nightly report pipeline 主要围绕 pytest process 和 `BenchmarkReport`，NCU 需要额外汇总路径 |

### 3.4 适用边界

NCU 方案适合：

```text
1. 做 benchmark 基础设施的对照验证；
2. 分析 CUPTI / Kineto projection 问题；
3. 对少量关键 op 做 profiler-grade measurement；
4. 需要保留 kernel-level diagnostic 的场景。
```

NCU 方案不太适合直接作为默认 nightly backend，除非能接受更长运行时间和更复杂的 runner 维护成本。

## 4. 方案 B：CUPTI Single-Kernel Sampling + Multi-Kernel CUDA Event Fallback

### 4.1 实现方法

该方案保留现有 benchmark 主路径：

```text
warmup
  -> torch.profiler CUPTI trial
  -> parse Kineto CUDA activity
  -> BenchmarkReport / JUnit / nightly report
```

但改变 CUPTI sample 的可用性判断：

```text
1. 先做一个很小的 profiled classification pass，例如 min(2, n_repeat)。
2. classification 要求：
   - 每个 projected classification window 都存在；
   - 每个 projected call window 内 exactly one business CUDA kernel；
   - 所有 classification window 的 business kernel name 一致。
3. 如果 classification 成功，说明该 callable 满足 single-kernel logical call contract。
4. 正式 CUPTI trial 中，不再要求 projected regions == n_repeat；
   而是统计 window 内 sampled business kernel count。
5. 如果 count 在 [1, n_repeat]，使用 total_kernel_time / sampled_count。
6. 如果 multi-kernel、empty、count > n_repeat、classification 失败，fallback CUDA event。
```

这个方案的关键是只扩大 single-kernel op 的 CUPTI 可用范围：

```text
single-kernel op:
  49 captured kernels / 50 repeats
    -> 49 complete sampled calls
    -> total_kernel_time / 49

multi-kernel op:
  kernel count 不能推出 logical call count
    -> fallback CUDA event
```

### 4.2 正确性补充：不会统计半截 kernel duration

CUPTI / Kineto 的 CUDA kernel activity event 是 kernel 粒度事件。只要一个 kernel event 被捕获，它的 duration 应该对应完整 kernel launch，而不是被 projected annotation window 裁剪后的半截时间。

因此，single-kernel op 出现 `49/50` projection mismatch 时，更合理的理解是捕获到了 49 次完整 sampled calls，而不是捕获到了 50 次 call 里的某些半截 kernel。方案 B 正是利用这一点：只在 classification 证明“一次 logical call 只有一个 business kernel”后，才允许用 sampled kernel count 作为 denominator。

### 4.3 优点

| 优点 | 说明 |
| --- | --- |
| 改动小 | 保留现有 `bench_kernel`、pytest、JUnit、nightly report 主流程 |
| 和现有 CI 兼容 | 不需要外部 NCU runner，也不需要新增 profiler 依赖 |
| 运行时间接近现状 | 只增加很小的 classification pass；不会像 NCU 那样对每个 kernel replay/采 metric |
| 解决主要 fallback 来源 | single-kernel fast ops 是 CUDA event launch overhead 最明显的受害者，`49/50` 可继续使用 CUPTI |
| 风险边界清晰 | multi-kernel 不做 kernel-count 推断，保守走 CUDA event |

### 4.4 缺点

| 缺点 | 说明 |
| --- | --- |
| 仍依赖 Kineto / CUPTI | 没有彻底绕开 torch.profiler 和 Kineto event parsing |
| classification 是经验 contract | 它证明 profiled classification sample 是 single-kernel；如果 callable 后续 data-dependent dispatch 到 multi-kernel，仍需 fallback 或额外诊断 |
| multi-kernel 无法获得 pure kernel timing | multi-kernel op 继续使用 CUDA event，会包含 launch / host-side overhead |
| kernel name 仍参与 single-kernel 判定 | kernel name 不用于 dispatch matching，但用于判定 classification 是否稳定 single-kernel |
| 不能作为 profiler 诊断替代品 | 它是 benchmark timing fix，不是完整 kernel-level analysis tool |

### 4.5 适用边界

该方案适合成为默认 benchmark 后端策略：

```text
1. single-kernel op 尽量使用 CUPTI kernel duration；
2. projection 少量漏计时，允许用 sampled kernel count 作为 denominator；
3. multi-kernel / 不确定 dispatch / ambiguous sample 走 CUDA event；
4. NCU 只作为后续诊断工具或独立实验 runner。
```

## 5. 两个方案的核心差异

| 维度 | NCU backend | CUPTI single-kernel + CUDA event fallback |
| --- | --- | --- |
| profiler 位置 | 外部 Nsight Compute launcher | 进程内 torch.profiler / Kineto |
| 匹配方式 | NVTX range | projected record_function window + CUDA activity |
| 是否依赖 projection | 否 | 是，但不再要求 `regions == n_repeat` |
| 是否依赖 kernel name | 不依赖 kernel name 抓取；可保留作诊断 | single-kernel classification 使用 kernel name 判断稳定性 |
| single-kernel `49/50` | 可按 49 个 NVTX iteration / kernel row 计算 | 可按 49 个 sampled kernels 计算 |
| multi-kernel | 可抓到 range 内多个 kernel，但 duration sum 不等于 wall latency | 直接 fallback CUDA event |
| 运行时间 | 明显更长 | 接近现有 benchmark |
| CI 集成复杂度 | 高 | 低 |
| 诊断能力 | 强 | 中等 |
| 默认 nightly 适配 | 不理想 | 更适合 |

## 6. 推荐决策

推荐后续默认路线采用 **方案 B：CUPTI single-kernel sampling + multi-kernel CUDA event fallback**。

理由是：

```text
1. 当前最紧急的问题是大量 single-kernel fast ops 因 projection mismatch fallback 到 CUDA event。
2. single-kernel op 可以通过 classification 建立 kernel count == sampled call count 的前提。
3. multi-kernel op 已经决定走 CUDA event，不在 CUPTI sampled-count 路径里冒险。
4. 该方案对现有 benchmark / CI / nightly report 改动最小。
5. NCU 虽然更适合诊断 projection，但作为默认 benchmark backend 成本太高。
```

因此，短期实现应收敛为：

```text
default benchmark:
  CUPTI for classified single-kernel samples
  CUDA event for multi-kernel or ambiguous samples

optional diagnostic:
  NCU runner for selected nodeids / selected ops
```

## 7. 后续工作

### 7.1 短期实现应处理

```text
1. 保留现有 per-repeat profiler window，不引入 big window。
2. 增加小规模 profiled classification。
3. 对 single-kernel sample 使用 sampled count。
4. 对 multi-kernel / ambiguous sample fallback CUDA event。
5. 在 warning / report 中保留 fallback 原因。
```

### 7.2 后续 issue 可处理

```text
1. 将 timing backend、fallback reason 和 CUPTI mismatch 诊断持久化到 benchmark result / JUnit / perf history。
2. regression、best 和 improvement 只比较可比 timing backend；旧 history 中缺失 backend 的记录应标为 unknown。
3. 增加可选 NCU diagnostic runner，只跑指定 nodeid / op。
4. 在 report 中统计 CUPTI / CUDA event / NCU diagnostic 的比例。
5. 对 dynamic-dispatch op 记录 classification kernel names 和 fallback 原因。
6. 如果未来需要 multi-kernel pure GPU wall latency，再评估 nsys 或 CUPTI activity start/end span，而不是 kernel duration sum。
```

## 8. 结论

NCU 方案从机制上更干净地绕开 Kineto projection，适合作为诊断和对照；但它运行成本、runner 复杂度和 CI 集成成本都更高。

CUPTI single-kernel sampling + multi-kernel CUDA event fallback 方案不试图解决所有 profiler 问题，而是把可安全使用 CUPTI 的范围收窄到 single-kernel callable。它可以让 `49/50` 这类 partial projection sample 正常产生 kernel-only latency，同时避免对 multi-kernel op 做不可靠 denominator 推断。

因此，后续默认 benchmark 路线应选择 CUPTI single-kernel sampling 方案；NCU 保留为后续可选 diagnostic backend。
