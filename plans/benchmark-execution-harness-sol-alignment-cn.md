# TileOps Benchmark Execution Harness 调研与 SOL 对齐目标

更新日期：2026-08-04

## 1. 目标

这份笔记只讨论 benchmark execution harness，不重新展开 timing backend 的取舍。当前方向是保留 CUPTI activity timing 原则，并把 execution harness 的设计目标收敛成四个主需求和一个次要目标。

主需求：

```text
1. 环境稳定
2. 严格串行
3. 重复测量前后无污染
4. 可观测、可追溯
```

次要目标：

```text
5. 运行成本可控
```

这里默认 TileOps 与 baseline 使用同一套 benchmark 执行口径。后续所有具体实现，例如 native CUPTI discovery、L2 flush、input address policy、strict serial runner，都应该服务于这些需求。

参考资料：

- NVIDIA SOL-ExecBench README: https://github.com/NVIDIA/SOL-ExecBench
- NVIDIA SOL-ExecBench timing implementation: https://github.com/NVIDIA/SOL-ExecBench/blob/main/src/sol_execbench/core/bench/timing.py
- NVIDIA SOL-ExecBench input allocator: https://github.com/NVIDIA/SOL-ExecBench/blob/main/src/sol_execbench/core/bench/io.py
- SOL-ExecBench technical report: https://arxiv.org/abs/2603.19173

## 2. 主需求一：环境稳定

判断：这一项 TileOps 当前问题不大，但还需要把环境信息完整落到 artifact 中。

TileOps 现状：

- Nightly 使用官方 runner 镜像。
- Workflow 验证 GPU current graphics clock 为 1500 MHz。
- 使用共享 cache path，例如 `/ci-cache`、`TILELANG_CACHE_DIR`、`TRITON_CACHE_DIR`。
- 安装流程使用镜像内预装依赖，TileOps 自身通过 CI 脚本安装。

我们的需求：

- GPU 型号、driver、CUDA、torch、tilelang、runner image、clock、cache path 都应该可追溯。
- benchmark 结果要能区分冷 cache、热 cache、cache miss、不同镜像或不同依赖状态。
- 环境不稳定时，不应该产出和正式 nightly 混用的 latency。

SOL 可借鉴点：

- SOL 官方评测在固定 GPU 与锁频环境中运行。
- SOL trace 会记录 evaluation environment，便于结果复现和排查。
- SOL 默认 benchmark config 明确记录 warmup / iterations / seed / lock clocks 等参数。

我们的解法：

- 继续保留 1500 MHz clock 校验。
- 在 TileOps benchmark artifact 中记录 runner image、clock、driver、CUDA、torch/tilelang 版本和 cache env。
- 明确 cache 状态对 benchmark 数据的影响，至少在 debug artifact 中保留 cache path 与关键 miss 线索。

## 3. 主需求二：严格串行

判断：这是当前最需要详细研究的主需求。#1761 引入的 per-file 子进程隔离解决了 crash/OOM/hang 影响面问题，但 runner 为了吞吐加入了 prewarm 和后台 teardown，这导致执行过程不是完全严格串行。

TileOps 现状：

- `scripts/ci/run_benchmarks.py` 每个 `bench_*.py` 启动一个 pytest child。
- child 先执行 `pytest --collect-only -q <bench_file>`，然后等待 parent 释放后再正式运行 pytest。
- 默认 `--prewarm=4`，即后续 benchmark file 的 collect/import 可以和当前 benchmark file 的执行重叠。
- child 写回 pytest rc 后，`wait_result()` 就返回；interpreter teardown 可能继续在后台运行。
- parent 将未退出 child 放入 `lingering`，最多等待 `--teardown-timeout`。

已知风险：

- collect/import 被假设为 GPU-silent；如果某个 benchmark module import 时初始化 CUDA，就会和当前测量阶段重叠。
- 上一个 child 的 interpreter teardown 可能继续持有 CUDA context；下一个 child 已经进入 warmup/timing 时可能发生冲突。
- 在 GPU `Exclusive_Process` 模式下，这个问题会直接暴露为 `cudaErrorDevicesUnavailable`。
- 在 default compute mode 下，即使不报错，也需要验证是否会影响 timing 稳定性。

SOL 可借鉴点：

- SOL 强调 serialized GPU benchmarking 和 isolated subprocess execution。
- SOL evaluation subprocess 是明确的一次 solution evaluation，不通过 pytest prewarm 隐藏后续 import 成本。
- 失败 worker 可以被隔离处理，不要求后续 benchmark 与前一个 worker teardown 重叠。

我们的解法：

- 为 nightly 增加 strict serial runner mode：不 prewarm 后续 benchmark file；前一个 child 完全退出后再启动或释放下一个正式 benchmark。
- 保留 per-file isolation 的 crash/OOM/hang 防护价值，但避免它破坏 GPU 独占语义。
- 增加 instrumentation，记录 child collect start/end、run start/end、pytest rc、process exit、teardown deadline。
- 实测 strict serial 与当前 prewarm mode 对总耗时和 latency 稳定性的影响。

小规模实验记录：

| 日期 | 环境 | 对比 | 测例 | 结果 |
| --- | --- | --- | --- | --- |
| 2026-08-04 | `ghcr.io/tile-ai/tileops-runner:65dbc98-torch2.10`；GPU0 H200；`Exclusive_Process`；1500 MHz；共享 cache | default prewarm runner | `bench_independent_elementwise.py`、`bench_logical_reduce.py`、`bench_gemm.py` | 退出码 1，总耗时 154s。第一个文件通过；后两个文件在创建 CUDA tensor 时出现 `cudaErrorDevicesUnavailable`。 |
| 2026-08-04 | 同上 | strict serial runner | 同上 | 退出码 0，总耗时 172s。三个文件全部通过；每个文件后等待到 `teardown complete` 再启动下一个 child。 |
| 2026-08-04 | 同上；default prewarm runner；collect 阶段加 GPU-silent probe | collect/import GPU-silent 检查 | 同上 | 三个 child collect 完成时均为 `torch_loaded=True`、`cuda_initialized=False`、`gpu_rows=[]`；随后第二、第三个文件在正式 pytest 创建 CUDA tensor 时仍出现 `cudaErrorDevicesUnavailable`。 |
| 2026-08-04 | 官方 runner 镜像；GPU3 H200；1500 MHz；共享 cache；strict serial；`--prewarm 0`；host pmon 采样 | 全量 `benchmarks/ops` | 59 个 benchmark file | 总 wall time 2357.58s；未出现 `cudaErrorDevicesUnavailable`；pmon 463 个采样点中没有同时出现多个 GPU compute process；5 个 file 因 native CUPTI attribution fail-closed。 |

初步解读：

- 在 GPU `Exclusive_Process` 下，当前 default prewarm runner 可以触发后续 child 拿不到 CUDA device 的问题。
- strict serial runner 能消除这组小规模测例中的 device unavailable 失败。
- collect/import 在这组三个文件上没有初始化 CUDA，也没有在 collect 完成时出现在 GPU compute app 列表中；因此当前证据不支持“prewarm collect 阶段抢 GPU”是直接原因。
- 全量实验中有 7 个 benchmark file 在 collect/import 阶段已经 `cuda_initialized=True`，但 strict serial 下它们没有和其他 benchmark file 的测量阶段重叠；后续 artifact 仍应记录这类信息。
- 剩余风险点是 child release 后正式 pytest 与上一个 child 的 CUDA context / interpreter teardown 重叠；但这个 root cause 不再作为主方案 blocker。
- 主路径采用 strict serial。prewarm 仅保留为未来成本优化方向；只有在额外证明不会影响 GPU context 和 latency 稳定性后再考虑恢复。

## 4. 主需求三：重复测量前后无污染

判断：这一项很难从单一错误日志直接看出问题，需要从 execution policy 本身分析。重点是输入地址、输出状态、L2/cache 状态、flush 与同步边界是否稳定。NV SOL 在这里有多项可借鉴做法。

TileOps 现状：

- `bench_kernel` 默认 `n_warmup=10`、`n_repeat=50`、`n_trials=3`。
- warmup 中执行 `cache.zero_(); _run(i)`，最后 `torch.cuda.synchronize()`。
- 每个 timed repeat 前执行 `cache.zero_(); torch.cuda.synchronize()`，再进入被测调用。
- L2 flush buffer 使用实际 L2 size；当前 main 没有调用 `cudaCtxResetPersistingL2Cache()`。
- 输入 tensor 默认预 clone 3 份 pool，按 `i % 3` 轮换；总输入超过 1 GiB 时跳过 clone 并记录 `inputs_cloned=False`。
- case 结束后，pytest hook 调用 `gc.collect()` 和 `torch.cuda.empty_cache()`。
- 多数 TileOps benchmark 先构造 op、完成 `tune=True` 触发的 autotune，再进入 `bench_kernel()`。正式方向使用 CUPTI activity timing，CPU 侧 JIT/cache miss 时间不会直接进入 latency；因此 TileOps 当前没有显著的独立预编译需求。

我们的需求：

- 每次 repeat 前的 cache/input/output 状态要尽量一致。
- flush/setup work 不能进入被测 user activity sequence。
- repeat 之间不能通过固定 data pointer、输出残留、allocator 状态或 L2 命中状态产生隐藏耦合。
- 大输入无法 clone 时，要在结果中显式标记，不和正常地址扰动数据混用。
- warmup 仍然保留，但它是 steady-state 边界的一部分：用于让 CUDA context/module/library、allocator 和外部 baseline lazy state 在 discovery/timing 前稳定，而不是作为单独的预编译目标。

SOL 可借鉴点：

- 每次 warmup、discovery、timed iteration 之前都走统一的 `prepare_iteration()`。
- `prepare_iteration()` 中先 setup args，再 reset persisting L2 cache，再 clear L2 buffer。
- L2 buffer 大小约为 `2 * L2_cache_size`。
- `ShiftingMemoryPoolAllocator` 预分配 input/output pool，每次 iteration 返回不同 `data_ptr`，同时保持 seed 可复现。
- SOL 使用 discovery sequence 区分用户 activity 与 setup/cache noise。

我们的解法：

1. **统一 prepare path**
   warmup、discovery、timing 都调用同一套 `prepare_iteration()` 语义。prepare 负责提供本轮输入/输出 view、reset persisting L2、clear L2 buffer，并在进入 user attribution window 前完成必要同步。

2. **采用 SOL-style cache policy**
   每次 user call 前执行：

   ```text
   setup args / outputs
   cudaCtxResetPersistingL2Cache()
   clear L2 buffer
   torch.cuda.synchronize()
   ```

   L2 buffer 默认按 SOL 当前实现使用约 `2 * L2_cache_size`。这样比当前单纯 `cache.zero_()` 更接近 SOL 口径，也避免 persisting L2 状态成为隐性变量。

3. **采用 SOL-style input address policy**
   将当前 3-clone pool 收敛为 preallocated shifting allocator 或等价机制。原则是每个 warmup / discovery / timed iteration 都拿到不同 `data_ptr`，同时不把 allocation / clone 成本放入正式计时窗口。

4. **固定 attribution window 顺序**
   native CUPTI timing 的顺序固定为：

   ```text
   prepare_iteration()
   start_cpu = cupti.get_timestamp()
   runner(args)
   torch.cuda.synchronize()
   end_cpu = cupti.get_timestamp()
   ```

   flush / setup work 必须在 `start_cpu` 之前完成并 drain；正式 attribution 只从 `[start_cpu, end_cpu]` 中选择 discovery 得到的 expected user activity sequence。

验证项：

- 用 native CUPTI activity 证明 flush memset/kernel 不进入 expected user sequence。
- 对 fixed-address、3-clone、shifting allocator 做小范围 latency 分布对比；这个实验用于量化收益，不阻塞采用 SOL-style address policy。

本轮实验结果：

| 日期 | 环境 | 测例 / policy | sequence | 结果 |
| --- | --- | --- | --- | --- |
| 2026-08-04 | 官方 runner 镜像；GPU0 H200；共享 cache；native CUPTI probe | `torch.add`；current flush + fixed address | 1 kernel | timing 50/50；`dropped=0`；flush kernel 50 个，全部在 attribution window 外；inside/outside kernel name 无重叠；mean 9.089 us，std 0.257 us，p90 9.344 us。 |
| 2026-08-04 | 同上 | `torch.add`；current flush + 3-clone | 1 kernel | timing 50/50；`dropped=0`；flush 全部在窗口外；mean 9.013 us，std 0.257 us，p90 9.213 us。 |
| 2026-08-04 | 同上 | `torch.add`；SOL-style flush + fixed address | 1 kernel | timing 50/50；`dropped=0`；flush 全部在窗口外；mean 9.096 us，std 0.252 us，p90 9.309 us。 |
| 2026-08-04 | 同上 | `torch.add`；SOL-style flush + 3-clone | 1 kernel | timing 50/50；`dropped=0`；flush 全部在窗口外；mean 9.045 us，std 0.318 us，p90 9.658 us。 |
| 2026-08-04 | 同上 | `torch.add`；SOL-style flush + shifting address | 1 kernel | timing 50/50；`dropped=0`；flush 全部在窗口外；mean 9.049 us，std 0.258 us，p90 9.213 us。 |
| 2026-08-04 | 同上 | GDN forward；current flush + fixed address | 7 kernels | timing 50/50；`dropped=0`；flush 全部在窗口外；mean 300.617 us，std 12.864 us，p90 310.140 us。 |
| 2026-08-04 | 同上 | GDN forward；SOL-style flush + fixed address | 7 kernels | timing 50/50；`dropped=0`；flush 全部在窗口外；mean 302.267 us，std 13.345 us，p90 314.841 us。 |

结论：

- 在 single-kernel 和 multi-kernel case 上，`prepare_iteration()` 中的 flush work 都没有泄入 user activity sequence。
- SOL-style `cudaCtxResetPersistingL2Cache() + 2 * L2 cache clear` 不破坏 native CUPTI attribution。
- `fixed address / 3-clone / shifting address` 在这组小范围 `torch.add` 中结果接近；正式方案仍采用 shifting allocator，因为它更符合“repeat 之间无地址复用污染”的需求。
- TileOps 选择 CUPTI activity timing 后，即便发生 cache miss 或 CPU 侧 JIT，也不会直接进入正式 latency；初始化边界不再作为独立目标，只在本需求下约束 discovery/timing 前的 steady-state。

## 5. 主需求四：可观测、可追溯

判断：这里一定有问题。之前后端大量 fallback 到 CUDA event 时，我们没有第一时间从正式 report / history 中看出来。即使后续只保留一个 native CUPTI backend，错误、降级、归因失败、异常隔离信息也必须输出。

TileOps 现状：

- `BenchmarkReport` 记录 latency、tflops、bandwidth 等性能字段。
- fallback 时可能记录 `timing`，但历史数据里曾存在 CUPTI 与 CUDA event 混用。
- `profile_run.log`、JUnit properties、nightly report、perf history 之间的信息粒度不一致。
- runner 能为 timeout child 输出 py-spy dump，但 timing attribution 失败的细节不一定进入长期 artifact。

我们的需求：

- 每条结果必须记录 timing backend 和关键执行参数。
- native CUPTI 失败时不能静默产出看似正常的 latency。
- activity sequence mismatch、CUPTI dropped records、CUDA error、OOM、timeout、teardown crash 都要有明确 failure reason。
- 异常要能定位到 benchmark file、pytest nodeid、op、tag、workload、trial/repeat。
- 一个 file/case 的 crash、hang、OOM 不能污染后续 benchmark，也不能让 artifact 无法解释。

SOL 可借鉴点：

- SOL trace 明确记录 evaluation status、performance、environment、extra message。
- SOL 使用 expected activity sequence / counts 做 validation；不满足时显式失败。
- subprocess evaluation 使失败 worker 与其他 workload 隔离。

我们的解法：

1. **正式 backend fail closed**

   新 benchmark 主路径只保留 native CUPTI activity attribution。正式 nightly 中，如果 discovery、activity collection、sequence matching、count validation 或 CUPTI buffer flush 出错，不静默 fallback 到 CUDA event，也不产出可比较 latency。

2. **diagnostic 与正式结果分离**

   CUDA event、CPU wall、NCU 或额外 probe 只能作为 diagnostic backend。它们可以进入 debug artifact，但必须显式标记，不能进入同一条正式 performance history。

3. **每条结果记录 timing provenance**

   `BenchmarkReport`、JUnit properties 和 perf history 至少保存：

   ```text
   timing_backend
   n_warmup / n_repeat / n_trials
   expected_activity_sequence
   expected_activity_counts
   attributed_calls
   dropped_records
   input_policy
   cache_policy
   failure_reason
   ```

4. **失败可定位**

   失败信息必须能定位到：

   ```text
   benchmark file
   pytest nodeid
   op / tag / workload
   trial / repeat
   child pid / runner lifecycle
   ```

5. **异常隔离继续保留**

   per-file subprocess isolation 仍然保留，用于限制 crash、hang、OOM 的影响面。strict serial 只改变 child 调度顺序，不取消隔离。

6. **全量实验中的真实 fail-closed 结果**

   strict serial + native CUPTI + SOL-style prepare path 的全量实验中，失败集中在 attribution 语义本身，而不是 GPU unavailable：

   ```text
   bench_convolution.py          discovery sequence inconsistent / no sequence
   bench_elementwise_manifest.py timing attributed 48/50
   bench_grouped_gemm.py         timing attributed 39/50
   bench_norm.py                 timing attributed 47/50、49/50；discovery inconsistent / no sequence
   bench_vector_norm.py          timing attributed 40/50、48/50、49/50
   ```

   这些失败没有静默 fallback 到 CUDA event，也没有生成可混用 latency。下一步需要改进 native CUPTI attribution 策略，例如允许完整 sequence 的部分样本进入统计，或对 discovery 不稳定的 workload 做更明确的分类与诊断。

暂缓项：

- perf history 的 schema 迁移可以单独设计；但在 native CUPTI PR 中至少要保证新字段进入当前 artifact，避免再次出现 backend/fallback 不可见的问题。

本轮实验结果：

| 日期 | 环境 | 故障注入 | 结果 |
| --- | --- | --- | --- |
| 2026-08-04 | 官方 runner 镜像；GPU0 H200；共享 cache；native CUPTI probe | `dropped_records` | 明确报错：`CUPTI dropped 1 records during timing`。 |
| 2026-08-04 | 同上 | `wrong_expected_sequence` | 明确报错：`CUPTI timing found no complete expected kernel sequence`。 |
| 2026-08-04 | 同上 | `sample_count_mismatch` | 明确报错：`CUPTI timing attributed 10/11 complete expected kernel sequences`。 |
| 2026-08-04 | 同上 | `partial_activity_trace` | 明确报错：`CUPTI timing attributed 9/10 complete expected kernel sequences`。 |

结论：native CUPTI 正式路径可以 fail closed；activity dropped、sequence 错误、sample count 不足都不会静默生成 latency。

## 6. 次要目标：运行成本可控

判断：最后再扫。当前优先级是先把前四个主需求做准，再看能不能优化运行时间。

TileOps 现状：

- 当前 runner 用 prewarm 隐藏后续 pytest collect/import 成本。
- native CUPTI / SOL-style attribution 相比 Kineto 路径会增加 discovery、timestamp window、sequence matching、validation 等成本。
- 全量 benchmark 时间对 nightly 可用性很重要。

我们的需求：

- 严谨性优先于吞吐，但 full nightly 不能无限变慢。
- strict serial、native CUPTI、shifting allocator、extra artifact 都要有成本评估。
- debug-only instrumentation 不应该无条件进入正式 nightly 热路径。

SOL 可借鉴点：

- SOL 默认配置清楚：warmup、iterations、seed、methodology。
- SOL 把 correctness、evaluation、timing、trace 输出分阶段管理，便于控制每阶段成本。

我们的解法：

1. **正确性路径优先**

   默认 nightly 先采用 strict serial + native CUPTI + SOL-style prepare path。prewarm 不再作为 correctness 方案的一部分。

2. **prewarm 降级为未来成本优化**

   当前实验已经说明 default prewarm 在 `Exclusive_Process` 下会破坏严格串行。collect/import 在已测文件上是 GPU-silent，因此继续深挖 root cause 更像 runner forensic；即使暂时研究不清，也不影响主方案。后续只有在证明不影响 GPU context 和 latency 稳定性后，才考虑作为可选 fast path。

3. **先测全量正确性与总耗时**

   在主需求收敛后，先跑一轮 strict serial + native CUPTI 全量 benchmark，得到：

   ```text
   total wall time
   per-file time
   pass/fail files
   attribution failure count
   artifact size
   ```

4. **heavy diagnostics 默认关闭**

   release-time GPU probe、完整 child lifecycle trace、NCU 对照、CPU wall 四路测量等诊断只在 debug mode 或 failure artifact 中启用，不进入默认 nightly 热路径。

暂缓项：

- prewarm fast path 的收益和风险后续再评估；当前不继续把它作为 blocker。

全量实验记录：

| 日期 | 环境 | 配置 | 结果 |
| --- | --- | --- | --- |
| 2026-08-04 | 官方 runner 镜像；GPU3 H200；1500 MHz；共享 cache | strict serial；`--prewarm 0`；native CUPTI；SOL-style prepare path；CUDA event fallback disabled | 59 个 file 全部执行；总 wall time 2357.58s；child elapsed sum 2215.47s；5 个 file fail-closed；无 `cudaErrorDevicesUnavailable`。 |

耗时分布：

```text
top 5 files  = 1289.42s，占 child elapsed 58.2%
top 10 files = 1539.44s，占 child elapsed 69.5%
top 15 files = 1678.13s，占 child elapsed 75.7%
runner / docker / teardown 等非 child elapsed 差值约 142.11s，占 wall time 6.0%
```

最重文件：

```text
bench_grouped_gemm_baselines.py 640.42s
bench_gqa.py                    259.13s
bench_gqa_sliding_window.py     159.41s
bench_gemm.py                   148.39s
bench_moe_fused_moe.py           82.08s
```

结论：当前 strict serial + native CUPTI 的全量成本约 39.3 分钟，成本可控；主要耗时来自少数重 benchmark file，而不是 runner 串行化本身。prewarm 即使恢复，理论收益也主要来自约 142s 的非 child elapsed 和部分 collect/import 隐藏成本；在 attribution 正确性稳定之前，优化优先级低于 native CUPTI attribution 策略收敛。

## 7. 实验矩阵与状态

本轮先按机制收敛顺序完成小规模验证：重复测量前后无污染、可观测性和 strict serial runner。全量 benchmark cost 需要等 native CUPTI + strict serial + SOL-style prepare path 合成到同一实现后再测。

| 优先级 | 实验 | 目标 | 对比 / 测例 | 主要指标 | 状态 |
| --- | --- | --- | --- | --- | --- |
| 1 | Prepare / attribution boundary | 验证 SOL-style prepare path 不污染 user activity sequence。 | `torch.add` single-kernel；GDN forward multi-kernel；检查 `prepare_iteration -> start_ts -> runner -> sync -> end_ts`。 | expected activity sequence 不包含 L2 flush；attributed calls == repeats；dropped records == 0。 | 已完成：single-kernel 和 7-kernel case 都是 timing 50/50、`dropped=0`，flush 全部在 attribution window 外。 |
| 2 | Cache / address policy | 验证 SOL-style persisting L2 reset、L2 clear 和 shifting allocator 的稳定性。 | fixed address、3-clone、shifting allocator；current flush vs reset persisting L2 + `2 * L2_cache_size` clear。 | latency mean/std/p90、outlier、地址策略是否引入稳定偏差。 | 已完成小范围验证：三种 address policy 在 `torch.add` 上结果接近；SOL-style cache policy 不破坏 attribution。 |
| 3 | Artifact / failure observability | 验证错误不会静默混入正式性能数据。 | 构造 CUPTI dropped、sequence mismatch、sample count mismatch、partial trace。 | 错误是否 fail closed；是否给出明确 failure reason。 | 已完成：四类故障都明确报错，没有静默生成 latency。 |
| 4 | Full benchmark cost | 在正确性路径稳定后评估全量运行成本。 | strict serial + native CUPTI + SOL-style prepare path；fallback disabled；全量 `benchmarks/ops`。 | 全量耗时、单文件耗时分布、pass/fail files、GPU 干扰情况。 | 已完成一轮：wall time 2357.58s；无 GPU unavailable；5 个 file 因 native CUPTI attribution fail-closed。 |

实验执行原则：

- 先小规模验证机制，再跑全量。
- 每轮记录代码版本、runner 镜像、GPU clock、cache path 和关键 env。
- 正式结论只来自同一 commit、同一 runner 镜像、同一 cache policy 下的对比。
- 如果实验中发现新的污染源，先回到对应主需求章节修正文档，再继续后续实验。

## 8. 细节索引

下面的表只作为实现排查索引，不作为正文主线。

| 时序环节 | 归属需求 | TileOps 现状 | SOL 可借鉴点 | 下一步 |
| --- | --- | --- | --- | --- |
| Runner 启动 | 环境稳定 | workflow 验证 1500 MHz；使用官方 runner 镜像和共享 cache。 | 固定评测环境与 lock clocks。 | 补充 image / clock / cache provenance。 |
| File 调度 | 严格串行 | 每个 benchmark file 一个 pytest child；默认 `--prewarm=4`。 | serialized GPU benchmarking。 | 增加 strict serial mode。 |
| Collect/import | 严格串行 | child collect-only 可与当前 benchmark 重叠。 | 不通过 pytest prewarm。 | 检查 collect 是否 GPU-silent。 |
| Teardown | 严格串行 | pytest rc 写回后 interpreter teardown 可后台继续。 | dedicated subprocess isolation。 | strict serial 下等待 child 完全退出。 |
| Warmup | 无污染 / steady-state | 10 次 warmup，最后同步。 | warmup 使用统一 prepare path。 | 统一 warmup/discovery/timing prepare。 |
| L2/cache | 无污染 | `cache.zero_()` 冲刷实际 L2 size buffer。 | reset persisting L2 + clear `2 * L2_cache_size` buffer。 | 评估 persisting L2 reset 和 buffer size。 |
| Input/output | 无污染 | 3-clone pool，超过 1 GiB 跳过。 | shifting allocator，每轮不同 data pointer。 | 评估替换为 shifting allocator。 |
| Discovery | 可观测 / 无污染 | Kineto 路径没有 native expected sequence discovery。 | discovery expected activity sequence/counts。 | 引入 native CUPTI discovery。 |
| Timed repeat | 严格串行 / 无污染 | 每个 repeat 包 `record_function`，依赖 projected regions。 | CUPTI timestamp window + sequence attribution。 | 替代 Kineto projection。 |
| Aggregation | 可观测 | fallback/backend 信息不够完整进入 history。 | trace status / performance / environment。 | 记录 timing provenance 和 failure reason。 |
