# TileOps Benchmark Execution Harness 调研与 SOL 对齐目标

更新日期：2026-08-04

## 1. 目标

这份笔记只讨论 benchmark execution harness，不重新展开 timing backend 的取舍。当前方向是保留 CUPTI activity timing 原则，并把 execution harness 的设计目标收敛成四个主需求和两个次要目标。

主需求：

```text
1. 环境稳定
2. 严格串行
3. 重复测量前后无污染
4. 可观测、可追溯
```

次要目标：

```text
5. 初始化开销移出正式窗口
6. 运行成本可控
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

修改目标：

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

修改目标：

- 为 nightly 增加 strict serial runner mode：不 prewarm 后续 benchmark file；前一个 child 完全退出后再启动或释放下一个正式 benchmark。
- 保留 per-file isolation 的 crash/OOM/hang 防护价值，但避免它破坏 GPU 独占语义。
- 增加 instrumentation，记录 child collect start/end、run start/end、pytest rc、process exit、teardown deadline。
- 实测 strict serial 与当前 prewarm mode 对总耗时和 latency 稳定性的影响。

小规模实验记录：

| 日期 | 环境 | 对比 | 测例 | 结果 |
| --- | --- | --- | --- | --- |
| 2026-08-04 | `ghcr.io/tile-ai/tileops-runner:65dbc98-torch2.10`；GPU0 H200；`Exclusive_Process`；1500 MHz；共享 cache | default prewarm runner | `bench_independent_elementwise.py`、`bench_logical_reduce.py`、`bench_gemm.py` | 退出码 1，总耗时 154s。第一个文件通过；后两个文件在创建 CUDA tensor 时出现 `cudaErrorDevicesUnavailable`。 |
| 2026-08-04 | 同上 | strict serial runner | 同上 | 退出码 0，总耗时 172s。三个文件全部通过；每个文件后等待到 `teardown complete` 再启动下一个 child。 |

初步解读：

- 在 GPU `Exclusive_Process` 下，当前 default prewarm runner 可以触发后续 child 拿不到 CUDA device 的问题。
- strict serial runner 能消除这组小规模测例中的 device unavailable 失败。
- 这轮只验证了 runner 机制和失败模式，不能直接外推到全量 latency 稳定性；下一步需要扩大到更多 benchmark files，并记录每个 child 的 collect/run/exit 时间线。

## 4. 主需求三：重复测量前后无污染

判断：这一项很难从单一错误日志直接看出问题，需要从 execution policy 本身分析。重点是输入地址、输出状态、L2/cache 状态、flush 与同步边界是否稳定。NV SOL 在这里有多项可借鉴做法。

TileOps 现状：

- `bench_kernel` 默认 `n_warmup=10`、`n_repeat=50`、`n_trials=3`。
- warmup 中执行 `cache.zero_(); _run(i)`，最后 `torch.cuda.synchronize()`。
- 每个 timed repeat 前执行 `cache.zero_(); torch.cuda.synchronize()`，再进入被测调用。
- L2 flush buffer 使用实际 L2 size；当前 main 没有调用 `cudaCtxResetPersistingL2Cache()`。
- 输入 tensor 默认预 clone 3 份 pool，按 `i % 3` 轮换；总输入超过 1 GiB 时跳过 clone 并记录 `inputs_cloned=False`。
- case 结束后，pytest hook 调用 `gc.collect()` 和 `torch.cuda.empty_cache()`。

我们的需求：

- 每次 repeat 前的 cache/input/output 状态要尽量一致。
- flush/setup work 不能进入被测 user activity sequence。
- repeat 之间不能通过固定 data pointer、输出残留、allocator 状态或 L2 命中状态产生隐藏耦合。
- 大输入无法 clone 时，要在结果中显式标记，不和正常地址扰动数据混用。

SOL 可借鉴点：

- 每次 warmup、discovery、timed iteration 之前都走统一的 `prepare_iteration()`。
- `prepare_iteration()` 中先 setup args，再 reset persisting L2 cache，再 clear L2 buffer。
- L2 buffer 大小约为 `2 * L2_cache_size`。
- `ShiftingMemoryPoolAllocator` 预分配 input/output pool，每次 iteration 返回不同 `data_ptr`，同时保持 seed 可复现。
- SOL 使用 discovery sequence 区分用户 activity 与 setup/cache noise。

修改目标：

- 统一 warmup、discovery、timing 的 prepare path。
- 引入或评估 `cudaCtxResetPersistingL2Cache()`。
- 评估把 3-clone pool 替换为 SOL-style shifting allocator。
- 明确 flush、start timestamp、runner、post sync、end timestamp 的顺序，并用实验验证 flush 不泄入 attribution window。
- 对 fixed-address、3-clone、shifting allocator 做小范围 latency 分布对比。

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

修改目标：

- `BenchmarkReport`、JUnit 和 perf history 都保存 timing provenance。
- native CUPTI backend 记录 expected activity sequence/count、sampled calls、dropped records、failure reason。
- 默认正式路径 fail closed；diagnostic 或 fallback 必须显式标记，不能进入同一可比历史。
- runner artifact 记录 child lifecycle 和异常隔离信息。

## 6. 次要目标一：初始化开销移出正式窗口

判断：这一项现在应该基本有覆盖，而且参考 SOL 的做法相对直接；它和 timing backend 相对独立，可以在主需求稳定后继续收紧。

TileOps 现状：

- `bench_kernel` 有 10 次 warmup。
- TileLang/Triton 编译、autotune、CUDA context init、module load、library handle init 通常应在第一次调用或 warmup 中发生。
- 共享 cache 能减少 nightly 中的重复编译成本。

我们的需求：

- JIT、compile、autotune、context init、module load、library handle init、allocator 扩容不进入正式 timed repeat。
- 如果某些 lazy init 只能在被测 callable 第一次调用时触发，应通过 first-call 或 warmup 明确吸收。
- 初始化失败和 cache miss 应进入 debug 信息，而不是表现为 latency regression。

SOL 可借鉴点：

- SOL evaluation 在正式 timing 前先做 user function correctness / round-0 call，注释中说明 round 0 也用于吸收 JIT/compiler 线程启动。
- SOL timing 有明确 warmup 阶段，warmup work 不进入 CUPTI timing result。

修改目标：

- 明确 TileOps benchmark first-call、warmup、discovery、timing 的边界。
- 对可能 lazy init 的 op/baseline 增加诊断字段或 debug logging。
- 保持 compile/autotune cache 与正式 timing window 解耦。

## 7. 次要目标二：运行成本可控

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

修改目标：

- 先用 strict serial + native CUPTI 跑全量，拿到正确性和总耗时基线。
- 再评估 prewarm 是否值得以可选 mode 保留。
- 把 heavy diagnostics 放在 debug mode 或 failure artifact 中。

## 8. 实验计划

后续实验按主需求优先级推进。前六组用于验证 execution harness 的正确性和可观测性；最后再评估运行成本。

| 优先级 | 实验 | 目标 | 对比 / 测例 | 主要指标 |
| --- | --- | --- | --- | --- |
| 1 | Strict serial runner | 验证 #1761 引入的 prewarm / teardown overlap 是否影响测量稳定性。 | `default prewarm runner` vs `strict serial runner`；先小规模，再全量。 | 总耗时、失败率、`cudaErrorDevicesUnavailable`、latency 分布、child collect/run/exit lifecycle。 |
| 2 | Native CUPTI attribution | 验证 SOL-style native CUPTI discovery / attribution 是否能稳定覆盖 single-kernel 和 multi-kernel op。 | single-kernel 小/中/大；multi-kernel GQA / GDN。 | discovery sequence、attributed count、dropped records、latency mean/std/p90。 |
| 3 | Flush window boundary | 验证 L2 flush 不泄入被测 attribution window。 | 正确顺序 `flush -> sync -> start_ts -> runner -> sync -> end_ts` 与若干错误顺序对照。 | activity sequence 是否包含 flush memset/kernel；expected sequence 是否稳定。 |
| 4 | L2 policy | 评估是否需要 SOL-style `cudaCtxResetPersistingL2Cache()` 和 `2 * L2_cache_size` buffer。 | 仅 `cache.zero_()`、`reset persisting L2 + cache.zero_()`、不同 buffer size。 | memory-bound kernel latency mean/std/p90；activity sequence 是否受影响。 |
| 5 | Input address policy | 判断当前 3-clone pool 是否足够，是否需要替换为 shifting allocator。 | fixed address、3-clone pool、SOL-style shifting allocator。 | latency mean/std/p90、outlier、是否存在稳定偏差。 |
| 6 | Artifact / failure observability | 验证错误不会静默混入正式性能数据。 | 构造 CUPTI dropped、sequence mismatch、CUDA error、timeout/OOM 等场景。 | JUnit、`profile_run.log`、perf history 是否记录 backend、sequence、sampled count、failure reason、nodeid/op/tag/workload。 |
| 7 | Full benchmark cost | 在正确性路径稳定后评估全量运行成本。 | 当前 main / strict serial + native CUPTI / 必要 debug mode。 | 全量耗时、单文件耗时分布、artifact 体积、nightly 可接受性。 |

实验执行原则：

- 先小规模验证机制，再跑全量。
- 每轮记录代码版本、runner 镜像、GPU clock、cache path 和关键 env。
- 正式结论只来自同一 commit、同一 runner 镜像、同一 cache policy 下的对比。
- 如果实验中发现新的污染源，先回到对应主需求章节修正文档，再继续后续实验。

## 9. 细节索引

下面的表只作为实现排查索引，不作为正文主线。

| 时序环节 | 归属需求 | TileOps 现状 | SOL 可借鉴点 | 下一步 |
| --- | --- | --- | --- | --- |
| Runner 启动 | 环境稳定 | workflow 验证 1500 MHz；使用官方 runner 镜像和共享 cache。 | 固定评测环境与 lock clocks。 | 补充 image / clock / cache provenance。 |
| File 调度 | 严格串行 | 每个 benchmark file 一个 pytest child；默认 `--prewarm=4`。 | serialized GPU benchmarking。 | 增加 strict serial mode。 |
| Collect/import | 严格串行 | child collect-only 可与当前 benchmark 重叠。 | 不通过 pytest prewarm。 | 检查 collect 是否 GPU-silent。 |
| Teardown | 严格串行 | pytest rc 写回后 interpreter teardown 可后台继续。 | dedicated subprocess isolation。 | strict serial 下等待 child 完全退出。 |
| Warmup | 无污染 / 初始化 | 10 次 warmup，最后同步。 | warmup 使用统一 prepare path。 | 统一 warmup/discovery/timing prepare。 |
| L2/cache | 无污染 | `cache.zero_()` 冲刷实际 L2 size buffer。 | reset persisting L2 + clear `2 * L2_cache_size` buffer。 | 评估 persisting L2 reset 和 buffer size。 |
| Input/output | 无污染 | 3-clone pool，超过 1 GiB 跳过。 | shifting allocator，每轮不同 data pointer。 | 评估替换为 shifting allocator。 |
| Discovery | 可观测 / 无污染 | Kineto 路径没有 native expected sequence discovery。 | discovery expected activity sequence/counts。 | 引入 native CUPTI discovery。 |
| Timed repeat | 严格串行 / 无污染 | 每个 repeat 包 `record_function`，依赖 projected regions。 | CUPTI timestamp window + sequence attribution。 | 替代 Kineto projection。 |
| Aggregation | 可观测 | fallback/backend 信息不够完整进入 history。 | trace status / performance / environment。 | 记录 timing provenance 和 failure reason。 |
