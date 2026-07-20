# AKO-Style 调试 GQA Decode Kernel 技术总结

日期：2026-06-10

讨论对象：2026-06-11 组内分享

## 摘要

本报告总结本轮将 AKO-style 方法应用到 TileOps GQA decode kernel 调试中的过程、设计取舍和阶段性结论。报告主线分三步：

```text
1. 先讲 AKO：
   它如何把 agent 的发散能力放进可验证、可记录的优化闭环里。

2. 再分析 GQA decode：
   它为什么适合 AKO，哪些地方又限制了 AKO 的 kernel-internal 收益。

3. 最后讲本轮实践：
   我们如何变体化 AKO，用它调试 TileOps GQA decode，并得到 dispatch / kernel 两层结论。
```

核心结论：

```text
1. 在当前 dense contiguous GQA decode case 中，AKO-style 方法的主要收益来自 dispatch policy，
   而不是产出一个全面强于 upstream 的 non-WS kernel。

2. 对 Qwen3.5 group=8，AKO-style sweep 发现 split=32 明显优于 upstream/default split=16：
       S=4K:    1.19x
       S=8K:    1.17x
       S=16K:   1.33x
       S=32K:   1.57x
       S=64K:   1.77x
       S=128K:  1.91x

3. WS 的有效区间被收窄到 Qwen3.5 group=8, S=4K, split=32。
   对 Llama4 group=5，目前不建议默认启用 WS。

4. kernel 内部优化有小收益，但不是主收益来源。
   当前保留的内部优化包括：
       v002: split_length shared mask
       v003: combine 复用 glse_vec
       v007: denominator-form combine
   其中 v007 在推荐点上的几何平均收益约 1.0027x，是小而真实的 cleanup。
```

因此，本轮最重要的经验不是“AKO 自动生成了更强 kernel”，而是：

```text
AKO-style 方法帮助我们把一个复杂 kernel 问题拆成可审计的 policy surface 和 variant archive，
并在较大组合空间中定位了真正有收益的决策层级。
```

## 1. AKO 项目简介

AKO 可以理解为一套 agentic kernel optimization harness：它不是只枚举几个 autotune knob，而是把 agent、benchmark、reference、profiler、知识来源、archive 和 keep/reject 决策放进同一个闭环里。

它的核心思想是：保留 agent 提出新方向的泛化能力，但用 workload、correctness、baseline、scoring 和 archive 把每一轮优化约束成可复现、可审计的工程实验。

选择 GQA decode 的原因是它同时包含 kernel body、kernel-family selection 和 dispatch 策略；这次实践也为后续调试 GDN / linear attention kernel 预演了一套更可复用的实验制度。

## 2. 原版 AKO 方法概括

原版 AKO 的核心不是一个新的 kernel 搜索算法，而是一个 agentic kernel optimization harness。它把 coding agent 放进一个有 benchmark、profiler、reference、skill、archive 和 trap 记录的环境中，让 agent 持续执行闭环优化。

### 2.1 AKO Harness 流程图

更准确地说，AKO 由三类机制组成：

```text
┌─────────────────────────────────────────────────────────────────────┐
│ 一、问题定义与实验契约                                                │
│                                                                     │
│  A. 对象契约: operator / kernel family / 可归因子面                  │
│  B. 实验契约: workload rows / reference / benchmark protocol         │
│  C. 比较契约: baseline / scoring / keep-reject 口径                  │
│  D. 执行与记录契约: allowed files / frozen logic / archive format      │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  │  定义清楚以后，agent 才能无歧义地执行实验
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 二、单轮优化闭环                                                      │
│                                                                     │
│  1. 确认 baseline                                                    │
│     输入: workload rows / reference / current best                   │
│     动作: correctness + benchmark + profile if needed                │
│     产物: 可信 baseline record                                       │
│                                                                     │
│  2. 提出 hypothesis                                                  │
│     输入: baseline record / profile signal / archive / hints         │
│     动作: 选择一个可验证的单一改动方向                               │
│     产物: hypothesis + expected effect                               │
│                                                                     │
│  3. 生成 variant                                                     │
│     输入: hypothesis / allowed files / child workspace               │
│     动作: 修改 kernel / wrapper / config 中的最小必要部分            │
│     产物: code diff + variant id                                     │
│                                                                     │
│  4. 执行 gate                                                        │
│     输入: variant / benchmark adapter / frozen scoring               │
│     动作: compile → correctness → benchmark → profiler               │
│     产物: result record + failure reason if any                      │
│                                                                     │
│  5. 做 decision                                                      │
│     输入: result record / scoring rule / traps                       │
│     动作: keep / reject / debug / extend                             │
│     产物: decision + next-round prompt                               │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  │  每一轮的结果必须结构化写回
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 三、跨轮记忆与 campaign 管理                                          │
│                                                                     │
│  A. Archive record                                                   │
│     保存: variant id / diff / hypothesis / result / decision          │
│                                                                     │
│  B. Best lineage                                                     │
│     更新: current best / accepted change / baseline for next round    │
│                                                                     │
│  C. Trap memory                                                      │
│     记录: dead-end / correctness failure / benchmark artifact         │
│                                                                     │
│  D. Campaign controller                                              │
│     产出: unexplained bucket / next-round prompt / child workspace    │
│                                                                     │
│  下一轮从 archive 出发，而不是从空白上下文重新猜方向                  │
└─────────────────────────────────────────────────────────────────────┘
```

这里的 workload rows 指一行一行可执行的实验用例。每一行固定一组 model / B / S / group / dtype / layout / backend / split / WS 设置，用来保证不同 variant 在同一张实验表上比较。

### 2.2 三个 Harness 核心问题

这幅图里最能体现 AKO 特色的是单轮优化闭环中的两个步骤：

- **2. 提出 hypothesis**
  agent 从代码、profile、benchmark surface、archive、skill 和用户 hint 中，
  生成一个新的、可验证的优化方向。

- **5. 做 decision**
  agent 不只读 latency，而是综合 correctness、benchmark、profile、复杂度、
  trap 和 campaign 状态，决定 keep / reject / rerun / pivot。

这两个步骤是 agent 发挥泛化性的地方。AKO 的设计哲学不是把 agent 变成一个固定 autotuner，而是允许 agent 在这两个位置使用经验、类比、推理和外部资料；同时用 problem contract、gate 和 archive 把这种自由度约束成可复现、可审计的工程过程。

原版 AKO / AKO4X 的回答不是设计一个复杂的显式搜索算法，而是把 agent 的自由度放进一组工程约束里。这里有三个核心问题。

#### 2.2.1 Agent 的知识从哪里来

AKO 不希望 agent 凭空“灵感优化”。这里要区分两类输入：一类是问题上下文，另一类才是知识来源。

当前 kernel / wrapper / benchmark、correctness reference、tolerance 和 workload 定义，主要提供的是边界：它们告诉 agent 现在到底在优化什么、什么算正确、什么口径下比较性能。真正帮助 agent 产生新方向的知识，更多来自 per-DSL / per-tool SKILL、历史 archive、TRAPS、failed variants、profiler / compile log 的解释、用户提供的 knowledge / hints，以及必要时的外部资料。

这里有一类知识尤其重要：用户提供的知识。用户往往知道模型结构、业务 workload、哪些路径不能动、哪些 trick 不可接受、哪些现象值得优先解释。这些信息不是 benchmark 自己能推出的，所以在 AKO4ALL / AKO4X 里会通过 prompt、`HINTS.md`、knowledge 文件或 task description 进入系统。它相当于 harness 暴露给用户的接口：用户把领域知识和约束注入进来，agent 再把这些信息和代码、profile、历史 archive 一起使用。

因此，agent 的 hypothesis 不是从空白处生成的，而是在“问题边界 + 历史经验 + 用户意图 + DSL/tool skill + profiling 解释”共同形成的上下文里生成的。这个设计保留了 agent 的泛化能力，但减少了纯猜测。

#### 2.2.2 如何避免过度发散

agent 可以提出方向，但不能随便改变问题本身。AKO 先固定 workload、reference、baseline、scoring、target GPU 和 frozen logic；这样 agent 不能通过改题目、改评分、改 baseline 来制造“优化”。

然后，AKO 要求每个想法都落成一次可验证的 iteration。bench 之后必须写 `ITERATIONS.md`，必要时 bench 前还要写 `Expected`，让“我以为它会快的原因”先被记录下来。接着通过 compile、correctness、benchmark、profile 和 variance / audit gate，把一个 hypothesis 判成 keep、reject、rerun、defer 或 pivot。

最后，失败和误判不会被遗忘。archive 保存 baseline、variants 和 round 结果，`TRAPS.md` 保存 silent bug 和 methodology trap。下一轮 agent 仍然可以发散，但它必须带着这些历史证据发散，而不是重新把旧错路再走一遍。

#### 2.2.3 如何平衡深度和广度

AKO 的策略不是把所有方向平均试一遍。只要一条路径持续产生可信收益，就可以继续深挖；但如果连续若干轮没有明显超过 current best，就不能继续盲试。AKO4ALL 的默认规则是连续 3 次没有至少 3% 改进时，必须暂停，重新 profile、回看 `ITERATIONS.md`、搜索相关资料，并检查哪些 axis 已经试过。

换句话说，深挖需要正证据，换路需要负证据。停止也不能只靠“感觉没戏”：要么 iteration cap 到了，要么有 physical floor 证据，要么多个 distinct directions 已经被记录为耗尽。

AKO4X 的 master 还会处理下一轮方向。上一轮的 `Open directions` 只是 forensic signal，不是下一轮 checklist；master 要结合 cost、scope 和 mechanism 重新组织 prompt。这样可以避免把上一轮已经剪枝的方向原样转发给下一个 child。

所以，AKO 对 agent 行为的约束不是“少想一点”，而是：

```text
允许 hypothesis 生成阶段发散；
要求每一次实际 round 都进入可复现的 contract、log、gate 和 archive；
要求停滞、放弃、换路都有文档化证据。
```

### 2.3 对本轮 GQA Decode 的启发

把 AKO 的设计哲学放到 GQA decode 上，第二章只需要说明一件事：在进入具体实验之前，我们要先给 agent 标出边界，并提供足够的领域知识。至于我们具体如何组织 benchmark、如何记录 variant、如何做轻量变体化，放到第四章展开。

#### 2.3.1 先标记归因边界

首先要标记的是归因边界。GQA decode 不能简单分成“kernel 内”和“kernel 外”，因为 split count、WS path、backend path 本身也属于 kernel 实现体系的一部分。更有用的分法是两层：一层看 kernel body 内部怎么写，另一层看 dispatch 时选择哪类 kernel family 和哪组参数。

kernel body 内部，agent 需要知道哪些地方属于具体实现细节。例如 mainloop、online softmax、PV accumulation、combine path，以及 partial workspace 的 dtype、layout 和 writeback。这些问题回答的是“同一类 kernel 里面怎么算得更好”。

dispatch / kernel-family selection 则回答另一个问题：在某个 workload 下，应该选哪条实现路径。这里包括 non-WS split path 还是 WS producer-consumer path，split count 取多少，backend/path 选哪个，以及不同 scenario、S bucket 下的启用边界。

comparison 口径也要归入这一层，因为它决定收益怎么归因。比如 local variant 比 upstream default 快，不一定说明 kernel body 更强；还需要和 upstream manually matched split 比。如果 matched split 之后基本持平，那么收益就应该归因到 dispatch / kernel-family selection，而不是归因到 kernel body。

#### 2.3.2 再提供领域知识

其次要提供 GQA decode 的领域知识。agent 需要知道 Llama4 / Qwen3.5 的 group size、Hkv、D 和典型 S bucket，也要知道这一轮只覆盖 B=1 dense contiguous KV decode。它还需要知道 split-KV 主循环、partial O / LSE、combine 各自负责什么，WS path 是 producer-consumer overlap，而不是普通 non-WS 的小改动。

最后还要提供实验和历史知识。correctness 要对 torch SDPA math reference；upstream default dispatch 和 upstream manually matched split 要分开比较；fp32 partial 更像 stability/debug path，不能默认当作 performance path。已经证明收益很小的 variant、来自 benchmark / dispatch 口径变化的结果、容易产生 attribution trap 的方向，也应该作为历史知识交给 agent。

#### 2.3.3 本轮使用原则

因此，本轮把 AKO 用到 GQA decode 的关键，不是先让 agent 自由写 kernel，而是先让 agent 明确：哪些东西不能动，哪些收益必须分层归因，哪些背景知识来自用户和实验记录。第四章再讨论我们如何把这些边界和知识落成一个轻量 AKO-style 调试流程。

## 3. GQA Decode 对 AKO 的适配性分析

GQA decode 适合用 AKO-style 方法处理，但并不是因为它一定能让 agent 写出一个全面更强的 kernel body。更准确地说，它适合 AKO，是因为它有明显的多层决策面：既有 kernel body 内部实现问题，也有 kernel-family / dispatch selection 问题，还需要严格区分 correctness、baseline 和归因口径。

同时，这个 case 也有一个天然限制：现有 upstream GQA decode kernel 已经比较成熟，几个主路径的 kernel 内部优化程度较高。如果只在原 kernel family 内做局部 cleanup，收益可能不会特别明显。AKO 在这里的价值，更多体现在帮助我们拆清楚问题、避免错误归因、找到真正有收益的决策层级。

### 3.1 为什么 GQA Decode 适合用 AKO

第一，GQA decode 的 kernel body 本身包含多个阶段，不同阶段的性能敏感点不同。split-KV kernel 至少包括：

```text
1. QK score 计算；
2. mask / valid length 处理；
3. online softmax / logsumexp；
4. PV accumulation；
5. partial O 和 partial LSE 写出；
6. combine kernel 读 partial，完成跨 split 归一化和最终输出。
```

这些阶段的性能敏感点不同：

```text
QK/PV 主循环:
    受 block_H、block_N、group size、register pressure、shared memory layout 影响。

partial workspace:
    fp16 / fp32 决定带宽和数值稳定性取舍。

combine:
    对 split count 敏感。split=16/32 时，combine 开销虽然小，但在短 S 或小 group 场景中会变得可见。

WS path:
    producer / consumer overlap 可能减少短 S latency，但也可能引入额外 synchronization 和 combine 成本。
```

这意味着 agent 不能只看一条 latency 曲线就判断应该改哪里。它需要结合代码结构、benchmark surface、correctness 风险和必要的 profiler 信号来提出 hypothesis，这正是 AKO 适合发挥的地方。

第二，GQA decode 的实现空间不只是 kernel body 内部参数。它还包含 kernel-family / dispatch selection：

```text
scenario:
    Llama4 group=5 和 Qwen3.5 group=8 可能落在不同策略区间。

S bucket:
    短 S 和长 S 的主循环工作量、combine 占比、workspace 成本不同。

split / chunk:
    split 太小会降低并行度，split 太大又会增加 partial workspace 和 combine 成本。

backend / path:
    non-WS split path、WS producer-consumer path、FA3 / FlashInfer / upstream path
    都可能在不同 bucket 有不同 crossover。

tile / block:
    block_N、chunk_len、tail handling、occupancy 和 memory coalescing 互相耦合。
```

这种空间不是普通单 kernel autotune 能完全覆盖的。AKO 的优势是可以把代码、benchmark、用户 hint 和历史结果组织在一起，让 agent 不只枚举 knob，而是去判断“当前收益到底属于 kernel body、kernel-family selection，还是 dispatch 边界”。

第三，GQA decode 容易出现归因误判。比如某个 local variant 比 upstream default 快，并不自动说明 kernel body 更强；可能只是 split choice 更合适。只有做 upstream manually matched split 之后仍然领先，才能说 kernel body 本身有优势。AKO 的 archive、decision log 和 attribution check 正好适合解决这类问题。

### 3.2 GQA Decode 对 AKO 的不利点和边界

GQA decode 也不是一个“AKO 必然大幅改进 kernel body”的理想场景。它的主要不利点在于：现有几个核心 kernel path 已经有较高优化程度，留给局部 intra-kernel cleanup 的空间有限。

从 kernel 内部看，可调方向当然存在：

```text
tile shape / work decomposition:
    block_H、block_N、num_split、threads、num_stages。

split-KV partial state:
    partial O / LSE 的 dtype、layout、写出方式和 combine 方式。

mask / sequence boundary:
    split_length、valid length、tail split、非整齐 split 的处理方式。

combine kernel:
    跨 split 做 max / exp / sum / weighted output。

workspace pipeline:
    producer / consumer overlap、workspace traffic、同步成本。
```

但这些方向大多是在成熟 kernel family 上做局部调整。如果 upstream mainloop、split-KV partial、combine 等路径已经接近局部最优，那么单纯改一两个源码细节，未必能带来明显收益。这个时候，AKO 如果只做 intra-kernel variant，很可能看到的是小幅 cleanup，而不是数量级提升。

另一个边界是 profiler 需求。kernel-internal optimization 往往需要 NCU、generated CUDA/PTX/SASS、register/shared memory/occupancy 等证据。如果只用 benchmark latency，agent 很难判断一个源码改动是真正改善了硬件行为，还是被 compiler lowering 掩盖，甚至引入了新的开销。

因此，本 case 对 AKO 的定位应该是：

```text
适合 AKO 的部分:
    多层策略空间；
    dispatch / kernel-family selection；
    workload bucket 和 split choice；
    correctness / benchmark / upstream 对比归因；
    dead-end 和 attribution trap 记录。

不应过度期待的部分:
    在成熟 upstream kernel body 上，仅靠轻量源码 patch 获得大幅 intra-kernel 提升。

如果要深入 kernel body:
    必须升级为 profiler-driven、codegen-aware、family-aware 的 AKO loop。
```

### 3.3 本章结论

所以，GQA decode 对 AKO 的适配性是“双面的”。

一方面，它的组合空间复杂，很适合用 AKO 来组织知识、拆分边界、记录假设和做归因。另一方面，现有 kernel body 已经较强，轻量 intra-kernel optimization 的边际收益可能有限。这个判断会影响第四章的变体化设计：本轮应该优先把 workload、dispatch、upstream 对比和 attribution check 做清楚，再谨慎评估 kernel-internal cleanup 的真实收益。

## 4. 我们对 AKO 的变体化设计

### 4.1 原版 AKO 与本轮轻量变体的对照

我们没有直接照搬 AKO4X 的多 agent campaign，而是保留 AKO 的核心闭环，做了一个更适合当前 GQA decode 调试节奏的轻量变体。

对照如下：

```text
1. Workload 定义

   原版 AKO:
       面向 benchmark suite，通常由 adapter 提供 list_workloads / run / profile。

   本轮变体:
       显式固定 GQA decode workload surface：
           Llama4 group=5
           Qwen3.5 group=8
           S=4K 到 128K
           split=16 / 32，以及必要细扫

   取舍:
       不追求通用 benchmark suite，先保证当前 policy surface 可解释、可复现。

2. Benchmark adapter

   原版 AKO:
       通过统一 adapter 抽象不同 benchmark，agent 只需要调用标准接口。

   本轮变体:
       使用 benchmarks.ops.attention.bench_gqa_decode_policy_microbench，
       统一输出 JSONL，方便跨 variant、跨 backend、跨 split 比较。

   取舍:
       adapter 还没有产品化，但已经满足本轮 sweep、check、perf 对比需要。

3. Correctness gate

   原版 AKO:
       每个 variant 必须通过 reference / tolerance，避免 reward hacking。

   本轮变体:
       对 torch SDPA math reference 做 spot-check；
       fp16 decode 接受 upstream fast-partial 同量级误差，约 <= 2e-4。

   取舍:
       使用小而高价值的 correctness sweep，优先覆盖 Llama4 split=16 和 Qwen split=32。

4. Baseline / comparison

   原版 AKO:
       维护当前 best variant 和 benchmark baseline，所有 variant 都与同协议结果比较。

   为什么本轮必须强化这一点:
       本轮研究对象包含 dispatch 策略，不只是单个 kernel body。
       同一个 workload 下，我们要横向比较 upstream default、upstream matched split、
       local non-WS、local WS，以及不同 split / backend path。
       如果 baseline 口径不固定，agent 很容易把“选中了更合适的 path”
       误判成“写出了更强的 kernel body”。

   本轮变体:
       显式区分三类 baseline：
           upstream/default split=16
           upstream 手动 split=32
           本地 protected kernel commit

   取舍:
       这是本轮最关键的适配。
       Baseline / comparison 是横向比较的坐标系；
       它决定每个收益应该归因到 kernel body、kernel-family selection，
       还是 dispatch boundary。

5. Variant 隔离

   原版 AKO:
       master / child 多 workspace campaign；child 自由尝试，master 归档结果。

   为什么本轮必须保留这一点:
       dispatch 研究会同时产生很多 kernel variant：
       split 不同、WS / non-WS 不同、backend path 不同、combine 写法也可能不同。
       这些 variant 必须从同一个可信状态出发，才能横向比较。
       如果 rejected patch 混进下一轮，后面的性能数字就无法判断是当前 hypothesis 的收益，
       还是前面残留改动的副作用。

   本轮变体:
       没有多 workspace campaign。
       先提交 protected commit，再用小 patch 尝试 variant；
       keep 则提交，reject 则回滚，保持主线干净。

   取舍:
       牺牲自动化隔离，换取更低工程成本和更强人工审查。
       Variant 隔离是横向比较的实验卫生；
       它保证每次 decision 都能绑定到一个清楚的 code diff 和 hypothesis。

6. Profiler / sanitizer

   原版 AKO:
       profiler 和 sanitizer 是固定工具入口，停滞时会重新 profile。

   本轮变体:
       主要依赖 benchmark surface、代码阅读、upstream diff 和历史经验；
       NCU 尚未进入固定 loop。

   取舍:
       足够支撑当前 dispatch / combine 级别判断；
       但对后续 GDN / linear attention，profiler 应该系统化接入。

7. Knowledge / direction source

   原版 AKO:
       有 per-DSL / per-tool SKILL catalog。
       这些 SKILL 是 harness 的一部分，也允许用户扩展；
       覆盖 Triton / CUDA / CuTe DSL / TileLang / C++ / profiler-ncu / sanitizer 等。

   为什么本轮也需要这一项:
       agent 的方向不是凭空发散出来的。
       它需要知道哪些资料可以作为 hypothesis 的来源，
       也需要知道哪些历史结论会约束后续搜索。
       对 GQA decode 来说，这一点尤其重要：
       我们既要读 kernel body，也要读 benchmark surface 和 upstream/local diff，
       否则很容易把 dispatch 问题误当成 kernel body 问题。

   本轮变体:
       没有独立 skill catalog，但不是纯手工拍脑袋。
       方向来源被显式固定为几类资料：
           upstream vs local diff
               例如 non-WS 与 upstream 的差异主要在 partial workspace 和 combine。
           benchmark surface
               例如 Qwen group=8 在 split=32 上的收益来自系统性 sweep，
               不是单点猜测。
           kernel 结构分析
               combine、partial output、split_length mask、block_H / group mismatch
               都来自代码路径阅读。
           历史 dead-end
               v004-v009 的 reject 结果会约束后续不要重复尝试同类改动。
           用户 hint 和领域 tuning 常识
               例如减少 global read、调整 combine threads、减少 wasted rows。

   取舍:
       本轮没有把这些知识做成可调用的 SKILL catalog，
       但已经把“方向从哪里来”写成了显式接口；
       未来如果迁移到 GDN，可以把 recurrent scan、layout、numerics、profiler 经验沉淀成 skill。

8. Archive / report

   原版 AKO:
       自动归档 best、dead-end、traps、profile summary，形成 cross-round memory。

   为什么本轮也需要这一项:
       dispatch 策略研究会产生大量横向比较结果。
       如果只记最终 best，就无法复盘为什么某条 path 被保留，
       也无法解释某个收益究竟来自 split choice、WS path，还是 kernel body 改动。

   本轮变体:
       使用 markdown log + git commit + JSONL result 组成轻量 archive：
           gqa_decode_ako_style_tuning_plan.md
           gqa_decode_internal_kernel_ako_log.md
           gqa_decode_microbenchmark_report.md
           gqa_decode_ako_experience_report.md

   取舍:
       这不是原版 AKO 那种自动归档系统，
       但也不是无结构的手工笔记。
       JSONL 保存可复算结果，git commit 保存代码状态，
       markdown 保存 hypothesis、result、decision 和 reason。
       它的不足是还需要人工同步和整理，
       但已经能支撑本轮的跨 variant 记忆和归因复盘。
```

这个轻量变体更适合当前目标：

```text
1. 快速建立 GQA decode policy surface；
2. 避免多 agent / 多 workspace 的工程成本；
3. 保持每个结论可解释、可审计；
4. 显式区分 kernel 本体收益和 dispatch policy 收益；
5. 为后续 GDN / linear attention 设计更完整 AKO harness 做准备。
```

## 5. 阶段性成果

### 5.1 Dispatch policy 成果

本轮最重要的成果是明确了当前 dense contiguous GQA decode 的 dispatch boundary。

```text
Llama4 group=5:
    S=4K-128K:
        tileops_split split=16
    WS:
        当前不默认启用

Qwen3.5 group=8:
    S=4K:
        tileops_ws split=32
    S>=8K:
        tileops_split split=32
```

相对 upstream/default split=16，Qwen3.5 group=8 的收益为：

```text
S=4K:    1.19x
S=8K:    1.17x
S=16K:   1.33x
S=32K:   1.57x
S=64K:   1.77x
S=128K:  1.91x
```

但需要强调：

```text
如果 upstream 也手动使用 split=32，
我们的 non-WS kernel 基本与 upstream-style split path 持平。
```

因此，这个收益应被归因于 dispatch policy，而不是 kernel 本体全面超越 upstream。

### 5.2 Kernel 内部成果

本轮保留的 kernel 内部改动：

```text
v002 split_length shared mask
v003 combine reuse glse_vec
v007 denominator-form combine
```

其中 v007 的正式结果：

```text
recommended points geo mean vs v003: 1.0027x

Llama4 split=16 non-WS:
    geo mean: 0.9993x

Qwen3.5 non-WS split=32:
    geo mean: 1.0061x
```

这个结果说明：

```text
kernel-internal AKO 可以带来小幅 cleanup 收益，
但当前主收益仍不在 kernel 内部。
```

### 5.3 Correctness 与 upstream 关系

我们额外验证了 upstream GQADecodeKernel 对 torch SDPA math reference 的 correctness：

```text
scenario: Llama4 group=5, Qwen3.5 group=8
S:        4K, 8K, 32K
split:    16, 32
dtype:    fp16

max_diff 最大约 1.83e-4
```

这说明 upstream split path 本身是正确且很强的。后续报告中不应把 upstream 作为“错误 baseline”处理。

## 6. 反思

### 6.1 AKO 的价值边界

本轮最重要的反思是：AKO 的价值不一定体现为“自动写出一个更强 kernel”。

在这个 case 中，它的价值主要体现在：

```text
1. 把复杂组合空间变成可审计 policy surface；
2. 让 dispatch boundary 从经验判断变成数据结论；
3. 避免把 correctness audit、debug path、kernel body 改动和 dispatch 收益混在一起；
4. 系统记录 rejected variants，减少重复试错；
5. 为后续更复杂 kernel 建立实验制度。
```

如果只用“是否产生全面强于 upstream 的 kernel”评价 AKO，本轮会显得收益有限；但如果评价“是否帮助我们在复杂空间中定位真正收益来源”，本轮是成功的。

### 6.2 对 autotune 的补充关系

常规 autotune 擅长在固定 kernel family 内搜索局部参数：

```text
block_N
num_stages
threads
num_split
```

但本轮的主要机会是：

```text
scenario-specific split policy
WS vs non-WS boundary
backend selection
upstream/default policy 修正
```

因此，AKO-style 方法不是替代 autotune，而是把 autotune 放到更大的决策面里：

```text
先用 AKO-style loop 找正确 policy surface，
再对每个稳定区域做局部 autotune。
```

### 6.3 为什么 kernel 内部优化不够好

本轮 kernel-internal AKO 的收益较小，需要单独反思。这里不能简单归因于“AKO 不擅长 kernel 内部优化”，更准确的说法是：我们这轮轻量 AKO 的信息输入、搜索边界和验证工具，更适合发现 policy / dispatch 问题，不足以系统打开一个已经较成熟的 kernel body。

具体原因有两类。

第一类是问题本身的客观限制：

```text
1. upstream split path 本身已经很强。
   correctness spot-check 证明 upstream 不是错误 baseline；
   性能上，当 upstream 也手动使用 split=32 时，Qwen non-WS 基本与本地持平。
   这说明可挖空间不是“明显 bug 修复”，而是很窄的 micro-architecture 空间。

2. 当前 GQA decode dense contiguous case 的主收益不在 kernel body。
   对 Qwen，真正的大收益来自 split=32 policy；
   对 Llama，upstream/default split=16 已经接近最优。
   当主要瓶颈被 policy 解释后，kernel-internal variant 只能在小比例开销里找收益。

3. split decode 的内部优化容易被 combine、workspace traffic 和 launch/JIT 噪声稀释。
   v002/v003/v007 主要作用在 mask、combine、denominator-form cleanup 这类局部路径；
   对长 S，主循环占比更高，combine cleanup 不容易显著改变总 latency；
   对短 S，noise、workspace 和 dispatch 影响又更明显。

4. correctness 约束迫使我们回到保守 fast path。
   fp32 partial 提高了稳定性直觉，但实际带来带宽和 combine 成本；
   最后 non-WS/WS 都回到 fp16 partial fast path，说明这类数值路径改动不是免费收益。
```

第二类是我们这轮 AKO 变体执行上的不足：

```text
1. profiler 没有进入固定闭环。
   我们主要靠 benchmark surface、代码 diff 和 spot-check 推断瓶颈；
   缺少固定 NCU 指标来回答：
       是 memory-bound 还是 instruction-bound？
       combine 的真实占比是多少？
       register pressure / occupancy / shared memory bank conflict 有没有恶化？
       TileLang lowering 后生成的实际指令是否符合假设？

2. kernel-internal 搜索空间没有被形式化。
   dispatch surface 被枚举得比较清楚：
       scenario × S × split × WS/non-WS
   但 kernel 内部没有同等清楚地定义：
       block_H / block_N / threads / num_stages
       partial dtype / layout / store path
       combine reduction form
       index/mask hoisting
       shared/register staging
   因此 agent 更像在做少量手工 variant，而不是系统探索 kernel body。

3. 没有建立 low-level evidence chain。
   一个 kernel-internal 假设应该从：
       source change
       generated CUDA / PTX / SASS
       NCU metric
       latency result
   四层同时闭合。
   本轮大多数内部 variant 只闭合了 source change 和 latency result，
   中间的 codegen / hardware metric 证据不足。

4. TileLang DSL 降低了改动成本，也遮蔽了部分硬件细节。
   AKO 可以快速改 TileLang source，
   但如果不检查 lowering 结果，agent 很难判断一个源码级 cleanup 是否真的减少指令、
   改善 memory transaction，或只是被 compiler 改写成等价甚至更差的形式。

5. 早期把 stability path 和 performance path 混在了一起。
   fp32 partial 是一个合理的 correctness / stability 方向，
   但它不应该被默认当作 performance candidate。
   这说明 kernel-internal AKO 需要先命名路径：
       debug/stability path
       production fast path
       experimental path
   否则实验结果容易被不同目标混淆。

6. 没有投入新的 kernel family。
   本轮内部优化主要是在 upstream-style split kernel 上做 cleanup；
   没有真正设计 small-M / group-aware kernel family，
   也没有系统尝试 CUDA / MMA / WGMMA / warp-specialized 等实现范式切换。
   如果 upstream family 已经接近局部最优，只做局部 patch 很难带来数量级收益。
```

所以，本轮更准确的结论是：

```text
AKO 在 kernel 内部没有失败，但我们只给了它“轻量源码变体 + benchmark 验证”的条件；
在一个强 upstream kernel 上，这不足以产生显著 kernel-body 超越。

如果下一轮目标是 kernel-internal optimization，
AKO loop 必须升级成 profiler-driven、codegen-aware、family-aware 的流程。
```

对应的改进方向是：

```text
1. 每个 kernel-internal variant 必须绑定一个 profiler question；
2. 每个保留或拒绝的 variant 都记录 NCU 指标和 generated code 证据；
3. 把 debug/stability path 与 production fast path 明确分开；
4. 先做 workload bucket 归因，再决定优化 combine、mainloop、workspace 还是 launch overhead；
5. 对 small-M / group-aware 新 kernel family 单独建 campaign，而不是混在 cleanup variant 中；
6. 对 GDN 这类后续 kernel，从第一天就把 profiler、codegen dump、TRAPS.md 放进 AKO contract。
```

### 6.4 当前方法的不足

本轮轻量版 AKO 仍有不足：

```text
1. profiler 使用不够系统。
   主要依赖 benchmark surface 和代码分析，NCU 尚未进入固定 loop。

2. variant archive 还比较手工。
   目前靠 md log 和 git commit，尚未形成自动化 archive structure。

3. trap catalog 还不完整。
   已记录 correctness / fp32 partial / compile failure 等问题，
   但还没有形成独立 TRAPS.md。

4. dispatch policy 还没有正式接入生产代码。
   当前主要是实验结论，需要后续实现和更广覆盖测试。

5. workload surface 仍有限。
   目前集中在 dense contiguous KV、B=1 主线、Llama4/Qwen3.5 两类 group。
```

### 6.5 对 GDN / linear attention 的启发

后续调试 GDN 等 linear attention kernel 时，建议复用本轮方法，但做得更系统：

```text
1. 先定义 workload taxonomy；
2. 建立统一 JSONL benchmark adapter；
3. 明确 correctness reference 和 tolerance；
4. 把 dispatch policy 和 kernel-internal variant 分开记录；
5. 每个 variant 都写 hypothesis / result / decision；
6. 对 profiler 结果建立固定入口；
7. 对 dead-end 建立 TRAPS.md；
8. 周期性生成 report，而不是只保留流水账。
```

对 linear attention 尤其要注意：

```text
1. recurrence / scan / blockwise state 的 correctness 更容易出隐性错误；
2. dispatch boundary 可能比 GQA decode 更复杂；
3. kernel 内部 pipeline 和 memory layout 可能需要更强 profiler 支撑；
4. stable numeric path 与 performance path 必须分开命名和验证。
```

## 7. 建议的讨论问题

明日讨论可以围绕以下问题展开：

```text
1. 我们是否接受当前 GQA decode 的结论：
   主收益来自 dispatch policy，而非 kernel 本体全面超越 upstream？

2. Qwen3.5 group=8 split=32 policy 是否应该进入正式 dispatch？

3. WS 是否只在 Qwen3.5 S=4K split=32 启用？

4. 是否需要为 fp32 partial 保留显式 debug / stability config？

5. 如果继续做 kernel-internal AKO，是否必须把 NCU、generated CUDA/PTX/SASS、
   low-level metric 和 TRAPS.md 纳入固定闭环？

6. 后续是否投入 small-M / group-aware 新 kernel family？

7. GDN / linear attention 的 AKO harness 是否需要从一开始就加入 NCU 和 TRAPS.md？
```

## 8. 结论

本轮 AKO-style GQA decode 调试证明：

```text
1. 对复杂 kernel，优化对象不应只定义为 kernel body；
   dispatch policy 也必须是一等公民。

2. AKO-style loop 的最大价值是建立可重复、可审计、可积累的实验制度。

3. 在本 case 中，AKO-style 方法发现了常规单 kernel autotune 不容易覆盖的 policy 收益。

4. kernel 内部优化仍有价值，但在本轮轻量 AKO 条件下，
   它只产生了小幅 cleanup 收益，尚不足以超越强 upstream kernel body。

5. 若下一轮目标是 kernel-internal optimization，
   AKO loop 必须升级成 profiler-driven、codegen-aware、family-aware 的流程。

6. 这套流程值得迁移到 GDN / linear attention，但需要更系统的 profiler 和 trap archive。
```

一句话总结：

```text
这次 AKO 应用的成功点，不是“生成了一个神奇 kernel”，
而是“帮助我们把 GQA decode 的复杂优化空间拆清楚，并把真正的收益边界找出来”。
```

## 参考资料

```text
1. AKO official project page:
   https://tongminglaic.github.io/AKO/

2. AKO4X GitHub:
   https://github.com/TongmingLAIC/AKO4X

3. AKO4X official docs used in this report:
   docs/closed-loop.md
   master/MASTER.md
   templates/closed-loop-scope.md
   templates/iterations.md
   templates/retrospective.md
   templates/agent/lessons-convention.md

4. AKO4ALL GitHub:
   https://github.com/TongmingLAIC/AKO4ALL

5. AKO4ALL protocol files used in this report:
   SKILL.md
   HINTS.md
   ITERATIONS.md

6. 本轮实验记录:
   gqa_decode_ako_style_tuning_plan.md
   gqa_decode_policy_modeling_plan.md
   gqa_decode_microbenchmark_report.md
   gqa_decode_internal_kernel_ako_log.md
```
