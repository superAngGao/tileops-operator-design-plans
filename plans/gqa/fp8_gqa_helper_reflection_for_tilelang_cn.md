# FP8 GQA helper 使用过程复盘：和 TileLang 朋友一起讨论的材料

日期：2026-05-21

这份材料想请 TileLang 开发团队帮忙看一看：当前 TileOps FP8 GQA kernel 里还有不少 C++/CUTE/inline PTX helper，是否有机会用更高比例的 TileLang 原生 DSL 表达，把实现写得更简洁、更自然，也更便于后续维护。

为方便讨论，本文先复盘我们从尽量使用 TileLang 原生表达，到逐步引入 helper 的过程。重点是把当时遇到的问题、解决思路、已验证的路线和仍不确定的边界讲清楚，方便大家站在 TileLang 工具设计的角度一起判断：

```text
当前 FP8 GQA kernel 是否可以用更少 helper、
更高比例的 TileLang 原生 DSL 完成？
如果可以，怎样调整写法、抽象边界或 TileLang 表达方式会更合适？
```

这份材料默认读者只会看 PR #1511 中的当前 FP8 GQA 实现。历史上的中间实验不会作为代码参照出现，因此下文尽量用文字描述它们的设计意图、现象和结论，不列具体 kernel / helper / probe 名字。这样讨论重点可以放在当前实现还能否写得更 TileLang-native，而不是让大家在多个历史版本之间来回对照。

我们目前的理解是：helper 的引入不是为了绕开 TileLang 写主体 kernel，而是为了先固定 FP8 GQA 中几类当时我们没有找到稳定原生表达的硬件契约：FP8 RS WGMMA register fragment、V shared-memory swizzle/TMA descriptor、P/V/O combined layout contract，以及 WGMMA async wait 与 ptxas scoreboard 边界。

PR #1511 中的当前实现仍然高度依赖 TileLang 表达外层结构：persistent tile ownership、GQA head mapping、producer/consumer warpgroup 分工、TMA/barrier 管理、QK WGMMA、online softmax、descale、loop 和大部分 scalar update。helper 主要集中在 FP8 相比 FP16/BF16 新增的 layout/fragment 边界上。

## 1. 背景：为什么 FP8 GQA 比 FP16/BF16 难很多

FP16/BF16 GQA 的主要挑战是把 QK、softmax、PV、TMA 和 barrier 排成有效 pipeline。TileLang 对这类结构比较自然：shared tile、WGMMA、online softmax、TMA stage 和 producer/consumer 分工都能在 DSL 中表达。

FP8 GQA 多了几层硬件契约：

- Q/K/V 以 FP8 存储和参与 Tensor Core，QK accumulator 后需要乘 `q_descale * k_descale`，PV/O 路径还要处理 `v_descale`。
- PV 必须走 FP8 RS WGMMA：`P` 是 register operand A，`V` 是 shared operand B。
- `P` 来自 QK 的 accumulator C layout，但 PV 需要 RS-A register fragment layout。二者不是同一种 lane/register mapping。
- `V` 不能只是 row-major `[N,D]` 放进 shared；它必须变成 WGMMA B descriptor 能解释的 transposed / swizzled physical layout。
- TMA producer 写 shared 的 swizzle、WGMMA descriptor 读 shared 的 swizzle、helper 生成的 physical bytes 必须完全一致。
- FA3-style FP8 V path 还会引入 V-column permutation，因此 O accumulator / output 侧必须做对应 unpermute。
- 如果试图做 consumer 内部 overlap，`wait_group<1>` 后读取 QK accumulator，同时让 PV outstanding，会触发 ptxas 对 WGMMA async accumulator 的保守 dependency barrier。

所以 FP8 不是把 FP16 kernel 的 dtype 换成 FP8。它要求 DSL/lowering 同时看懂：

```text
scale contract
shared physical swizzle contract
TMA tensor-map contract
RS WGMMA register-fragment contract
P/V/O combined permutation contract
WGMMA async group / accumulator lifetime contract
```

这些合同任意一个错，结果不是轻微性能差，而是直接 correctness 错、NaN、或者 SASS 序列化导致 overlap 消失。

## 2. 当前 kernel 的分层：TileLang 负责什么，helper 负责什么

当前 FP8 GQA kernel 不是纯手写 CUDA，也不是完全由 TileLang high-level op lowering 出来的单层实现，而是分层实现。

TileLang 仍然负责：

- persistent tile 调度和 tile ownership；
- batch/head/block 的 program id 映射；
- GQA query-head 到 KV-head 的映射；
- producer / consumer warpgroup 分工；
- shared buffer 分配和 K/V stage 轮转；
- TMA copy、barrier、empty/full slot 协议；
- QK WGMMA；
- online softmax、mask、descale 读取和 scalar update；
- 大部分 O / LSE 生命周期；
- wrapper、autotune/config、测试入口。

helper 负责：

- `acc_s` 到 PV-A register fragment 的 FP8 pack / reorder；
- BN224 下 7 个 K32 PV atom 的 register list 和 descriptor；
- FA3-style 或 TileLang-native 的 V transpose / swizzle；
- PV RS WGMMA 的 raw register/shared descriptor contract；
- raw O accumulator layout、FA3 output unpermute、epilogue store；
- 一些更裸的 `wgmma.wait_group`、operand fence、anchor wait；
- byte-level / raw shared dump 与对比。

当时采用 helper 的主要原因不是“TileLang 不能写 kernel 主体”，而是：

```text
TileLang 能表达逻辑矩阵操作，
但 PR #1511 中的 FP8 GQA 实现需要保证硬件看到的 physical layout / register fragment
和 WGMMA/TMA descriptor byte-exact、lane-exact 对齐。
```

## 3. helper 引入过程的主线

### 3.1 第一阶段：先尽量使用 TileLang 原生路径

早期默认正确路径更接近 TileLang-native contract：

```text
QK:
  Q/K shared -> TileLang WGMMA -> acc_s

P:
  acc_s -> softmax -> p_shared[M,N] FP8
  p_shared[:, k:k+32] -> p_fp8 K32 fragment

V:
  producer-side V movement / staging
  v_tc_smem[D,N] -> v_tc_shared[D,32]

PV:
  TileLang K32 RS WGMMA
  acc_delta -> acc_o update
```

这个路径的优点是 correctness 容易打通。它显式经过 shared-P 和 K32 staging，天然绕开了一些 fragment-to-fragment layout conflict。

但它有性能问题：

- P 多了 shared round-trip；
- V full tile 到 K32 slice 多了 staging movement；
- K32 分段让 PV 结构更碎；
- 很难逼近 FA3 那种 P register conversion + V LDSM/STSM + PV RS 的紧凑路径。

当时的直觉是：应该尽可能把 P 保持在 register 中、把 V 变成 WGMMA-B 能直接消费的 shared physical layout、让 PV 走更完整的 RS WGMMA。

### 3.2 第二阶段：尝试让 P 保持在寄存器中

目标是把：

```text
acc_s -> p_shared -> T.copy -> p_fp8
```

改成：

```text
acc_s.data -> 外部寄存器转换逻辑 -> p_fp8.data
```

这里踩到几个点：

1. `T.copy(acc_s, p_frag)` 会遇到 layout conflict。QK 输出是 WGMMA accumulator D layout，PV 需要 matrix A register fragment layout。二者本来就不是同一张 register mapping。
2. 传 fragment 给 extern helper 时，用 `T.address_of(acc_s[0,0])` 不适合 accumulator fragment；更稳的是传 raw `.data`。
3. `__shfl_sync` 的语义容易写错。不能让 destination lane 直接索引 source lane 的不同位置，因为 source lane 实际传出的是它自己算的 index value。第一版 helper 用常量 index 循环保证 correctness，后来再压成按 lane mapping 的版本。

结果：

- correctness 证明了 `.data + extern` 这类路线可行；
- 但性能上这条“每个 K32 slice 重建 P fragment”的路线不是最终方案；
- 它每个 K32 slice 引入大量 `SHFL/PRMT`，没有像 FA3 那样通过 `convert_layout_acc_Aregs<TiledMmaPV>` 原地 reinterpret + 少量 permutation 完成；
- 因此这条尝试更多是 correctness 小实验，不是理想性能主线。

想请 TileLang 朋友一起判断的点：

TileLang 如果要减少这类 helper，需要一等表达 “accumulator C fragment -> RS-A fragment” 的 layout conversion，而不是把它退化成普通 tensor copy 或大量手写 shuffle。

理想能力类似：

```python
p_regs = T.convert_acc_to_wgmma_a(
    acc_s,
    mma_shape="m64n128k32",
    dtype="fp8",
    source_role="qk_accumulator_c",
    target_role="pv_rs_a",
)
```

或者至少让 `T.Fragment` / layout system 能显式描述 WGMMA A 和 C 的 PTX fragment layout，并让 lowering 生成必要的 shuffle/byte_perm。

### 3.3 第三阶段：V transpose / swizzle 的表达尝试

PV 的 V 是 shared operand B。这里不是“做一个 transpose”这么简单，而是：

```text
global V[N,D]
  -> TMA/source shared tile
  -> LDSM.T / PRMT / STSM 或其它 movement
  -> WGMMA-B descriptor 期待的 shared physical layout
```

TileLang generated CUDA 明确显示，当前 FP8 PV 有两类 descriptor：

```cpp
// K32 staging path
tl::initialize_wgmma_descriptor<3, 1, 16>(desc_b, v_tc_shared);
// layout_type=3 -> SWIZZLE_32B

// full-PV path
tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, v_tc_smem);
// layout_type=1 -> SWIZZLE_128B
```

这带来一个重要边界：

```text
不能把 v_tc_smem[D,128] 写成 SWIZZLE_128B，
再用普通 TileLang logical copy 切成 v_tc_shared[D,32]，
然后交给 SWIZZLE_32B descriptor。
```

这样会把 swizzled physical bytes 当普通逻辑矩阵读走，结果必然错。

#### 3.3.1 试过的 TileLang 原生 swizzle 表达

当时为了尽量不用 helper，试过这些 TileLang 原语 / layout：

- `T.annotate_layout({dst: tilelang.layout.make_full_bank_swizzled_layout(dst)})`
- `make_wgmma_swizzled_layout`
- `make_half_bank_swizzled_layout`
- `make_quarter_bank_swizzled_layout`
- `make_swizzled_layout(k_major=True)`
- `make_swizzled_layout(k_major=False)`
- shared logical shape 从 `[block_n, dim]` 改成 `[dim, block_n]`
- 自定义 `tilelang.layout.Layout([128,128], perm_n/perm_d)`
- `T.tma_copy(..., dst_with_candidate_layout)`，并用 `disable_tma=False/True` 对照

这些实验要回答的问题不是“名字上是不是 swizzle”，而是：

```text
T.tma_copy 写出的 shared physical bytes
是否 byte-exact 等于后续 LDSM/STSM 路径需要的 source contract？
```

#### 3.3.2 关键结果：一种标准 source layout 可行，另一种自定义 source permutation 不适合直接由 TMA 复刻

我们后来分清了两套 source contract：

```text
FA3 source:
  V[n,d] -> sVt(d,n) / SmemLayoutVt
  -> FA3 LDSM.T + PRMT + STSM
  -> SmemLayoutVtMma

TileLang full-PV source:
  V[n,d] -> 自定义 source permutation
  -> LDSM.T + PRMT + STSM movement
  -> TileLang full-swizzle PV-B layout
```

对 FA3 `sVt` 入口，TileLang/TMA 的 standard full-bank 或 wgmma swizzle 可以 byte-exact 复刻：

```text
direct_tma:full   equal_to_pack=True, mismatched_bytes=0
direct_tma:wgmma  equal_to_pack=True, mismatched_bytes=0
```

这说明 `TMA -> FA3 sVt` 这一步是可行的。

但对 TileLang-native 的那套自定义 source permutation，内置 swizzle 都不等价：

```text
none                                      mismatch 14336
make_full_bank_swizzled_layout / wgmma    mismatch 15680
make_half_bank_swizzled_layout            mismatch 15888
make_quarter_bank_swizzled_layout         mismatch 15920
make_swizzled_layout(k_major=True)        mismatch 15680
make_swizzled_layout(k_major=False)       mismatch 15990 or alignment assert
```

自定义 `tilelang.layout.Layout([128,128], perm_n/perm_d)` 可以 byte-exact 表达 correctness：

```text
logical (n,d) -> physical perm_n(n) * 128 + perm_d(d)
```

但 generated CUDA fallback 成普通 loop store，不走 TMA：

```cpp
for (int i = 0; i < 128; ++i) {
  src_smem[perm_expr(i, threadIdx.x)] = v[i * 128 + threadIdx.x];
}
```

TileLang 也提示：

```text
Came across unsupported swizzle layout for src: v, dst: src_smem,
fallback to normal copy
```

所以这个自定义 Layout 只能证明 swizzle 语义正确，不能作为性能实现替代旧的 shared-to-shared 预打包过程。

进一步的外部 PTX TMA 小实验也确认：对完整 `128x128` FP8 V tile，合法 standard tensor-map 参数没有一个能一步复刻这套 `perm_n/perm_d` physical layout。32B/64B swizzle 还会受到 boxDim span 限制。

想请 TileLang 朋友一起判断的点：

- 如果目标是 FA3 contract，TileLang 已经接近能表达 `TMA -> sVt`，但还需要 P/O 也留在 FA3 contract 中，不能只替换 V。
- 如果目标是 TileLang-native contract，可能不适合把 TMA 直接复刻这套自定义 source permutation 当成唯一方向。另一个更自然的方向是：TMA 落到硬件支持的 source layout，然后由 DSL/lowering 或 helper 生成 TileLang PV-B 需要的 exact physical layout。
- DSL 需要能区分“logical layout”和“WGMMA/TMA physical layout”，并提供 byte-level debug / descriptor inspection 能力。

### 3.4 第四阶段：FA3 contract 与 TileLang-native contract 分叉

一个重要反思是：FA3 的 P/V/O 是一套闭环 contract，不能只搬其中一段。

FA3 FP8 row-major V 路径大致是：

```text
P:
  QK-C accumulator layout
  -> permute_Cregs_fp8
  -> convert_layout_acc_Aregs<TiledMmaPV>
  -> FP8 PV-A regs

V:
  TMA -> sVt / SmemLayoutVt
  -> LDSM.T + PRMT + STSM
  -> sV / SmemLayoutVtMma

O:
  PV raw accumulator has permuted columns
  -> permute_output_fp8
  -> logical O
```

TileLang-native full-PV contract 则是：

```text
P:
  TileLang PV-A fragment layout

V:
  TileLang full-PV SWIZZLE_128B B layout

O:
  TileLang acc_delta / acc_o fragment layout
```

因此：

```text
FA3 的 V swizzle 只能配 FA3 的 P register permutation 和 O unpermute。
TileLang 的 V layout 需要 TileLang 自己的 P/O contract。
```

我们曾经把 FA3-style V movement 直接接到 TileLang full-PV descriptor，结果 correctness 明显不对。后来一个只验证 PV 子图的小实验说明：

```text
same descriptor <1,1,64> / SWIZZLE_128B:
  TileLang 原生 full-PV 路径                 correct
  直接接入 FA3-style V movement              wrong
  按 TileLang descriptor 物理公式写出的版本   correct
```

这说明 full-PV WGMMA descriptor 本身没错；问题在于那条外部 V movement 写出的 `STSM` target physical layout 不等于 TileLang descriptor 正在解释的 layout。

后来我们用一条更贴近 TileLang descriptor 的写法显式写：

```text
dst(d, n) = d * 128 + (((n >> 4) ^ (d & 7)) << 4) + (n & 15)
src(d, n) = n * 128 + d
```

它能和 TileLang 原生 full-PV 路径在 byte/结果上对齐。但这只是 scalar contract implementation，不是高性能 implementation。

想请 TileLang 朋友一起判断的点：

TileLang 可以考虑把这类 path 显式建模成两种合法 contract：

1. **FA3-closed contract**：P/V/O 都在 FA3 raw fragment/layout 内闭环，TileLang 不在中途解释 raw O。
2. **TileLang-native contract**：P/V/O 都按 TileLang fragment/layout 闭环，V helper 输出 TileLang descriptor 能解释的 physical bytes。

最需要小心的是混用：

```text
FA3 V layout + TileLang P/O layout
```

它在单个 movement 层面看起来“像 swizzle”，但整体 PV 数学不成立。

### 3.5 第五阶段：收敛到 PR #1511 中的当前实现

当前准备给大家看的 PR #1511，是这轮实验收敛出的最终形态。

它已经做到：

- producer / consumer warpgroup ping-pong；
- V 走 TMA load；
- V layout 走外部 layout movement；
- PV 走外部 RS WGMMA / raw accumulator 路径；
- P/PV/O 更接近 FA3 contract；
- SASS 中 QGMMA 更连续，避免了 overlap 版本的大量 `DEPBAR/ARRIVE`。

但它仍然只实现了 FA3 pipeline 的一半：

```text
已经有:
  producer/consumer warpgroup ping-pong
  V TMA staging
  已经验证过的 P/V/O layout contract

还没有做到:
  consumer warpgroup 内部的 QK / softmax / PV overlap
```

consumer 内部仍偏串行：

```text
QK[n]
wait QK[n]
softmax[n]
PV[n]
update O
```

没有做到理想的：

```text
issue QK[n]
issue PV[n-1]
wait_group<1>
softmax[n]
wait_group<0>
update O
```

## 4. 为什么 overlap 尝试没有转成性能收益

当时确实尝试过 source-level overlap。TileLang source、generated CUDA 和 PTX 里都能看到想要的顺序：

```text
issue QK[n]
issue PV[n-1]
wgmma.wait_group.sync.aligned 1
read acc_s
```

但最终 SASS 中出现大量：

```text
WARPGROUP.DEPBAR
WARPGROUP.ARRIVE
```

ptxas warning 也提示：

```text
wgmma.mma_async instructions are serialized due to non wgmma instructions
reading accumulator registers between start and end of the pipeline stage
```

这里的根因更接近 WGMMA async scoreboard，而不是 TileLang source 顺序错。

`wait_group<1>` 的语义是：保留最近 1 个 WGMMA group pending，等待更早的 group 完成。我们的意图是：

```text
QK[n]       older group
PV[n-1]    newer group
wait<1>    QK done, PV may still be flying
read acc_s
```

但 ptxas 必须保证 scalar code 不会读到仍被 in-flight WGMMA 写的 accumulator，也不会破坏后续 WGMMA 仍要读写的 register fragment。如果它无法证明：

- `wait<1>` 后读的是已经完成的 QK accumulator；
- PV 写的是独立 accumulator；
- extern helper 内外的 raw register live range 不 alias；
- direct-to-O update 不会影响 outstanding group；

它就会插 dependency barrier，把 GMMA stream 分段。

我们做过 fresh-delta / visible-PV 旁路尝试：

```text
QK[n]   -> acc_s
PV[n-1] -> acc_delta
wait_group<1>
softmax(acc_s)
wait_group<0>
acc_o = acc_o * ss + acc_delta * v_scale[n-1]
```

这个 contract 更清晰，避免 PV 直接写旧 `acc_o`。但在当时的 TileLang 表达里，我们还没有找到高效表达 FP8 RS A fragment load 和 V TMA layout 的方式。我们试过一条按 K32 atom 暴露 PV 的路径，correctness 可以做，但性能很差；也试过更底层地把多个 K32 atom 放到同一个 WGMMA group 里，语义上可行，但需要手写 flat local buffer 和 manual `ptx_ldmatrix` destination offset，也不是性能路径。

想请 TileLang 朋友一起判断的点：

减少 helper 不只需要 layout API，还需要更明确的 WGMMA async contract：

- 命名/分离 accumulator group；
- 表达 `wait_group<1>` 后哪些 registers 可读；
- 告诉 lowering QK/PV accumulators 不 alias；
- 避免 extern/helper 边界让 ptxas 失去 alias/lifetime 信息；
- 或提供一个 higher-level “begin PV / wait PV / update O” primitive，让 compiler 控制完整 live range。

## 5. 当前 helper 列表按表达边界分类

下面不是精确函数名清单，而是按我们当时遇到的 DSL/硬件表达边界归类。

### 5.1 P register conversion helper

用途：

```text
QK accumulator C layout
  -> softmax P
  -> FP8 RS-A register fragment
```

引入原因：

- `T.copy(acc_s, p_frag)` layout conflict；
- high-level `T.wgmma_gemm` 对 FP8 RS full chunk 不稳定；
- shared-P round-trip correct 但多 movement；
- 当前 TileLang 无一等 API 表达 FA3-style `permute_Cregs_fp8 + convert_layout_acc_Aregs + FP8 pack`。

可以一起评估的表达方向：

- WGMMA C/A fragment role-aware layout conversion；
- FP8 pack with fixed register order；
- raw fragment mapping dump；
- 能区分 K32 default 和 full-PV / BN224 contract。

### 5.2 V transpose / swizzle helper

用途：

```text
V[N,D] source shared
  -> transposed / swizzled WGMMA-B physical layout
```

引入原因：

- WGMMA B descriptor 需要 exact physical bytes；
- K32 uses `SWIZZLE_32B`，full-PV uses `SWIZZLE_128B`；
- FA3 风格的标准 source layout 可以用 TileLang standard swizzle 产生，但 TileLang-native 的自定义 source permutation 不能由 standard TMA descriptor 直接产生；
- custom Layout correctness 可表达但 fallback 成普通 loop store。

可以一起评估的表达方向：

- 显式 TMA tensor map view / destination layout construction；
- 对 unsupported TMA swizzle 是否 fallback、如何提示，提供更明确的观察方式；
- raw shared byte dump / compare；
- WGMMA descriptor inspection；
- `LDSM.T + PRMT + STSM` 这类 shared-matrix movement primitive，或者可组合 intrinsic；
- 能把 source layout 和 target descriptor layout 绑定检查。

### 5.3 PV RS WGMMA helper

用途：

```text
P regs + V shared descriptor -> raw O accumulator
```

引入原因：

- BN224 是 7 个 K32 atom；
- descriptor、base offset、register list 不适合手写猜；
- current high-level path 对 FP8 RS `M64,N128,K224` 不够可靠；
- helper 内能保证 P/V/O contract 不跨回 TileLang fragment layout。

可以一起评估的表达方向：

- first-class FP8 RS WGMMA for K > 32；
- `ldmatrix_a_full` 或 `dst_ki` offset；
- descriptor base offset / ki offset 可观察；
- 支持 full chunk A fragment 填充；
- 能生成连续 WGMMA group，并控制 wait/commit。

### 5.4 O raw layout / unpermute / epilogue helper

用途：

```text
raw PV accumulator
  -> FA3 output unpermute
  -> row/col shared staging or direct global store
  -> acc_o / output
```

引入原因：

- FA3 V layout 可能 permute columns，O 必须 unpermute；
- raw PTX PV 写出的 O register order 不等于 TileLang `acc_delta.data` 可直接解释的 order；
- `CUTE tOrO -> TileLang acc_delta.data` 是真实 correctness boundary；
- 如果中途跨回 TileLang fragment layout，会出错或引入 shared bridge。

可以一起评估的表达方向：

- WGMMA C accumulator raw layout 到 logical O 的 mapping；
- output permutation primitive；
- coalesced epilogue store；
- 允许复用 V shared memory 做 epilogue scratch 的 typed/raw alias contract。

### 5.5 WGMMA wait / anchor helper

用途：

```text
更明确地发出 wait_group / fence / anchor，控制 ptxas 行为
```

引入原因：

- source-level overlap 可写，但 SASS 被 conservative scoreboard 打碎；
- extern boundary、accumulator read、direct-to-O update 会让 ptxas 保守；
- 需要实验不同 wait/fence/anchor 边界。

可以一起评估的表达方向：

- WGMMA async group 的 high-level 表达；
- accumulator lifetime/noalias annotation；
- begin/wait/update 分段 primitive；
- 可检查 generated PTX/SASS 中是否保留预期 wait structure。

## 6. 可以一起讨论的几个方向

这一节不是在预设哪些 helper 应该被替换，也不是给出优先级结论。我们只是把当前实现里几个 helper 边界拆开，说明每一类如果想更 TileLang-native，大概会牵涉到哪些语义。最终是否值得上移到 TileLang 原生表达、应该以什么形式表达，还是希望请 TileLang 团队从 DSL 设计和 lowering 可维护性的角度判断。

### 6.1 方向一：descriptor / raw byte debug 工具

这类能力不直接改变 kernel 主路径，更多是帮助双方更快对齐问题边界：

- dump shared physical bytes；
- dump WGMMA descriptor fields；
- dump fragment role/layout mapping；
- generated CUDA/PTX 中标出 descriptor `layout_type`、leading/stride/base offset；
- 对 `T.tma_copy + Layout` 是否真的走 TMA 给出更明确的可观察信号。

这类能力对我们很有帮助，因为它能帮助使用者判断“是逻辑错、layout 错、还是 lowering fallback”。但它是否应该成为 TileLang 原生调试能力，或者只是作为我们这类 kernel 的本地验证工具，也想听听大家的判断。

### 6.2 方向二：V shared physical layout 的表达方式

这里想讨论的是：V shared physical layout 是否适合由 TileLang DSL 直接表达，还是继续让当前实现通过外部 movement 固定。相关语义大致包括：

```text
TMA -> FA3 sVt
TMA -> hardware-supported source layout
source -> target WGMMA-B layout via LDSM/PRMT/STSM primitive
```

如果这些语义适合进入 TileLang，那么 V transpose/swizzle helper 可能可以减少；如果不适合，也可以保留 helper，但让 TileLang 更清楚地知道它的输入输出 layout contract。

这里我们踩过的坑是：只有一个名字叫 `swizzle` 的 layout 还不够，关键是它和后续 WGMMA descriptor 是否 byte-exact 对齐。

一种仅供讨论的 API 思路是以 target descriptor 为中心：

```python
v_b = T.make_wgmma_b_smem(
    logical_shape=(D, N),
    dtype="fp8",
    descriptor="m64n128k32_rs_b",
    swizzle="128B",
)

T.tma_copy(v_global_view, v_source, ...)
T.transform_smem(v_source, v_b, op="ldsm_t_prmt_stsm")
T.assert_descriptor_matches(v_b, pv_descriptor)
```

### 6.3 方向三：P accumulator -> RS-A 的表达方式

这里想讨论的是：QK accumulator 到 PV RS-A register fragment 的转换，是否适合被 TileLang 作为 fragment role conversion 来表达。

如果只是把 helper 翻译成很多 DSL-level shfl，可能会像那条寄存器化 P 尝试一样 correct 但慢。更理想的是 DSL/lowering 能生成接近 FA3 的：

```text
accumulator reinterpret
FP32 -> FP8 pack
small local permutation / byte_perm
```

这可能需要 PTX fragment layout 内建知识，或者接入类似 CUTE layout algebra 的 fragment mapping。也可能大家会判断，这类转换更适合作为外部 intrinsic，只在 TileLang 里声明语义边界。

### 6.4 方向四：PV RS WGMMA 边界怎么表达

这里不是说一定要完全去掉 PV helper，而是想请大家判断这段边界应该放在哪里。如果希望把更多 PV 逻辑上移到 TileLang，可能会涉及：

- FP8 RS WGMMA；
- K > 32 chunk；
- P regs 是 raw fragment list；
- B descriptor offset 正确；
- raw O accumulator layout 可继续被 epilogue consume；
- begin/wait/update 可拆分而不触发 ptxas 序列化。

这也许最终更适合成为 TileLang intrinsic / dialect op，也可能继续保留外部 helper、但让 TileLang 看到更明确的 dependency 和 layout 信息。这里希望听 TileLang 团队对抽象边界的建议。

### 6.5 方向五：consumer 内部 overlap 的表达方式

这不只是 DSL expressiveness，还涉及 ptxas 的最终 SASS 调度。

DSL 也许可以帮助提供更清晰的 lifetime/noalias 信息，但是否能稳定避免 `WARPGROUP.DEPBAR/ARRIVE`，需要最小 repro 和 SASS 级验收。这个方向尤其希望听听大家认为 TileLang 层面能帮到哪里、哪些部分更接近 backend/ptxas 行为边界。

## 7. 想请 TileLang 朋友一起讨论的问题

下面这些问题可以作为讨论 agenda。

1. 从 TileLang 的设计角度看，WGMMA fragment role 是否适合作为一等概念暴露？

   例如同样是 per-thread register list，能否区分：

   ```text
   QK accumulator C fragment
   PV RS-A fragment
   PV accumulator C fragment
   output logical fragment
   ```

2. `T.copy(fragment_a, fragment_b)` 在 role/layout 不兼容时，是否有更适合 TileLang 风格的提示或 lowering 方式？

   我们遇到过 layout conflict、silently no-write 和 fallback 几种表现，想请大家帮忙判断这些现象分别对应什么 DSL 语义边界。

3. `ldmatrix_a` 的 destination K offset 是否适合通过 TileLang 原语暴露？

   BN224/K224 需要填多个 K32 atom 的 A fragment。我们当时因为没有找到 `dst_ki` 一类表达，只能手写 flat local buffer + `ptx_ldmatrix` offset。

4. FP8 RS WGMMA `K > 32` 是否适合成为 TileLang 的 first-class lowering 场景？

   当前 lower-level emitter 能发 K224，但周边 A fragment load、descriptor、wait/fence 还要手写。

5. TMA destination layout 是否能区分：

   ```text
   correctness-custom Layout
   hardware-supported TMA swizzle
   fallback normal copy
   ```

   对这类性能敏感路径，是否可以给用户一个选项：在不走 TMA/bulk path 时给出更明确提示，而不是悄悄变成普通 loop store。

6. WGMMA descriptor introspection 是否符合 TileLang 的调试模型？

   例如在 debug mode 输出：

   ```text
   layout_type: 32B/64B/128B
   leading offset
   stride offset
   base offset
   ki offset
   ```

7. “this smem buffer physical layout matches this WGMMA descriptor” 这类 assertion 是否适合放进 TileLang 调试/验证工具？

   我们很多 bug 都是 helper 写出的 bytes 和 descriptor 解释不一致。

8. FA3-style combined contract 是否适合由 TileLang 表达，还是更适合继续作为外部 intrinsic/helper？

   即 P/V/O 三者作为闭环，不要求中途回到 TileLang logical fragment。比如：

   ```text
   acc_s.data + v_smem -> helper/dialect op -> acc_o.data
   ```

   但 DSL 仍知道这个 op 的 dependency、wait、resource usage。

9. 对 WGMMA async overlap，accumulator noalias/lifetime annotation 是否是 TileLang 层面能帮助表达的东西？

   目标是让 ptxas 更容易相信：

   ```text
   wait_group<1> 后读取的是 older QK accumulator，
   newer PV accumulator 仍可 pending。
   ```

10. `T.call_extern` ABI 层对 raw fragment/register 输入是否需要更明确的约定？

    当前 `.data` 可用，但对我们来说更像经验路径。想请大家帮忙判断哪些 fragment `.data` 可以被当作稳定 ABI，哪些最好不要这样用。

最后，我们最希望请大家帮忙看的不是“把所有 helper 都消灭”，而是当前这些 helper 边界里，有哪些其实可以更自然地表达在 TileLang 里；哪些继续保留为外部 intrinsic/helper 更合适；以及如果要减少 helper，当前实现应该优先调整写法、layout 表达，还是抽象边界。

如果大家觉得有必要，我们可以再把某一段单独拆成最小例子继续讨论。这封材料先尽量只保留背景、现象和我们遇到的困惑，避免把太多中间版本和验证脚本一起丢给大家。
