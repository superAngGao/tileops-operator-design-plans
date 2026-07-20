# FA3 FP8 Swizzle Contract for GQA

日期：2026-05-12

这份文档只回答一件事：**FP8 GQA 里每个矩阵到底应该在什么阶段、以什么 layout/swizzle 形态出现；FA3 是不是这样做的；TileOps 现在做到哪一步；如果用 PTX 写，应该写哪一段。**

## 一句话结论

FP8 attention 里没有一个孤立的 “the swizzle”。真正的 correctness contract 是一整条链：

```text
Q/K:
  GMEM [seq,d] -> k-major SMEM -> QK WGMMA

P:
  QK FP32 accumulator -> softmax -> FP8 PV-A register fragment

V:
  GMEM V[n,d]
    -> FA3 sVt(d,n) / TMA-side shared layout
    -> LDSM.T + byte_perm + STSM
    -> FA3 sV(d,n) / PV-B shared layout，且 V columns 被 permute

O:
  PV accumulator 继承同一个 column permutation
    -> epilogue unpermute O columns
    -> global O[m,d]
```

FA3 是把这条链作为一个闭环 contract 来做的。TileOps 现在已经验证了其中几块，但当前最快正确 full kernel 仍是 TileLang-native 路线，里面还有 shared-P、K32 staging、accumulator bridge 等桥。

## 术语先分清

我们之前一直说 “swizzle”，其实混了两种东西：

| 名称 | 出现位置 | 含义 |
| --- | --- | --- |
| descriptor swizzle | TMA tensor map 或 WGMMA shared descriptor，例如 32B/64B/128B | 硬件用来解释 shared-memory tile 地址的 swizzle mode |
| FA3 V transpose permutation | `sVt -> LDSM.T -> byte_perm -> STSM -> sV` | tile-wise transpose + packed FP8 byte reorder + bank-friendly STSM value layout；它会 permute V columns，所以 O columns 也要 unpermute |

FA3 论文里讨论 V 的 layout 问题，更接近第二个：**完整的 V transpose / permutation / O unpermute 流程**，不是单个 descriptor enum。

## 路线一：FA3 contract 矩阵形态

这是当前已经在 PV 子图里跑通、接下来要 streamline 的路线。这里 P/V/O 必须成套出现，不能和 TileLang contract 混搭。

| 矩阵/阶段 | 逻辑形状 | FA3 contract 需要的 layout/swizzle | FA3 是否这样做 | TileOps 当前状态 |
| --- | --- | --- | --- | --- |
| `Q` in QK | `Q[M,D]` | FP8 WGMMA k-major SMEM；`D` 是 QK 的 WGMMA-K | 是。`SmemLayoutQ` 用 `GMMA::Major::K` | 对齐。QK shared swizzle 不是当前问题 |
| `K` in QK | `K[N,D]`，消费时等价 `K^T` | FP8 WGMMA k-major SMEM；`D` 是 QK 的 WGMMA-K | 是。`SmemLayoutK` 用 `GMMA::Major::K` | 对齐。K swizzle 不是当前主要问题 |
| `P` / PV operand A | `P[M,N]`，softmax 后 | QK-C accumulator layout -> `permute_Cregs_fp8` -> `convert_layout_acc_Aregs<TiledMmaPV>` -> FP8 PV-A registers | 是 | PV 子图里已跑通；要把实现从 CUTE/oracle 形态改成 PTX register reorder + FP8 pack |
| `V` in GMEM | `V[N,D]`，`D` 连续 | 普通 row-major V 输入 | 是 | 对齐 |
| `V` TMA destination | 逻辑上看成 `sVt[D,N]` | `SmemLayoutVt = ss_smem_selector<TmaMajorV=MN, Element, D, N>` + `Step<_2,_1,_3>`；入口可由 full/wgmma descriptor swizzle 表达 | 是。FA3 把 V TMA global view 构造成 `(K,N)`，落到 `sVt` | 已证明可行：`make_full_bank_swizzled_layout` / `make_wgmma_swizzled_layout` byte-match `fp8_pack_v_128x128_fa3_vt`；要删掉中间 pack bridge |
| `V` / PV operand B | 逻辑上是 `V^T[D,N]` | `sV / SmemLayoutVtMma = ss_smem_selector<MmaMajorV=K, Element, D, N>` + `Step<_1,_2,_3>`；由 `LDSM.T + byte_perm + STSM` 产生，带 V-column permutation | 是 | FA3 helper/probe 已跑通；要用 PTX 替换 CUTE helper |
| `O` PV accumulator | `O[M,D]` | 因为 V columns 被 permute，O columns 也被 permute；写回/累加前必须 `permute_output_fp8` | 是 | PV 子图已跑通；现在还存在 `acc_delta_shared -> T.copy -> TileLang acc_delta` 出口 bridge，要继续消掉 |

## 路线二：TileOps / TileLang-native contract 矩阵形态

这是后续第二阶段再打开的路线。它的 P/V/O contract 和 FA3 不同，所以 P swizzle 也必须换，不能沿用 FA3 的 `permute_Cregs_fp8 + convert_layout_acc_Aregs<TiledMmaPV>`。

| 矩阵/阶段 | 逻辑形状 | TileOps contract 需要的 layout/swizzle | 是否与 FA3 相同 | TileOps 当前状态 |
| --- | --- | --- | --- | --- |
| `Q` in QK | `Q[M,D]` | 默认/k-major QK shared layout | 基本相同 | 已正确，先不动 |
| `K` in QK | `K[N,D]`，消费时等价 `K^T` | 默认/k-major QK shared layout | 基本相同 | 已正确，先不动 |
| `P` / PV operand A | `P[M,N]`，softmax 后 | 必须生成 TileOps/TileLang PV-A fragment 期望的 raw register layout；K32 default 和 full-PV contract 还不同 | 不同 | default 走 shared-P round-trip；register-P helper 有但不是最终形态 |
| `V` default K32 operand B | `V^T[D,N]` 的 K32 slice | `v_tc_smem[D,N] -> v_tc_shared[D,32] -> TileLang K32 PV-B operand` | 不同 | 当前最快 `ws_shared_p / ws_default` 使用这条，正确但有 staging movement |
| `V` full-PV operand B | `V^T[D,N]` full tile | TileLang full-PV `SWIZZLE_128B` B layout | 不同 | `tl_full` 正确；`pack_ldsm_tl_full` 正确但 prepack 太慢 |
| `tl_ldsm_src` source | `tl_ldsm_src[N,D]` physical bytes | `fp8_pack_v_128x128_tl_ldsm_src` 的自定义 `perm_n/perm_d` source permutation | 不是 FA3 `sVt` | 只能服务 TileLang full-PV LDSM helper；显式 PTX/TMA probe 说明 standard tensor-map 不能直接复刻 |
| `O` accumulator/update | `O[M,D]` | TileLang `acc_delta` / `acc_o` fragment layout | 不同 | 当前路径正确；如果走 TileOps contract，P/V/O 都要按这套重写 |

所以从数学上看，两条路线都在算：

```text
P[M,N] @ V[N,D] -> O[M,D]
```

但 tensor core 的 physical operand-A、operand-B、accumulator/output contract 不同。FA3 的 P swizzle 只能配 FA3 `sV / SmemLayoutVtMma` 和 FA3 O unpermute；TileOps 的 V layout 需要 TileOps 自己的 P swizzle。

## 路线一图解：FA3 contract

这一节只画 FA3 contract。这里的重点是：Q/K 比较普通，真正成套绑定的是 `P/V/O`。

### FA3-Q：QK operand A

```text
logical Q[M,D]

          D / head_dim, contiguous, WGMMA-K
        +-----------------------------------+
   M    | q00 q01 q02 ... q0D              |
 query  | q10 q11 q12 ... q1D              |
 rows   | ...                               |
        +-----------------------------------+

FA3 shared contract:
  Q -> SmemLayoutQ
  major = GMMA::Major::K
  K dimension = D

tensor core sees:
  operand A for QK, k-major shared tile
```

结论：这和我们默认 QK contract 对齐，先不改。

### FA3-K：QK operand B

```text
logical K[N,D]

          D / head_dim, contiguous, WGMMA-K
        +-----------------------------------+
   N    | k00 k01 k02 ... k0D              |
 key    | k10 k11 k12 ... k1D              |
 rows   | ...                               |
        +-----------------------------------+

QK math:
  Q[M,D] @ K^T[D,N] -> S[M,N]

FA3 shared contract:
  K -> SmemLayoutK
  major = GMMA::Major::K
  K dimension = D

tensor core sees:
  operand B for QK, k-major shared tile
```

结论：K 也不需要特殊 V-style transpose。

### FA3-S/P：QK output 到 PV operand A

```text
QK WGMMA output:

logical S[M,N] / P[M,N]

          N / sequence block
        +-----------------------------------+
   M    | s00 s01 s02 ... s0N              |
 query  | s10 s11 s12 ... s1N              |
 rows   | ...                               |
        +-----------------------------------+

physical register state right after QK:
  tSrS = FP32 C accumulator fragment
  layout = QK-C accumulator layout

after mask/softmax:
  tSrS still logical P[M,N]
  layout still QK-C accumulator layout

FA3 PV-A conversion:
  tSrS
    -> permute_Cregs_fp8(tSrS)
    -> convert_layout_acc_Aregs<TiledMmaPV>
    -> FP8 pack
    -> tOrP

tensor core sees:
  operand A for PV
  layout = FA3 PV-A register layout
```

可以把这一步画成：

```text
P logical [M,N]

    QK-C register layout                    PV-A register layout
  +----------------------+    permute     +----------------------+
  | p elements in QK-C   |  ----------->  | p elements packed    |
  | accumulator order    |  reinterpret   | as FP8 A registers   |
  +----------------------+  + fp8 pack    +----------------------+

注意：
  这不是 shared swizzle。
  这是 register fragment layout transform。
```

更具体地说，这一步可以拆成两张图。

第一张图：每个 thread 自己手里的 QK-C accumulator raw registers。对 `64x128` tile，当前每个 thread 有 `64` 个 FP32 accumulator value，可以按 8 组画：

```text
before permute_Cregs_fp8, per thread:

raw acc_s index:

group 0:  [ 0] [ 1] [ 2] [ 3] [ 4] [ 5] [ 6] [ 7]
group 1:  [ 8] [ 9] [10] [11] [12] [13] [14] [15]
group 2:  [16] [17] [18] [19] [20] [21] [22] [23]
group 3:  [24] [25] [26] [27] [28] [29] [30] [31]
group 4:  [32] [33] [34] [35] [36] [37] [38] [39]
group 5:  [40] [41] [42] [43] [44] [45] [46] [47]
group 6:  [48] [49] [50] [51] [52] [53] [54] [55]
group 7:  [56] [57] [58] [59] [60] [61] [62] [63]
```

FA3 `permute_Cregs_fp8` 对每个 group 做同一个 `float2` 交换：

```text
swap:
  [8*g + 2, 8*g + 3]  <->  [8*g + 4, 8*g + 5]
```

所以：

```text
after permute_Cregs_fp8, per thread:

group 0:  [ 0] [ 1] [ 4] [ 5] [ 2] [ 3] [ 6] [ 7]
group 1:  [ 8] [ 9] [12] [13] [10] [11] [14] [15]
group 2:  [16] [17] [20] [21] [18] [19] [22] [23]
group 3:  [24] [25] [28] [29] [26] [27] [30] [31]
group 4:  [32] [33] [36] [37] [34] [35] [38] [39]
group 5:  [40] [41] [44] [45] [42] [43] [46] [47]
group 6:  [48] [49] [52] [53] [50] [51] [54] [55]
group 7:  [56] [57] [60] [61] [58] [59] [62] [63]
```

这张图只描述 **本 thread raw registers 内部** 的交换。没有跨 lane shuffle，也没有 shared memory。

第二张图：permuted `tSrS` 被重新解释成 PV-A operand register view。

```text
permuted FP32 QK-C registers

  raw acc_s.data
        |
        |  convert_layout_acc_Aregs<TiledMmaPV>
        v

PV-A register view tOrP_acc

  tOrP_acc logical axes:
    outer:  PV-A register atom layout
    m:      M tile fragment coordinate
    k:      N/K tile fragment coordinate for PV

  same underlying registers,
  different interpretation for PV operand A.
```

然后做 FP32 -> FP8 pack：

```text
tOrP_acc FP32 values
  -> cvt.rn.satfinite.e4m3 / e4m3x2
  -> tOrP FP8 registers
  -> PV WGMMA operand A
```

抽象成图：

```text
                  same acc_s.data storage
              +---------------------------+
QK-C view     |  FP32 P in QK-C order     |
              +---------------------------+
                         |
                         | permute_Cregs_fp8
                         v
              +---------------------------+
permuted      |  FP32 P in FA3 bridge     |
QK-C view     |  order                    |
              +---------------------------+
                         |
                         | convert_layout_acc_Aregs
                         v
              +---------------------------+
PV-A view     |  FP32 P as PV-A fragment  |
              +---------------------------+
                         |
                         | FP32 -> FP8 pack
                         v
              +---------------------------+
tensor core   |  FP8 tOrP registers       |
operand A     +---------------------------+
```

这里最容易犯错的是把最后的 `tOrP` 当成 TileLang `p_fp8.data`。它不是。它只保证在 **FA3 PV-A + FA3 PV-B + FA3 O unpermute** 这一整组 contract 里被 tensor core 正确解释。

### FA3-V：GMEM 到 TMA source view

原始 V 是普通 row-major：

```text
logical / GMEM V[N,D]

          D / head_dim contiguous
        +-----------------------------------+
   N    | v00 v01 v02 ... v0D              |
 seq    | v10 v11 v12 ... v1D              |
 rows   | v20 v21 v22 ... v2D              |
        | ...                               |
        +-----------------------------------+
```

但 PV GEMM 是：

```text
P[M,N] @ V[N,D] -> O[M,D]
```

对 PV WGMMA 来说，V 的 K 维是 `N`，所以 FA3 把 V 的 TMA view 改成 `(D,N)`：

```text
TMA logical view mVt_TMA / gVt_TMA:

          N / sequence block, PV-WGMMA-K
        +-----------------------------------+
   D    | v00 v10 v20 ... vN0              |
 head   | v01 v11 v21 ... vN1              |
 dim    | v02 v12 v22 ... vN2              |
        | ...                               |
        +-----------------------------------+

logical shape:
  sVt[D,N]
```

### FA3-sVt：TMA destination shared layout

```text
FA3 TMA destination:

  sVt(d,n)

logical:
          n / sequence
        +-----------------------------------+
   d    | v00 v10 v20 ...                  |
        | v01 v11 v21 ...                  |
        | v02 v12 v22 ...                  |
        | ...                               |
        +-----------------------------------+

physical/shared contract:
  SmemLayoutVt =
    ss_smem_selector<TmaMajorV = GMMA::Major::MN, Element, D, N>
    tiled with Step<_2,_1,_3>

descriptor-level:
  128x128 FP8 can be expressed by full/wgmma swizzle.
```

这就是我们已经证明可以 TMA-direct 的入口 layout。

### FA3-sV：PV operand B

FA3 producer warpgroup 把 `sVt` 变成 `sV`：

```text
sVt / SmemLayoutVt
  -> LDSM.T
  -> __byte_perm(0x6420 / 0x7531)
  -> STSM with bank-friendly value layout
  -> sV / SmemLayoutVtMma
```

结果：

```text
logical sV[D,N] = V^T[D,N]

          N / PV-WGMMA-K
        +-----------------------------------+
   D    | v00 v10 v20 ...                  |
        | v01 v11 v21 ...                  |
        | v02 v12 v22 ...                  |
        | ...                               |
        +-----------------------------------+

physical/shared contract:
  SmemLayoutVtMma =
    ss_smem_selector<MmaMajorV = GMMA::Major::K, Element, D, N>
    tiled with Step<_1,_2,_3>

extra:
  STSM layout permutes V columns in D dimension for fewer bank conflicts.
```

可以抽象画成：

```text
pure transpose would be:
  d order:  d0 d1 d2 d3 d4 d5 ...

FA3 bank-friendly sV is:
  d order:  perm(d0) perm(d1) perm(d2) ...

tensor core consumes:
  P_FA3[M,N] @ V_FA3[N,D_perm]
```

所以 FA3 的 O 也会带同一个 column permutation。

### FA3-O：PV output and unpermute

```text
PV math inside tensor core:

  P_FA3[M,N] @ V_FA3[N,D_perm]
      -> tOrO[M,D_perm]

register state:
  tOrO = PV C accumulator fragment
  columns are permuted because V columns were permuted

FA3 epilogue:
  permute_output_fp8(tOrO)
      -> O[M,D] logical column order
```

图上看：

```text
before unpermute:

          D_perm
        +-----------------------------------+
   M    | o0,p0  o0,p1  o0,p2 ...          |
        | o1,p0  o1,p1  o1,p2 ...          |
        +-----------------------------------+

after permute_output_fp8:

          D logical
        +-----------------------------------+
   M    | o00 o01 o02 ...                  |
        | o10 o11 o12 ...                  |
        +-----------------------------------+
```

到这里，FA3 contract 的 P/V/O 是闭环的。

## 路线二图解：TileOps / TileLang-native contract

这条路线不是 FA3。它也能正确算 `P @ V`，但 operand-A、operand-B、O accumulator 的 physical contract 是 TileLang 自己的。

### TileOps-Q / TileOps-K

QK 部分和 FA3 基本一致：

```text
Q[M,D] and K[N,D]
  D contiguous
  QK WGMMA K dimension = D
  default/k-major shared layout is OK
```

所以 TileOps contract 真正分叉也是从 PV 开始。

### TileOps-P：PV operand A

当前 default 正确路径：

```text
QK output / acc_s FP32
  -> softmax
  -> p_shared[M,N] FP8
  -> T.copy(p_shared[:, pv_k:pv_k+32], p_fp8)
  -> TileLang K32 PV-A fragment
```

图上看：

```text
P logical [M,N]

          N
        +-----------------------------------+
   M    | p00 p01 p02 ... p0N              |
        | p10 p11 p12 ... p1N              |
        +-----------------------------------+

default K32 slicing:

  [N=0..31]   -> p_fp8 for WGMMA ki=0
  [N=32..63]  -> p_fp8 for WGMMA ki=1
  [N=64..95]  -> p_fp8 for WGMMA ki=2
  [N=96..127] -> p_fp8 for WGMMA ki=3

physical/register contract:
  TileLang K32 PV-A fragment layout
```

如果后续走 TileOps-native register-P 优化，目标不是 FA3 P layout，而是：

```text
acc_s.data
  -> TileLang K32/full-PV 的 PV-A raw fragment layout
```

这就是为什么 TileOps contract 下 P swizzle 要另写。

### TileOps-V default：K32 operand B

当前最快 default 路线里，V 最终按 K32 slice 喂给 tensor core：

```text
producer builds v_tc_smem[D,N]

          N full block
        +-----------------------------------+
   D    | v00 v10 v20 ... vN0              |
        | v01 v11 v21 ... vN1              |
        | ...                               |
        +-----------------------------------+

consumer per pv_k:
  v_tc_smem[:, 0:32]    -> v_tc_shared[D,32]
  v_tc_smem[:, 32:64]   -> v_tc_shared[D,32]
  v_tc_smem[:, 64:96]   -> v_tc_shared[D,32]
  v_tc_smem[:, 96:128]  -> v_tc_shared[D,32]

tensor core sees:
  TileLang K32 PV-B shared layout
```

这不是 FA3 `sV / SmemLayoutVtMma`。

### TileOps-V full-PV：TileLang SWIZZLE_128B operand B

full-PV 实验路线想一次喂完整 `N=128`：

```text
V logical [N,D]
  -> v_tc_smem[D,N]
  -> TileLang full-PV B descriptor
```

目标 physical contract：

```text
TileLang full-PV B layout:
  shape logical = [D,N]
  descriptor = SWIZZLE_128B / <1,1,64> style
```

`tl_full` 手写 transpose 能正确写出这个 layout。`pack_ldsm_tl_full` 也正确，但多了 prepack，太慢。

### TileOps-tl_ldsm_src：只服务 full-PV LDSM helper 的 source layout

这不是 tensor core operand B 本身，而是 TileLang full-PV LDSM helper 的 source bytes：

```text
global/logical V[n,d]
  -> row-major v_smem[n,d]
  -> fp8_pack_v_128x128_tl_ldsm_src
  -> tl_ldsm_src physical bytes
  -> LDSM.T + PRMT + STSM
  -> TileLang full-PV B layout
```

`tl_ldsm_src` micro-tile:

```text
physical row pN -> logical n offset
  pN:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
  n :  0  1  4  5  8  9 12 13  2  3  6  7 10 11 14 15

physical col pD -> logical d offset
  pD:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
  d :  0  8  1  9  2 10  3 11  4 12  5 13  6 14  7 15
```

这和 FA3 `sVt` 不一样，也不是我们 streamline FA3 contract 的目标。

### TileOps-O

TileOps-native 路线希望 PV output 直接回到 TileLang accumulator/update contract：

```text
TileLang PV WGMMA
  -> acc_delta fragment
  -> acc_o += acc_delta * v_scale
```

图上看：

```text
O logical [M,D]

          D
        +-----------------------------------+
   M    | o00 o01 o02 ... o0D              |
        | o10 o11 o12 ... o1D              |
        +-----------------------------------+

physical/register contract:
  TileLang acc_delta / acc_o layout
```

因为 TileOps-native V 没有采用 FA3 的 V-column permutation contract，所以这里也不应该调用 FA3 `permute_output_fp8`。如果未来 TileOps-native helper 自己引入新的 V column permutation，那必须同步定义自己的 O unpermute。

## QK 乘积寄存器应该怎么处理

QK 的输出是：

```text
S[M,N] = Q[M,D] @ K^T[D,N]
```

它一开始放在 QK WGMMA 的 FP32 C accumulator registers 里，也就是 FA3 源码里的 `tSrS`。这一段不要理解成 shared-memory descriptor swizzle；它是 **register fragment layout** 问题。

处理顺序应该是：

```text
QK WGMMA
  -> tSrS FP32 accumulator, QK-C layout
  -> mask / scale / online softmax
  -> P still in tSrS logical [M,N]
  -> permute_Cregs_fp8(tSrS)
  -> convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout())
  -> cast/convert to FP8 tOrP
  -> feed PV as operand A
```

要点：

1. **QK accumulate 阶段不需要改寄存器 swizzle**。先让 QK 正常产出 `tSrS`，softmax 也在这个 logical accumulator view 上做。
2. **寄存器 permutation 发生在 softmax 之后、PV 之前**。FA3 在 row-major FP8 V 路径里调用 `permute_Cregs_fp8(tSrS)`，然后用 `convert_layout_acc_Aregs<TiledMmaPV>` 把 QK-C accumulator layout 解释成 PV-A register layout。
3. **这一步必须和 V/O contract 配套**。如果 V 走 FA3 的 `sVt -> sV` column-permuted contract，P 也要走 FA3 的 `permute_Cregs_fp8 + convert_layout_acc_Aregs`。如果 P 仍然走 TileLang shared-P/T.copy contract，那就不是 FA3 闭环，中间会多一次 bridge。

所以，对“QK 乘积放的寄存器，swizzle 应该怎么做”的短答是：

```text
QK 输出寄存器先不动；
softmax 后，把 P 从 QK-C accumulator layout
register-level permute 成 PV-A operand layout。

FA3 公式：
  tSrS -> permute_Cregs_fp8(tSrS)
       -> convert_layout_acc_Aregs<TiledMmaPV>
       -> FP8 tOrP
```

这也是当前应该先改的第一步。原因是：

```text
Q/K shared swizzle 已经自然匹配 FP8 QK WGMMA；
第一个真正跨 contract 的点，是 P 从 QK-C accumulator layout
变成 PV-A operand layout。
```

当前 TileOps 最稳的正确路径用 shared-P round-trip：

```text
tSrS / acc_s
  -> softmax
  -> p_shared FP8
  -> T.copy(p_shared slice, p_fp8)
  -> TileLang PV-A fragment
```

FA3 要消掉的是这座桥：

```text
tSrS / acc_s
  -> softmax
  -> register-level permute/convert
  -> PV-A FP8 registers
```

所以第一步的验收标准应该是一个独立 P-register probe：

```text
输入：同一份 softmax 后 P[M,N]
路径 A：shared-P -> T.copy -> p_fp8
路径 B：acc_s.data -> register permute/convert -> p_fp8
要求：路径 B 的 PV-A fragment 被后续 PV WGMMA 正确解释，PV-only cos = 1.0
```

注意这里的目标不是让 raw bytes 和 TileLang `p_fp8.data` 逐项相等；如果走 FA3 closed contract，目标是 **PV WGMMA 解释后的数学结果一致**。raw layout 可以不同，但 P/V/O 三段必须属于同一个 contract。

### 这一步具体怎么用 PTX 写

实现上不要依赖 CUTE。CUTE helper 只能当 oracle，用来生成/验证 mapping；真正路径应该是一个 PTX/inline-asm extern helper：

```text
acc_s.data
  -> PTX register reorder: QK-C accumulator layout -> PV-A layout
  -> PTX FP32 -> FP8 pack
  -> inline PTX WGMMA RS: A=P registers, B=FA3 sV shared descriptor
  -> PTX output column unpermute / O update
```

第一小步是只做 `acc_s -> P registers`。FA3 的 `permute_Cregs_fp8` 对 row-major FP8 V 来说是每个 thread 本地的 64-bit swap，不需要 shared memory，也不应该先写 `p_shared`。按当前 `64x128` raw fragment，可以先实现成：

```cpp
__device__ __forceinline__ void p_permute_cregs_ptx(float* acc) {
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    // swap float2(acc[8*i + 2 : 8*i + 4])
    // with float2(acc[8*i + 4 : 8*i + 6])
    uint64_t a, b;
    asm volatile("mov.b64 %0, {%1, %2};"
                 : "=l"(a)
                 : "f"(acc[8 * i + 2]), "f"(acc[8 * i + 3]));
    asm volatile("mov.b64 %0, {%1, %2};"
                 : "=l"(b)
                 : "f"(acc[8 * i + 4]), "f"(acc[8 * i + 5]));
    asm volatile("mov.b64 {%0, %1}, %2;"
                 : "=f"(acc[8 * i + 2]), "=f"(acc[8 * i + 3])
                 : "l"(b));
    asm volatile("mov.b64 {%0, %1}, %2;"
                 : "=f"(acc[8 * i + 4]), "=f"(acc[8 * i + 5])
                 : "l"(a));
  }
}
```

这对应我们现在 helper 里的 raw 版本：

```text
fp8_permute_cregs_64x128:
  swap float2 at acc[8*i+2] and acc[8*i+4]
```

第二小步是把 permuted FP32 P pack 成 PV-A 需要的 FP8 register list。这里不能假设 TileLang `p_fp8.data` 的 raw order 就是 FA3 PV-A order。正确做法是先用 oracle 生成一个静态 mapping：

```text
p_reg[k] = fp8_pack(acc[src0[k]], acc[src1[k]], acc[src2[k]], acc[src3[k]])
```

其中 `src*` 来自 FA3 `convert_layout_acc_Aregs<TiledMmaPV>` 的寄存器解释。生成表之后，运行时只用 PTX：

```cpp
__device__ __forceinline__ uint32_t cvt_4xf32_to_4xe4m3(
    float a, float b, float c, float d) {
  uint16_t lo, hi;
  uint32_t out;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
               : "=h"(lo) : "f"(a), "f"(b));
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
               : "=h"(hi) : "f"(c), "f"(d));
  asm volatile("mov.b32 %0, {%1, %2};"
               : "=r"(out) : "h"(lo), "h"(hi));
  return out;
}

template <int NREG>
__device__ __forceinline__ void p_pack_regs_ptx(float* acc, uint32_t (&a)[NREG]) {
  #pragma unroll
  for (int r = 0; r < NREG; ++r) {
    a[r] = cvt_4xf32_to_4xe4m3(
        acc[P_SRC[r][0]], acc[P_SRC[r][1]],
        acc[P_SRC[r][2]], acc[P_SRC[r][3]]);
  }
}
```

这里 `P_SRC` 是必须先反推/生成的表。不要手猜。生成方法：

1. 用现有 CUTE helper 作为一次性 oracle，dump 每个 thread 的 `tOrP_acc(i)` 对应哪个 `acc_s[j]`。
2. 固化成 `constexpr int P_SRC[NREG][4]`。
3. PTX helper 只读这张表，不再包含 CUTE。
4. byte/probe 验证 PTX `P registers` 被同一个 PV WGMMA 解释后，结果和 oracle 一致。

第三小步才是 inline PTX WGMMA。形态是：

```text
wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3
  D registers: O accumulator
  A registers: 上一步 pack 出来的 P register list
  B descriptor: FA3 sV / SmemLayoutVtMma shared descriptor
```

`K=128` 时发 4 个 `k32` WGMMA group，等价于当前 `for ki in 0..3`。descriptor 和寄存器列表不要手写猜测，第一版直接从当前生成的 CUTE PV helper PTX/SASS 抽取 opcode、operand count、descriptor bit layout，然后替换 A register production 为上面的 PTX pack。

短期实现步骤改成：

1. 保留 TileLang 侧 softmax，softmax 后不要 `T.copy(acc_s, p_shared)`。
2. 写 `_probe_fp8_p_reg_ptx.py`：同一份 `acc_s.data`，比较 CUTE oracle PV 结果和 PTX P-pack + PTX/同等 PV 结果。
3. 用 oracle 生成 `P_SRC`，实现 `p_permute_cregs_ptx + p_pack_regs_ptx`。
4. 接 inline PTX WGMMA，先只做 PV-only probe，要求 `cos_base = 1.0`。
5. 再把 O unpermute 和 `acc_o += acc_delta * v_scale` 也放进同一个 PTX helper，避免 `acc_delta_shared -> T.copy -> TileLang acc_delta`。

## FA3 具体怎么做

FA3 源码里关键常量是：

```cpp
Transpose_V = Is_FP8 && !V_colmajor;
MmaMajorV = GMMA::Major::K;
TmaMajorV = GMMA::Major::MN;
```

对 row-major FP8 V，FA3 的路径是：

```text
GMEM V[N,D]
  -> TMA view as (D,N)
  -> sVt / SmemLayoutVt
  -> producer WG: LDSM.T + __byte_perm(0x6420/0x7531) + STSM
  -> sV / SmemLayoutVtMma
  -> PV WGMMA
  -> permute_output_fp8(tOrO)
```

关键 STSM layout 是：

```cpp
using STSM_value_shape  = Shape<_1, _4, _2, _2>;
using STSM_value_stride = Stride<_0, _1, _4, _8>;
using STSM_divide_shape = Shape<_8, _16>;
```

FA3 源码注释说，另一套 STSM value layout 不会 permute V columns，但有 bank conflict，会稍慢。因此 FA3 选择现在这套 bank-friendly layout，代价是 O columns 也被 permute，所以 epilogue 要 unpermute O columns。

主要参考：

- FA3 paper: https://papers.neurips.cc/paper_files/paper/2024/file/7ede97c3e082c6df10a8d6103a2eebd2-Paper-Conference.pdf
- `/home/ga/TileOPs/.github/runner/vendor/flash-attention/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`
- `/home/ga/TileOPs/.github/runner/vendor/flash-attention/hopper/utils.h`
- `/home/ga/TileOPs/tileops/kernels/attention/_fp8_gqa_helper.h`

## TileOps 已经证明了什么

### 已证明：FA3 `sVt` 入口 layout 可以由 TMA 直接产生

已有 probe：

```text
direct_tma:full     equal_to_pack = True, mismatched_bytes = 0
direct_tma:wgmma    equal_to_pack = True, mismatched_bytes = 0
```

含义：TileLang/TMA 的 standard full-bank 或 wgmma swizzle，可以产生和下面 helper 一样的 raw bytes：

```cpp
fp8_pack_v_128x128_fa3_vt(v_smem, v_vt_smem)
```

所以 `TMA -> FA3 sVt` 这一步是可行的。

### 已证明：TileLang `tl_ldsm_src` 不是 FA3 `sVt`

新的 byte probe：

```text
source:tl_ldsm_src_vs_fa3_svt: equal=False mismatched_bytes=15680
post_transpose:tl_tc_vs_fa3_tc: equal=False mismatched_bytes=14336
```

含义：`fp8_pack_v_128x128_tl_ldsm_src` 是 TileLang-native prepack source layout，不是 FA3 source layout。

### 已证明：FA3-style PV unit 单独可以正确

关键 probe 结果：

```text
fa3_from_acc             cos_base = 1.0000
fa3_pretrans_from_acc    cos_base = 1.0000
fa3_pretrans_to_acc      cos_base ~= -0.009
```

含义：

- P/V/O 都留在 FA3 contract 里时，PV unit 可以正确。
- 把 FA3/CUTE raw O registers 直接塞回 TileLang accumulator layout 会失败。
- `CUTE tOrO -> TileLang acc_delta.data` 是一个真实 correctness boundary，不是简单 raw copy。

更准确地说，现在已经跑通的是 **PV 子图的全 FA3 contract**：

```text
P: acc_s.data -> FA3 PV-A register layout
V: FA3 sV / SmemLayoutVtMma
O: FA3 PV accumulator -> permute_output_fp8
```

这组是自洽且正确的。剩下的问题不是“FA3 contract 是否正确”，而是这组正确 contract 的入口/出口和实现方式还低效：

```text
1. V 入口可能还有 row-major v_smem -> fp8_pack_v_128x128_fa3_vt -> sVt 的 pack bridge
   已证明 TMA direct to FA3 sVt 可行，应该删这个 bridge。

2. P 入口现在从 TileLang acc_s.data 进入 helper
   语义正确，但实现上还依赖 CUTE/oracle 形态；目标是 PTX register reorder + FP8 pack。

3. O 出口现在还可能是 tOrO -> acc_delta_shared -> T.copy -> TileLang acc_delta -> acc_o
   这是最大 correctness/性能 bridge 之一。正确方向是把 O unpermute 和 acc_o 更新也留在同一个 PTX/helper 内。
```

### 当前最快正确 full kernel 还不是完整 FA3 contract

当前 TileOps 最快正确 baseline 仍是：

```text
ws_shared_p / ws_default
```

它是 TileLang-native correctness contract：shared-P + K32 PV staging。它正确，但有 FA3 没有的 movement bridges。

### 2026-05-12 FA3 oracle correctness / benchmark 快照

用 night docker + physical GPU1，对 `use_fa3_pv_extern=True, use_fa3_v_tma_direct=True` 路线重新和官方 FA3 对齐：

```text
S=1024 tileops_fa3_direct_vs_fa3 cos=0.999627 max_abs=0.001133 mean_abs=0.000161
S=4096 tileops_fa3_direct_vs_fa3 cos=0.999565 max_abs=0.000622 mean_abs=0.000091
```

这说明当前 FA3-contract TMA-direct full-kernel path 对官方 FA3 是正确的。注意：手写 dense reference 容易和 FA3 scale/softmax-FP8 口径不一致；后续 FA3 路线 correctness 以官方 FA3 output 为 oracle。

同一环境 quick benchmark：

```text
llama8b-1k:
  tileops_ws_shared_p                 0.185031 ms   92.85 TFLOP/s
  tileops_ws_fa3_pv_extern            0.588714 ms   29.18 TFLOP/s
  tileops_ws_fa3_pv_extern_tma_direct 0.250381 ms   68.61 TFLOP/s
  fa3                                 0.027878 ms  616.24 TFLOP/s

llama8b-4k:
  tileops_ws_shared_p                 3.747701 ms   73.35 TFLOP/s
  tileops_ws_fa3_pv_extern_tma_direct 4.817191 ms   57.06 TFLOP/s
  fa3                                 0.536586 ms  512.27 TFLOP/s
```

解释：

1. `TMA direct -> FA3 sVt` 是明确有效的：1k 下从 `0.588714 ms` 降到 `0.250381 ms`，说明删掉 `v_smem -> fp8_pack_v_128x128_fa3_vt -> sVt` 这座 V 入口桥收益很大。
2. 但 FA3-contract path 仍慢于当前 `ws_shared_p`，说明剩下的大头不是 V pack bridge，而是 PV helper 实现和 O 出口 bridge。
3. 下一步优化优先级应是：PTX 化 P register reorder/pack + PV WGMMA，随后把 `tOrO -> acc_delta_shared -> T.copy -> acc_delta` 出口桥并入同一个 helper。

为了不改坏现有路径，已经新增独立入口：

```text
GQAFwdFP8Fa3ContractKernel
```

第一版行为等价于：

```text
GQAFwdFP8WsPersistentKernel(
  config={
    "use_fa3_pv_extern": True,
    "use_fa3_v_tma_direct": True,
  }
)
```

但后续 PTX 化 P/PV/O 只接到这个新 kernel 上，不改默认 `ws_shared_p`，也不直接改旧的 `GQAFwdFP8WsPersistentKernel` 行为。

验证：

```text
pytest -q tests/ops/attention/test_gqa_fp8.py -k "ws_fa3_contract_kernel" -s
  -> 1 passed

GQAFwdFP8Fa3ContractKernel vs official FA3, S=1024:
  cos = 0.999627
  max_abs = 0.001133

quick benchmark, llama8b-1k:
  tileops_ws_fa3_contract 0.285677 ms 60.14 TFLOP/s
```

## 两条候选路线

### 路线 A：FA3 闭环 contract

这是最符合 FA3 论文/源码的路线。

```text
Producer:
  TMA V[N,D] with logical view (D,N)
    -> sVt / FA3 SmemLayoutVt
  PTX/CuTe-equivalent LDSM.T + PRMT + STSM
    -> sV / FA3 SmemLayoutVtMma

Consumer:
  QK acc S/P
    -> permute_Cregs_fp8
    -> convert_layout_acc_Aregs<TiledMmaPV>
    -> FP8 P operand-A regs
  PV WGMMA(P, sV)
    -> tOrO with permuted columns
    -> permute_output_fp8
    -> O update/store
```

这条路线的关键要求是：**不要在 PV 中途回到 TileLang fragment layout**。除非我们显式实现了 raw fragment reorder，否则中间跨回 TileLang 就会遇到 `fa3_pretrans_to_acc` 那类错误。

### 路线 B：TileLang-native 当前最快 contract

这是从当前最快正确路径出发。

```text
Producer:
  V[N,D]
    -> current TileLang V movement / K32 staging

Consumer:
  QK acc
    -> shared-P 或当前 register-P helper
  PV WGMMA in TileLang K32/full-PV descriptor layout
    -> TileLang acc_delta
    -> acc_o update
```

如果在这条路线上用 PTX 优化 V，那么 PTX helper 必须输出 **TileLang B operand 期望的 exact physical layout**。不能直接把 FA3 `sV` 拿来接 TileLang PV；除非 P 和 O 也一起改成 FA3 contract。

## PTX 应该怎么写

PTX 目标不应该再是：

```text
让 TMA 直接复刻 tl_ldsm_src
```

这条路已经试过：对完整 `128x128` FP8 tile，合法 standard tensor-map 参数表达不出 `tl_ldsm_src` 的自定义 `perm_n/perm_d`。

PTX 应该在两种目标里选一个：

1. **FA3-closed helper**：实现 `sVt -> sV`，并且 P/O 也留在 FA3 PV contract 里。
2. **TileLang-native helper**：从硬件支持的 source layout 出发，写出 TileLang PV-B 需要的 exact physical layout。

底层 PTX atom 是类似的。

### shared pointer 转 32-bit shared address

```cpp
__device__ __forceinline__ uint32_t smem_u32addr(void const* ptr) {
  uint32_t addr;
  asm volatile(
      "{ .reg .u64 u64addr;\n"
      "  cvta.to.shared.u64 u64addr, %1;\n"
      "  cvt.u32.u64 %0, u64addr;\n"
      "}\n"
      : "=r"(addr)
      : "l"(ptr));
  return addr;
}
```

### LDSM.T

```cpp
asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
    "{%0, %1, %2, %3}, [%4];\n"
    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
    : "r"(src_addr));
```

### packed FP8 byte reorder

```cpp
uint32_t x0, x1;
asm volatile("prmt.b32 %0, %1, %2, 0x6420;\n" : "=r"(x0) : "r"(upper), "r"(lower));
asm volatile("prmt.b32 %0, %1, %2, 0x7531;\n" : "=r"(x1) : "r"(upper), "r"(lower));
```

### STSM

```cpp
asm volatile(
    "stmatrix.sync.aligned.m8n8.x4.shared.b16 "
    "[%0], {%1, %2, %3, %4};\n"
    :
    : "r"(dst_addr), "r"(x0), "r"(x1), "r"(x2), "r"(x3));
```

具体 PTX 语法可能要按当前 CUDA/PTX 版本微调，但目标指令就是这些。

## FA3-equivalent lane schedule

对 `D=128, N=128`，FA3 使用：

```text
LDSM:
  thread_shape  = Shape<32,4,1,1>
  thread_stride = Stride<4,1,0,0>
  value_shape   = Shape<2,2,1,4>
  value_stride  = Stride<1,2,16,4>
  divide_shape  = Shape<64,8>

STSM:
  thread_shape  = Shape<8,4,4,1>
  thread_stride = Stride<4,1,32,0>
  value_shape   = Shape<1,4,2,2>
  value_stride  = Stride<0,1,4,8>
  divide_shape  = Shape<8,16>
```

PTX helper 必须复刻这个 lane schedule。形态应该是：

```cpp
template <typename FP8>
__device__ void fa3_v_transpose_ptx(FP8* sVt, FP8* sV) {
  int tid = threadIdx.x & 127;
  int lane = tid & 31;
  int warp = tid >> 5;

  // 遍历和 FA3 一样的 logical tiles:
  //   source flat_divide(sVt, Shape<64,8>)
  //   dest   flat_divide(sV,  Shape<8,16>)
  //
  // 每个 tile 中，每个 lane 的 source/dest row address
  // 必须由上面的 FA3 tiled-copy layout 推导出来。
  for (int tile = 0; tile < NUM_TRANSPOSE_TILES; tile += 2) {
    uint32_t src0 = smem_u32addr(sVt + fa3_svt_offset_for_lane(tid, tile + 0));
    uint32_t dst0 = smem_u32addr(sV  + fa3_sv_offset_for_lane(tid, tile + 0));

    uint32_t r0, r1, r2, r3;
    ldmatrix_x4_trans_b16(src0, r0, r1, r2, r3);

    uint32_t x0 = prmt(r0, r1, 0x6420);
    uint32_t x1 = prmt(r0, r1, 0x7531);
    uint32_t x2 = prmt(r2, r3, 0x6420);
    uint32_t x3 = prmt(r2, r3, 0x7531);

    stmatrix_x4_b16(dst0, x0, x1, x2, x3);
  }
}
```

这里最重要的是：

```text
fa3_svt_offset_for_lane
fa3_sv_offset_for_lane
```

这两个不能随便写成 row-major 公式。它们必须编码 FA3 的 LDSM/STSM tiled-copy layout。

实际建议：

1. 先用 CuTe 生成 constexpr lane offset table，把表贴进 PTX helper。
2. PTX helper 用这些 table 做 address 计算。
3. byte-for-byte 对齐当前 `fp8_transpose_v_128x128_fa3_src_ldsm_stsm`。
4. 等 PTX helper 过了，再考虑手推公式替代表。

这样比直接手推所有 lane mapping 更稳。

## 验收标准

### 如果走 FA3-closed PTX

```text
1. TMA/full-bank -> sVt 等于 fp8_pack_v_128x128_fa3_vt
2. PTX sVt->sV 等于 fp8_transpose_v_128x128_fa3_src_ldsm_stsm
3. PV-only fa3_pretrans_from_acc 保持 cos_base=1.0
4. full kernel 在 1k/4k correctness 通过
5. 不再引入 acc_delta_shared round-trip 或 O raw-layout mismatch
```

### 如果走 TileLang-native PTX

```text
1. PTX V movement output 等于当前 TileLang B operand bytes
2. tl_full / K32 default correctness 不变
3. movement microbench 接近裸 LDSM/STSM 成本，而不是 pack+LDSM 成本
4. full kernel 至少追平/超过 ws_shared_p，否则只是 probe，不是主线优化
```

## 当前建议

不要继续把主线放在：

```text
PTX/TMA 直接复刻 tl_ldsm_src
```

这条路已经暴露出 tensor-map expressiveness 问题。

更 coherent 的 PTX 路线是：

```text
TMA direct to FA3 sVt
  -> PTX LDSM.T/PRMT/STSM to FA3 sV
  -> P 和 O 也留在同一个 FA3 PV helper
  -> helper 内部做 O unpermute
  -> 再回到 TileLang/global
```

因此第一个真正有用的 PTX 交付物不是 full kernel 重写，而是 byte-exact 替换：

```cpp
fp8_transpose_v_128x128_fa3_src_ldsm_stsm
```

等这个 PTX helper 和 CuTe helper 完全一致后，再把 FA3 P/O bridge 一起搬进同一个 extern helper，避免在 PV 中途跨 TileLang 和 FA3 raw fragment layout。

## 2026-05-12 进展：P operand-A no-CUTE pack

新增独立 probe：

```text
_probe_fp8_fa3_p_areg_mapping.py
_probe_fp8_fa3_p_pack_no_cute.py
_probe_fp8_fa3_pv_ptx_unit.py
```

结论一：FA3 的 `QK-C accumulator -> PV operand-A` 源顺序已经确认。128 个线程完全一致；按 `ki=0..3` 切片后也是连续 16 个 FP8 标量一组：

```text
P_AREG_SRC_BY_KI =
  ki0: 00 01 04 05 02 03 06 07 08 09 12 13 10 11 14 15
  ki1: 16 17 20 21 18 19 22 23 24 25 28 29 26 27 30 31
  ki2: 32 33 36 37 34 35 38 39 40 41 44 45 42 43 46 47
  ki3: 48 49 52 53 50 51 54 55 56 57 60 61 58 59 62 63
```

也就是每 8 个 FP32 accumulator 里：

```text
before: 0 1 2 3 4 5 6 7
after : 0 1 4 5 2 3 6 7
```

结论二：不含 CUTE 的 P pack 已经 byte-exact 对齐 CUTE oracle。

```text
external_tensor: equal=True mismatch=0
cute_rmem      : equal=True mismatch=0
```

这里 `cute_rmem` 比 `external_tensor` 更关键：它 dump 的是 CUTE `make_tensor_like<Element>(p_acc)` 后真正进入 PV WGMMA 的 register-memory bytes。它和 no-CUTE `uint32_t p_regs[16]` 完全一致，所以现在可以把 P pack 从“疑点列表”里拿掉。

当前 no-CUTE P pack 公式可以写成：

```cpp
src(i):
  base = i & ~7
  w    = i & 7
  map  = [0, 1, 4, 5, 2, 3, 6, 7][w]
  return base + map

for reg in 0..15:
  p_regs[reg] = pack_fp8(acc_s[src(4*reg + 0)],
                         acc_s[src(4*reg + 1)],
                         acc_s[src(4*reg + 2)],
                         acc_s[src(4*reg + 3)])
```

结论三：第一版 `wgmma_rs(P regs, TileLang full-swizzle desc_B)` 仍然不等于 FA3 CUTE PV。

```text
ptx_perm    max_abs=1.115173 mean_abs=0.234881 cos=0.015197
ptx_no_perm max_abs=1.209717 mean_abs=0.233368 cos=0.024962
```

这说明错误不在 P pack。更可能的边界是：

```text
1. FA3 sV 的 CUTE GMMA descriptor 不等于 TileLang initialize_wgmma_descriptor<1,1,64>
2. 或者 C raw accumulator layout / O unpermute 仍不能直接接 TileLang acc_delta.data
```

已有 `_probe_fp8_atom_pv_descriptor.py` 也支持这个判断：TileLang atom/full-bank descriptor 接 FA3 V movement 时 `cos_ref ~= 0.04`，和正确 FA3 PV contract 不同。

下一步应先 dump / 复刻 FA3 CUTE `tOrV(_,_,ki)` 的 B descriptor，而不是继续改 P。P 的 no-CUTE register pack 已经足够硬。

## 2026-05-12 进展：V descriptor 已对齐 FA3

新增 probe：

```text
_probe_fp8_fa3_v_desc.py
```

它在同一个 kernel 内做：

```text
V row-major
  -> fp8_pack_v_128x128_fa3_vt
  -> fp8_transpose_v_128x128_fa3_src_ldsm_stsm
  -> dump FA3 CUTE tOrV(_,_,ki) descriptor
  -> dump TileLang initialize_wgmma_descriptor<1,1,64> descriptor
```

结果四个 `ki` 完全一致：

```text
ki=0
  fa3 start=64 lead=1 stride=64 base=0 layout=1 size=1
  tl   start=64 lead=1 stride=64 base=0 layout=1 size=1
ki=1
  fa3 start=66 lead=1 stride=64 base=0 layout=1 size=1
  tl   start=66 lead=1 stride=64 base=0 layout=1 size=1
ki=2
  fa3 start=68 lead=1 stride=64 base=0 layout=1 size=1
  tl   start=68 lead=1 stride=64 base=0 layout=1 size=1
ki=3
  fa3 start=70 lead=1 stride=64 base=0 layout=1 size=1
  tl   start=70 lead=1 stride=64 base=0 layout=1 size=1
```

所以，前面 `ptx_perm / ptx_no_perm` 错，不是因为 V descriptor 没对齐。现在 FA3 PV 的三个入口里：

```text
P operand-A register bytes: 已对齐
V operand-B shared bytes: 由 FA3 helper 产生，已在正确路径里验证
V operand-B GMMA descriptor: 已对齐
```

剩下的错点应转向：

```text
PV WGMMA output C raw register layout
  -> TileLang acc_delta.data
  -> O unpermute / row-col store
```

换句话说，下一步不是继续改 V，而是把 O/C raw layout 的 bridge 拆掉或复刻 FA3 row-col store。

## 2026-05-12 进展：PTX PV + FA3 O store 已接入新 kernel

继续拆 O/C 后确认：

```text
_probe_fp8_fa3_p_pack_no_cute.py
  cute_rmem_by_ki:
    ki0/ki1/ki2/ki3 mismatch = 0

_probe_fp8_fa3_pv_raw_o.py
  CUTE raw tOrO vs PTX raw O:
    max_abs = 0
    mean_abs = 0
    cos = 1.0
```

所以前面 `ptx_perm / ptx_no_perm` 错误的真正原因不是 P/V/WGMMA，而是：

```text
PTX/CUTE raw O registers
  -> 直接塞进 TileLang acc_delta.data
```

这一步 layout contract 不匹配。解决方式是让 PTX PV helper 自己完成 FA3 的 O 解释和 row-col shared store：

```text
acc_s.data
  -> no-CUTE P regs
  -> PTX wgmma_rs(P, FA3 sV descriptor)
  -> raw O registers
  -> FA3 permute_output_fp8
  -> FA3 row-col store to acc_delta_shared
  -> TileLang T.copy(acc_delta_shared, acc_delta)
```

新增 helper：

```text
fp8_pv_ptx_unit_from_acc_pretransposed_to_smem_64x128x128
```

PV-only probe：

```text
ptx_perm    max_abs=1.115173 mean_abs=0.234881 cos=0.015197
ptx_no_perm max_abs=1.209717 mean_abs=0.233368 cos=0.024962
ptx_smem    max_abs=0        mean_abs=0        cos=0.99999994
```

随后把它接入 `GQAFwdFP8Fa3ContractKernel`，通过开关启用：

```text
config = {"use_ptx_pv": True}
```

验证：

```text
pytest -q tests/ops/attention/test_gqa_fp8.py -k "ws_fa3_contract_kernel" -s
  -> 2 passed

S=1024 ptx_pv_vs_fa3:
  cos      = 0.9996273517608643
  max_abs  = 0.001132965087890625
  mean_abs = 0.0001609630126040429
```

quick benchmark, H200 GPU1, `llama8b-1k`, FA3 scale:

```text
tileops_ws_fa3_contract        0.267309 ms  64.27 TFLOP/s
tileops_ws_fa3_contract_ptx_pv 0.242386 ms  70.88 TFLOP/s
```

这一步已经去掉了 PV 内部的 CUTE WGMMA 实现路径，保留的是 FA3 row-col shared-store bridge。下一步才是继续消掉：

```text
acc_delta_shared -> T.copy(acc_delta_shared, acc_delta)
```

也就是把 `acc_o += O_delta * v_scale` 一并放进 helper，避免 raw O 再跨回 TileLang fragment layout。

## 2026-05-12 进展：新 kernel 砍掉 `acc_delta_shared -> acc_delta` 大 bridge

新开入口：

```text
GQAFwdFP8Fa3ContractPtxAccKernel
```

它在 FA3 contract 内继续把 PV 后半段收进 PTX/helper：

```text
acc_s.data
  -> no-CUTE P regs
  -> PTX wgmma_rs(P, FA3 sV descriptor)
  -> raw O delta registers
  -> FA3 permute_output_fp8 / row-col view
  -> acc_o raw registers: acc_o = acc_o * ss + O_delta * v_scale
  -> final FA3 row-col store
  -> output
```

也就是说，原来的大 bridge：

```text
raw O delta
  -> acc_delta_shared[64,128] fp32
  -> T.copy(acc_delta_shared, TileLang acc_delta)
  -> TileLang acc_o += acc_delta * v_scale
```

在这个新 kernel 里已经不走了。

这次踩到的关键点：

```text
ss.data / ls.data 不能按 ss[row] / ls[row] 直接解释
```

`ss` 和 `ls` 是 TileLang fragment vector，它们的 raw fragment layout 不是 row-major vector。直接在 PTX/CUTE helper 里用 `ss[row]` / `ls[row]` 会导致只有 row 0/1 正常，其余 row 出现 inf。修正方式是保留一个很小的 row-major shared bridge：

```text
T.copy(ss, ss_shared[64])
T.copy(ls, ls_shared[64])
helper 读取 ss_shared[row] / ls_shared[row]
```

这只搬 64 个 float，不是原来的 64x128 `acc_delta_shared` 大 round-trip。

验证：

```text
pytest -q tests/ops/attention/test_gqa_fp8.py -k "ws_fa3_contract_kernel_ptx_acc" -s
  -> 1 passed

pytest -q tests/ops/attention/test_gqa_fp8.py -k "ws_fa3_contract_kernel" -s
  -> 3 passed

pytest -q tests/ops/attention/test_gqa_fp8.py -s
  -> 12 passed
```

S=1024 对官方 FA3：

```text
finite   = True
cos      = 0.9996273517608643
max_abs  = 0.001132965087890625
mean_abs = 0.0001609630126040429
lse finite = True
```

H200 GPU1 quick benchmark, `llama8b-1k`, FA3 scale：

```text
tileops_ws_fa3_contract         0.286659 ms   59.93 TFLOP/s
tileops_ws_fa3_contract_ptx_pv  0.242180 ms   70.94 TFLOP/s
tileops_ws_fa3_contract_ptx_acc 0.056032 ms  306.61 TFLOP/s
fa3                             0.027859 ms  616.67 TFLOP/s
```

结论：

1. `acc_delta_shared -> T.copy -> TileLang acc_delta` 确实是大头，砍掉后速度从约 0.242 ms 到 0.056 ms。
2. 新路径 correctness 与官方 FA3 对齐，误差和 ptx_pv 版本一致。
3. 现在还剩一个出口问题：`o_shared` 临时用 float32，最终 `T.copy(o_shared, output)` 因 dtype 不同 fallback 到 normal copy。下一步可以把 final store 也放进 helper，直接从 FA3 raw `acc_o` 写 fp16/bf16 output，去掉这个 warning 和最后一段 shared output staging。

## 2026-05-12 收工状态和明日计划

### 当前最快正确版本

截至今天收工，最快的正确 FA3-contract FP8 路线是：

```text
GQAFwdFP8Fa3ContractPtxAccKernel
```

当前 contract：

```text
Q/K: TileLang 默认 TMA/shared layout，QK WGMMA 正确
P:   从 QK acc_s.data 用 no-CUTE register reorder/pack 成 FA3 PV operand A
V:   TMA-direct 到 FA3 sVt contract，再经 FA3 LDSM.T + PRMT + STSM 到 PV operand B
PV:  PTX wgmma_rs，raw O 与 CUTE tOrO byte-exact
O:   helper 内按 FA3 raw accumulator layout 做 online accumulation
ss/ls: 临时通过 64-float row-major shared bridge 进入 helper
out: FA3 raw acc_o -> float32 o_shared -> TileLang T.copy 到 fp16/bf16 output
```

已经确认不再经过：

```text
raw O delta -> acc_delta_shared[64,128] -> T.copy -> TileLang acc_delta
```

### 正确性基线

必须继续用较长序列验证，不能只看 S=128：

```text
pytest -q tests/ops/attention/test_gqa_fp8.py -s
  -> 12 passed

S=1024 vs official FA3:
  finite   = True
  cos      = 0.9996273517608643
  max_abs  = 0.001132965087890625
  mean_abs = 0.0001609630126040429
  lse finite = True
```

### 性能基线

H200 GPU1，nightly docker，`llama8b-1k`，FA3 scale：

```text
tileops_ws_fa3_contract         0.286659 ms   59.93 TFLOP/s
tileops_ws_fa3_contract_ptx_pv  0.242180 ms   70.94 TFLOP/s
tileops_ws_fa3_contract_ptx_acc 0.056032 ms  306.61 TFLOP/s
fa3                             0.027859 ms  616.67 TFLOP/s
```

当前离官方 FA3 还有约 2x。今天已经证明 `acc_delta_shared` 大 bridge 是主要问题之一，但不是最后一个问题。

### 明天优先级

| 优先级 | 项目 | 当前问题 | 预期做法 | 验收 |
|---|---|---|---|---|
| P0 | final output store | 现在 `acc_o -> float32 o_shared -> T.copy output`，有 dtype mismatch fallback warning | 写 helper 直接从 FA3 raw `acc_o` cast/store 到 fp16/bf16 output，删除 `o_shared_1/2` 的 ptx_acc 路径 | S=1024 vs FA3 数值不变；`test_gqa_fp8.py` 全过；bench 是否低于 0.056 ms |
| P1 | `ss/ls` 小 bridge | 现在还要 `T.copy(ss/ls, shared[64])` | 若值得做，再解 TileLang vector fragment raw layout，或让 helper 接收更稳定的 row-major vector contract | correctness 不变；确认收益是否可测 |
| P2 | 剩余 2x gap 定位 | ptx_acc 仍慢于 FA3 | 用 benchmark/NCU 分段看 final store、producer V movement、softmax、barrier、register pressure | 列出下一批最高收益 bridge/latency source |
| P3 | 清理实验开关 | 当前新 kernel 是 experimental entry | 保持旧入口不动；只在 ptx_acc 稳定后考虑默认策略或命名整理 | API/export/test/bench 清晰 |

### 明天第一刀：direct output store

要新增的 helper 形态大概是：

```text
fp8_fa3_raw_acc_store_output_64x128(
    acc_o_raw,
    ls_shared,
    flags,
    output_ptr,
    output_stride...
)
```

它应复用今天已经正确的 FA3 row/col mapping：

```text
tAccO raw
  -> convert_layout_acc_rowcol
  -> FA3 column unpermute mapping
  -> divide by ls[row]
  -> cast to output dtype
  -> global output[row, col]
```

注意事项：

1. 不要重新引入 `o_shared[64,128]`。
2. output dtype 至少覆盖 fp16/bf16。
3. 保持 `GQAFwdFP8Fa3ContractPtxAccKernel` 独立，旧 `ptx_pv` 和 CUTE extern contract 不动。
4. 先跑 S=128 smoke，再跑 S=1024 vs FA3，最后跑 quick bench。

## 2026-05-13 direct output store 实验

今天按昨天 P0 计划新建了一个 kernel：

```text
GQAFwdFP8Fa3ContractPtxAccDirectStoreKernel
```

它基于当前最快正确的 `GQAFwdFP8Fa3ContractPtxAccKernel`，只改最后 output store：

```text
旧 ptx_acc:
  FA3 raw acc_o
    -> float32 o_shared[64,128]
    -> T.copy(o_shared, output fp16/bf16)

新 direct_store:
  FA3 raw acc_o
    -> helper 内 row/col view
    -> divide by ls[row]
    -> cast fp16/bf16
    -> global output
```

新增 helper：

```text
fp8_fa3_raw_acc_store_global_64x128(
    acc_o_raw,
    ls_shared,
    flags,
    output_ptr,
    output_row_stride
)
```

它复用 FA3-contract 已经验证正确的 accumulator 解释方式：

```text
tAccO raw
  -> convert_layout_acc_rowcol
  -> FA3 output column unpermute
  -> O[row, col] / ls[row]
  -> cast/store global
```

### 正确性

nightly docker，H200 GPU1：

```text
pytest -q tests/ops/attention/test_gqa_fp8.py -k "ws_fa3_contract_kernel_ptx_acc" -s
  -> 2 passed, 11 deselected

pytest -q tests/ops/attention/test_gqa_fp8.py -k "ws_fa3_contract_kernel" -s
  -> 4 passed, 9 deselected

pytest -q tests/ops/attention/test_gqa_fp8.py -s
  -> 13 passed
```

S=1024 vs official FA3：

```text
finite   = True
cos      = 0.9996273517608643
max_abs  = 0.001132965087890625
mean_abs = 0.0001609630126040429
lse finite = True
```

也就是说 direct output store 没有改坏 FA3-contract correctness。

### 性能

H200 GPU1，nightly docker，`llama8b-1k`，FA3 scale：

```text
tileops_ws_fa3_contract_ptx_acc              0.056074 ms  306.38 TFLOP/s
tileops_ws_fa3_contract_ptx_acc_direct_store 0.071305 ms  240.93 TFLOP/s
fa3                                          0.027894 ms  615.89 TFLOP/s
```

结论：direct global output store 正确，也去掉了 `o_shared` 和 dtype mismatch fallback warning，但性能反而更慢。当前判断是 helper 内按 FA3 raw accumulator 做 global store 的写出模式不够 coalesced；原来的 `float32 o_shared -> T.copy output` 虽然看起来冗余，但 TileLang 的 final copy 更像一个优化过的 shared-to-global store path。

所以当前最快正确版本仍然是：

```text
GQAFwdFP8Fa3ContractPtxAccKernel
```

而不是 direct-store 版本。

### 重要踩坑

direct store 里不能用：

```text
output.access_ptr("w", offset=dynamic_expr)
```

这个写法在测试里会导致输入侧 `q_fp8` 被污染，reference FA3 结果出现 NaN。正确写法是从具体 tensor element 取地址：

```text
T.address_of(output[tile_b, row_base, tile_h, 0])
```

然后把 row stride 作为显式参数传给 helper：

```text
output_row_stride = heads * dim
```

另一个试过但没有保留的方向是 typed shared store：

```text
FA3 raw acc_o -> fp16/bf16 shared -> T.copy output
```

它能编译，但 correctness 失败，`cos ~= 0.0549`。原因更像是 same-dtype `T.copy` 对 shared/global 的 layout contract 和我们手写 store 到 shared 的 physical layout 不一致。因此这个实验已经撤掉，没有留下入口。

### 下一步

output store 方向短期先收住。direct store 证明了 correctness contract 没问题，但性能不是当前最高收益点。

下一步应转向 P2：系统定位 `ptx_acc` 到官方 FA3 剩余约 2x gap：

```text
ptx_acc 当前:
  0.056 ms

official FA3:
  0.028 ms
```

优先看：

1. producer 侧 V movement / TMA / LDSM.T + PRMT + STSM 是否仍有额外等待。
2. consumer 侧 PTX PV issue/latency 是否和 FA3 对齐。
3. softmax、barrier、pipeline overlap 和 register pressure。
4. `ss/ls` 64-float 小 bridge 是否可测，而不是凭直觉拆。

## 2026-05-13 output store A 实验：FA3-style epilogue store

今天继续 output store 方向，先按方案 A 新建了一个 kernel：

```text
GQAFwdFP8Fa3ContractPtxAccFa3EpilogueStoreKernel
```

它不改已有最快正确 kernel，只在 `ptx_acc` 的最终输出路径上替换：

```text
旧 ptx_acc:
  FA3 raw acc_o
    -> fp32 o_shared[64,128]
    -> TileLang T.copy(o_shared, output fp16/bf16)

A / fa3_epilogue_store:
  FA3 raw acc_o
    -> cast to fp16/bf16 CUTE accumulator fragment
    -> FA3 permute_output_fp8_Vcolmajor
    -> STSM 写入 swizzled fp16/bf16 smem_o
    -> 每线程 16B，按 row/col coalesced store 到 global output
```

关键点是：只把 dtype 改成 fp16/bf16 shared 并不够。FA3 的 V 路径带有 column permutation，所以 output fragment 写 shared 之前必须补：

```text
permute_output_fp8_Vcolmajor(tOut)
```

第一次没做这一步时，kernel 能编译但 correctness 失败：

```text
S=128 smoke cos ~= 0.1876
```

加上 `Vcolmajor` permutation 后，smoke correctness 通过。

### A 的实现形态

新增 helper：

```text
fp8_fa3_raw_acc_store_smem_cute_64x128(
    acc_o_raw,
    ls_shared,
    flags,
    o_shared_fp16_or_bf16
)

fp8_fa3_o_smem_store_global_cute_64x128(
    o_shared_fp16_or_bf16,
    output_ptr,
    output_row_stride
)
```

第一段复用 FA3/CUTE epilogue 的核心 contract：

```text
partition_fragment_C(TiledMmaPV, Shape<64,128>)
  -> convert_layout_acc_rowcol
  -> divide by ls[row]
  -> cast OutT
  -> permute_output_fp8_Vcolmajor
  -> make_tiled_copy_C(SM90_U32x4_STSM_N)
  -> swizzled smem_o
```

第二段暂时不用 CUTE dynamic gmem tensor，因为 TileLang extern 里 `output_row_stride` 是 runtime int，直接构造 CUTE gmem stride 会触发模板推导问题。当前用手写 coalesced pattern：

```text
128 epilogue threads
每线程每次写 8 个 fp16/bf16 = 16B
每 row 8 个线程覆盖 64 columns
两个 col_block 覆盖 128 columns
row += 16 覆盖 64 rows
```

也就是：

```text
row = tid / 8 + {0,16,32,48}
col = (tid % 8) * 8 + {0,64}
store output[row, col:col+8]
```

shared 读地址用 FA3 smem layout 反查：

```text
SmemLayoutO = tile_to_shape(
  composition(Swizzle<3,3,3>, Layout<Shape<8,64>, Stride<64,1>>),
  Shape<64,128>)
```

### 正确性

nightly docker，H200 GPU1：

```text
pytest -q tests/ops/attention/test_gqa_fp8.py -k "fa3_epilogue_store" -s
  -> 1 passed, 13 deselected

pytest -q tests/ops/attention/test_gqa_fp8.py -k "ptx_acc" -s
  -> 3 passed, 11 deselected
```

S=1024 vs official FA3：

```text
finite   = True
cos      = 0.9996273517608643
max_abs  = 0.001132965087890625
mean_abs = 0.0001609630126040429
lse finite = True
```

也就是说 A 没有改变数值 contract，和旧 `ptx_acc` 对 FA3 的误差一致。

### 性能

H200 GPU1，nightly docker，`llama8b-1k`，FA3 scale：

```text
tileops_ws_fa3_contract_ptx_acc                    0.056054 ms  306.49 TFLOP/s
tileops_ws_fa3_contract_ptx_acc_fa3_epilogue_store 0.054662 ms  314.29 TFLOP/s
tileops_ws_fa3_contract_ptx_acc_direct_store       0.071696 ms  239.62 TFLOP/s
fa3                                                0.027885 ms  616.10 TFLOP/s
```

结论：A 有提升，但不大，约 2.5%。它验证了我们的直觉：

```text
不是直接按 raw accumulator owner lane 写 global。
而是先做 FA3 output fragment permutation / STSM staging，
再按 coalesced store pattern 写 global。
```

当前最快正确版本更新为：

```text
GQAFwdFP8Fa3ContractPtxAccFa3EpilogueStoreKernel
```

### 下一步 B：register exchange direct store

B 的目标是进一步去掉 `smem_o`：

```text
FA3 raw acc_o
  -> cast fp16/bf16
  -> permute_output_fp8_Vcolmajor 等价 register movement
  -> warp-level value exchange
  -> 直接 16B coalesced global store
```

但 B 不能复用旧 direct store。旧 direct store 是“owner lane 各写各的 global 元素”，正确但不 coalesced，所以慢。B 必须让负责 coalesced output segment 的 lane 拿到对应 8 个值。

需要先确认两件事：

1. 对每个目标 `(row, col:col+8)`，8 个值的 owner lane 是否都在同一个 warp 内。
2. `permute_output_fp8_Vcolmajor` 后的 owner mapping 是否可以用少量 `shfl.sync` 表达，而不是退化成大量泛化 gather。

如果 owner mapping 跨 warp，B 就需要 shared 或 warpgroup-level staging，本质上会回到 A。所以下一步先写一个 mapping/probe，而不是马上写性能版 B。

## 2026-05-13 FA3 FP8 hdim128 的 mainloop tile size

重新对了一遍 FA3 hopper 代码，结论是：**FA3 FP8 hdim128 非 causal 路径的 mainloop tile 确实比我们当前 TileOps FP8 路径更大，主要大在 N 维。**

FA3 的 tile size 来自：

```text
tile_size_fwd_sm90(
  headdim=128,
  headdim_v=128,
  is_causal=false,
  is_local=false,
  element_size=1,   // FP8
  v_colmajor=false,
  paged_kv_non_TMA=false,
  softcap=false
)
```

对应 `tile_size.h` 的 FP8 分支：

```text
headdim <= 128:
  return {128, 224, true, true}
```

所以 FA3 的 CTA/mainloop logical tile 是：

```text
QK:
  Q [M=128, K=128]
  K [N=224, K=128]
  S/P [M=128, N=224]

PV:
  P [M=128, K=224]
  V [K=224, D=128]
  O [M=128, D=128]
```

FA3 代码里随后构造：

```text
TileShape_MNK    = Shape<kBlockM, kBlockN, kHeadDim>
TileShape_MNK_PV = Shape<kBlockM, kHeadDimV, kBlockN>
```

因此 hdim128 FP8 非 causal 时就是：

```text
TileShape_MNK    = Shape<128, 224, 128>
TileShape_MNK_PV = Shape<128, 128, 224>
```

另外，因为 `MmaPV_is_RS=true`、`IntraWGOverlap=true`，且：

```text
AtomLayoutQK = Layout<Shape<kBlockM / 64, 1, 1>>
```

在 `kBlockM=128` 时，FA3 的 tiled MMA 在 M 维上有两个 64-row atom，也就是两个 MMA warpgroups 合作覆盖一个 128-row tile。

我们当前 TileOps FP8 WS 路径：

```text
block_m = 128
half_m  = 64
block_n = 128
```

consumer 实际上拆成两个 half：

```text
QK/PV half1:
  M=64, N=128, D=128

QK/PV half2:
  M=64, N=128, D=128
```

helper 里的 FA3-contract PV tile 也固定是：

```text
TileShapePV = Shape<64, 128, 128>
```

所以差异不是“单条 WGMMA instruction 更大”这么简单，而是 CTA/mainloop contract 不同：

```text
FA3:
  logical CTA tile M=128, N=224
  两个 MMA WG 合作一个 tile

TileOps 当前:
  logical CTA tile M=128, N=128
  但 consumer 分成两个 M=64 的独立半 tile
```

这会影响性能判断：

1. FA3 每个 K/V tile 覆盖 `N=224`，比我们的 `N=128` 更长，循环次数更少。
2. FA3 在一个 tile 内对两个 64-row atom 统一做 QK/PV/softmax pipeline；我们是两个 half consumer，各自维护 `acc_s/acc_o/ss/ls`。
3. 之前“先别考虑 block”的判断对 swizzle 排查是合理的，因为 FP16 路径已和 FA3 接近；但 FP8 当前 FA3-contract kernel 的绝对性能 gap，已经不能完全排除 block_n=128 vs 224 的影响。

后续如果要验证 block size 的影响，最直接实验是新开一个 kernel 做：

```text
block_n = 224
half_m = 64
V smem / TMA / transpose / PV helper 全部支持 N=224
```

但这不是小改：现在很多 helper 和 shared layout 写死 `128x128`，尤其 V transpose、P fragment、PV PTX unit、output mapping 都默认 `N=128`。因此短期还是先把 FA3-contract 的 movement/store/pipeline 做干净；block_n=224 可以作为独立 P2/P3 实验。

## 2026-05-13 TileOps 尝试 block_n=224 的可行性检查

今天直接用 nightly docker / H200 GPU1 做了两个 probe，结论是：**TileLang/WGMMA 不是完全不支持 224，但当前 TileOps FP8 kernel 不能直接把 `block_n` 改成 224。**

### Probe 1：原生 TileLang WS 路径

配置：

```text
GQAFwdFP8WsPersistentKernel(
  seq_len=896,
  heads=2,
  heads_kv=1,
  dim=128,
  config={"block_n": 224}
)
```

结果：

```text
TileLang compile: OK
runtime launch: Failed to set the allowed dynamic shared memory size to 260096
```

这说明 TileLang 前端 / WGMMA lowering 至少能接受 `N=224` 这种 shape；但我们当前 WS kernel 的 shared memory 分配太大，launch 不起来。

主要原因是当前 kernel 有很多按 `block_n` 放大的双 buffer：

```text
k_smem_0/1      [block_n, 128]
v_smem_0/1      [block_n, 128]      // 非 FA3 direct path
v_tc_smem_0/1   [128, block_n]
p_shared_1/2    [64, block_n]       // shared-P path
q_shared_1/2    [64, 128]
o_shared_1/2    [64, 128]
```

`block_n=128` 时还能压在上限内；`block_n=224` 时 dynamic smem 到了 `260096` bytes，超过可设置上限。

### Probe 2：当前最快 FA3-contract / PTX accumulate 路径

配置：

```text
GQAFwdFP8Fa3ContractPtxAccKernel(
  seq_len=896,
  heads=2,
  heads_kv=1,
  dim=128,
  config={"block_n": 224}
)
```

结果在 TileLang builder 阶段失败：

```text
Check failed: continuous % (vector_size * 8) == 0
continuous=224, vector_size=16
```

这个更像是 TileLang layout helper / swizzle annotation 的限制，而不是 NVIDIA WGMMA 的硬件限制。当前 FA3-contract 路径里有：

```text
T.annotate_layout({
  v_vt_smem: tilelang.layout.make_full_bank_swizzled_layout(v_vt_smem)
})
```

`v_vt_smem` shape 是：

```text
[dim=128, block_n=224]
```

TileLang 的这个 swizzle helper 要求某个连续维满足 `vector_size * 8 = 128` 对齐；`224 % 128 != 0`，所以被拒绝。

### 额外硬约束：我们的 helper 全部是 128 特化

即使绕过上面的 builder 检查，当前最快路径也不是改一个 config 就能正确，因为 helper 大量写死：

```text
VTranspose128x128
VTranspose128x128Fa3Src
fp8_transpose_v_128x128_*
fp8_pv_ptx_unit_accumulate_fa3_raw_64x128x128
fp8_fa3_raw_acc_store_*_64x128
TileShapePV = Shape<64,128,128>
```

也就是说目前的 FA3-contract path 实际 contract 是：

```text
P [64,128] @ V [128,128] -> O [64,128]
```

如果外层 `block_n=224`，helper 仍只消费 128-wide K/V tile，数学上一定不对。

### scale 也要重看

当前 TileOps block-quant 路径默认：

```text
scale_block = 128
k_scale[..., n_idx]
v_scale[..., n_idx]
```

当 `block_n=128` 时，`n_idx` 和 quant scale block 对齐。`block_n=224` 后，一个 attention tile 会跨越多个 128-scale block：

```text
N tile: [0,224) = scale block [0,128) + [128,224)
```

如果用 FA3-compatible per-head scale，benchmark 里 scale 在 N 上是常量，暂时没问题。但如果走 TileOps block128 scale，就必须在 QK/PV 内部按 scale 子段处理，不能继续每个 `n_idx` 用一个 scale。

### 结论

短结论：

```text
TileLang 可能能 lower N=224 WGMMA；
但当前 TileOps FP8 kernel 不能直接支持 block_n=224。
```

阻塞点按优先级是：

1. 当前 native WS path shared memory 超限：`260096` bytes。
2. 当前 FA3-contract path 的 TileLang full-bank swizzle annotation 拒绝 `continuous=224`。
3. 当前 V transpose / PV / output helpers 全部是 `128x128` / `64x128x128` 特化。
4. block128 quant scale 和 `block_n=224` 不天然对齐。

如果要认真做 `block_n=224`，建议不要在现有 kernel 上硬改，而是新建一个实验 kernel：

```text
GQAFwdFP8Fa3ContractPtxAccBN224Kernel
```

目标 contract：

```text
QK:
  half consumer: S [64,224]

PV:
  P [64,224] @ V [224,128] -> O [64,128]
```

需要同步做：

1. 新写 `VTranspose128x224Fa3Src`，对齐 FA3 的 LDSM/STSM layout。
2. 新写 `fp8_pv_ptx_unit_accumulate_fa3_raw_64x128x224`，内部循环 7 个 `K=32` WGMMA。
3. 避开 TileLang `make_full_bank_swizzled_layout([128,224])` 的限制：要么用 CUTE/PTX TMA descriptor，要么先不用 TMA-direct sVt，保留一个可验证但较慢的 staging path。
4. 重算 shared memory，尽量 overlap 或删掉不必要 buffer，否则会超过单 CTA smem 上限。
5. 只在 FA3 scale mode 先验证 correctness；block128 scale 另开处理。

## 2026-05-13 shared memory 利用率论断的实证

前面说“FA3 能用 `block_n=224`，不是因为硬件允许它超限，而是因为它的 shared-memory storage contract 更紧”，现在有两个层面的实证。

### 实证 1：TileOps 当前 native WS `block_n=224` 真实 launch 失败

用 nightly docker / H200 GPU1 probe：

```text
GQAFwdFP8WsPersistentKernel(
  seq_len=896,
  heads=2,
  heads_kv=1,
  dim=128,
  config={"block_n": 224}
)
```

TileLang compile 可以过，但 runtime launch 失败：

```text
Failed to set the allowed dynamic shared memory size to 260096
```

这个 `260096` bytes 是 CUDA runtime 在设置 kernel dynamic smem attribute 时给出的真实需求，不是手算估计。

### 实证 2：FA3 同类 FP8 hdim128 kernel 的 `SharedStorageSize = 191488`

我直接按 FA3 的模板实例化了非 causal / 非 local / hdim128 / e4m3 / `V_colmajor=false` 的 SM90 forward kernel，打印 `AttnKernel::SharedStorageSize`：

```text
kBlockM 128
kBlockN 224
MmaPV_is_RS 1
IntraWGOverlap 1
mainloop.TensorStorage 189440
mainloop.smem_v 57344
mainloop.smem_vt 57344
mainloop.smem_q 16384
mainloop.smem_k 57344
epilogue.TensorStorage 32768
AttnKernel.SharedStorageSize 191488
```

对应源码依据：

```text
tile_size_fwd_sm90:
  FP8 hdim <= 128 -> {128, 224, MmaPV_is_RS=true, IntraWGOverlap=true}

mainloop TensorStorageTransposeV:
  smem_v   = 57344 bytes  # [D=128, N=224, stages=2] FP8
  smem_vt  = 57344 bytes
  smem_q   = 16384 bytes  # [M=128, D=128] FP8
  smem_k   = 57344 bytes  # [N=224, D=128, stages=2] FP8
  smem_p   = 0 bytes      # MmaPV_is_RS=true, P 留在 register

epilogue TensorStorage:
  smem_o   = 32768 bytes  # [M=128, D=128] bf16
```

FA3 kernel-level storage 不是把 mainloop 和 epilogue 简单相加，而是在 `flash_fwd_kernel_sm90.h` 里用 union overlap：

```text
SharedStorage.tensors:
  union {
    mainloop storage
    epilogue storage
  }
```

并且源码注释明确说希望 `smem_o` 和 `smem_v` 起始位置对齐，只在 `sizeof(smem_o) > sizeof(smem_v)` 时补 padding。这里 `smem_o=32768 < smem_v=57344`，所以 epilogue 不额外吃掉一份 32KB 大 buffer。

### 结论修正

所以更准确的说法是：

```text
FA3 的 block_n=224 没有超过硬件 shared memory 上限；
它的实际 SharedStorageSize 约 187KB，能 launch。

TileOps 当前 native WS 直接改 block_n=224 会请求 260096 bytes，
这个才超过了可设置的 dynamic shared memory 上限。
```

二者差距大约：

```text
260096 - 191488 = 68608 bytes
```

主要来源不是 `block_n=224` 本身，而是当前 TileOps path 同时保留了更多 bridge/staging buffer：

```text
P shared staging
V original staging
V tc staging
V vt / pretranspose staging
acc_delta_shared / o_shared round-trip
两个 half_m consumer 的双份 output staging
```

FA3 省下来的关键点：

1. `P` 是 RS：QK output 直接作为 PV operand A register fragment，不落 `smem_p`。
2. `smem_o` 和 `smem_v` union overlap：final store 的 shared buffer 不和 mainloop V buffer 叠加。
3. V transpose contract 是 mainloop 内部的一套固定 `smem_vt -> smem_v` pipeline，不额外保留 TileLang bridge buffer。
4. kernel storage 是围绕 FA3 contract 一次设计的，不是 TileLang layout 和 FA3 layout 之间反复跨桥。

## 2026-05-13 在 TileLang 实现里需要做的具体操作

目标不是一次把所有东西改成 FA3，而是按风险拆成两层：

```text
Layer A: 先把当前 block_n=128 FA3-contract path streamline 干净
Layer B: 再新建 block_n=224 kernel，对齐 FA3 tile size
```

### Layer A：当前 block_n=128 path 的 TileLang 操作

当前最快正确入口：

```text
GQAFwdFP8Fa3ContractPtxAccFa3EpilogueStoreKernel
```

已经做到：

```text
QK output acc_s
  -> PTX helper 直接作为 PV operand A raw register
  -> FA3 V layout smem
  -> raw acc_o
  -> FA3-style epilogue store helper
```

还可以继续做的 TileLang 侧操作：

| 目标 | 当前 TileLang 形态 | 需要改的操作 | 预期收益 |
|---|---|---|---|
| 删除 `o_shared_1/2` 额外 alloc | `use_ptx_pv_fa3_epilogue_store` 仍 alloc `[64,128] out_dtype` 两份 | 不再 `T.alloc_shared(o_shared_1/2)`；final store 阶段把 `v_tc_smem_0/1.access_ptr(...)` 当作 epilogue smem scratch 传给 `tl::fp8_fa3_raw_acc_store_smem_cute_64x128` 和 `tl::fp8_fa3_o_smem_store_global_cute_64x128` | 省 32KB shared allocation；更接近 FA3 `smem_o` overlap `smem_v` |
| 删除 `ls_shared_1/2` | final store 前 `T.copy(ls, ls_shared)`，helper 从 smem 读 lse/normalizer | 改 PTX store helper ABI：直接接收 `ls.data` fragment/register，而不是 `ls_shared.access_ptr("r")` | 省少量 smem 和一次 copy；主要是清 bridge |
| 删除 `ss_shared_1/2` | PV accumulate 前 `T.copy(ss, ss_shared)`，helper 从 smem 读 softmax rescale | 改 `fp8_pv_ptx_unit_accumulate_*` ABI：直接接收 `ss.data` fragment/register | 省少量 smem 和一次 copy；避免每个 N tile 的 shared round-trip |
| 保持 `P` RS | `use_fa3_pv_extern + use_ptx_pv_accumulate` 已经不 alloc `p_shared` | 不要回退到 `T.copy(acc_s, p_shared)`；所有新 helper 都继续从 `acc_s.data` raw register 生成 PV operand A | 防止重新引入最大的一块 P bridge |
| V TMA direct 到 FA3 Vt | `use_fa3_v_tma_direct` 尝试过，但对 `N=224` 被 TileLang swizzle helper 卡住；`N=128` 可以继续验证 | 对 `N=128` 先保留/验证 `T.tma_copy -> v_vt_smem`，然后外部 PTX 做 `v_vt -> v_tc`；如果正确且更快，删 `v_smem -> v_vt` repack path | 省一轮 shared-to-shared V repack |

其中最现实的下一步是：

```text
reuse v_tc_smem_0/1 as o_shared_1/2 scratch
```

因为对 `block_n=128`：

```text
v_tc_smem_i: [128,128] fp8  = 16KB
o_shared_i : [64,128] bf16 = 16KB
```

大小刚好相同。consumer 完成最后一个 PV 后，`v_tc_smem_i` 不再需要作为 V operand，可以作为 epilogue scratch。TileLang 写法上不要做 typed `T.copy`，而是继续用 extern PTX helper，把 `v_tc_smem_i.access_ptr("w")` 当 raw shared pointer。

需要注意同步：

```text
最后一次 consumer PV 读完 v_tc_smem
  -> arrive v_empty
  -> 确保 producer 不再覆盖该 smem stage
  -> reuse v_tc_smem_i as output scratch
```

初版可以用保守的 `T.sync_threads` 或现有 barrier 做保护；之后再收紧。

### Layer B：新建 block_n=224 kernel 的 TileLang 操作

建议新建入口，不改坏现有最快正确 kernel：

```text
GQAFwdFP8Fa3ContractPtxAccBN224Kernel
```

TileLang kernel 参数：

```text
block_m = 128
half_m  = 64
block_n = 224
dim     = 128
threads = 384
```

需要新增/改造的操作：

| 模块 | TileLang 里要做什么 | 外部 PTX / helper 要做什么 |
|---|---|---|
| QK | `acc_s_1/2 = T.alloc_fragment([64,224], accum_dtype)`；`T.wgmma_gemm(q_shared, k_smem, acc_s, transpose_B=True)` 保持 | 暂时不用改，先验证 TileLang QK 对 N=224 的 lowering |
| Online softmax | `online_softmax(..., block_n=224)`；`for i,j in T.Parallel(64,224)` scale | 如果 FA3 scale mode，`k_scale/v_scale` 每 tile 常量；block128 scale 另开处理 |
| P as PV operand A | 不 alloc `p_shared`；不做 `T.copy(acc_s, p_shared)` | 新增 `tl::fp8_pv_ptx_unit_accumulate_fa3_raw_64x128x224`，从 `acc_s.data` raw QK output 做 FA3/operand-A register permutation |
| PV WGMMA | TileLang 只负责调用 extern helper，传 `acc_s.data`、`v_tc_smem.access_ptr("r")`、`ss/scale`、`acc_o.data` | helper 内部按 K=224 做 7 个 K=32 WGMMA group，而不是当前 4 个 K=32 |
| V source smem | 目标保留两份：`v_vt_smem_i [128,224]` 和 `v_tc_smem_i [128,224]` | 新增 `fp8_transpose_v_128x224_fa3_*`：`LDSM.T + PRMT + STSM`，输出 FA3 PV operand B layout |
| TMA V -> Vt | 不用 `make_full_bank_swizzled_layout([128,224])` 这条 TileLang helper 路径；它已实测 `continuous=224` 失败 | 两条路：A. 给 TileLang lowering 增加 FA3/CuTe Vt descriptor 支持；B. extern PTX TMA，传 raw tensor-map descriptor + barrier + smem ptr |
| output store | 不 alloc `o_shared_1/2` | 复用 `v_tc_smem_0/1` 的 shared bytes 做 two-half epilogue scratch，新增/复用 `64x128` store helper |
| shared memory | 不再同时保留 `v_smem`、`v_vt_smem`、`v_tc_smem`、`o_shared`、`p_shared` | 目标接近 FA3：Q + K + Vt + V + small barriers/scheduler，P/O 不额外叠加大 buffer |

### TileLang compiler / runtime 可能需要的能力

如果只写 kernel Python 层，能做一部分；但要真正对齐 FA3 `block_n=224`，大概率需要补 TileLang 能力：

1. **shared memory alias / lifetime reuse**

   当前 `T.alloc_shared` 倾向于每个 buffer 独立分配。FA3 用 C++ union 让 `smem_o` overlap `smem_v`。TileLang 里可以先用“extern helper raw pointer 复用已有 buffer”绕过 typed alias；长期最好有类似：

   ```text
   T.alloc_shared(..., reuse=some_buffer)
   ```

   或者 pass 做 shared allocation lifetime analysis。

2. **自定义 TMA descriptor / swizzle layout**

   当前 `T.tma_copy` 的 swizzle 由 `T.annotate_layout` 推导。`make_full_bank_swizzled_layout([128,224])` 已实测被拒绝。BN224 需要：

   ```text
   FA3 SmemLayoutVt
     -> TMA descriptor swizzle/stride
     -> cp.async.bulk.tensor
   ```

   这可能要加 TileLang primitive，或者让 extern PTX helper 能拿到 tensor-map descriptor 指针。

3. **extern PTX helper ABI 支持 fragment/register 输入**

   为了删 `ss_shared/ls_shared`，helper 需要能接收：

   ```text
   acc_s.data
   ss.data
   ls.data
   acc_o.data
   ```

   而不是把标量 fragment 先 `T.copy` 到 shared。

4. **224-specialized helper family**

   当前 helper 名字和 contract 都是 128 特化：

   ```text
   fp8_transpose_v_128x128_*
   fp8_pv_ptx_unit_accumulate_fa3_raw_64x128x128
   fp8_fa3_raw_acc_store_*_64x128
   ```

   BN224 至少要新增：

   ```text
   fp8_transpose_v_128x224_fa3_*
   fp8_pv_ptx_unit_accumulate_fa3_raw_64x128x224
   ```

   output 仍是 `64x128`，store helper 可以复用。

## 2026-05-13 新建 Layer A 实验 kernel：reuse V smem 做 output scratch

按“不要改坏最快正确基线”的原则，第一步没有改现有：

```text
GQAFwdFP8Fa3ContractPtxAccFa3EpilogueStoreKernel
```

而是新增了一个实验入口：

```text
GQAFwdFP8Fa3ContractPtxAccFa3EpilogueReuseVSmemKernel
```

它只打开一个新 flag：

```text
use_ptx_pv_fa3_epilogue_reuse_v_smem = True
```

### 改动内容

旧 `fa3_epilogue_store`：

```text
acc_o raw registers
  -> tl::fp8_fa3_raw_acc_store_smem_cute_64x128
  -> o_shared_1/2 [64,128] out_dtype
  -> tl::fp8_fa3_o_smem_store_global_cute_64x128
  -> global output
```

新 `reuse_v_smem`：

```text
acc_o raw registers
  -> tl::fp8_fa3_raw_acc_store_smem_cute_reuse_64x128
  -> v_tc_smem_0/1 raw bytes reused as output scratch
  -> tl::fp8_fa3_o_smem_store_global_cute_reuse_64x128
  -> global output
```

也就是说：

```text
不再 alloc:
  o_shared_1 [64,128] out_dtype
  o_shared_2 [64,128] out_dtype

复用已有:
  v_tc_smem_0 [128,128] fp8 raw bytes
  v_tc_smem_1 [128,128] fp8 raw bytes
```

对 `block_n=128`：

```text
o_shared_i : 64 * 128 * 2B = 16KB
v_tc_smem_i: 128 * 128 * 1B = 16KB
```

大小刚好一致。

### helper ABI 细节

不能直接把 `v_tc_smem.access_ptr()` 传给原 helper，因为 `v_tc_smem` 的静态类型是 FP8 pointer，C++ 模板会把 `OutT` 推成 FP8，和 output store 需要的 fp16/bf16 不一致。

所以新增了两层 reuse wrapper：

```cpp
fp8_fa3_raw_acc_store_smem_cute_reuse_64x128(
    acc_o, ls, flags, ScratchT* o_smem, OutT* output)

fp8_fa3_o_smem_store_global_cute_reuse_64x128(
    ScratchT* o_smem, OutT* output, output_row_stride)
```

其中 `ScratchT` 从 `v_tc_smem` 推导，`OutT` 从 global `output` 推导，内部再把 scratch raw pointer reinterpret 成 `OutT*`。

### 验证

nightly docker / H200 GPU1：

```text
pytest -q tests/ops/attention/test_gqa_fp8.py -k "fa3_epilogue_reuse_v_smem" -s
```

结果：

```text
1 passed, 14 deselected
```

长序列 S=1024 对官方 FA3：

```text
finite True True
cos 0.9996287226676941
max_abs 0.00115203857421875
mean_abs 0.00016127641720231622
lse_finite True
```

回归测试：

```text
pytest -q tests/ops/attention/test_gqa_fp8.py -k "ptx_acc" -s
```

结果：

```text
4 passed, 11 deselected
```

### 性能

quick bench / llama8b-1k / FA3 scale：

```text
tileops_ws_fa3_contract_ptx_acc_fa3_epilogue_store
  0.054172 ms  317.13 TFLOP/s

tileops_ws_fa3_contract_ptx_acc_fa3_epilogue_reuse_v_smem
  0.054339 ms  316.16 TFLOP/s

fa3
  0.027770 ms  618.66 TFLOP/s
```

结论：

```text
reuse V smem 可以正确删除独立 o_shared_1/2 allocation；
但 quick bench 没有速度收益，反而在噪声范围内略慢。
```

所以这一项对 `block_n=224` 的 shared memory budget 有意义，但不是当前 `block_n=128` 下 2x performance gap 的主因。下一步优先级应转向：

```text
1. 删除 ss_shared / ls_shared round-trip
2. 或系统看 V producer / transpose / barrier timeline
3. 再开 BN224 新 kernel
```

## 2026-05-13 BN224 新 kernel 第一轮

按照“BN224 必须新建 kernel，不污染现有最快正确路径”的原则，新增了实验入口：

```text
GQAFwdFP8Fa3ContractPtxAccBN224Kernel
```

这版不是从现有 WS persistent 大 kernel 继续堆 flag，而是单独开一条瘦路径，先验证 BN224 的核心 contract：

```text
CTA tile:
  M = 64
  N = 224
  D = 128

QK:
  q_shared [64,128]
  k_smem   [224,128]
  acc_s    [64,224]

V:
  T.copy/TMA-like V -> v_smem [224,128] row-major
  tl::fp8_transpose_v_128x224_ldsm_stsm
  -> v_tc_smem [128,224] FA3/PV operand-B layout

PV:
  tl::fp8_pv_ptx_unit_accumulate_fa3_raw_64x128x224
  internally 7 * K32 WGMMA groups

O:
  reuse v_tc_smem as FA3 epilogue output scratch
```

同时新增了两个 helper family：

```text
VTranspose128x224
fp8_transpose_v_128x224_ldsm_stsm

fp8_acc_to_fa3_p_regs_64x224_no_cute
fp8_pv_ptx_unit_accumulate_fa3_raw_64x128x224
```

### 为什么没有继续挂在 generic WS kernel 上

一开始尝试在现有 `_gqa_fwd_fp8_ws_persistent_kernel` 里加：

```text
use_fa3_v_rowmajor_ldsm_transpose
block_n == 224 ? helper224 : helper128
```

但 TileLang eager frontend 不会把这些实验 flag 当 Python 常量剪枝，而是把所有分支都展开进生成源码。结果直接触发：

```text
SyntaxError: too many statically nested blocks
```

这说明 BN224 不能再塞进现有“大而全”的 WS kernel。后续 BN224 应保持独立瘦 kernel，等 contract 稳定后再考虑合并。

### 当前阻塞点

用 nightly docker / H200 GPU1 跑 S=896：

```text
GQAFwdFP8Fa3ContractPtxAccBN224Kernel
case = batch=1, seq=896, heads=32, heads_kv=8, dim=128
scale_mode = fa3
```

现在能过 Python 语法和 TileLang frontend 嵌套限制，但在 device lowering 阶段失败：

```text
Fatal: InternalError:
Check failed: (info.defined()) is false:
Cannot find memory info of local.fragment
```

伴随 warning：

```text
MemoryInfo for scope = local.fragment is undefined
```

当前最可疑对象是：

```text
acc_s = T.alloc_fragment([64, 224], accum_dtype)
```

也就是 TileLang 对 `N=224` 的 accumulator fragment / local.fragment storage info 可能没有定义完整。这个阻塞点和之前两个问题不同：

```text
不是 shared memory 超限；
不是 TMA full-bank swizzle annotation 拒绝 continuous=224；
而是 local.fragment lowering 对 [64,224] 这类 shape 的支持问题。
```

### 已确认未影响旧路径

同一轮改动后重新跑：

```text
pytest -q tests/ops/attention/test_gqa_fp8.py -k "fa3_epilogue_reuse_v_smem" -s
```

结果：

```text
1 passed, 14 deselected
```

所以现有最快正确路径仍然可用。

### 下一步选择

BN224 继续推进有两条路：

1. 查 TileLang lowering：为什么 `T.alloc_fragment([64,224], "float")` 没有 `local.fragment` memory info。先做一个最小 repro，只保留 QK `acc_s[64,224]`，确认是不是 fragment shape 限制。
2. 绕开 TileLang accumulator fragment：把 QK 也改成外部 PTX/CUTE helper，自己管理 raw QK/P fragment layout。这更接近 FA3，但工作量明显更大。

建议先做 1。它能判断 BN224 是否还能继续利用 TileLang QK，还是必须进一步 PTX 化。

## 2026-05-13 BN224 第二轮：先跑通，再定位 correctness

继续使用 nightly docker / GPU1。

### 1. local.fragment 不是 BN224 的根因

先做了最小 repro，分别测试：

```text
alloc_fragment([64,N]) + T.clear + T.copy
QK WGMMA -> acc_fragment([64,N])，不 copy acc
QK WGMMA -> acc_fragment([64,N]) -> T.copy(acc,out)
```

N sweep：

```text
N = 128, 160, 176, 192, 224, 256
```

结果全部 OK。说明：

```text
T.alloc_fragment([64,224], "float") 本身可以 lowering；
TileLang QK WGMMA 也可以产出 [64,224] accumulator；
T.copy(acc_s, out) 也不是问题。
```

所以上一轮看到的：

```text
Cannot find memory info of local.fragment
```

不是简单的 fragment shape 限制。

### 2. 真正的 lowering 问题在 epilogue helper 读 acc_o.data

进一步拆 repro：

```text
QK + online_softmax + BN224 PV helper
```

可以编译运行。

但一旦接上：

```text
fp8_fa3_raw_acc_store_smem_cute_reuse_64x128(acc_o.data, ...)
```

或者：

```text
fp8_fa3_raw_acc_store_global_64x128(acc_o.data, ...)
```

就复现：

```text
MemoryInfo for scope = local.fragment is undefined
Cannot find memory info of local.fragment
```

原因不是 global pointer，也不是 shared scratch，而是这个瘦 kernel 里 `acc_o` 只被 extern helper 读写，TileLang 没有给它建立普通 fragment layout map。

临时解决办法：

```python
acc_o_layout_seed = T.alloc_shared([64, 128], "float")
...
T.copy(acc_o, acc_o_layout_seed)
T.call_extern(..., acc_o.data, ...)
```

这次 `T.copy` 只是 layout seeding；它让 TileLang 先认识 `acc_o` 的 fragment layout。代价是多一次 64x128 float shared copy，后面要想办法去掉。

加上这个 seed 后：

```text
GQAFwdFP8Fa3ContractPtxAccBN224Kernel
S=896, H=32, Hkv=8, D=128
```

可以编译并运行：

```text
finite(output) = True
finite(lse)    = True
```

### 3. correctness 仍然不对

和 FA3 FP8 GQA 对比：

```text
S=896, H=32, Hkv=8, D=128
cos      = -0.0785426
max_abs  = 0.0424271
mean_abs = 0.0095660
```

用自定义 ceil(scale_blocks) 输入继续测单块/多块：

```text
S=224: cos = 0.0011342
S=448: cos = 0.0083625
S=896: cos = 0.0368517
```

单块 S=224 已经错，所以不是 online softmax 跨 block 合并的问题。

### 4. P pack 初步排除

把 BN224 的 P pack 从手写 no-CUTE 版本：

```text
fp8_acc_to_fa3_p_regs_64x224_no_cute
```

换成 CUTE layout reference：

```text
TileShapeQK = Shape<64,224,128>
TileShapePV = Shape<64,128,224>
convert_layout_acc_Aregs<TiledMmaPV>
```

结果完全不变：

```text
cos 仍然是 -0.0785426
```

这说明当前错误更可能不在 P raw register pack，至少不是这一个手写 permutation 导致的。

### 5. 当前最可疑：V 128x224 transpose / WGMMA descriptor offset

BN224 当前 V 路径是：

```text
v_smem [224,128] row-major
  -> fp8_transpose_v_128x224_ldsm_stsm
  -> v_tc_smem [128,224] FA3/MMA layout
  -> 7 个 K32 wgmma_rs
```

PV helper 里仍然使用：

```cpp
initialize_wgmma_descriptor<1, 1, 64>(desc_b, v_tc_smem);
desc_b + ((ki * 32) >> 4)
```

这个在 128x128 已经验证正确，但对 128x224 不一定仍然成立。下一步应集中验证：

```text
1. fp8_transpose_v_128x224_ldsm_stsm 是否真的覆盖并正确写出 224 个 K columns；
2. v_tc_smem 的 128x224 CUTE layout 下，ki=0..6 的 descriptor offset 是否仍然是 ki*32 bytes / 16；
3. 如果 descriptor offset 不对，改成按 CUTE layout 生成每个 K32 tile 的 smem pointer/descriptor；
4. 如果 transpose 覆盖不对，先把 224 拆成 128 + 96 两段验证。
```

### 当前状态

```text
现有最快正确路径：
  GQAFwdFP8Fa3ContractPtxAccFa3EpilogueReuseVSmemKernel
  correctness OK，性能无明显回退。

BN224 新 kernel：
  可以编译运行；
  output/lse finite；
  correctness wrong；
  当前优先查 V 128x224 transpose / descriptor。
```

## 2026-05-13 BN224 第三轮：dump descriptor 后定位并修复 correctness

这轮按“把矩阵/descriptor dump 出来看”的方向继续查。先没有直接 dump 全量 `64x224`/`128x224` 数据，而是 dump 最关键的矩阵访问 contract：PV WGMMA operand B 的 7 个 K32 V descriptor。

### 1. dump 出来的 descriptor 矩阵

新增 debug helper：

```text
fp8_dump_fa3_v_desc_64x128x224
```

它比较两套 descriptor：

```text
FA3/CUTE:
  thr_mma.partition_fragment_B(sV)(_,_,ki)

TileOps 手写:
  initialize_wgmma_descriptor<1,1,64>(desc, v_tc_smem)
  desc + ((ki * 32) >> 4)
```

dump 结果：

```text
columns:
fa3_start fa3_lbo fa3_sbo fa3_base fa3_layout fa3_size |
tl_start  tl_lbo  tl_sbo  tl_base  tl_layout ok

ki=0 [1856, 1, 16, 0, 3, 1 | 1856, 1, 64, 0, 1, 1]
ki=1 [2112, 1, 16, 0, 3, 1 | 1858, 1, 64, 0, 1, 1]
ki=2 [2368, 1, 16, 0, 3, 1 | 1860, 1, 64, 0, 1, 1]
ki=3 [2624, 1, 16, 0, 3, 1 | 1862, 1, 64, 0, 1, 1]
ki=4 [2880, 1, 16, 0, 3, 1 | 1864, 1, 64, 0, 1, 1]
ki=5 [3136, 1, 16, 0, 3, 1 | 1866, 1, 64, 0, 1, 1]
ki=6 [3392, 1, 16, 0, 3, 1 | 1868, 1, 64, 0, 1, 1]
```

差值：

```text
fa3_minus_tl =
[[   0, 0, -48, 0, 2],
 [ 254, 0, -48, 0, 2],
 [ 508, 0, -48, 0, 2],
 [ 762, 0, -48, 0, 2],
 [1016, 0, -48, 0, 2],
 [1270, 0, -48, 0, 2],
 [1524, 0, -48, 0, 2]]
```

这就是错误点：

```text
BN128 时手写 desc + ki*32/16 可以工作；
BN224 时 V 的 FA3/CUTE layout descriptor 不再等价于这个线性 offset。
```

尤其是：

```text
FA3/CUTE layout_type = 3, stride_byte_offset = 16
手写 TL  layout_type = 1, stride_byte_offset = 64
```

所以 PV 读到的是错误的 V shared layout。之前单块 `S=224 cos ~= 0` 正是这个原因。

### 2. 修复方式

把 BN224 PV helper 从手写 descriptor：

```cpp
tl::GmmaDescriptor desc_b;
initialize_wgmma_descriptor<1, 1, 64>(desc_b, v_tc_smem);

for ki in 0..6:
  wgmma_rs(..., uint64_t(desc_b + ((ki * 32) >> 4)), ...)
```

改成 CUTE partition 出来的 descriptor：

```cpp
using VConfig = VTranspose128x224<FP8T>;
Tensor sV = make_tensor(make_smem_ptr(v_tc_smem), VConfig::SmemLayoutVtMma{});
Tensor tOrV = thr_mma.partition_fragment_B(sV)(_, _, _, _0{});

for ki in 0..6:
  cute::GmmaDescriptor desc_b = tOrV(_, _, ki)(0);
  wgmma_rs(..., uint64_t(desc_b), ...)
```

P pack 也保留为 CUTE reference 版本：

```text
fp8_acc_to_pv_a_frag_64x224_cute
```

这能避免继续用 BN128 的 raw-register permutation 外推 BN224。

### 3. correctness 修复结果

小头数 sweep：

```text
S=224, H=8, Hkv=2:
  cos      = 0.9999987
  max_abs  = 0.0001221
  mean_abs = 0.0000179

S=448, H=8, Hkv=2:
  cos      = 0.9998862
  max_abs  = 0.0014381
  mean_abs = 0.0000994

S=896, H=8, Hkv=2:
  cos      = 0.9997842
  max_abs  = 0.0012112
  mean_abs = 0.0001342
```

目标 case：

```text
S=896, H=32, Hkv=8:
  finite(output/lse) = True / True
  cos      = 0.9997576
  max_abs  = 0.0012970
  mean_abs = 0.0001354
```

所以 BN224 contract 现在正确。

### 4. 当前性能，仅作参考

quick benchmark，`S=896,H=32,Hkv=8,D=128`：

```text
tileops_ws_fa3_contract_ptx_acc_fa3_epilogue_reuse_v_smem:
  0.048652 ms, 270.35 TFLOP/s

tileops_ws_fa3_contract_ptx_acc_bn224:
  0.095309 ms, 138.01 TFLOP/s

FA3:
  0.023926 ms, 549.74 TFLOP/s
```

BN224 这版慢是预期的：

```text
1. 它是 slim single-consumer kernel，不是现有 WS persistent pipeline；
2. 还有 T.copy(acc_o, acc_o_layout_seed) 这次 layout seed shared copy；
3. 当前目标是验证 BN224 contract，不是最终性能。
```

### 5. 旧正确路径未被改坏

重新跑：

```text
pytest -q tests/ops/attention/test_gqa_fp8.py -k "fa3_epilogue_reuse_v_smem" -s
```

结果：

```text
1 passed, 14 deselected
```

### 下一步

现在 BN224 的 correctness 已经成立，后续优化顺序建议：

```text
1. 继续只围绕 P/V swizzle 推进：
   P: QK output accumulator -> PV operand A register layout；
   V: row-major V tile -> FA3 sV / WGMMA operand B descriptor layout。

2. 把 BN224 P/V swizzle contract 移回真正的 WS persistent pipeline：
   不能塞回原 generic mega-kernel；
   应该新开一条更瘦的 BN224 WS kernel，避免 too many statically nested blocks。

3. 再测 BN224 是否真的有性能价值：
   只有进入 WS pipeline 后，BN224 和 BN128 的比较才公平。

备注：`acc_o_layout_seed` 是 slim debug kernel 的 TileLang lowering workaround，不是性能主线。不要继续围绕它做优化判断；真正要改的是 P/V swizzle contract。
```

## 2026-05-13 BN224 WS：只切 P/V swizzle 后跑通

这轮把方向重新收敛到 P/V swizzle：

```text
不继续围绕 output seed 做主线优化；
只把 BN224 的 P/V contract 搬进 WS persistent 路径。
```

### 改动

没有新写一整份 WS kernel，而是在现有 WS FA3/PV 分支中按 `block_n == 224` 切换 P/V 相关 helper：

```text
P:
  fp8_pv_ptx_unit_accumulate_fa3_raw_64x128x224
  内部使用 fp8_acc_to_pv_a_frag_64x224_cute

V:
  producer 从 v_smem [224,128] 直接调用
  fp8_transpose_v_128x224_ldsm_stsm

PV operand B descriptor:
  在 224 helper 内用 CUTE partition_fragment_B(sV) 取每个 K32 的真实 descriptor
```

同时：

```text
T.warpgroup_fence_operand(acc_s, num_regs=112)
```

用于 `[64,224]` QK accumulator。

新增 entry point：

```text
GQAFwdFP8Fa3ContractPtxAccBN224WsKernel
```

它只是现有 WS kernel 的 BN224 FA3-contract 配置封装：

```text
block_n = 224
use_fa3_pv_extern = True
use_fa3_v_tma_direct = False
use_ptx_pv = True
use_ptx_pv_accumulate = True
use_ptx_pv_fa3_epilogue_store = True
use_ptx_pv_fa3_epilogue_reuse_v_smem = True
```

注意：这里暂时没有使用 `use_fa3_v_tma_direct`。也就是说，当前 V 仍然是：

```text
TMA V -> row-major v_smem [224,128]
v_smem -> fp8_transpose_v_128x224_ldsm_stsm -> v_tc_smem [128,224]
```

还不是最终的：

```text
TMA V -> 直接落到 FA3 sV / operand-B-compatible layout
```

### correctness

用 WS BN224 和 FA3 对比：

```text
S=896, H=32, Hkv=8, D=128
finite(output/lse) = True / True
cos      = 0.9997576
max_abs  = 0.0012970
mean_abs = 0.0001354
```

说明 P/V swizzle contract 搬进 WS 后仍然正确。

### performance quick bench

nightly docker / GPU1 / H200：

```text
case = b1_s896_h32_hkv8_d128

BN128 WS current fastest-correct:
  tileops_ws_fa3_contract_ptx_acc_fa3_epilogue_reuse_v_smem
  0.048557 ms, 270.89 TFLOP/s

BN224 WS:
  tileops_ws_fa3_contract_ptx_acc_bn224_ws
  0.056457 ms, 232.98 TFLOP/s

FA3:
  0.024106 ms, 545.65 TFLOP/s
```

BN224 WS 现在比 BN128 WS 慢。这个结果不说明 BN224 方向错，说明当前 BN224 还没有拿到真正的 V swizzle producer 优势：

```text
1. BN224 当前仍然先把 V TMA 到 row-major v_smem；
2. 然后再用 LDSM/STSM 转成 FA3 sV；
3. 没有做到 TMA 直接落到 FA3 sV layout；
4. 还保留了 BN224 epilogue 前的 acc_o layout seed workaround。
```

### regression

旧最快正确路径 smoke：

```text
pytest -q tests/ops/attention/test_gqa_fp8.py -k "fa3_epilogue_reuse_v_smem" -s
```

结果：

```text
1 passed, 14 deselected
```

### 下一步

继续围绕 V swizzle 做，不再偏到 output：

```text
1. 先 dump/验证 BN224 producer 里的 v_smem -> v_tc_smem 是否和 FA3 sV 完全等价；
2. 然后尝试让 TMA 直接写到这个 FA3 sV / operand-B layout；
3. 如果 TileLang TMA descriptor 不能表达，就只在这一段用 PTX/TMA descriptor；
4. 目标是删掉 row-major V -> FA3 sV 的 shared-to-shared transpose/repack movement。
```

## 2026-05-13: 新 kernel，PTX TMA 直接落 BN224 FA3 Vt

今天推进的是 producer 侧 V swizzle：

```text
旧 BN224 WS:
  TMA/copy V -> row-major v_smem [224,128]
  v_smem -> fp8_transpose_v_128x224_ldsm_stsm -> v_tc_smem [128,224]

新 BN224 WS TMA-V:
  PTX TMA V -> FA3 SmemLayoutVt source [128,224]
  fp8_transpose_v_128x224_fa3_src_ldsm_stsm -> v_tc_smem [128,224]
```

### 关键点

TileLang 的 layout annotation 直接做 BN224 V TMA 会卡在：

```text
continuous=224, vector_size=16
TMA bulk copy cannot support a swizzled global layout with inner_box_dim_ > 32
```

所以这一步没有继续依赖 `T.tma_copy + annotate_layout`，而是显式创建 CUtensorMap，并用 inline PTX 发：

```text
cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint
```

真实 V 是 `[B,S,Hkv,D]`，直接用 2D descriptor 不够，因为固定 `head_kv` 时 sequence stride 是 `Hkv * D`。对齐 FA3 的做法是把 V 描述成 4D logical tensor：

```text
logical TMA view: [D, Hkv, S, B]
stride bytes:     [1, D, Hkv*D, S*Hkv*D]
box:              [128, 1, 224, 1]
coords:           [0, head_kv, n_start, batch]
swizzle:          128B
```

新增 helper / probe / kernel：

```text
VTranspose128x224Fa3Src
fp8_transpose_v_128x224_fa3_src_ldsm_stsm
fp8_tma_load_4d_ptx
_probe_fp8_ptx_tma_to_fa3_svt_bn224.py
_probe_fp8_ptx_tma_to_fa3_svt_bn224_4d.py
GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel
```

### byte-level probe

2D 和 4D 探针都验证了最终 `v_tc_smem` 和旧正确 producer 完全一致：

```text
direct_tma:128B: equal_to_baseline=True mismatched_bytes=0
direct_tma_4d:   equal_to_baseline=True mismatched_bytes=0
```

其中 `TMA_SWIZZLE_NONE` 不匹配，`TMA_SWIZZLE_128B` 匹配。

### correctness

新 kernel 和旧 BN224 WS 在 `S=896,H=32,Hkv=8,D=128` 上 bitwise 等价：

```text
finite(output/lse) = True / True
old_new_cos        = 1.0
old_new_max_abs    = 0.0
old_new_mean_abs   = 0.0
lse_max_abs        = 0.0
```

### quick bench

nightly docker / GPU1 / H200：

```text
case = b1_s896_h32_hkv8_d128

BN224 WS:
  tileops_ws_fa3_contract_ptx_acc_bn224_ws
  0.056035 ms, 234.73 TFLOP/s

BN224 WS + PTX TMA direct V:
  tileops_ws_fa3_contract_ptx_acc_bn224_ws_tma_v
  0.054691 ms, 240.50 TFLOP/s

FA3:
  0.023968 ms, 548.79 TFLOP/s
```

结论：TMA 直接落 FA3 Vt 是正确方向，端到端约 `+2.5%`，但还远没解释和 FA3 的差距。说明剩下瓶颈大概率还在 PV/producer-consumer overlap/epilogue bridge，而不是这一处 V source layout 本身。

### 下一步

```text
1. 看新 TMA-V kernel 的 generated CUDA/SASS，确认 V producer 真的少掉 row-major v_smem path；
2. 对比 old BN224 vs TMA-V 的 producer 指令区间，确认收益被哪里吃掉；
3. 继续砍 consumer/PV bridge，优先看 acc_o layout seed / FA3 epilogue store 相关 workaround；
4. 如果要继续提 BN224，需要同时看 occupancy/smem footprint，而不是只看 V TMA。
```

## 2026-05-13: TMA-V lowering check + PV/bridge 初查

### TMA-V 是否真的消掉 row-major V source

导出了 old/new BN224 WS generated CUDA 对比：

```text
old BN224 WS:
  tl::tma_load(v_desc -> smem offset 57344/86016)
  tl::fp8_transpose_v_128x224_ldsm_stsm(
      row-major v_smem -> v_tc_smem offset 114688/143360)

new BN224 WS TMA-V:
  tl::fp8_tma_load_4d_ptx(v_desc -> smem offset 114688/143360)
  tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm(
      FA3 Vt source -> v_tc_smem offset 57344/86016)
```

所以 lowering 后确实没有再走 row-major `v_smem` source。旧的：

```text
TMA V -> row-major source -> LDSM/STSM -> v_tc
```

已经换成：

```text
PTX TMA V -> FA3 Vt source -> LDSM/STSM -> v_tc
```

但要注意：BN224 旧版本已经没有 BN128 那种 `fp8_pack_v_128x128_fa3_vt` 完整 shared-to-shared pack pass。也就是说这一步主要修正 TMA destination layout，而不是删除一个完整 repack pass。因此收益只有约 `2.5%` 是合理的。

### 和 FA3 的 producer/overlap 差异

FA3 FP8/Transpose_V producer 用两条 pipeline：

```text
pipeline_vt: TMA V -> sVt
pipeline_v : LDSM/STSM transpose sVt -> sV
```

关键流程是：

```text
load_V(n)           # TMA 到 sVt
load_K(n)
...
copy_Vt_to_V(n+1)  # 等 Vt ready，然后 transpose 到 sV，并 commit 给 consumer
```

我们的 WS producer 目前仍然更线性：

```text
load K(n)
load V(n-1) -> transpose V(n-1)
commit v_full
```

虽然 V 对 consumer 有一拍提前，但 producer 里 `K(n)` 和 `V(n-1)+transpose` 仍在同一个 producer WG 中串行组织。FA3 的 pipeline 状态更细：Vt TMA 和 V transpose 分开建 pipeline，consumer 等的是最终 `pipeline_v`。

### 和 FA3 的 consumer/PV overlap 差异

FA3 `IntraWGOverlap` 的核心是让：

```text
QK(n+1) WGMMA
PV(n)   WGMMA
```

在同一个 forward step 里交叠发出。代码形态上是先发下一轮 QK，然后在 `warpgroup_wait` 前发当前 PV：

```text
flash::gemm(QK next, wg_wait=-1)
...
flash::gemm(PV current, wg_wait=-1)
warpgroup_wait<1>()
```

我们当前 consumer 是：

```text
QK(n)
T.wait_wgmma(0)
softmax / P swizzle
wait V(n)
fp8_pv_ptx_unit_accumulate...
  internally commit/wait PV WGMMA
next n
```

所以 PV 在 helper 内部被同步完成，无法和下一轮 QK 交叠。这很可能比 TMA-V source layout 更接近当前主差距。

### 当前 bridge 列表

| bridge | 当前形态 | 影响 | 备注 |
|---|---|---|---|
| P swizzle bridge | `fp8_acc_to_pv_a_frag_64x224_cute(acc_s -> p_frag)` | 每轮 PV 前 register pack | 正确但仍是额外 register movement |
| PV delta bridge | PV WGMMA 先写 `delta_storage[64]`，再 `acc_o = acc_o * ss + delta * v_scale` | 不能直接让 WGMMA 累到 O；每轮有 delta fragment 和 row/col add | FA3-scale 下 `v_scale` 是 per-head 常量，可以尝试改成 raw O accumulate，最后 store 再乘 v_scale |
| epilogue layout seed | BN224 仍有 `T.copy(acc_o, acc_o_layout_seed)` | generated CUDA 里仍存在一段 shared store | 看起来是为了稳定 TileLang fragment lowering/layout 的 workaround |
| output scratch bridge | `fp8_fa3_raw_acc_store_smem_cute_reuse_64x128` 复用 `v_tc_smem` 当 O scratch | 需要 shared store + sync + global vector store | 目前是唯一能跑且 smem 不爆的 store path |

### epilogue bridge 快速实验

nightly docker / GPU1 / H200 / `S=896,H=32,Hkv=8,D=128`：

```text
reuse_v_smem:
  correct
  0.054723 ms, 240.36 TFLOP/s

o_shared:
  build ok, launch fail
  Failed to set allowed dynamic shared memory size to 254976

direct_global:
  lowering fail
  Cannot find memory info of local.fragment
```

结论：现有开关不能直接替换掉 output bridge。`reuse_v_smem` 不是美观方案，但它绕开了 dynamic smem 爆掉的问题。要继续砍这里，需要单独写一个不依赖 TileLang `local.fragment` lowering 的 PTX/CUDA store helper。

### 下一步建议

优先级我建议这样排：

```text
1. 先做 FA3-scale 专用 PV accumulate：
   acc_o 先按 ss rescale；
   PV WGMMA 直接 accumulate 到 acc_o；
   v_scale 延迟到 epilogue/store 阶段乘。

2. 这样可以移除 delta_storage[64] + delta*v_scale rowcol add，
   更接近 FA3 的 tOrO 累加方式。

3. 然后再考虑 QK/PV overlap：
   需要把 PV helper 拆开，不在 helper 内部 wait；
   consumer loop 改成 QK(n+1) 和 PV(n) 交叠。

4. 最后再处理 output store：
   direct global 或 register-side store pattern 需要绕过 TileLang local.fragment lowering。
```

## 2026-05-13: FP8 WS 和 FP16 GQA WS 的 fence/barrier 对照

用户提醒以后，我们重新对了一遍：当前 FP8 WS 的 fence/同步结构确实还没有和 GQA WS FP16 版本对齐。这里的差别不只是某个 `warpgroup_fence_operand` 参数，而是 consumer 的 WGMMA overlap contract 不一样。

### FP16 GQA WS 的 consumer contract

FP16 WS 在 consumer 里有两组关键同步：

```text
cross-consumer named barrier:
  tl::tileops_barrier_arrive_named(1/2, 256)
  T.sync_threads(barrier_id=1/2, arrive_count=256)

WGMMA overlap wait:
  tl::wait_wgmma_anchor<1>(...)  # 等 QK，但允许 PV 仍在飞
  tl::wait_wgmma_anchor<0>(...)  # 等 PV 完成
```

主循环从第二个 N block 开始是这个形态：

```text
wait K(n)
launch QK(n)
wait V(n-1)
launch PV(n-1)
arrive named barrier for sibling consumer WG

wait_wgmma_anchor<1>()
fence acc_s
arrive k_empty
softmax QK(n) -> P(n)

wait_wgmma_anchor<0>()
fence acc_o
arrive v_empty
rescale O by ss(n)
prepare P(n)
```

也就是说 FP16 WS 明确在同一个 consumer WG 内交叠：

```text
QK(n)  和  PV(n-1)
```

相关代码位置：

```text
tileops/kernels/attention/gqa_fwd_ws.py:204-239
```

### 当前 FP8 WS 的 consumer contract

当前 FP8 WS 主循环是更串行的：

```text
wait K(n)
launch QK(n)
T.wait_wgmma(0)
fence acc_s
arrive k_empty
scale + online softmax
pack/swizzle P for PV
wait V(n)
call fp8 PV helper
arrive v_empty
```

相关代码位置：

```text
tileops/kernels/attention/gqa_fwd_fp8.py:915-1044
```

FP8 的 PV helper 内部又是同步完成的：

```text
pack P fragment
launch 7 x PV WGMMA for BN224
wgmma.commit_group
wgmma.wait_group 0
wgmma.fence
acc_o = acc_o * ss + delta * v_scale
```

相关代码位置：

```text
tileops/kernels/attention/_fp8_gqa_helper.h:1784-1855
```

所以当前 FP8 版本没有像 FP16 WS 那样把下一轮 QK 和上一轮 PV 放在同一个 WGMMA group window 里。PV 延迟会完整落在主循环关键路径上，这和我们怀疑“PV 延迟是大头”的现象一致。

### fence 本身的对齐情况

| 阶段 | FP16 WS | 当前 FP8 WS | 是否对齐 |
|---|---|---|---|
| QK 后读 `acc_s` | `wait_wgmma_anchor<1>` 或 `T.wait_wgmma(0)` 后 `warpgroup_fence_operand(acc_s, 64)` | `T.wait_wgmma(0)` 后 `warpgroup_fence_operand(acc_s, 112 if BN224 else 64)` | fence 有，但等待策略不同 |
| PV 后读/更新 `acc_o` | `wait_wgmma_anchor<0>` 后 `warpgroup_fence_operand(acc_o, 64)` | PV helper 内部 `wait_group 0` + `wgmma.fence`，然后 helper 内 row/col 更新 `acc_o` | 不同；FP8 helper 把 wait 和 O 更新封装死了 |
| K smem release | QK 对 `acc_s` fence 后 `barrier_arrive(k_empty)` | QK 对 `acc_s` fence 后 `barrier_arrive(k_empty)` | 基本对齐 |
| V smem release | PV 对 `acc_o` fence 后 `barrier_arrive(v_empty)` | PV helper 返回后 `barrier_arrive(v_empty)` | 语义上安全，但不能 overlap |
| output scratch -> global | `T.copy(acc_o, q_shared)` + `fence_proxy_async` + 128-thread sync + global copy | FA3/raw store helper 到 scratch + `fence_proxy_async` + 128-thread sync + global copy | 大方向类似，但 FP8 多了 FA3 output layout/unpermute bridge |

### 结论

现在答案是：

```text
是的，当前 FP8 WS 的 fence/barrier/overlap contract 和 GQA WS FP16 不一样。
```

更准确地说：

```text
producer 侧 TMA-V direct 已经基本确认；
consumer 侧仍是 serial QK -> PV；
FP16 WS 是 overlapped QK(n) + PV(n-1)。
```

这解释了为什么我们砍掉或修正 V source layout 后只有小幅收益：真正的大块 latency 可能还在 PV helper 内部同步，以及没有复用 FP16 WS 的 WGMMA overlap schedule。

### 后续计划

下一步不应该先继续微调 output store，而应该新建一个 FP8 overlap kernel，专门对齐 FP16 WS 的 consumer contract：

```text
1. 保留当前正确 BN224 TMA-V direct kernel 作为 baseline。
2. 新建 FP8 BN224 overlap kernel，不改坏现有最快正确版本。
3. 把 PV helper 拆成 begin/wait/update 三段，至少先做到：
   - begin: pack P + launch PV WGMMA，不 wait；
   - wait/update: wait PV，fence O/delta，然后做 O update；
4. consumer loop 改成：
   - launch QK(n)
   - launch PV(n-1)
   - wait QK -> softmax/pack P(n)
   - wait PV -> update O/release V(n-1)
5. correctness 先用长序列，比如 S >= 896，再和 FA3 / 当前 baseline 对齐。
6. performance 用 nightly docker / GPU1 对比：当前 BN224 TMA-V、overlap kernel、FA3。
```

这个方向会比继续只改 TMA-V 或 output store 更接近 FP16 WS 和 FA3 的核心调度策略。

## 2026-05-13: 对齐 FA3 的 P 矩阵处理方式

这次专门对了一遍 FA3 在 FP8 / transpose-V 路径里怎么处理 P，也就是：

```text
QK output / scores accumulator
  -> online softmax 得到 P
  -> PV WGMMA operand A
```

结论先写在前面：

```text
FA3 FP8 路径强制 MmaPV_is_RS=true。
所以 P 不写 shared memory；P 是 register-side operand A。
```

FA3 代码里有两个直接约束：

```text
static_assert(!(!MmaPV_is_RS && Is_FP8), "MmaPV must be RS if FP8");
static_assert(!(!MmaPV_is_RS && Transpose_V), "MmaPV must be RS if Transpose_V");
```

也就是说，对于我们现在关心的 FP8 transpose-V 路径，FA3 不允许 P 走 SS / shared P。

### FA3 的 P lifecycle

FA3 overlap 主循环里，P 的生命周期是：

```text
第一次：
  QK(0) -> tSrS
  online_softmax(tSrS)
  if FP8 && !V_colmajor:
      permute_Cregs_fp8(tSrS)
  tOrP_acc = view(tSrS.data, convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout))
  tOrP = make_tensor_like<fp8>(tOrP_acc)
  convert_type_out(tOrP_acc, tOrP)

循环中：
  launch PV(prev P = tOrP, prev V)
  同时 launch QK(curr) -> 新的 tSrS
  wait QK
  online_softmax(curr tSrS)
  if FP8 && !V_colmajor:
      permute_Cregs_fp8(curr tSrS)
  convert_type_out(view(curr tSrS, tOrP.layout), tOrP)
  rescale O
```

关键点：

```text
tOrP 是一个 FP8 register tensor；
上一轮 PV 用它时，当前 QK 写的是另一个 tSrS；
当前 softmax 完成后，再把当前 P 覆盖写回同一个 tOrP buffer，供下一轮 PV 用。
```

所以 FA3 并不是：

```text
P -> shared -> PV
```

而是：

```text
P(fp32 acc regs)
  -> C-reg permutation
  -> reinterpret as PV operand-A layout
  -> fp8 register tensor tOrP
  -> PV RS WGMMA
```

### P 的两个 register layout 变换

FA3 对 P 做的第一步不是普通转置，而是先修正 FP8 C accumulator layout：

```text
tSrS: QK output / softmax 后的 FP32 accumulator

if Is_FP8 && !V_colmajor:
    permute_Cregs_fp8(tSrS)
```

这一步在 transpose-V 路径会触发。它的意义是把 QK 的 C-register raw layout 调整到后续 FP8 PV-A 转换能解释的形态。

然后 FA3 做：

```text
tOrP_acc = make_tensor(
    tSrS.data(),
    convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout())
)
```

这一步是纯 register-view 变换：

```text
同一批 tSrS raw registers
  从 QK C accumulator layout
  解释成 PV operand-A accumulator-compatible layout
```

最后：

```text
tOrP = make_tensor_like<Element>(tOrP_acc)
convert_type_out(tOrP_acc, tOrP)
```

这一步才把 FP32 P downcast/pack 成 FP8 PV operand A register fragment。

### V_colmajor 路径和 transpose-V 路径不同

FA3 这里有一个容易混淆的分支：

```text
if Is_FP8 && !V_colmajor:
    permute_Cregs_fp8(tSrS)

if Is_FP8 && V_colmajor:
    permute_Aregs_fp8(tOrP)
```

我们现在主要对齐的是普通 V 输入 `[seq, d]`、kernel 内 transpose V 的路径，也就是：

```text
V_colmajor = false
Transpose_V = true
```

因此 P 侧应对齐的是：

```text
permute_Cregs_fp8(tSrS)
convert_layout_acc_Aregs<TiledMmaPV>
convert_type_out -> tOrP
```

而不是 `permute_Aregs_fp8(tOrP)`。`permute_Aregs_fp8` 是 V_colmajor=true 的另一条路径。

### 和我们现在代码的对照

当前正确的 serial BN224 helper 基本采用了 FA3 的 P contract：

```text
fp8_acc_to_pv_a_frag_64x224_cute(acc_s, p_frag)
  内部做：
    permute_Cregs_fp8(acc)
    convert_layout_acc_Aregs<TiledMmaPV>
    convert_type_out -> p_frag

PV:
  wgmma.rs(p_frag, V_tc_smem, delta)
  wait
  permute_output_fp8(delta)
  acc_o = acc_o * ss + delta * v_scale
```

也就是说，serial 版本里 P 的 swizzle/pack 大方向是和 FA3 transpose-V 路径对齐的。

新建的 overlap kernel 目前为了让 P 跨过“当前 QK 和上一轮 PV overlap”这个窗口，尝试了：

```text
p_pv_local_1/2: 每线程 local[112] fp8
pack current P -> local
下一轮 PV 从 local 读回 p_frag -> wgmma.rs
```

这个方向是在模拟 FA3 的 `tOrP` register tensor，但还不是完全等价：

```text
FA3: tOrP 是活在寄存器里的 CUTE fragment，PV 直接消费。
我们: local[112] 很可能被 TileLang/LLVM 当作 local memory 数组处理，存在额外 load/store 或 spilling 风险。
```

更重要的是，FA3 在 transpose-V 路径的 PV 结果最后还会做：

```text
if Is_FP8 && !V_colmajor:
    permute_output_fp8(tOrO)
```

serial helper 当前是在每次 delta 上做 `permute_output_fp8(delta)` 后再累到 `acc_o`。如果 overlap kernel 直接让 PV WGMMA 累到 `acc_o`，就必须保证：

```text
acc_o 的 raw layout 始终是 FA3 PV output raw layout；
最终 store 前执行一次等价的 permute_output_fp8(acc_o)；
中间 rescale_o 只按 row 乘 scale，不依赖已经 unpermute 的列顺序。
```

否则会出现：

```text
P 看起来对了，
V 也看起来对了，
但 O 的 raw register layout 没有按 FA3 contract 收尾，
结果仍然错。
```

### 后续应该怎么改 overlap kernel

下一步应优先把 overlap kernel 改成更接近 FA3 的 accumulator contract：

```text
1. P 仍然用 fp8_acc_to_pv_a_frag_64x224_cute 打包。
2. 不走 shared P；shared P 会炸 smem，也不是 FA3 FP8 contract。
3. 尽量让 p_frag/p local 保持 register fragment 语义，避免大 shared scratch。
4. PV 可以直接 accumulate 到 acc_o，但 acc_o 应保持 raw PV-O layout。
5. rescale O 时按 row 做乘法，允许 raw layout。
6. tail / store 前对 acc_o 做一次 permute_output_fp8(acc_o)，再进入现有 output store。
```

这也解释了目前 overlap 尝试为什么会“P/V 都像是在改对的方向，但 correctness 仍然不稳”：我们之前 serial helper 的 correctness 依赖的是每个 delta 先 `permute_output_fp8`，而 direct-accumulate overlap 版本绕开了这一步。

## 2026-05-13: 尝试把 FP8 P 改成 FP16 `acc_s_cast` 同构 fragment

目标是把 FP8 overlap kernel 的 P carrier 从：

```text
p_pv_local = T.alloc_local([112], fp8)
```

改成更接近 FP16 WS 的：

```text
p_pv_frag = T.alloc_fragment([half_m, block_n], fp8)
```

也就是：

```text
FP16:
  softmax(acc_s)
  T.copy(acc_s, acc_s_cast)
  PV(acc_s_cast, V, acc_o)

FP8 目标:
  softmax(acc_s)
  FA3 pack/swizzle(acc_s.data -> p_pv_frag.data)
  PV(p_pv_frag.data, V_tc, acc_o)
```

### 实验结果

尝试 1：

```text
p_pv_frag = T.alloc_fragment([112], fp8)
extern(pack, acc_s.data, p_pv_frag.data)
extern(PV, p_pv_frag.data, ...)
```

结果：

```text
Cannot find memory info of local.fragment
```

说明 TileLang 当前不能直接 lowering 一个只通过 extern raw `.data` 消费的一维 fragment。

尝试 2：

```text
p_pv_frag = T.alloc_fragment([half_m, block_n], fp8)
extern(pack, acc_s.data, p_pv_frag.data)
extern(PV, p_pv_frag.data, ...)
```

结果仍然是：

```text
Cannot find memory info of local.fragment
```

尝试 3：

为了让 TileLang 建立 fragment layout，在每次 FA3 pack 前加一个 layout seed：

```text
T.copy(acc_s, p_pv_frag)
extern(pack, acc_s.data, p_pv_frag.data)
```

这个版本可以 compile/run：

```text
compiled
ran finite=True
```

但 correctness 不过。和 serial BN224 FA3-contract baseline 对比，S=896：

```text
out_cos      ~= 0.505
out_max_abs  ~= 0.689
out_mean_abs ~= 0.0312
lse_max_abs  ~= 0.356
```

`lse` 也变了，说明问题不只是 PV/O output permute。更可能是这个 layout seed copy 或 fragment liveness 让 TileLang 的 fragment allocator / layout inference 干扰了 `acc_s` / softmax 相关寄存器。

### 当前判断

概念上，FP8 应该对齐 FP16 的 `acc_s_cast`：

```text
acc_s      = 当前 QK/softmax FP32 fragment
p_pv_frag = 上一轮 P 的 FP8 PV operand-A fragment
```

但 TileLang 目前不能直接表达：

```text
extern writes p_pv_frag.data
extern later consumes p_pv_frag.data
```

除非用 `T.copy(acc_s, p_pv_frag)` 让 layout inference 看到这个 fragment；而这个 seed copy 目前会破坏 correctness 或至少引入未理解的 register/layout alias 风险。

下一步应该查两件事：

```text
1. 是否能给 p_pv_frag 显式 annotate/register layout，而不是靠 T.copy seed。
2. 是否能把 pack+PV 合并成一个 TileLang 可见的 fragment op，或在 C++ helper 内保持 FA3 tOrP 生命周期。
```

## 2026-05-13: 关于 `make_mma_load_layout` 是否能解决 P fragment 问题

今天重新对齐 TileLang 的 RS WGMMA contract：

```text
SS: A/B 都来自 shared descriptor
RS: A 来自 register fragment，B 来自 shared descriptor
```

TileLang native RS 路径确实有一个显式解法：

```text
p_frag = T.alloc_fragment((M, K), fp8)
T.annotate_layout({
    p_frag: pv_emi.make_mma_load_layout(p_frag, matrix="A"),
})
T.copy(p_shared, p_frag)
T.wgmma_gemm(p_frag, v_shared, acc_o, ...)
```

这个 annotation 告诉 TileLang：

```text
p_frag[i, k] 的 logical element
  -> 哪个 lane
  -> 哪个 local register index
```

也就是硬件 `wgmma.mma_async` RS variant 对 operand A 期待的寄存器排布。

### 它能解决什么

它可以解决“TileLang 不知道这个 fragment layout”的一类问题，比之前用：

```text
T.copy(acc_s, p_pv_frag)
```

做 layout seed 更干净。

本地 probe 已经证明，下面这种 TileLang native contract 可以工作：

```text
extern(acc_s -> p_frag.data)
T.wgmma_gemm(p_frag, v_tc_shared, acc_o)
```

前提是：

```text
1. p_frag 已经被 annotate 成 TileLang WGMMA RS operand-A layout
2. extern 写入 p_frag.data 的 raw order 正好就是这个 layout
3. v_tc_shared 的 shared layout 也正好是 TileLang 这次 WGMMA 期待的 B layout
```

### 它还不能直接证明什么

它不能自动完成 FA3 的：

```text
acc_s FP32 QK-C layout
  -> permute_Cregs_fp8
  -> convert_layout_acc_Aregs<TiledMmaPV>
  -> convert_type_out FP8
  -> FA3 PV operand-A registers
```

`make_mma_load_layout` 描述的是 TileLang emitter 的 WGMMA RS-A layout；FA3 的 `convert_layout_acc_Aregs<TiledMmaPV>` 描述的是 CUTE `TiledMmaPV` 的 RS-A layout。二者很可能都面向同一个硬件 RS operand-A contract，但必须用同一组 tile/atom 参数验证 raw register order 是否完全一致。

### 对当前路线的判断

这个方法值得试，而且应该作为下一步最小实验：

```text
1. 新建 kernel，不改现有正确 kernel。
2. 给 p_pv_frag 显式 annotate:
   p_pv_frag: pv_emi.make_mma_load_layout(p_pv_frag, matrix="A")
3. 移除 T.copy(acc_s, p_pv_frag) seed。
4. 保留 extern FA3 pack 写 p_pv_frag.data。
5. 先继续用当前 PTX/FA3 PV helper 消费 p_pv_frag.data，验证是否能 compile。
6. 如果 compile 过，再和 serial BN224 FA3-contract baseline 对 correctness。
7. 如果 correctness 不过，dump/对比 p_pv_frag raw order，判断 TileLang RS-A layout 和 FA3 TiledMmaPV Aregs layout 是否不等价。
```

所以结论不是“它已经解决当前问题”，而是：

```text
它可能解决当前的 fragment layout/lowering 阻塞；
但是否解决 FA3 P swizzle 正确性，要看 TileLang RS-A layout 是否等价于 FA3 TiledMmaPV Aregs layout。
```

### 进一步校正：这不是当前 FA3 路线的正解

回看前面的 contract 记录后，需要把判断再收紧：

```text
我们之所以手动做 P/V swizzle，
正是因为 TileLang native 的自动 fragment/shared layout
不是当前 FA3 contract 需要的 layout。
```

`make_mma_load_layout` 能生成的是：

```text
TileLang emitter 自己的 WGMMA RS operand-A layout
```

而当前 FA3 路线需要的是：

```text
FA3 / CUTE TiledMmaPV 的 operand-A layout
```

这两个不能默认等价。尤其当前文档前面已经确定：

```text
FA3 contract:
  P: permute_Cregs_fp8 + convert_layout_acc_Aregs<TiledMmaPV>
  V: FA3 sVt -> LDSM.T/STSM -> FA3 sV
  O: FA3 PV output raw layout + FA3 output unpermute

TileLang-native contract:
  P: TileLang PV-A fragment layout
  V: TileLang PV-B shared layout
  O: TileLang accumulator/update layout
```

所以 `make_mma_load_layout` 只有两种合理用途：

```text
1. 作为诊断工具：
   比较 TileLang RS-A raw layout 和 FA3 TiledMmaPV Aregs raw layout 是否相同。

2. 作为第二阶段 TileLang-native contract 的入口：
   如果决定切到 TileLang-native P/V/O，全套 layout 都要跟着换。
```

它不应该被当成当前 FA3 streamline 路线的主要修复手段。当前 FA3 路线仍然应该坚持：

```text
手写 / PTX / C++ helper 生成 FA3 需要的 P register layout；
V 也保持 FA3 sV contract；
O 按 FA3 output contract 收尾。
```

### 再校正：PTX 写寄存器时 `annotate_layout` 不参与数据重排

如果 P 的写入是：

```text
T.call_extern(...)
  -> inline PTX / C++ helper
  -> 写 p_pv_frag.data 的 raw register/storage order
```

那么 `T.annotate_layout` 不会改变这些 PTX 写入的物理顺序。它只是 TileLang 编译期 layout inference 的提示，主要影响 TileLang 自己生成的：

```text
T.copy(...)
T.wgmma_gemm(...)
```

这类 op 如何索引 fragment。对于 extern/PTX：

```text
PTX 写了哪个 raw register / local slot，就是哪个 raw register / local slot。
```

所以当前 FA3 P 的真实要求应该写成一个 raw ABI：

```text
p_frag raw storage:
  FP8[112] = 28 x uint32

for ki in 0..6:
  WGMMA RS operand A reads:
    reinterpret_cast<uint32_t*>(p_frag) + ki * 4
```

也就是说，pack helper 必须把第 `ki` 个 K32 tile 的 FA3 PV-A registers 写到：

```text
p_regs[ki * 4 : ki * 4 + 4]
```

然后 PV helper 用同一套 `TiledMmaPV` / descriptor contract 读它。这里 correctness 只取决于：

```text
1. PTX/C++ pack 写出的 flat order 是否等于 FA3 TiledMmaPV Aregs order；
2. PV WGMMA 读 A 的 flat offset 是否和 pack helper 完全一致；
3. 发 WGMMA 前是否有正确的 wgmma operand fence，防止寄存器写被重排到 WGMMA 之后。
```

当前 `fp8_pv_ptx_unit_begin_accumulate_from_p_frag_fa3_raw_64x128x224` 的消费端已经是这种形式：

```text
uint32_t* p_regs = reinterpret_cast<uint32_t*>(p_frag);
for ki in 0..6:
  wgmma_rs(p_regs + ki * 4, desc_b(ki), acc_o, scale_d=true)
```

因此下一步不是依赖 `annotate_layout` 自动变换 P，而是验证/固定这个 raw ABI：

```text
pack(acc_s -> p_frag) 写出的 p_regs[0..27]
  == FA3 tOrP raw register order

PV(p_frag -> acc_o) 读取的 p_regs[0..27]
  == 同一个 FA3 tOrP raw register order
```

`annotate_layout` 如果还有价值，也只是让 TileLang lowering 接受这个 fragment 变量；它不是 correctness 的来源。

## 2026-05-13: 新增 local-P overlap 实验 kernel

用户提醒：不要把实验做成老 kernel 的 branch。按这个原则，这轮没有给旧 overlap factory 加参数，而是复制出一条独立入口：

```text
GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapLocalPKernel
```

对应 factory：

```text
_gqa_fwd_fp8_fa3_contract_bn224_ws_overlap_local_p_kernel
```

它从旧 overlap 实验 kernel 机械复制而来，只改 P storage / helper ABI：

```text
旧 overlap:
  p_pv_frag = T.alloc_fragment([64,224], fp8)
  T.copy(acc_s, p_pv_frag)              # layout seed
  fp8_pack_p_fa3_raw_64x128x224(acc_s.data, p_pv_frag.data)
  fp8_pv_ptx_unit_begin_accumulate_from_p_frag(... p_pv_frag.data ...)

新 local-P overlap:
  p_pv_local = T.alloc_local([112], fp8)
  fp8_pack_p_fa3_raw_64x128x224_to_local_p_experimental(
      acc_s.data, p_pv_local.data)
  fp8_pv_ptx_unit_begin_accumulate_from_local_p_experimental_64x128x224(
      p_pv_local.data, ...)
```

也就是明确测试这个 raw ABI：

```text
extern pack 写 local P buffer
extern PV 读同一个 local P buffer
```

而不是让 TileLang fragment layout inference 参与 P。

### 编译 / correctness 结果

nightly docker / GPU1，`S=896,H=16,Hkv=4,D=128`，和当前正确 BN224 TMA-V baseline 比：

```text
compiled = yes
finite(output/lse) = True / True
out_cos      = 0.4806168
out_max_abs  = 0.0145569
out_mean_abs = 0.0020484
lse_max_abs  = 0.0007257
```

补充：加上实验专用 wrapper helper 名字后，又用 zero input 做了一次 docker/GPU1 smoke：

```text
compiled_and_ran True True
```

结论：

```text
1. local P buffer 跨 extern pack/PV 可以 compile/run；
2. LSE 基本一致，说明 QK/softmax 主路径没有炸；
3. O 明显不对，问题更集中在 PV/O raw accumulation、overlap rescale、或 final FA3 store contract；
4. 这不是正确 kernel，只是 ABI/lowering 实验入口。
```

下一步不要再纠结 `annotate_layout`。应继续把问题缩小：

```text
A. 做一个非 overlap 的 local-P split helper 对照：
   acc_s -> local P -> PV -> delta/acc_o
   验证 local P raw ABI 本身是否 byte/math 等价。

B. 如果 A 正确，再查 overlap 里的 O raw accumulation:
   - begin_accumulate_from_p_ptr 是否需要和成功 helper 一样先打 delta；
   - rescale 是否应该作用在 permuted / unpermuted O layout；
   - final store helper 读到的 acc_o layout 是否和当前 begin_accumulate 输出一致。
```

## 2026-05-13: local-P overlap correctness 重新定位

今天调试时先用了一个太小的 persistent case：

```text
B=1, S=896, H=8, Hkv=2, D=128
logical_tiles = B * Hkv * ceil(S / 128) * (H / Hkv)
              = 1 * 2 * 7 * 4
              = 56
NUM_SMS = 132
```

这个 case 的逻辑 persistent tile 数小于 `NUM_SMS`。因此它会暴露/放大 persistent underfill 下的映射问题，不能拿来判断当前 overlap pipeline 的 correctness。它表现为：

```text
old overlap / local-P overlap:
  lse 只在部分 head group 错；
  错误 head group 与 blockIdx.x 超出 logical tile 数后的映射高度相关。
```

用户提醒后，改用至少 132 个 logical tiles 的 case。两个测试都在 nightly docker / GPU1 上跑，和正确 baseline `GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel` 比。

### Case A

```text
B=1, S=1792, H=16, Hkv=4, D=128
logical_tiles = 1 * 4 * ceil(1792 / 128) * 4 = 224
```

结果：

```text
old overlap:
  cos      = 0.5525047
  out_mean = 0.0030583
  out_max  = 0.0233765
  lse_max  = 0.0

local-P overlap:
  cos      = 0.9994095
  out_mean = 0.0001463
  out_max  = 0.0021057
  lse_max  = 0.0
```

### Case B

```text
B=2, S=896, H=16, Hkv=4, D=128
logical_tiles = 2 * 4 * ceil(896 / 128) * 4 = 224
```

结果：

```text
old overlap:
  cos      = 0.5377232
  out_mean = 0.0044308
  out_max  = 0.0327148
  lse_max  = 0.0

local-P overlap:
  cos      = 0.9995017
  out_mean = 0.0001809
  out_max  = 0.0031128
  lse_max  = 0.0
```

结论：

```text
1. current local-P overlap 在足量 persistent tiles 下，QK/softmax 完全对齐 baseline：
   lse_max = 0。

2. old overlap 的 LSE 也能对齐，但 O 明显错，说明它的问题主要在 PV/O update 公式或 raw O layout；
   这和之前修 local-P 时发现的 direct accumulate/rescale 问题一致。

3. local-P overlap 的 O 已经接近正确：
   cos ~= 0.9994-0.9995，max diff ~= 2e-3 到 3e-3。
   这说明：
     acc_s -> local P -> PV delta -> acc_o update
   这条 raw ABI 基本成立。

4. 后续 correctness 测试必须满足：
   logical_tiles >= NUM_SMS
   否则不能把结果作为 pipeline correctness 证据。
```

下一步：

```text
A. 用足量 logical tiles case 继续看 local-P overlap 的剩余 O diff；
B. 对比 FP16 WS 和 FP8 local-P overlap lowered CUDA/PTX，但只用 logical_tiles >= NUM_SMS 的 case；
C. 如果目标是性能，local-P overlap 可以作为当前继续推进的实验入口，old overlap 先不再作为 correctness 参考。
```

## 2026-05-13: local-P overlap quick speed

在确认 `logical_tiles >= NUM_SMS` 后，开始测速度。命令形态：

```text
docker / GPU1 / nightly
bench_gqa_fp8.py --quick
impls:
  tileops_ws_fa3_contract_ptx_acc_bn224_ws_tma_v
  tileops_ws_fa3_contract_ptx_acc_bn224_ws_overlap_local_p
  fa3
scale-mode:
  fa3
```

### Speed 表

```text
case                         impl                         latency_ms   TFLOPS
b1_s1792_h16_hkv4_d128       ws_tma_v                     0.121837     215.92
b1_s1792_h16_hkv4_d128       overlap_local_p              0.145993     180.19
b1_s1792_h16_hkv4_d128       fa3                          0.039187     671.31

b2_s896_h16_hkv4_d128        ws_tma_v                     0.054425     241.68
b2_s896_h16_hkv4_d128        overlap_local_p              0.107843     121.97
b2_s896_h16_hkv4_d128        fa3                          0.023953     549.14

b1_s3584_h32_hkv8_d128       ws_tma_v                     1.005388     209.33
b1_s3584_h32_hkv8_d128       overlap_local_p              1.235379     170.36
b1_s3584_h32_hkv8_d128       fa3                          0.379762     554.17
```

### 当前判断

```text
1. local-P overlap correctness 已经接近可用，但 quick speed 没有赢 ws_tma_v。

2. 这说明当前 overlap_local_p 的结构还没有把 QK(n) / PV(n-1) overlap 转化成收益；
   额外成本可能来自：
     - acc_s -> local P 的 pack；
     - PV begin/wait/update helper 拆分后的 wait/fence/anchor；
     - delta_local 额外寄存器压力；
     - output/update 仍然串在 consumer 关键路径上；
     - 当前 schedule 没有真正把 PV latency 藏到 QK/softmax 下面。

3. ws_tma_v 仍然是当前最快正确 TileOps FP8 路径。

4. FA3 仍然快很多：
   当前 TileOps ws_tma_v 大约是 FA3 的 38%-44% 速度；
   overlap_local_p 大约是 FA3 的 22%-31% 速度。
```

下一步性能分析应该看：

```text
A. 用 NCU / lowered SASS 对比 ws_tma_v vs overlap_local_p：
   - wgmma issue/wait 分布；
   - producer/consumer stall；
   - register pressure / occupancy；
   - shared memory traffic；
   - tensor core utilization。

B. 先别继续扩大 overlap 改动；
   现在更重要的是确认为什么 overlap_local_p 没有藏住 PV latency。
```

## 2026-05-13: 为什么 overlap_local_p 没藏住 PV latency

用户指出：横向 latency 对比意义有限，应该直接看为什么 PV latency 没被藏住。

先看当前 `overlap_local_p` 的 steady-state 结构：

```text
QK(n) WGMMA launch
PV(n-1) begin helper launch
wait_wgmma_anchor<1>     # 等 QK done，理论上 PV 还在飞
softmax(QK(n))
wait_wgmma_anchor<0>
PV wait/update helper
pack P(n) -> local P
```

理论上能藏住的是：

```text
PV(n-1) 已经 issue 进 tensor core 之后，到 wait<0> 之间的执行尾部
```

但不能藏住：

```text
1. PV begin helper 的 issue 开销；
2. PV begin 前清 delta 的开销；
3. begin/wait helper 之间为了保存 delta 的 live range；
4. local P 跨 iteration/helper 的 live range；
5. wait/update helper 读取 delta、更新 acc_o 的依赖。
```

### Lowering 证据

`overlap_local_p` lowered CUDA 里出现了两个跨 helper 的 per-thread C 数组：

```cpp
fp8_e4_t p_pv_local_1[112];
float    acc_delta_local_1[64];

fp8_e4_t p_pv_local_2[112];
float    acc_delta_local_2[64];
```

对应路径：

```cpp
fp8_pack_p_fa3_raw_64x128x224_to_local_p_experimental(
    acc_s_1, p_pv_local_1);

fp8_pv_ptx_unit_begin_from_local_p_experimental_64x128x224(
    p_pv_local_1, v_tc_smem, 4, acc_delta_local_1);

... softmax ...

fp8_pv_ptx_unit_wait_update_fa3_raw_64x128x224(
    acc_delta_local_1, 4, ss_shared, v_scale, acc_o_1);
```

这说明这条实验路径不是“纯寄存器持久化 P/delta”的最终形态，而是把 P 和 PV delta 做成了跨 helper live 的 local arrays。

### NCU 证据

case:

```text
B=1, S=1792, H=16, Hkv=4, D=128
logical_tiles=224 >= NUM_SMS
```

采集：

```text
ncu sections:
  LaunchStats
  WarpStateStats
  SpeedOfLight
```

关键指标：

```text
metric                                             ws_tma_v     overlap_local_p
duration_us                                        113.50       135.68
SM throughput                                      25.30%       22.64%
tensor pipe active                                 14.89%       12.39%
LSU inst pipe                                      3.87%        12.75%
MIO issued                                         2.35%        6.78%
L1/TEX throughput                                  31.09%       41.70%
L2 throughput                                      8.94%        37.75%
warp cycles / issued inst                          9.50        10.69
long scoreboard stall / issue active               3.46        5.06
wait stall / issue active                          0.84        1.41
registers/thread                                   168         168
```

判断：

```text
1. overlap_local_p 没有把 tensor core 利用率提高；
   tensor pipe active 反而从 14.89% 降到 12.39%。

2. 额外时间主要变成了 LSU / L1 / L2 / long-scoreboard；
   这非常符合 local arrays / stack spill / local-memory round-trip 的形态。

3. 也就是说，当前 overlap 不是在“藏 PV latency”，而是在用 local memory 保存：
     P(n-1)
     delta(n-1)
   然后 softmax 前后再读写它们。

4. 因为每个 CTA 只有 1 wave/SM，local-memory scoreboard 没有别的 CTA 可以遮掉；
   因此这部分依赖直接反映到 kernel latency。
```

### 直接结论

当前 `overlap_local_p` 失败的核心原因：

```text
为了拆 begin/wait，我们把 PV 的输入 P 和输出 delta 做成跨 softmax live 的 local arrays。
这些 arrays 在 lowering/ptxas 后带来明显 LSU/L1/L2/scoreboard 成本。
这部分成本大于 softmax 能隐藏的 PV tensor-core tail latency。
```

因此下一步不应该继续沿着 `T.alloc_local([112])` / `T.alloc_local([64])` 扩展，而应该改成真正的寄存器/fragment 形态：

```text
A. P:
   尽量使用 fragment/raw register ABI，而不是 T.alloc_local。

B. delta:
   使用真正的 accumulator fragment 保存 PV delta，
   或者用 helper/inline PTX 保证 WGMMA output regs 跨 softmax 不落 local memory。

C. 如果 TileLang 不允许 extern 直接写 fragment:
   就需要手写更完整的 PTX helper，
   把 PV begin、softmax overlap boundary、wait/update 的寄存器 contract 固定住。
```

## 2026-05-13 方案 2：raw CUDA/PTX kernel 路线

我们确认 `T.Kernel(prelude=...)` / `T.import_source()` 能注入 CUDA helper，但它仍然没有绕过
`T.call_extern()` 参数 lowering。也就是说：

```text
prelude/import_source:
  可以把 helper/device function 放进 generated CUDA；
  但传参仍要经过 TileLang IR/lowering。

T.alloc_fragment(...).data:
  如果作为 extern raw pointer 传入，
  仍然可能卡在 local.fragment memory info / layout inference。
```

今天的两个失败实验正好验证了这个边界：

```text
OverlapFragmentDelta:
  P 用 fragment
  delta 用 fragment
  结果：compile fail
  原因：acc_delta fragment layout / local.fragment memory info 无法由 TileLang 推断。

OverlapFragPLocalDelta:
  P 用 fragment
  delta 暂时用 local
  结果：仍然容易卡在 fragment raw pointer lowering；
  即使补 T.copy 也不是最终想要的纯 register contract。
```

因此我们选择方案 2：

```text
用 T.CUDASourceCodeKernel / raw CUDA source
直接写一个独立 CUDA/PTX kernel，
让 P、delta、acc_o 都由 CUDA local register arrays / inline PTX 明确控制，
不再让 TileLang fragment lowering 管这段 register contract。
```

### 已完成 smoke

新增了一个 raw CUDA smoke entry：

```text
GQAFwdFP8Fa3ContractRawCudaSmokeKernel
```

它目前只做：

```text
q/k/v/q_scale/k_scale/v_scale 作为输入参数进入 raw CUDA kernel
output/lse 作为 TileLang out_idx=[6,7] 输出
raw CUDA kernel 把 output 和 lse 清零
```

GPU1 nightly docker 结果：

```text
case: B=1, S=1792, H=16, Hkv=4, D=128, out=bfloat16
compile: pass
launch: pass
output shape: [1, 1792, 16, 128], max_abs=0
lse shape:    [1, 16, 1792], max_abs=0
```

这个结果说明：

```text
1. T.CUDASourceCodeKernel 的 ABI / 参数顺序 / dtype 写回已经跑通；
2. TileLang wrapper 可以负责分配 output/lse 和 launch；
3. 后续可以把真正的 FA3-contract FP8 pipeline 逐段搬进 raw CUDA。
```

### 下一步拆分

不要一口气把整个 kernel 搬完，按下面顺序推进：

```text
1. raw CUDA source 里复刻 producer/consumer 的 persistent tile mapping，
   先只写 lse/output sentinel，确认 tile ownership 和输出坐标。

2. 搬 QK：
   使用和当前 TileLang FP8 WS 相同的 Q/K shared layout 与 QK WGMMA；
   输出 acc_s 寄存器后，先把部分 debug matrix 写到 global/smem 验证。

3. 搬 P pack：
   在 raw CUDA 中把 acc_s -> FA3 P operand-A register layout，
   不经过 TileLang fragment。

4. 搬 V path：
   复用当前已正确的 TMA V -> FA3 Vt -> LDSM/STSM -> V_tc 思路，
   或直接在 raw CUDA 中发 PTX TMA + transpose helper。

5. 搬 PV overlap：
   P(n-1) register + V_tc -> delta register；
   同时 QK(n)/softmax(n) 执行；
   wait PV 后执行：
     acc_o = acc_o * ss + delta * v_scale[n-1]
   delta 必须保持为寄存器 live range，不能回到 local-memory round-trip。

6. 搬 epilogue/store：
   先复用已有 FA3 raw accumulator store helper；
   再评估是否需要 register-side output unpermute/coalesced store。
```

### raw CUDA tile-map 已完成

新增第二个 raw CUDA entry：

```text
GQAFwdFP8Fa3ContractRawCudaTileMapKernel
```

它开始复刻 FP16/FP8 WS 的 persistent tile ownership：

```text
gridDim.x = NUM_SMS
for tile = blockIdx.x; tile < logical_tiles; tile += gridDim.x:
    tile dims = [batch, heads_kv, ceildiv(seq_len, 128), groups]
    group/head mapping:
        tile_h = tile_hkv * groups + tile_g
    row range:
        row_base = tile_m * 128
```

当前 kernel 还不做 QK/PV，只写 sentinel：

```text
sentinel = b * 1000000 + hkv * 100000 + group * 10000 + m_tile
output[b, row, head, d] = sentinel
lse[b, head, row]       = sentinel
```

GPU1 nightly docker 验证：

```text
case: B=1, S=1792, H=16, Hkv=4, D=128, out=bfloat16
logical_tiles = 1 * 4 * 14 * 4 = 224 >= NUM_SMS

compile: pass
launch: pass
lse_max_err: 0
lse_samples:
  h0,row0      = 0
  h0,row127    = 0
  h0,row128    = 1
  h1,row0      = 10000
  h4,row0      = 100000
  h15,row1791  = 330013
output finite: true
```

这个结果说明：

```text
raw CUDA 路线里，TileLang wrapper -> raw CUDA kernel 的参数 ABI 已经通；
persistent tile loop / M tile / GQA head mapping 已经通；
下一步可以在这个 tile ownership 里搬 QK skeleton。
```

### raw CUDA QK scale/head mapping debug 已完成

新增第三个 raw CUDA entry：

```text
GQAFwdFP8Fa3ContractRawCudaQKFirstKeyDebugKernel
```

它仍然不是性能 kernel，只做一个可精确验证的 QK 子问题：

```text
对每个 (b, head, row):
    hkv = head // groups
    score_first_key =
        dot(q_fp8[b,row,head,:].float(),
            k_fp8[b,0,hkv,:].float())
        * q_scale[b,head,row//128]
        * k_scale[b,hkv,0]
        * (1 / sqrt(128))

写入:
    lse[b,head,row] = score_first_key
```

GPU1 nightly docker 验证：

```text
case: B=1, S=1792, H=16, Hkv=4, D=128, out=bfloat16
reference: PyTorch 直接用同一份 FP8 q/k + q_scale/k_scale 复算

max_err:  1.49e-08
mean_err: 1.49e-11

samples:
  h0,row0      =  0.0473500043
  h0,row128    = -0.0233551096
  h1,row0      = -0.0012164364
  h4,row0      =  0.0399109907
  h15,row1791  =  0.0155732511
```

这个结果说明 raw CUDA 路线里的这些 contract 已经对齐：

```text
1. q/k FP8 raw pointer reinterpret 成 __nv_fp8_e4m3 可正确转 float；
2. q layout = [B, S, H, D]；
3. k layout = [B, S, Hkv, D]；
4. GQA head mapping = hkv = head // groups；
5. q_scale index = [B, H, row//128]；
6. k_scale index = [B, Hkv, n//128]，本 debug 固定 n=0；
7. QK softmax scale = 1/sqrt(128)。
```

下一步可以把这个 debug 标量 QK 扩展成：

```text
先算一个完整 128x224 score tile 的 debug 版本；
再替换成 shared-memory + WGMMA QK。
```

### raw CUDA full QK score tile debug 已完成

新增第四个 raw CUDA debug entry：

```text
GQAFwdFP8Fa3ContractRawCudaQKTileDebugKernel
```

它把 `n_idx=0` 的完整 QK score tile 写出，输出 shape 是：

```text
scores: [B, H, M_tiles, 128, 224], dtype=float32
lse:    [B, H, S], 其中 lse[b,h,row] = scores[b,h,m_tile,row_in_tile,0]
```

计算公式：

```text
for each (b, head, m_tile, local_row, col in 0..223):
    row = m_tile * 128 + local_row
    hkv = head // groups

    scores[b,head,m_tile,local_row,col] =
        dot(q_fp8[b,row,head,:].float(),
            k_fp8[b,col,hkv,:].float())
        * q_scale[b,head,row//128]
        * k_scale[b,hkv,col//128]
        * (1 / sqrt(128))
```

GPU1 nightly docker 验证：

```text
case: B=1, S=1792, H=16, Hkv=4, D=128
scores shape: [1, 16, 14, 128, 224]
reference: PyTorch 直接复算同一份 FP8 q/k + q_scale/k_scale

score max_err: 0
score mean_err: 0
lse max_err:   0

samples:
  h0,  mt0,row0,col0     =  0.0473500043
  h0,  mt0,row0,col128   = -0.0121341385
  h0,  mt1,row0,col0     = -0.0233551096
  h1,  mt0,row3,col17    =  0.0923205316
  h4,  mt0,row5,col5     = -0.0345584750
  h15, mt13,row127,col223=  0.0806739628
```

这个结果说明：

```text
raw CUDA 路线已经对齐完整 128x224 QK score tile 的数学 contract：
  Q/K FP8 decode
  GQA head mapping
  row/col scale indexing
  BN224 K-block col indexing
  output tile coordinate
```

下一步：

```text
把 full-tile debug 的标量 dot-product 替换成:
  gmem Q/K -> shared
  shared descriptors
  FP8 WGMMA SS QK

目标不是先追速度，而是让 WGMMA 输出的 acc_s 与这个 full-tile debug scores 对齐。
```

### raw CUDA QK WGMMA tile debug 已对齐

新增第五个 raw CUDA debug entry：

```text
GQAFwdFP8Fa3ContractRawCudaQKWgmmaTileDebugKernel
```

它做的事情是：

```text
gmem Q/K FP8
  -> raw CUDA 写入 shared
  -> tl::initialize_wgmma_descriptor<1,1,64>
  -> tl::wgmma_ss FP8, m64n32k32, 7 个 N chunk 覆盖 BN224
  -> raw accumulator 按 64x32 chunk 写回 logical scores[128,224]
```

这里踩到一个很关键的点：不能把 Q/K 按 row-major 直接写到 shared。`initialize_wgmma_descriptor<1,1,64>` 对应的是 TileLang full-bank 128B swizzled shared contract；row-major fill 会出现局部正确、整体大错的现象：

```text
row-major shared fill:
  max_err  ~= 0.5068
  mean_err ~= 0.0612

典型现象:
  (row0,col0)   正确
  (row0,col16)  正确
  (row0,col128) 正确
  (row0,col1)   错
  (row1,col0)   错
```

原因不是 accumulator store，而是 Q/K shared physical layout 错。对 FP8 `[row, d]` 且 `d=128` 连续的 full-bank swizzle，物理地址需要写成：

```text
row_phase = row & 7
row_tile  = row >> 3
vec_col   = d >> 4      // 16 FP8 = 128 bit vector
vec       = d & 15

physical_index =
    row_tile * 1024
  + (row_phase * 8 + (vec_col ^ row_phase)) * 16
  + vec
```

这个公式和 TileLang lowering 里 `T.copy -> make_swizzled_layout(shared)` 的地址公式一致。修正 shared fill 后，WGMMA debug 与 scalar full-tile debug 对齐：

```text
case: B=1, S=1792, H=16, Hkv=4, D=128
scores shape: [1, 16, 14, 128, 224]
reference: RawCudaQKTileDebugKernel scalar dot

q_scale/k_scale = all ones:
  score max_err:  1.6253e-04
  score mean_err: 4.3006e-06
  lse max_err:    7.5534e-05

random q_scale/k_scale:
  score max_err:  2.3806e-04
  score mean_err: 3.7260e-06
  lse max_err:    1.1989e-04
```

这些误差来自 WGMMA FP8 累加顺序与 scalar float dot-product 的差异，数量级可接受。

当前 debug timing：

```text
RawCudaQKWgmmaTileDebugKernel:
  0.190 ms
```

这个 timing 不代表最终性能，因为 debug kernel 仍然用普通 CUDA loop 从 gmem 填 shared，并且把完整 score tile 写回 global；它只用于确认 raw CUDA/PTX QK contract。

到这里，raw CUDA 路线已经确认：

```text
1. Q/K gmem logical layout 正确；
2. Q/K scale indexing 正确；
3. QK WGMMA SS 指令序列正确；
4. FP8 full-bank 128B shared swizzle 公式正确；
5. 7 个 m64n32 chunk 拼成 BN224 的 accumulator contract 正确；
6. raw accumulator -> logical score tile 的 row/col store 正确。
```

下一步建议：

```text
把这个 QK WGMMA debug kernel 继续向真实 attention 推：
  1. 在 raw CUDA 内做 online softmax；
  2. 复用现有 fp8_acc_to_fa3_p_regs_64x224 / PTX PV helper；
  3. 先做单 stage / 无 overlap 的完整 correctness；
  4. 再搬 FP16 WS pipeline 的 producer/consumer overlap 与 barrier。
```

### raw CUDA QK WGMMA + local softmax tile debug 已对齐

新增第六个 raw CUDA debug entry：

```text
GQAFwdFP8Fa3ContractRawCudaQKWgmmaSoftmaxTileDebugKernel
```

它在上一节 QK WGMMA tile debug 的基础上继续做：

```text
QK WGMMA acc_s raw
  -> 按 64x32 chunk 写成 logical score tile
  -> 原地做 local softmax over N=224
  -> 输出 P tile: [B, H, M_tiles, 128, 224]
  -> lse 写 natural logsumexp，用于 debug 对比
```

注意：这个 kernel 目前验证的是 **单个 224-column K tile 内的 local softmax**，还不是跨所有 K/V block 的 online softmax。它的目的只是确认：

```text
QK WGMMA raw accumulator
  -> logical row/col score
  -> q_scale/k_scale/sm_scale
  -> softmax P
```

这一段数值和 layout contract 是对的。

实现细节：

```text
最开始尝试 score_smem[64,224]，但 static shared 超 48KB：
  q_smem  ~=  8KB
  k_smem  ~= 28KB
  score_smem ~= 56KB

ptxas 报：
  uses too much shared data (0x17000 bytes, 0xc000 max)

因此 debug kernel 改成：
  先把 WGMMA score 写到 output global 当 scratch
  同一个 CTA 内原地读回做 softmax
```

GPU1 nightly docker 验证：

```text
case: B=1, S=1792, H=16, Hkv=4, D=128
reference:
  PyTorch 用同一份 FP8 q/k 复算 scores[:, 0:224]
  再做 torch.softmax(scores, dim=-1)

random q_scale/k_scale:
  P max_err:     2.3134e-06
  P mean_err:    1.7997e-08
  lse max_err:   4.2915e-06
  lse mean_err:  4.4767e-07
  row_sum max:   9.5367e-07

samples:
  (h0,  mt0,row0,col0)   got 0.0045613260, ref 0.0045613735
  (h0,  mt0,row0,col1)   got 0.0043996619, ref 0.0043996782
  (h0,  mt0,row0,col128) got 0.0042325794, ref 0.0042325784
  (h1,  mt0,row3,col17)  got 0.0044901217, ref 0.0044901152
  (h15, mt13,row127,col223) got 0.0041238726, ref 0.0041238675
```

结论：

```text
raw CUDA/PTX 路线中，QK -> scale -> local softmax 的矩阵 contract 已经对齐。
```

下一步接 PV 的设计：

```text
1. QK 阶段需要 q_smem + k_smem，共约 36KB shared，已经能编译。
2. PV 阶段不能再额外放 v_smem + v_tc_smem，否则超过 48KB static shared。
3. 不能把最终 WS 主线写成“复用 k_smem 作为 v_tc_smem”：
   - K 需要 double buffer，consumer 读 K_i 做 QK 时，producer 还要 TMA 下一块 K_{i+1}；
   - 同一时间 consumer 还可能读上一块 V_i 做 PV，producer 写下一块 V_{i+1}；
   - 因此 K0/K1 和 V0/V1 的 live range 在 steady state 是交叠的，拿 K buffer 装 V 会破坏 overlap。
4. 正确的主线应该是 V 自己原位/半原位 swizzle：
   - TMA V 直接落到 V stage buffer 的 FA3 sVt/source layout；
   - 在同一个 V stage 内做 `sVt -> sV / Vtc`，最好复用同一块 dynamic shared 或同一个 V stage union；
   - K0/K1 保持专属于 QK double buffer；
   - V0/V1 保持专属于 PV double buffer；
   - output scratch 只能复用已经完成最后一次 PV 读取的 Vtc stage，不能提前占用下一轮 producer 要写的 V stage。
5. 如果只是做 single-tile PV correctness debug，可以临时用 k_smem 或 global scratch 省 shared；但这个 kernel 必须标成 debug，不用于性能/overlap 判断。
```

### 2026-05-14 V 原位 swizzle 实验

新增 probe：

```text
_probe_fp8_v_fa3_inplace_bn224.py
```

单次 transform 比较：

```text
out-of-place:
  sVt -> Vtc

in-place:
  buf as sVt -> same buf as Vtc
```

GPU1 nightly docker：

```text
fa3_v_inplace_bn224: equal_to_out_of_place=True mismatched_bytes=0
```

所以 `fp8_transpose_v_128x224_fa3_src_ldsm_stsm(buf, buf)` 这个 helper 本身不是马上错的；它在单 tile raw-byte 层面和 out-of-place 相同。

但是完整 WS kernel 里直接 alias `v_vt_smem_i == v_tc_smem_i` 仍然不正确：

```text
GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVInplaceKernel
  status: experimental
  test status: xfail
```

纠正后的判断：

```text
V stage 不能在 TMA 完成时 release/publish。

producer 侧必须是:
  TMA V -> sVt source
  raw_full
  sVt -> Vtc reorder
  v_full / publish to consumer

consumer 侧必须是:
  wait v_full
  PV reads Vtc
  v_empty / release stage
```

所以当前 full WS inplace 失败不能直接说明“原位破坏 overlap”。它更可能说明实验 kernel 没有把 `raw_full`、`v_full`、`v_empty` 三个节点的语义重新收紧。真正的主线应该是先做 correctness-only serialized inplace：

```text
producer: TMA -> raw_full -> reorder -> v_full
consumer: wait v_full -> PV -> v_empty
```

如果这条通过，再考虑如何把 producer 的下一块 TMA 与 consumer 的 PV overlap 回来。

### 2026-05-14 BN224 V-inplace barrier lowering check

新增 dump 脚本：

```text
/home/ga/TileOPs/_dump_fp8_bn224_inplace_barriers.py
```

导出的文件：

```text
/home/ga/TileOPs/.tmp/fp8_bn224_inplace_barriers/bn224_tma_v_out_of_place.cu
/home/ga/TileOPs/.tmp/fp8_bn224_inplace_barriers/bn224_tma_v_inplace.cu
/home/ga/TileOPs/.tmp/fp8_bn224_inplace_barriers/bn224_tma_v_out_of_place.ptx
/home/ga/TileOPs/.tmp/fp8_bn224_inplace_barriers/bn224_tma_v_inplace.ptx
```

对照结果：

```text
out-of-place:
  TMA -> v_vt_smem_0/1 at byte offsets 114688 / 143360
  LDSM/STSM -> v_tc_smem_0/1 at byte offsets 57344 / 86016

inplace:
  TMA -> v_stage_0/1 at byte offsets 57344 / 86016
  LDSM/STSM -> same v_stage_0/1
```

inplace producer 的 publish 顺序在 lowered CUDA/PTX 里是正确的：

```text
wait v_empty_slot, only after the first two fills
TMA V -> stage
wait v_raw_full
fp8_transpose_v_128x224_fa3_src_ldsm_stsm(stage, stage)
arrive v_full
```

per-slot empty 的 phase 公式 lower 成：

```cpp
v_empty_0[0].wait((((gi_vp >> 1) + 1) & 1));
v_empty_1[0].wait((((gi_vp >> 1) + 1) & 1));
```

这和“同一 slot 每两次 `gi_vp` 复用一次”的语义一致，不是当前看到的错误点。

consumer side 也没有提前 release V：

```text
v_full.wait
fp8_pv_ptx_unit_accumulate_fa3_raw_64x128x224
  -> wgmma.commit_group
  -> wgmma.wait_group 0
  -> wgmma.fence
v_empty_0/1.arrive
```

所以这次 lowering 对照的结论是：

```text
barrier 三节点语义目前看起来成立；
full WS inplace xfail 更像 V bytes 在真实 WS stage 下仍未被隔离验证，
或者是 alias/shared offset 与 helper layout 的组合问题。
```

下一步更有信息量的实验：

```text
1. standalone V probe 覆盖:
   n offset = 0, 224, 448, 672
   shared base = 57344, 86016

2. full WS debug dump:
   producer reorder 后把 v_stage bytes 写到 global scratch
   对比 out-of-place 的 v_tc_smem bytes

3. 如果 V bytes 完全相同，再看 PV helper 的 descriptor/base pointer 是否因 alias 后的 allocation 改变而读错。
```

### 2026-05-14 BN224 V-inplace raw-byte follow-up

后续补了 raw physical dump，结论比上一段更明确：

```text
单 stage / base=0:
  inplace == out-of-place

full-kernel exact offsets:
  out-of-place:
    Vt  stage0/1 = 114688 / 143360
    Vtc stage0/1 =  57344 /  86016

  inplace:
    Vt/Vtc stage0/1 = 57344 / 86016

结果:
  stage0 base=57344 同址变换不等价
  stage1 base=86016 同址变换等价
```

raw dump 结果：

```text
fa3_v_inplace_bn224_exact_offsets:
  equal_to_out_of_place=False
  mismatched_bytes=4096

stage0:
  n=0/224/448/672 都有 mismatch

stage1:
  n=0/224/448/672 都 byte-exact
```

这意味着当前 full WS inplace 数值错误的主嫌不再是 barrier contract，而是：

```text
fp8_transpose_v_128x224_fa3_src_ldsm_stsm(src, dst)
在 src == dst 且 base == 57344 时，会破坏 FA3 Vtc physical bytes。
```

因此 FA3 contract 路线下，V 的安全版本仍是：

```text
TMA V -> Vt high buffer
LDSM/STSM -> Vtc low buffer
PV reads Vtc
```

如果继续压 shared memory，只能做受控实验，例如：

```text
stage0 out-of-place + stage1 inplace
```

先确认 full kernel 错误是否完全来自 stage0，然后再考虑显式 scratch/phase split，而不是直接 identical src/dst。

### 2026-05-14 BN224 V-inplace producer barrier 修复

进一步实验推翻了 “找安全 base” 这个方向。正确解释是：

```text
当前 FA3 V transpose helper 的 out-of-place contract 是成立的；
直接令 src == dst 时，需要额外保证 producer WG 内部 read-before-write。
```

验证 helper：

```text
normal inplace:
  mismatch

noinline + memory fence:
  mismatch

barrier_each_iter:
  byte-exact

load_all_store_all:
  byte-exact
```

因此问题不是简单编译器 inline/alias 优化，而是同一 micro-iteration 内：

```text
部分线程先 STSM 写 dst
覆盖了其他线程尚未 LDSM 读取的 src
```

修法是在 helper 内部每轮 LDSM 后插入 producer WG barrier：

```ptx
bar.sync 15, 128
```

这个 barrier 只等 producer 的 128 个线程；不能换成 `__syncthreads()`，否则 full WS 里 consumer 线程不进入 helper，可能死锁。

新 kernel：

```text
GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVInplaceBarrierKernel
```

full WS correctness：

```text
out diff max = 0
lse diff max = 0
```

对应测试：

```text
tests/ops/attention/test_gqa_fp8.py::test_gqa_fwd_fp8_bn224_tma_v_inplace_barrier_matches_out_of_place
```

nightly Docker / GPU1 结果：

```text
1 passed
```

小型 timing：

```text
out_of_place      43.082 us
inplace_barrier  41.983 us
```

下一步应该用正式 benchmark 覆盖更长 seq_len，确认这个 producer WG barrier 版本的收益是否稳定。

### 2026-05-14 Barrier-inplace bench / pipeline 结构

bench 新增 impl：

```text
tileops_ws_fa3_contract_ptx_acc_bn224_ws_tma_v_inplace_barrier
```

quick bench 结果，`B=1,H=32,Hkv=8,D=128`：

```text
S=896:
  out-of-place      0.054630 ms
  inplace_barrier  0.054976 ms

S=1792:
  out-of-place      0.284726 ms
  inplace_barrier  0.289733 ms

S=3584:
  out-of-place      0.894529 ms
  inplace_barrier  0.867116 ms
```

lowered CUDA 显示当前三 WG pipeline 的结构已经存在：

```text
producer WG:
  TMA K[n]
  if n > 0:
    TMA V[n-1]
    V reorder
    release v_full[n-1]

consumer WG:
  wait K[n]
  QK
  softmax
  wait V[n]
  PV
  release v_empty[n]
```

也就是说，V producer 路径现在确实被放在独立 producer WG 中，和两个 consumer WG 分支并行推进。是否 “latency 真藏住了” 还需要下一步 timing instrumentation，看：

```text
wait k_full
QK
softmax/rescale
wait v_full
PV
```

尤其是 `wait v_full` 的时长。如果它接近 0，说明 producer 的 V TMA/reorder 跟得上；如果它仍然大，说明 pipeline 形状虽然对了，但 V readiness 仍然晚于 consumer 需求。

### 2026-05-14 latency hiding 复核：当前正确版还不是 FP16 式 QK/PV ping-pong

今天重新看了 NCU 和 lowered CUDA/PTX。注意：

```text
long scoreboard 高不一定是关键问题。
producer WG 有 128 threads，但实际工作线程更少，空等线程会污染 warp-state。
FP16 WS kernel 也会展示类似现象。
```

更可靠的判断是 tensor pipe active 和 WGMMA issue/wait 顺序。

`B=1,S=3584,H=32,Hkv=8,D=128`，nightly Docker / GPU1：

```text
impl                         gpu time      tensor active
FA3                          273 us        48.56%
TMA-V out-of-place           677 us        19.52%
TMA-V inplace-barrier        705 us        18.78%
overlap_frag_p_local_delta   775 us        17.24%
```

关键结论：

```text
当前正确版 tensor active 明显低于 FA3；
inplace V 只减少 producer shared movement，不能解决 TC issue/wait 编排；
已有 overlap 实验版还没有把 tensor active 拉起来。
```

当前正确主线 consumer loop 实际是：

```text
wait k_full[n]
issue QK[n]
wait QK[n]
softmax P[n]
wait v_full[n]
issue/accumulate PV[n]
release v_empty[n]
```

这只说明：

```text
producer 可以提前准备 V；
但 consumer 内部没有做到 PV[n-1] 与 QK[n] / softmax[n] 的 ping-pong。
```

我们真正要对齐 FP16 WS 的形态：

```text
have packed P[n-1]
issue QK[n]
issue PV[n-1]
wait QK[n]
softmax P[n] while PV[n-1] is flying
wait/update PV[n-1]
pack P[n] for next PV
```

下一步不继续盯 long scoreboard，而是新建 clock/timing kernel，量化：

```text
consumer wait k_full
QK issue/wait
softmax
wait v_full
PV issue/wait residual
producer V TMA + reorder
```

### 2026-05-14 clock64 分段结果：V ready 和 PV 都暴露在关键路径

已新建独立 timing kernel：

```text
/home/ga/TileOPs/_bench_fp8_bn224_pipeline_clock.py
```

它测的是当前正确主线：

```text
BN224 + TMA-V out-of-place + FA3 PV accumulate
B=1, S=3584, H=32, Hkv=8, D=128
nightly Docker / GPU1
```

consumer WG1 steady-state：

```text
wait_k_full                98.8 cycles
qk_issue_call            2075.7 cycles
wait_qk                    11.0 cycles
scale_scores              296.7 cycles
softmax+copy_ss          1976.1 cycles
wait_v_full              1175.9 cycles
pv_accumulate_call       2153.3 cycles
total                    7787.5 cycles
```

producer V path：

```text
wait_v_empty               71.1 cycles
v_tma_to_raw_ready        850.2 cycles
v_reorder                 695.5 cycles
v_total_to_full          1635.4 cycles
```

这说明现在的 FA3-contract 正确路径里，矩阵 layout contract 本身已经能跑通，但 pipeline contract 还没有完全对齐 FA3/FP16 WS：

```text
1. QK 显式 wait 剩余很小，wait_qk ~= 11 cycles。
2. V full wait 仍有 ~= 1176 cycles，producer V release 没有完全提前到 consumer 使用前。
3. PV accumulate ~= 2153 cycles，发生在 softmax 之后，没有被下一轮 QK/softmax 盖住。
```

因此下一步不是再随机改 swizzle，而是保留现有正确 layout contract，单独新建 delayed-PV/pingpong timing kernel：

```text
目标：
  P[n-1] 已经 pack 成 PV operand A；
  QK[n] 和 PV[n-1] 同轮 issue；
  softmax/pack P[n] 覆盖 PV[n-1] 的飞行时间；
  最后 wait/update PV[n-1]。

验证：
  用同一套 clock64 分段确认 pv wait residual 是否下降；
  再用 NCU 看 tensor active 是否从当前约 19% 往 FA3 的约 49% 靠近。
```

### 2026-05-14 Pingpong kernel 入口：contract 对齐，但实现还不够轻

已新增：

```text
GQAFwdFP8Fa3ContractPtxAccBN224WsPingpongKernel
```

它代表我们真正想要的 FA3/FP16-WS-style consumer contract：

```text
P[n-1] 已经 pack 成 PV operand A
PV[n-1] 与 QK[n] 同轮飞行
softmax(P[n]) 产出 ss[n]
wait/update PV[n-1]:
  acc_o = acc_o * ss[n] + delta[n-1] * v_scale[n-1]
pack P[n] 给下一轮 PV
```

正确性结果：

```text
S=896 smoke correctness: passed
S=896  ping-base cos = 0.99950695, max diff = 0.001808, lse diff = 0
S=3584 ping-base cos = 0.99932766, max diff = 0.001236, lse diff = 0
```

速度结果：

```text
B=1,S=3584,H=32,Hkv=8,D=128

serial TMA-V: 1.006023 ms
pingpong:     1.234084 ms
```

所以现在的状态是：

```text
layout contract: 仍然沿用 FA3 contract，正确性可接受；
pipeline contract: 已经有 delayed-PV 形状；
实现代价: 目前太重，尤其是 P fragment 和 acc_delta_local 跨 helper/loop 的保存。
```

下一步要查的不是 “P/V swizzle 是否又错了”，而是：

```text
1. p_pv_frag 是否保持在寄存器，还是被 TileLang 降成 local memory movement；
2. acc_delta_local 是否真的能作为 WGMMA C register fragment 跨 softmax live；
3. begin/wait_update 拆 helper 后，编译器有没有插入额外 load/store/fence；
4. 如果有，就把这一段继续收敛到更完整的 inline CUDA/PTX helper。
```
