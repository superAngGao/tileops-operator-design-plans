# 推理量化功能补齐计划

## 1. 计划定位

这份文档用于讨论 TileOps 是否需要系统补齐推理量化相关算子，以及第一批工作如何落地。它不是 manifest spec，也不冻结任何新 op 的公开 API；真正的 op contract 仍以后续 tracking issue、manifest PR 和实现 PR 为准。

这里的基本判断是：TileOps 当前已经有若干 FP8 相关能力，但量化支持还没有系统覆盖到现有推理算子 family 的主路径。我们不需要再抽出一个独立的 `quantized_inference` family；量化更像横切的 dtype / layout / scale contract，应该落回 TileOps 已有的大类中：

```text
attention: quantized Q/K/V, FP8 KV cache, quantized decode / prefill
moe: quantized routed/shared experts, activation quantization, scale-aware grouped GEMM
gemm / grouped_gemm: FP8 W8A8, weight-only W4A16 / W8A16, packed layout
elementwise / reduction / layout helpers: amax, quantize, dequantize, pack, scale update
normalization / activation: fused norm/activation + quantize where it is a real serving boundary
```

而主流推理需求正在变成一组完整路径：

```text
quantize / pack / scale metadata
  -> quantized GEMM / MoE experts / attention KV path
  -> fused dequant / activation / residual / output
  -> paged KV cache quantization and decode reuse
```

这份计划的目标不是把所有量化算法都搬进 TileOps，也不是建立一个和 `attention` / `moe` / `gemm` 平行的新 family；目标是在现有 family 中补齐推理热路径里的量化功能，并统一 scale metadata、packed layout、reference、test、benchmark 和 model-shaped workloads。

## 2. 模型侧需求

主流开源模型已经把量化推理放在核心部署路径中，不再只是离线压缩选项。

| 模型 / 系列 | 公开量化需求 | 对 TileOps 的启发 |
| --- | --- | --- |
| Qwen3 / newer Qwen MoE releases | 官方 Qwen benchmark 覆盖 BF16、FP8、GPTQ、AWQ；SGLang 路径中 AWQ 使用 awq_marlin backend，Transformers 路径存在 GPTQ/AWQ/FP8 性能差异。Qwen3.5 的具体 checkpoint / quantization contract 需要在 tracking issue 中单独核验 | 需要 weight-only INT4/INT8 GEMM、FP8 path、AWQ/GPTQ packed layout、Marlin-style serving workloads |
| Llama 4 | Scout BF16 权重可通过 on-the-fly INT4 放入单 H100；Maverick 同时发布 BF16 和 FP8 quantized weights | 需要 FP8 weight/activation GEMM、on-the-fly INT4 quant/dequant、MoE-friendly quantized experts |
| DeepSeek-V3 | 技术报告强调 FP8 mixed precision、1x128 activation tile scaling、128x128 weight block scaling，以及 MoE activation dispatch 的 FP8 化 | 需要 fine-grained FP8 quantize/dequant、group-scale GEMM、MoE activation quantization、scale-aware accumulation |
| Kimi K2 / Kimi K2 Thinking / Kimi3 方向 | Kimi K2 是 1T MoE、384 experts、top-8 routing；K2 Thinking 使用 QAT 后的 native INT4 weight-only quantization，主要应用于 MoE components。Kimi3 的具体 quantization contract 需要在 tracking issue 中单独核验 | 需要 INT4 MoE expert GEMM、compressed-tensors / packed-weight loader boundary、per-expert scale metadata、shared expert 与 routed expert 的混合路径 |
| vLLM / SGLang serving ecosystem | vLLM 支持 FP8、INT8、INT4、AWQ、GPTQ、Marlin、FP8 KV cache；KV cache quantization 支持 per-tensor 和 per-head scale | TileOps 需要对齐 serving engine 里已经稳定的 quantization contracts，而不是只提供孤立 fp8 cast kernel |

这张表只用于建立需求边界。具体模型版本、checkpoint 格式、engine 后端和 quantization metadata 需要在每个 tracking issue 中单独确认。

### 2.1 GQA / KV Cache 需求应按模型和后端来拆

GQA 的量化需求不能只写成“支持 FP8 KV cache”。对不同模型和后端来说，真正的 kernel 边界不同：

| 模型 / 后端 | Attention / cache 形态 | KV cache 量化需求 | 计算形态判断 | 对 TileOps 的第一批含义 |
| --- | --- | --- | --- | --- |
| Qwen3 / Qwen3.5 GQA / MoE 系列 | 标准 GQA + 长上下文；vLLM 已用 Qwen3 / Qwen3.5 做 FP8 KV-cache + FP8 attention 验证 | FP8 KV cache 是核心需求；需要 per-tensor 起步，并保留 per-head / calibrated scale 扩展 | 目标不是先把 K/V 解回完整 bf16 cache，而是在 attention kernel 中在线应用 scale；Hopper / Blackwell 后端可走 FA3 / FlashInfer 风格的 FP8 attention 路径 | 优先做 `GQADecodeWithFP8KVCacheFwdOp` 的 dense contract，再扩展 paged scale metadata |
| Llama 3.x / Llama 4 GQA 路径 | 标准 GQA，head_dim 通常更贴近主流 FA3 / FlashInfer 路径 | FP8 KV cache 用于降低 decode bandwidth / cache footprint | 长上下文 decode 主要是 bandwidth-sensitive；短上下文可能被固定 overhead 抵消 | 作为 baseline workload：dense non-paged、paged、per-tensor/per-head scale 都应覆盖 |
| vLLM / SGLang serving path | 引擎级 KV cache dtype / page table / calibration policy | 需要对齐 serving engine 的 scale granularity、paged metadata 和 skip-layer policy | 旧的 storage-only FP8 cache 会在线 dequant；新的 FP8 attention 路径把 scale 应用融合到 attention consumer 中 | TileOps op 不管理 allocator，只暴露 quantized KV read / scale-aware attention compute |
| DeepSeek-V3 / V4 MLA 路径 | MLA / latent cache，不是标准 GQA K/V cache | 需要 quantized latent/cache/state contract，而不是直接复用 GQA FP8 KV cache contract | FP8 mixed precision 与 MLA cache 压缩是相关但不同的问题 | 放到 MLA / attention owning family 的后续 tracking issue，不塞进第一批 GQA contract |
| Kimi KDA / FlashMLA / long-context 路径 | KDA / FlashMLA / prefill cache，和标准 GQA cache 不同 | 需要 KDA state / prefill-cache 的量化设计；非标准 backend 可能更依赖 calibration | vLLM 的经验显示非标准 attention backend 上 uncalibrated FP8 KV cache 可能有系统性精度下滑 | 作为 `linear_attn` / `sequence_modeling` 或 MLA-style cache quantization 的后续目标 |

因此，GQA 这一条线第一批不是“恢复历史上的 paged prefill FP8 KV cache fused op”，而是把现有能力重新组织成更干净的 contract：

```text
1. KV cache write:
   K/V -> K_fp8/V_fp8 + k_scale/v_scale

2. Dense GQA decode cache read:
   q(fp16/bf16) + K/V fp8 cache + scale -> o

3. Paged GQA decode cache read:
   q + fp8 pages + scale metadata + block_table + cache_seqlens -> o
```

计算上有三种层级，需要在 tracking issue 中明确：

```text
storage-only fallback:
  读 fp8 K/V -> 在线 dequant 到 fp16/bf16 中间值 -> 普通 attention math
  优点是 contract 简单；缺点是 latency gain 主要来自 cache footprint / bandwidth，不一定来自 Tensor Core compute。

fused scale-aware attention:
  读 fp8 K/V + scale -> 在 QK / PV 路径中直接应用 scale
  这是 decode hot path 更应该追求的形态。

full FP8 Q/K/V Tensor Core prefill:
  q/k/v 全部 fp8，QK / PV 更接近 FP8 Tensor Core 路径
  适合 prefill / large tile compute；它不是 KV cache decode contract 的替代品。
```

## 3. TileOps 当前能力与按 Family 拆分的缺口

### 3.1 已有能力

当前仓库中已经能看到这些量化相关能力：

| 能力 | 位置 | 状态判断 |
| --- | --- | --- |
| FP8 activation / KV-shaped quantization | `tileops/ops/fp8_quant.py`, `tileops/kernels/fp8_quant.py` | 有专用 op，输入形状是 `(batch, seq_len_kv, kv_group, index_dim)`，能产出 fp8 tensor 和 per-token/per-group scale；但还没有成为 attention KV cache 的完整 manifest contract |
| FP8 LightingIndexer | `tileops/ops/fp8_lighting_indexer.py` | attention indexing 相关局部能力 |
| FP8 GQA compute path | `GroupedQueryAttentionPrefillFP8TensorCoreFwdOp` in `tileops/manifest/attention.yaml`, `tileops/kernels/attention/gqa_fwd_fp8.py` | 已进入 attention manifest；底层 FP8 GQA kernel 能直接读取 fp8 Q/K/V 和 Q/K/V scale |
| MoE prepare/finalize abstraction | `tileops/ops/moe/abc.py` | 已预留 quantization / EP dispatch 位置，但默认实现仍是 pass-through |
| FP8 elementwise / selection benchmarks | `benchmarks/ops/bench_elementwise_fp8.py`, `benchmarks/ops/bench_independent_elementwise.py` | 可作为 dtype coverage，不是完整推理量化路径 |

### 3.2 按现有 Family 拆分的缺口

这一节是后续工作的主轴。它不是简单列“还缺什么”，而是把每个现有 family 需要补的量化能力拆成：

```text
current anchor -> missing contract -> first PR boundary -> not in scope
```

这样可以避免两个常见误区：

```text
1. 把量化重新包装成一个独立 family；
2. 直接从模型名跳到巨大 fused kernel，跳过 manifest / scale / layout / reference contract。
```

#### 3.2.1 总览

| Family / 方向 | 当前 anchor | 主要缺口 | 下一步动作 |
| --- | --- | --- | --- |
| `elementwise` / `reduction` / layout helpers | `FP8QuantOp`、fp8 elementwise benchmark | amax / scale update / dequant / pack metadata 不统一 | 先补共享 helper contract |
| `gemm` / `grouped_gemm` | `tileops/ops/gemm.py`、`tileops/ops/grouped_gemm.py`、benchmarks | 缺 manifest；缺 FP8 W8A8 / W4A16 layout contract | 先补 GEMM manifest，再挂量化 variant |
| `moe` | routing、permute、unpermute、grouped expert abstraction | 缺 quantized expert weights、activation quant、per-expert scale | 先做 compositional quantized expert path |
| `attention` | FP8 GQA compute path；FP8 LightingIndexer；KV-shaped `FP8QuantOp`；bf16/fp16 decode / paged decode | 底层已有 fp8 K/V + scale 读取能力；缺 decode/cache-facing manifest contract、paged scale metadata、reference / benchmark coverage | 按模型需求先做 dense FP8 KV cache read contract，再进 paged；storage-only dequant 只作为 fallback / reference |
| `normalization` / activation | RMSNorm / LayerNorm / fused add norm | 缺 norm/activation + quantize epilogue 的稳定边界 | 等 GEMM / MoE 明确消费后再抽 |
| `linear_attn` / `sequence_modeling` | GDN/KDA/GLA/SSD-style kernels in progress | 缺 quantized state/KV/cache representation | 先记录模型 contract，暂不抢第一批实现 |

这个拆分的核心是：量化功能不单独形成顶层 family。每个 op 回到它的计算语义 family 中，量化只作为 dtype / layout / metadata contract 出现在 signature、shape rules、dtype combos、workloads 和 roofline 中。

#### 3.2.2 Shared helpers：`elementwise` / `reduction` / layout

这是所有量化路径的共同前置层。TileOps 已经有 `FP8QuantOp` 和 fp8 elementwise benchmark，但它们还没有变成可被 `attention`、`moe`、`gemm` 共同引用的量化 tensor contract。

需要补齐的不是“更多 cast kernel”，而是这些可复用语义：

```text
amax / scale update:
  x -> scale_out

scale-aware quantize:
  x, optional scale_in -> x_quant, scale_out

scale-aware dequantize:
  x_quant, scale -> x_dequant

layout / pack helper:
  dense quantized tensor -> packed tensor, only when a consumer family needs it
```

第一批 PR 边界：

```text
FP8QuantizeFwdOp
FP8DequantizeFwdOp
scale shape rules
rounding / clamp / saturation policy
nonfinite policy
reference implementation and diagnostics
```

暂不提前做一个万能 `QuantizedTensorPackFwdOp`。pack / transpose / swizzle 只有在 `gemm`、`moe` 或 KV cache 的第一批 layout 确认需要复用时才独立出来；否则它应该先留在 owning consumer family 里。

#### 3.2.3 GEMM：`gemm` / `grouped_gemm`

这是推理量化最直接的计算主路径。Qwen AWQ/GPTQ/FP8、Llama INT4/FP8、Kimi INT4、DeepSeek FP8 最终都会落到 dense GEMM、linear 或 grouped GEMM 的 scale-aware compute 上。

当前问题分两层：

```text
manifest gap:
  TileOps 有 gemm / grouped_gemm op 和 benchmark，
  但 manifest 目录中还没有 gemm.yaml / grouped_gemm.yaml。

quantization gap:
  缺 FP8 W8A8 Linear；
  缺 W4A16 / W8A16 weight-only Linear；
  缺 AWQ / GPTQ / Marlin-like / compressed-tensors packed layout contract；
  缺 scale / zero metadata 的 shape rules。
```

第一批 PR 边界应该先从普通 GEMM manifest 开始：

```text
GemmFwdOp
GroupedGemmFwdOp
```

然后再挂量化 variant：

```text
FP8LinearW8A8FwdOp
WeightOnlyLinearW4A16FwdOp
QuantizedGroupedGemmFwdOp, if MoE needs a reusable grouped compute contract
```

不进入 TileOps kernel scope 的部分：

```text
AWQ / GPTQ calibration
QAT training
checkpoint conversion
engine loader policy
```

TileOps 接收的是已经量化好的权重、scale、zero-point 和 packed layout，并负责 serving-time fused dequant / matmul / accumulation / output。

#### 3.2.4 MoE：`moe`

MoE 是最需要量化补齐的 consumer family。TileOps 已有 routing、permute、unpermute、grouped expert abstraction，以及 prepare/finalize 中的 quantization hook；但 expert compute 主路径仍然主要是 bf16/fp16。

MoE 的量化缺口不能只写成“加一个 int4 GEMM”。真正的 op contract 需要同时覆盖：

```text
routed expert weights:
  w_gate_up, w_down 的 quantized layout

shared expert:
  是否共享同一 quantization policy，是否和 routed expert fused

activation quantization:
  dispatch 后 hidden_states 是否在线量化

per-expert scale metadata:
  scale shape 是否按 expert / group / block 组织

combine:
  topk_weights 和 unpermute / combine 是否和 dequant output fused
```

第一批应先做 compositional baseline，而不是直接写巨大 fused kernel：

```text
permute
  -> optional activation quantize
  -> quantized grouped GEMM
  -> activation
  -> quantized grouped GEMM
  -> unpermute / topk weighted combine
```

这个 baseline 的价值是把 correctness、layout、scale 和 model-shaped workload 先固定住。等它稳定后，再决定哪些边界值得 fused：

```text
dequant + grouped GEMM
activation + quantize
down projection + topk weighted combine
shared expert + routed expert
```

#### 3.2.5 Attention：`attention`

Attention 里已经有 FP8 GQA compute path，也有普通 GQA/MHA decode 和 paged decode。缺口不在“GQA 能不能读 fp8 K/V”这个底层 kernel 能力，而在 decode / KV cache 语义是否已经形成稳定的公开 op contract。

从模型需求看，这一节只把标准 GQA / MQA 的 KV cache 作为第一批目标。DeepSeek MLA、Kimi KDA / FlashMLA 这类非标准 cache/state 路径需要单独设计，不能强行塞进 GQA FP8 KV cache contract。

这里需要修正两个容易误解的点。

第一，TileOps 已经有 KV-shaped 的 `FP8QuantOp`。它的输入是：

```text
input_tensor: [batch, seq_len_kv, kv_group, index_dim]
```

输出是：

```text
scale_tensor:  [batch, seq_len_kv, kv_group]
output_tensor: [batch, seq_len_kv, kv_group, index_dim], dtype=float8_e4m3fn
```

第二，GQA FP8 compute kernel 已经能消费 fp8 Q/K/V 和 scale metadata；`GroupedQueryAttentionPrefillFP8TensorCoreFwdOp` 是当前公开 manifest anchor，底层 `gqa_fwd_fp8.py` 里有 `k_scale` / `v_scale` 读入和应用路径。因此 attention family 的缺口不是“完全没有 KV quantization kernel”，也不是“核心 GQA kernel 不能读 fp8 K/V”，而是这些已有能力还没有被提升成完整 KV cache decode / serving contract：

```text
1. 缺 attention-family manifest entry / ownership；
2. 缺 decode/cache-facing wrapper，把 FP8QuantOp 的 KV-shaped 输出接到 GQA FP8 compute；
3. 缺 dense decode / q_len=1 / cache-read 场景的 reference、shape rules 和 benchmark；
4. 缺 paged KV cache 下 scale metadata 与 block_table 的形状约定；
5. 缺 decode output correctness against bf16/fp16 KV reference；
6. storage-only dequant helper 只在 reference/debug 或非 fused fallback 需要时独立成 op，热路径优先走 fused scale-aware consumer。
```

第一批要把 KV cache 的 scale contract 和 decode/cache-facing wrapper 讲清楚：

```text
FP8KVCacheQuantizeFwdOp:
  K/V -> K_fp8, V_fp8, K_scale, V_scale
  may be a manifest-level wrapper or extension around the existing FP8QuantOp

GQADecodeWithFP8KVCacheFwdOp:
  Q + FP8 K/V cache + scale metadata -> O
  在 shape contract 覆盖 decode/cache-read 场景时复用现有 FP8 GQA compute
```

这里的计算目标不是先把整个 KV cache 解量化回 bf16/fp16 再调用普通 decode。第一版可以保留 storage-only fallback 作为 reference，但服务热路径应该是：

```text
Q(fp16/bf16) + K/V(fp8) + scale
  -> scale-aware QK / PV
  -> O(fp16/bf16)
```

完整 FP8 Q/K/V Tensor Core prefill 是另一条已有能力，适合 large prefill tile；它不能替代 decode 侧的 FP8 KV cache contract。

落地顺序应该是：

```text
1. dense / non-paged FP8 KV cache reference
2. decode attention correctness against bf16/fp16 KV reference
3. per-tensor and per-head scale
4. paged cache layout and block table integration
```

不应该把 paged allocator、block table policy、serving engine memory manager 塞进 TileOps quantization op。TileOps 只负责 quantized KV read / dequant / attention compute 这个 kernel boundary。

#### 3.2.6 Normalization / activation

Norm 和 activation 不应该为了“量化完整性”单独膨胀。它们只有在成为真实 serving boundary 时才进入第一批。

典型进入条件：

```text
RMSNorm output is immediately consumed by W8A8 Linear;
MoE activation output must be quantized before second expert GEMM;
residual + norm + quantize fusion is a measured bottleneck.
```

否则，普通 RMSNorm / LayerNorm / activation 仍归普通 `normalization` 或 elementwise family，不因为输入输出 dtype 是 fp8/int8 就变成新的量化算子线。

#### 3.2.7 Linear attention / sequence modeling

GDN、KDA、GLA、SSD、MLA 这类 state / cache / recurrence 算子未来会有量化需求，但它们的量化 contract 比 dense attention 更容易和模型实现绑定：

```text
state dtype
state scale granularity
low-rank cache layout
prepare / replay 中的 scale placement
decode state update correctness
```

因此这一类先不抢第一批实现。第一批只做需求记录和 contract inventory；等具体模型的 public implementation、checkpoint metadata、serving runtime contract 稳定后，再在 owning family 下设计量化 variant。

## 4. 共享 Metadata / Layout Contract

虽然不新建 `quantized_inference` family，但必须有跨 family 共享的 metadata 约定。否则 `attention`、`moe`、`gemm` 会各自发明不兼容的 scale shape 和 packed layout。

### 4.1 Quantized Tensor Contract

每个量化 op 的 manifest 都应显式写出：

```text
data dtype: fp8 / int8 / int4 / future fp4
scale dtype: fp32 or implementation-specific
zero_point: optional, dtype and shape explicit
group_shape: tensor / head / token_x_channel / block / expert / group_size
axis semantics: which tensor axes the scale groups cover
pack_order: physical packing order for int4/int8/fp8 block layouts
layout: dense, packed, paged, expert-major, token-major, marlin-like, compressed-tensors-like
compute_dtype: accumulator dtype
out_dtype: output dtype
rounding / clamp / saturation policy
```

### 4.2 Scale Contracts

```text
per-tensor
per-head
per-token x 128-channel
128x128 block
per-group weight scale, group_size = 32 / 64 / 128
per-expert scale
per-page or per-cache-block scale, if paged KV uses it
```

### 4.3 Family Ownership

| Op / 能力 | Manifest family 归属 | 说明 |
| --- | --- | --- |
| `FP8QuantizeFwdOp`, `FP8DequantizeFwdOp` | `elementwise` / helper track | 负责 scale-aware cast / dequant，不代表完整推理算子 |
| `QuantizedTensorPackFwdOp` | layout helper or owning consumer family | 只有被 Linear / MoE / KV cache 明确复用时才进入 |
| `FP8LinearW8A8FwdOp`, `WeightOnlyLinearW4A16FwdOp` | `gemm` | 线性层和 GEMM 语义，不单独建 quantized family |
| `QuantizedGroupedGemmFwdOp` | `grouped_gemm` | 可被 MoE 复用的低层 grouped GEMM 语义 |
| `QuantizedMoeExpertsFwdOp` | `moe` | MoE expert compute 的 dtype/layout variant；可调用 grouped GEMM，但 op 语义属于 MoE |
| `FP8KVCacheQuantizeFwdOp`, `GQADecodeWithFP8KVCacheFwdOp` | `attention` | `FP8KVCacheQuantizeFwdOp` 可以复用 / 包装现有 KV-shaped `FP8QuantOp`；decode/cache-facing op 应复用现有 FP8 GQA compute 的 fp8 K/V + scale 读取能力，并补齐 cache 语义 |
| Quantized MLA / KDA / linear attention | owning attention / linear-attention family | 等模型 contract 稳定后按原算子 family 扩展 |

### 4.4 不进入范围

| 类别 | 当前归类 | 不纳入原因 |
| --- | --- | --- |
| 普通 fp8 elementwise | `elementwise` | 只是 dtype coverage，不表达推理量化 contract |
| 普通 bf16/fp16 attention | `attention` | 没有 quantized tensor / scale metadata，仍是普通 attention |
| 普通 MoE routing / permute | `moe` | routing 本身不是量化；只有 quantized expert compute / activation dispatch 才进入 |
| 离线量化算法本身 | external tooling | AWQ/GPTQ calibration、QAT training、compressed-tensors conversion 不是 TileOps kernel |
| engine-level memory policy | runtime / serving engine | KV cache block allocation、TP/EP placement、checkpoint loader policy 不是 TileOps kernel |

## 5. 第一批核心目标：按现有 Family 落地

第一批不追求覆盖所有量化算法，而是沿着 3.2 的 family 拆分补齐最能支撑现代推理路径的关键缺口。每条线都落到现有 family 中。

### 5.0 优先级总表

| 优先级 | Family | 第一批目标 | 为什么先做 |
| --- | --- | --- | --- |
| P0 | `elementwise` / `reduction` / layout helpers | scale-aware FP8 quantize/dequantize、amax/scale update、必要的 pack contract | 其他 family 都依赖统一 scale 和 layout 语义 |
| P0 | `gemm` / `grouped_gemm` manifest | 先补 `GemmFwdOp` / `GroupedGemmFwdOp` manifest，再挂量化 variant | 没有 manifest source of truth，量化 Linear / MoE expert PR 会失去归属 |
| P1 | `gemm` | FP8 W8A8 Linear、W4A16 weight-only Linear | 这是 Qwen/Llama/Kimi/DeepSeek 量化推理最直接的 GEMM 热路径 |
| P1 | `moe` | quantized routed experts / shared expert contract | MoE 是 Qwen3、DeepSeek-V3、Kimi K2/K3、Llama 4 Maverick 的共同热点 |
| P2 | `attention` | 把已有 FP8 quant 和 FP8 GQA compute 收敛成 KV cache decode contract，再补 paged scale metadata | 长上下文 decode 的内存收益很明确，但 paged cache contract 需要和 serving engine 更仔细对齐 |
| P2 | `linear_attn` / `sequence_modeling` | quantized MLA/KDA/linear-attention state path | 模型需求明确，但 checkpoint / runtime contract 还需要逐个确认 |

### 5.1 `elementwise` / `reduction` / layout helpers：Scale-Aware Quantize / Dequantize

目标是把现有 `FP8QuantOp` 从局部形状扩展为 manifest 可描述的基础 helper。它们不是新 family，而是服务 `attention`、`moe`、`gemm` 的 shared primitives。

候选接口：

```text
FP8QuantizeFwdOp(
    x:        Tensor[..., H],
    scale:    Optional[Tensor[scale_shape]],
    mode:     enum("online_absmax", "provided_scale"),
    group:    enum("tensor", "head", "token_x_128", "block_128x128"),
    dtype:    enum("float8_e4m3fn", "float8_e5m2"),
) -> (x_fp8, scale_out)

FP8DequantizeFwdOp(
    x_fp8: Tensor[...],
    scale: Tensor[scale_shape],
    group: same_group_contract,
) -> x
```

第一版重点：

```text
per-tensor
per-head
per-token-per-128-channel
per-128x128 block, if used by GEMM path
```

这条线直接服务 DeepSeek-style fine-grained FP8、vLLM FP8 KV cache，以及现有 FP8 GQA / FP8Quant 的统一化。`QuantizedTensorPackFwdOp` 只有在 Linear / MoE / KV cache 的第一批 layout 明确需要时才进入，不提前创造宽泛 pack op。

### 5.2 `gemm` / `grouped_gemm`：Quantized Linear / GEMM

这是推理量化的主算子。第一批至少需要两个方向：

```text
FP8LinearW8A8FwdOp
WeightOnlyLinearW4A16FwdOp
```

候选接口：

```text
FP8LinearW8A8FwdOp(
    x_fp8:       Tensor[M, K],
    w_fp8:       Tensor[N, K] or packed Tensor,
    x_scale:     Tensor[x_scale_shape],
    w_scale:     Tensor[w_scale_shape],
    bias:        Optional[Tensor[N]],
    out_dtype:   enum("float16", "bfloat16", "float32"),
) -> y: Tensor[M, N]

WeightOnlyLinearW4A16FwdOp(
    x:           Tensor[M, K],                 # fp16/bf16
    w_packed:    Tensor[packed_shape],          # int4 packed
    w_scale:     Tensor[group_scale_shape],
    w_zero:      Optional[Tensor[group_shape]],
    layout:      enum("awq", "gptq", "marlin_like", "compressed_tensors"),
    out_dtype:   same_as(x),
) -> y: Tensor[M, N]
```

第一版不把 AWQ / GPTQ 校准算法放进 TileOps；只支持已经量化好的权重格式和高性能 fused dequant GEMM。

### 5.3 `moe`：Quantized Experts

MoE 是 Qwen3、DeepSeek-V3、Kimi K2/K3 和 Llama 4 Maverick 的共同热点。TileOps 已有 MoE routing / permute / expert abstraction，但 expert compute 仍主要是 bf16/fp16。

候选接口：

```text
QuantizedMoeExpertsFwdOp(
    hidden_states: Tensor[T, H],                    # fp16/bf16 or fp8
    w_gate_up:     QuantizedExpertWeights,
    w_down:        QuantizedExpertWeights,
    topk_ids:      Tensor[T, K_top],
    topk_weights:  Tensor[T, K_top],
    scales:        ExpertScaleMetadata,
    layout:        enum("fp8_w8a8", "int4_w4a16"),
) -> output: Tensor[T, H]
```

第一版需要覆盖：

```text
Qwen3-30B-A3B / 235B-A22B style E=128, top_k=8
DeepSeek-V3 style E=256, top_k=8
Kimi K2 style E=384, top_k=8, one shared expert
Llama 4 Maverick style E=128 MoE
```

这里的关键不是先写一个巨大 fused kernel，而是先把 quantized expert weight layout、scale metadata、shared/routed expert boundary 和 benchmark workloads 写清楚。

实现上需要区分两层：

```text
compositional baseline:
  permute -> optional activation quantize -> quantized grouped GEMM
  -> activation -> quantized grouped GEMM -> unpermute / topk weighted combine

fused target:
  fuse dequant + grouped GEMM + activation + topk weighted combine where it is profitable
```

### 5.4 `attention`：FP8 KV Cache and Decode Attention

长上下文 decode 的核心压力来自 KV cache。vLLM 已经把 FP8 KV cache 作为 serving 功能，支持 per-tensor 和 per-head scale。TileOps 这里不是从零开始：已有 KV-shaped `FP8QuantOp`，也已有能读取 fp8 Q/K/V 与 Q/K/V scale 的 FP8 GQA compute kernel。第一批工作的重点是把这些能力收敛成 attention family 中稳定的 decode / cache contract。

候选接口：

```text
FP8KVCacheQuantizeFwdOp(
    k:      Tensor[B, S, H_kv, D],
    v:      Tensor[B, S, H_kv, D],
    mode:   enum("per_tensor", "per_head"),
) -> (k_fp8, v_fp8, k_scale, v_scale)

# This can reuse or wrap the existing KV-shaped FP8QuantOp.

GQADecodeWithFP8KVCacheFwdOp(
    q:          Tensor[B, H, D],
    k_cache:    Tensor[paged_or_dense_fp8_layout],
    v_cache:    Tensor[paged_or_dense_fp8_layout],
    k_scale:    Tensor[scale_shape],
    v_scale:    Tensor[scale_shape],
    block_table: Optional[Tensor],
) -> o: Tensor[B, H, D]

# dense shape contract 覆盖 decode/cache-read 场景时，
# 复用现有 FP8 GQA compute。
```

第一版可以先做 dense reference / non-paged smoke，再进入 paged cache contract。不要把 paged allocator 和 block table policy 塞进 quantization op；TileOps 只负责可测的 quantized KV read / scale application / attention compute。是否需要独立 dequant op，取决于 reference/debug/fallback 是否需要；主热路径不应该为了形式完整而强制拆出 dequant。

## 6. 第二批目标

第一批稳定后，再考虑这些方向：

| 方向 | 进入条件 |
| --- | --- |
| INT8 W8A8 Linear | 有明确 serving demand 和参考后端；否则优先级低于 FP8 W8A8 与 INT4 W4A16 |
| NVFP4 / MXFP4 / microscaling | Blackwell / GB200 路径成熟，TileLang / PyTorch dtype contract 可表达 |
| Quantized LM Head | vocabulary projection 成为 memory bottleneck，且 checkpoint / engine 有对应 packed layout |
| Quantized MLA / Linear Attention | DeepSeek MLA、Kimi KDA、Qwen linear attention 的量化路径稳定后再单独设计 |
| INT4 KV Cache | 需要明确准确率 reference 和 serving engine contract；第一版只做 FP8 KV cache |
| fused quantize + transpose / pack | GEMM 或 KV path 确认存在独立瓶颈后再抽出 helper |

## 7. Manifest 与实现节奏

### Phase 0：Inventory and Shared Contract

新增本文档，明确量化能力在现有 family 中的归属、模型需求、缺口和共享 metadata / layout contract。同步检查现有 FP8Quant、FP8 GQA、MoE abstraction 应该补哪些 manifest 条目，以及是否需要调整 source metadata。

### Phase 1：Helper Primitive Manifest

在 `elementwise` / `reduction` / layout helper 方向新增或补齐基础 op：

```text
FP8QuantizeFwdOp
FP8DequantizeFwdOp
QuantizedTensorPackFwdOp, only if required by first Linear / MoE / KV layout
```

保持 `spec-only`，先收敛：

```text
scale shape rules
dtype_combos
grouping modes
rounding / clamp policy
nonfinite policy
reference implementation
```

### Phase 2：`gemm` / `grouped_gemm` Quantized Linear

先补 GEMM / grouped GEMM manifest 的普通 forward entry，再新增量化 variant：

```text
GemmFwdOp
GroupedGemmFwdOp
FP8LinearW8A8FwdOp
WeightOnlyLinearW4A16FwdOp
QuantizedGroupedGemmFwdOp, if MoE needs a reusable grouped GEMM contract
```

第一版只支持 inference forward，不支持 training backward。每个 op 必须明确：

```text
packed weight layout
scale / zero metadata
group size
accumulation dtype
output dtype
supported K/N alignment
model-shaped workloads
```

### Phase 3：`moe` Quantized Experts

在现有 MoE family 上扩展，不另起模型专属 op。

```text
QuantizedMoeExpertsFwdOp
QuantizedSharedFusedMoeFwdOp, if shared expert fusion is required
```

关键 gate：

```text
top_k=8
E=128/256/384
H=3072/7168
expert intermediate size from Qwen / DeepSeek / Kimi / Llama workloads
per-expert scale metadata
packed routed expert weights
shared expert precision policy
```

### Phase 4：`attention` FP8 KV Cache

先做：

```text
FP8KVCacheQuantizeFwdOp
GQADecodeWithFP8KVCacheFwdOp contract / wrapper
```

再评估：

```text
paged cache variant
per-head scale
mixed kv precision
int4 / fp4 future extension
```

### Phase 5：Release Hardening

对所有量化 op 建立统一 correctness 与 benchmark policy：

```text
absolute / relative error
cosine similarity only as diagnostic, not sole gate
calibration-free synthetic tests
model-shaped random tests
scale edge cases
overflow / underflow / saturation checks
throughput and bandwidth roofline
```

## 8. 测试与 Benchmark 原则

### 8.1 正确性

量化 op 不应只用 cosine similarity 作为 gate。建议至少记录：

```text
max_abs
max_rel with near-zero handling
mean_abs
cosine similarity
nonfinite count
saturation count for quantize ops
scale range
```

对 weight-only GEMM / MoE：

```text
reference = dequantize weights to fp32/bf16, then torch matmul
compare output after fused dequant GEMM
cover pathological scales and zero groups
```

对 KV cache：

```text
reference = fp16/bf16 KV attention
compare decode output, not just dequantized K/V
cover long-context scale drift
```

### 8.2 Benchmark

Benchmark 必须区分：

```text
quantization time
packed-weight load time
steady-state GEMM / MoE time
KV cache write time
KV cache decode read time
end-to-end fused path
```

不要把离线 AWQ/GPTQ calibration 算进 TileOps kernel benchmark。TileOps 只 benchmark serving-time kernels。

## 9. References

- Qwen3 Speed Benchmark, including BF16, FP8, GPTQ and AWQ results: https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html
- Llama 4 Scout / Maverick model card and quantization notes: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E
- Meta Llama 4 release blog: https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- DeepSeek-V3 Technical Report, FP8 training and fine-grained quantization: https://arxiv.org/html/2412.19437v1
- MoonshotAI Kimi K2 repository: https://github.com/moonshotai/kimi-k2
- Kimi K3 technical blog, KDA / QAT / prefill-cache context: https://www.kimi.com/blog/kimi-k3
- Kimi K2 Thinking model card, native INT4 quantization: https://huggingface.co/moonshotai/Kimi-K2-Thinking
- vLLM quantization documentation: https://docs.vllm.ai/en/latest/features/quantization/
- vLLM quantized KV cache documentation: https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
- vLLM FP8 KV-cache and attention quantization blog: https://vllm.ai/blog/2026-04-22-fp8-kvcache
- AWQ paper: https://arxiv.org/abs/2306.00978
