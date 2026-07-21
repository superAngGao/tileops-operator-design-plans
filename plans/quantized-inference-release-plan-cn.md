# 推理量化功能补齐计划

## 1. 计划定位

这份文档用于讨论 TileOps 是否需要系统补齐推理量化相关算子，以及第一批工作如何落地。它不是 manifest spec，也不冻结任何新 op 的公开 API；真正的 op contract 仍以后续 tracking issue、manifest PR 和实现 PR 为准。

这里的基本判断是：TileOps 当前已经有若干 FP8 相关能力，但量化支持还没有系统覆盖到现有推理算子 family 的主路径。我们不需要再抽出一个独立的 `quantized_inference` family；量化更像横切的 dtype / layout / scale contract，应该落回 TileOps 已有的大类中：

```text
attention: quantized Q/K/V, FP8 KV cache, quantized decode / prefill
moe: quantized routed/shared experts, activation quantization, scale-aware grouped GEMM
gemm / grouped_gemm: FP8 W8A8, weight-only W4A16 / W8A16, packed layout
quantization helpers: amax, quantize, dequantize, scale update, consumer-owned pack/layout helpers
normalization / activation: fused norm/activation + quantize where it is a real serving boundary
```

主流推理需求不是一条单一数据流，而是几条共享 scale / layout / metadata 约定的并行路径：

```text
Quantized weights:
  checkpoint metadata / pack
    -> weight-only or W8A8 GEMM
    -> bias / activation / residual / output cast

Activations:
  amax / scale generation / quantize
    -> GEMM or routed expert compute

K/V cache:
  quantized cache write
    -> paged FP8 KV storage
    -> scale-aware decode attention
```

这份计划的目标不是把所有量化算法都搬进 TileOps，也不是建立一个和 `attention` / `moe` / `gemm` 平行的新 family；目标是在现有 family 中补齐推理热路径里的量化功能，并统一 scale metadata、packed layout、reference、test、benchmark 和 model-shaped workloads。

## 2. 模型侧需求

多款主流开放权重模型已经提供官方量化 checkpoint 或量化部署方案，vLLM、SGLang 等开源 serving engine 也把量化纳入主要支持矩阵。量化因而不再只是离线模型压缩步骤，而是会进入 checkpoint、加载、内存布局和推理 kernel 的端到端部署路径。

| 模型 / 系列 | 公开量化需求 | 对 TileOps 的启发 |
| --- | --- | --- |
| Qwen3 / newer Qwen MoE releases | 官方 Qwen benchmark 覆盖 BF16、FP8、GPTQ、AWQ；SGLang 路径中 AWQ 使用 awq_marlin backend，Transformers 路径存在 GPTQ/AWQ/FP8 性能差异 | 需要 weight-only INT4/INT8 GEMM、FP8 path、AWQ/GPTQ packed layout、Marlin-style serving workloads |
| Qwen3.5 hybrid models | Qwen3.5-35B-A3B 是 Gated DeltaNet + 周期性 Gated Attention 的混合架构；FP8 KV cache 只覆盖 attention layers，DeltaNet recurrent state 量化属于另一类 contract | attention layers 走 GQA/KV-cache contract；DeltaNet state 量化归 `linear_attn` / `sequence_modeling` |
| Llama 4 | Scout BF16 权重可通过 on-the-fly INT4 放入单 H100；Maverick 同时发布 BF16 和 FP8 quantized weights | 需要 FP8 weight/activation GEMM、on-the-fly INT4 quant/dequant、MoE-friendly quantized experts |
| DeepSeek-V3 | 发布权重使用 FP8 E4M3、128x128 weight block scaling；技术报告还描述了训练系统中的 1x128 activation scaling、MoE dispatch activation FP8 和特殊 accumulation strategy | 推理 GEMM 先对齐发布权重和 activation scale contract；训练系统里的 MoE dispatch/accumulation 作为参考，不直接升格为通用 inference contract |
| Kimi K2 / Kimi K2 Thinking | Kimi K2 是 1T MoE、384 experts、top-8 routing；K2 Thinking 使用 post-training QAT 后的 native INT4 weight-only quantization，主要应用于 MoE components | 需要 INT4 MoE expert GEMM、compressed-tensors checkpoint schema / packed-weight compute boundary、per-expert scale metadata、shared expert 与 routed expert 的混合路径 |
| Kimi3 direction | Kimi3 使用 KDA 和 AttnRes；其 recurrent/state 量化不同于标准 GQA KV cache | KDA state / prefill-cache quantization 应归 `linear_attn` / `sequence_modeling`，不是第一批 GQA KV-cache contract |
| vLLM / SGLang serving ecosystem | vLLM 支持 FP8、INT8、INT4、AWQ、GPTQ、Marlin、FP8 KV cache；vLLM 当前支持 per-tensor KV scale，并在 Flash Attention backend + llm-compressor calibration 下支持 per-attention-head scale。SGLang 的 scale granularity 需按版本单独核验 | TileOps 需要对齐 serving engine 里已经稳定的 quantization contracts，而不是只提供孤立 fp8 cast kernel |

这张表只用于建立需求边界。具体模型版本、checkpoint 格式、engine 后端和 quantization metadata 需要在每个 tracking issue 中单独确认。

### 2.1 GQA / KV Cache 需求应按模型和后端来拆

GQA 的量化需求不能只写成“支持 FP8 KV cache”。对不同模型和后端来说，真正的 kernel 边界不同：

| 模型 / 后端 | Attention / cache 形态 | KV cache 量化需求 | 计算形态判断 | 对 TileOps 的第一批含义 |
| --- | --- | --- | --- | --- |
| Qwen3 GQA / MoE 系列 | 标准 GQA + 长上下文；vLLM 已用 Qwen3 做 FP8 KV-cache + FP8 attention 验证 | FP8 KV cache 是核心需求；需要 per-tensor 起步，并在 backend / calibration contract 明确时保留 per-attention-head scale 扩展 | 目标不是先把 K/V 解回完整 bf16 cache，而是在 attention kernel 中在线应用 scale；Hopper / Blackwell 后端可走 FA3 / FlashInfer 风格的 FP8 attention 路径 | 优先扩展 `GroupedQueryAttentionDecodeWithKVCacheFwdOp` 的 FP8 dense cache variant，再扩展 paged decode scale metadata |
| Qwen3.5 hybrid models | Gated DeltaNet + 周期性 Gated Attention；FP8 KV cache 只对应 attention layers，不对应 DeltaNet recurrent state | attention layers 需要 GQA KV cache 量化；DeltaNet state 需要独立 state/cache quant contract | 不能把整模写成标准 GQA；attention 和 recurrent state 属于不同 operator family | GQA KV cache 只覆盖 attention 子路径；DeltaNet 进入 `linear_attn` / `sequence_modeling` tracking issue |
| Llama 3.x / Llama 4 GQA 路径 | 标准 GQA；具体 workload 应明确 head_dim、num_q_heads、num_kv_heads、page size 和 cache layout | FP8 KV cache 用于降低 decode bandwidth / cache footprint | 长上下文 decode 主要是 bandwidth-sensitive；短上下文可能被固定 overhead 抵消 | 作为 baseline workload：dense non-paged、paged、per-tensor scale 都应覆盖；per-attention-head scale 需要显式 backend contract |
| vLLM / SGLang serving path | 引擎级 KV cache dtype / page table / calibration policy | 需要对齐 serving engine 的 scale granularity、paged metadata 和 skip-layer policy；vLLM 的 per-head KV scale 当前有 backend/calibration 限制，SGLang 需单独核验 | 旧的 storage-only FP8 cache 会在线 dequant；新的 FP8 attention 路径把 scale 应用融合到 attention consumer 中 | TileOps op 不管理 allocator，只暴露 quantized KV read / scale-aware attention compute |
| DeepSeek-V3 / V4 MLA 路径 | MLA / latent cache，不是标准 GQA K/V cache | 需要 quantized latent/cache/state contract，而不是直接复用 GQA FP8 KV cache contract | FP8 mixed precision 与 MLA cache 压缩是相关但不同的问题 | 放到 MLA / attention owning family 的后续 tracking issue，不塞进第一批 GQA contract |
| Kimi K2 / K2 Thinking | MLA latent cache + MoE；不是 KDA | 需要 MLA-style latent cache / state 量化，和 MoE INT4 expert GEMM | 不复用标准 GQA FP8 KV cache contract | MLA cache 进入 attention/MLA tracking issue；MoE INT4 进入 `moe` / `grouped_gemm` |
| Kimi3 KDA 路径 | KDA recurrent/sequence state，不是标准 GQA，也不是 FlashMLA | 需要 KDA state / prefill-cache 的量化设计 | 非标准 backend 可能更依赖 calibration 与模型实现约定 | 作为 `linear_attn` / `sequence_modeling` 的后续目标 |
| FlashMLA | DeepSeek MLA kernel/backend reference，不是模型架构类别 | 可作为 MLA-style cache/kernel 的参考实现 | 不应和 Kimi KDA 混为一类 | 只在 MLA tracking issue 中作为 backend/reference 讨论 |

因此，GQA 这一条线第一批不是“恢复历史上的 paged prefill FP8 KV cache fused op”，而是把现有能力重新组织成更干净的 contract：

```text
1. KV cache write:
   K/V -> K_fp8/V_fp8 + k_scale/v_scale

2. Dense GQA decode cache read:
   q(fp16/bf16) + K/V fp8 cache + scale -> o

3. Paged GQA decode cache read:
   q + fp8 pages + scale metadata + block_table + cache_seqlens -> o
```

计算实现应按两个维度区分：第一，FP8 KV cache 是否在 attention consumer 中融合读取和 scale application；第二，QK / P*V 使用 A16 还是 FP8 Tensor Core compute。tracking issue 至少应明确以下实现形态：

```text
materialized-dequant fallback:
  读 fp8 K/V cache -> 独立 dequantize 路径生成完整 fp16/bf16 K/V 中间 tensor
  -> 调用普通 attention kernel。

  该路径 contract 和 reference 实现较简单，适合作为 correctness fallback；
  但因为会写出并重新读取完整 A16 K/V，中间流量可能抵消大部分 decode latency 收益。
  主要价值是减小持久化 cache footprint，而不是 serving hot-path 性能。

fused dequantized attention with A16 compute:
  attention kernel 直接读取 fp8 K/V 和 scale；
  在 register / shared-memory tile 内完成反量化；
  随后使用 fp16/bf16 QK 和 P*V compute；
  不生成完整的 A16 K/V global-memory intermediate。

  该路径不一定使用 FP8 Tensor Core，但仍能保留 FP8 KV cache 带来的 HBM bandwidth 收益，
  是现实的 decode hot-path implementation。

fused FP8 Tensor-Core attention:
  attention kernel 直接消费 fp8 K/V；
  Q 可以保持 fp16/bf16 后在 kernel 内量化，也可以由上游直接提供 fp8 Q。
  QK 和/或 P*V 是否采用 FP8 Tensor Core、softmax intermediate dtype、
  scale granularity、scale placement 和 accumulation policy，均由具体 backend contract 决定。

  large-tile prefill 通常更容易发挥 FP8 Tensor Core 吞吐，但 FP8 Tensor-Core compute
  也可以用于 decode。现有 dense FP8 Q/K/V prefill kernel 不能替代 paged KV-cache decode
  contract；decode 路径仍需单独定义 page layout、block table、cache sequence length、
  scale metadata 和 q_len 边界。
```

无论选择哪种 fused implementation，scale metadata 都必须进入 contract，而不是留给 kernel 私下解释：

```text
scale granularity:
  per-tensor / per-attention-head / per-page / per-block / per-token-block

scale indexing dimension:
  scale 如何随 batch、head、page、KV block 或 token block 索引

scale placement:
  K scale 如何进入 QK score，是否能与 softmax scale 合并；
  V scale 如何进入 P*V partial accumulation，何时做 output rescale。

softmax interaction:
  如果 K scale 随 KV block 变化，score tile 必须在 online softmax 前应用对应 scale；
  如果 V scale 随 KV block 变化，PV contribution 必须在 online accumulation 合并前应用对应 scale。
```

### 2.2 GEMM / Linear 需求应按权重量化格式来拆

Linear / GEMM 是大多数模型量化最直接的落点。这里不要先抽象成“quantized matmul”，而要先区分权重格式、activation scale、packed layout 和 serving-time compute：

| 模型 / 后端 | 公开量化形态 | GEMM 侧核心需求 | 对 TileOps 的第一批含义 |
| --- | --- | --- | --- |
| Qwen3 / Qwen serving benchmark | BF16、FP8、GPTQ、AWQ 都是公开 benchmark 口径 | 需要 FP8 W8A8、W4A16 / W8A16 weight-only、AWQ/GPTQ packed layout 对齐 | 复用已有 `GemmFp8Op`；新增或扩展 weight-only GEMM physical layout contract |
| Llama 4 Scout / Maverick | Scout 提到 on-the-fly INT4；Maverick 发布 FP8 quantized weights | FP8 Linear 和 INT4 weight-only Linear 都是直接需求 | FP8 走已有 `GemmFp8Op` 扩展；INT4 走 `GemmWeightOnlyOp` / weight-only variant |
| DeepSeek-V3 | 发布权重说明强调 FP8 E4M3、weight 128x128 block scaling；技术报告还描述 activation 1x128 scaling 和训练系统中的 accumulation strategy | 需要 fine-grained FP8 GEMM：activation scale 与 weight block scale 都进入 compute contract | 第一批 FP8 GEMM 不能只支持 per-tensor scale，应把 1x128 / 128x128 作为目标 shape；训练系统里的 promotion/accumulation 作为参考 |
| Kimi K2 Thinking / Kimi3 方向 | Kimi K2 Thinking 使用 native INT4 weight-only，主要用于 MoE components | INT4 weight-only 更可能先落在 expert GEMM / grouped GEMM，而不是普通 dense Linear | weight-only GEMM contract 要能被 MoE expert path 复用 |
| vLLM / SGLang / Marlin-style backends | serving 生态需要同时对齐 AWQ/GPTQ 等量化方案、compressed-tensors 等 checkpoint metadata/schema，以及 Marlin 等 packed-weight compute backend | TileOps 不做离线量化或 loader policy，但需要接收已量化 packed weight、scale、zero metadata | loader / calibration 在 scope 外；kernel contract 只覆盖 serving-time unpack / scale application / matmul |

因此，GEMM / Linear 的第一批不是一个万能接口，而是三条稳定线：

```text
1. FP8 W8A8 GEMM / Linear:
   x_fp8, w_fp8, x_scale, w_scale -> y
   manifest: output_dtype, accumulation_dtype, accumulation_policy, scale_granularity

2. Weight-only INT4 / INT8 Linear:
   x(fp16/bf16), w_packed_int4/int8, scale, optional zero -> y
   manifest: packed_layout, scale_shape, zero_policy, output_dtype, accumulation_dtype

3. Quantized grouped GEMM:
   expert-routed activations + packed expert weights + per-expert/block scales -> expert output
```

TileOps 应该把 packed layout、scale shape、zero-point policy、accumulation dtype、output dtype 写进 manifest。AWQ/GPTQ 的离线校准与量化、QAT 训练、checkpoint conversion 和 loader policy 不进入 TileOps kernel scope。

### 2.3 MoE 需求应按 expert compute 路径来拆

MoE 的量化不是“routing 量化”，而是 routed token 到 expert GEMM 之间的 serving hot path。Qwen MoE、DeepSeek MoE、Kimi K2 / Kimi K2 Thinking、Llama 4 Maverick 都让 MoE 成为第一批需要重点覆盖的 consumer family。

| 模型 / 后端 | MoE 形态 | 量化压力点 | 对 TileOps 的第一批含义 |
| --- | --- | --- | --- |
| Qwen3 MoE / newer Qwen MoE | routed experts + top-k combine；公开 benchmark 覆盖多种量化格式 | expert weight-only / FP8 GEMM、dispatch 后 activation quant、top-k weighted combine | 先做 compositional path：permute -> quantized grouped GEMM -> activation -> quantized grouped GEMM -> unpermute/combine |
| DeepSeek-V3 MoE | 发布权重和技术报告提供 FP8 MoE 参考；训练系统中的 activation dispatch / combine policy 不能直接等同于 inference checkpoint contract | routed activations、per-expert / block scale、grouped GEMM accumulation | 需要 scale-aware grouped GEMM；是否在 EP dispatch 前量化 activation 应由目标 serving backend 和 benchmark 决定 |
| Kimi K2 / Kimi K2 Thinking / Kimi3 方向 | 大 MoE，top-k routing；Thinking 公开 native INT4 weight-only | INT4 expert weights、shared/routed expert policy、packed loader boundary | 优先扩展 `MoeGroupedGemmNopadFwdOp` / grouped GEMM 的 weight-only expert compute，再评估 fused expert variant |
| Llama 4 Maverick | MoE + FP8 quantized weights | FP8 expert weights 与 standard serving GEMM 路径复用 | MoE path 应能复用 FP8 GEMM / grouped GEMM contract |

第一批 MoE 不直接追求巨大 fused kernel。合理起点是把下面这些边界固定住：

```text
activation after routing:
  hidden_states_perm -> optional activation quantize

expert compute:
  grouped GEMM with per-expert / per-block scale metadata

combine:
  expert output -> weighted combine by topk_weights -> token order output
```

后续是否 fuse `dequant + grouped GEMM + activation + combine`，应由 benchmark 决定。

### 2.4 Activation / Norm / Helper 需求应按 consumer 来拆

Activation / norm / helper 不应该因为“量化”两个字就膨胀成独立主线。它们只有在下游 consumer 需要稳定 scale contract 时才成为第一批目标。

| Consumer | 为什么需要 helper | TileOps 应补的能力 |
| --- | --- | --- |
| FP8 GEMM / DeepSeek-style FP8 path | activation 需要按 1x128 或类似 tile 生成 scale | `amax + scale update + quantize`，并记录 nonfinite / saturation diagnostics |
| Weight-only GEMM | activation 通常保持 fp16/bf16；被量化的是 weight | weight unpack、dequant 和 scale application 应融合进 GEMM mainloop 或 partial-accumulation 路径；epilogue 主要处理 bias、activation、residual、output cast |
| MoE expert path | dispatch 后 token layout 改变，activation quantize 的位置会影响性能 | routed activation quantize helper、per-expert scale metadata |
| FP8 KV cache | KV write 需要 scale 生成；decode read 需要 scale-aware consumer | KV-shaped quantize、paged scale metadata；独立 dequant 只作为 reference/debug |
| RMSNorm / fused add norm 后接 W8A8 Linear | norm output 立刻进入 FP8 GEMM 时，norm+quantize 可能成为真实 serving boundary | 先记录为 optional fusion，等 GEMM consumer 稳定后再决定是否抽 `RMSNormQuantizeFwdOp` |

第一批 helper 的边界应更像 shared contract，而不是模型算子：

```text
amax / scale update
scale-aware FP8 quantize
scale-aware dequantize for reference/fallback
packed-layout helper only when GEMM/MoE/KV cache has confirmed reuse
```

这也是为什么本文不新建 `quantized_inference` family：helper 是横切能力，真正的 op 归属仍由 consumer family 决定。

## 3. 按现有 Family 划分的 Op / Dispatch 目录

这一章不再把“FP8/INT4”直接当作新 op 名字。TileOps 的 public op 名称应由数学语义和持久化数据结构决定；dtype、scale granularity、packed layout 和 backend 通常是同一个 op 的 variant。只有输入输出语义、状态/cache 写入方式、或持久化 physical layout contract 发生实质变化时，才新增 op。

| 变化 | 默认处理 |
| --- | --- |
| `GemmOp` 增加 FP8 dtype / scale | 扩展已有 op 或使用已有 `GemmFp8Op` |
| `GroupedQueryAttentionPrefillFwdOp` 增加 FP8 Tensor Core backend | 扩展已有 prefill op 的 backend / dtype variant |
| dense KV cache 变成 paged KV cache | 可以是不同 op，因为持久化 cache layout 和索引语义不同 |
| fp16 weight 变成 packed INT4 weight | 可以新增 weight-only op，因为物理输入 contract 根本不同 |
| MoE routed expert GEMM 增加 FP8 / INT4 expert weights | 优先扩展 `MoeGroupedGemmNopadFwdOp` 或 grouped GEMM variant |
| standalone debug dequant / materialized fallback | 优先作为 reference utility；只有 runtime 会调度时才进入 public manifest |

每个 family 下面分两层写：

```text
public op contract:
  稳定接口、输入输出、shape、metadata、是否扩展现有 op。

quantized dispatch / kernel variants:
  fp8 tensor core、storage-only fp8 cache、on-the-fly dequant、
  weight-only mainloop、scale indexing、packed layout 等实现路径。
```

### 3.1 Quantization Helpers：`quantization` plus owning-family helpers

上游 `main` 已经有 `quantization.yaml` 和 `FP8QuantOp`。这说明 narrow helper 可以有自己的 helper family；但完整推理算子仍应回到 consumer family，例如 `attention`、`gemm`、`moe`。

#### Public op contracts

| Op / contract | 输入 | 输出 | 量化语义 | 当前状态 |
| --- | --- | --- | --- | --- |
| `FP8QuantOp` | `input_tensor: [B, S, G, D]`, A16/FP32 | `output_tensor: fp8`, `scale_tensor: fp32` | KV-shaped / index-shaped FP8 quantize helper | 已有 manifest 和实现；第一批应明确 scale algebra、scale 是 dequant scale 还是 quant scale |
| `FP8Dequantize` reference utility | `x_fp8`, scale metadata | `x_a16` | correctness/debug/fallback；不默认作为 serving hot path public op | 可先作为 test/reference utility，不急着进 manifest |
| `Amax / scale update` helper | `x`, scale policy | `amax`, scale metadata | dynamic activation scale generation | 只有 W8A8 GEMM/MoE consumer 稳定后再决定是否独立成 op |

#### Quantized dispatch / kernel variants

| Variant | 作用 | 是否第一批 |
| --- | --- | --- |
| online amax + quantize fused path | `x -> fp8 + scale`，减少中间读写 | 是，作为 `FP8QuantOp` 扩展方向 |
| provided-scale quantize path | 复用已有 scale metadata | 是 |
| standalone dequant path | reference / debug / fallback | 低优先级 |
| generic pack helper | 只有两个以上 consumer 复用相同 physical layout 时再抽 | 暂不作为候选 op |

### 3.2 GEMM / Grouped GEMM：`gemm`

上游 `main` 已经有 `GemmOp`、`GemmFp8Op` 和 `GroupedGemmOp`。因此第一批不是补普通 GEMM manifest，也不是新增 `FP8LinearW8A8FwdOp`；重点是扩展已有 FP8 GEMM contract，并新增真正有不同 physical input contract 的 weight-only path。

#### Public op contracts

| Op / contract | 输入 | 输出 | 量化语义 | 相关模型 / 后端 | 当前状态 |
| --- | --- | --- | --- | --- | --- |
| `GemmOp` | A16 `a`, A16 `b` | A16 `d` | 普通 dense GEMM anchor | all dense Linear | 已有 manifest 和实现 |
| `GemmFp8Op` | `a_fp8`, `b_fp8`, `scale_a`, `scale_b`, optional bias | A16 `d` | FP8 W8A8 GEMM；已有 per-tensor / block-128 scale shape | DeepSeek-V3 FP8, Llama 4 Maverick FP8, Qwen FP8 path | 已有 manifest 和实现；待补 accumulation policy、scale algebra、更多 dtype/layout/backend coverage |
| `GroupedGemmOp` | grouped A16 activations / weights + group metadata | grouped output | 普通 grouped GEMM anchor | MoE expert GEMM | 已有 manifest 和实现；待补 quantized variants |
| `GemmWeightOnlyOp` or `GemmWeightOnlyW4A16Op` | A16 activation, packed INT4/INT8 weight, scale, optional zero metadata | A16 output | weight unpack / dequant / scale application 融进 mainloop 或 partial accumulation | Kimi K2 Thinking INT4, Llama 4 Scout INT4, AWQ/GPTQ/Marlin-like serving | 新增候选；必须先定义 physical storage layout |
| `GroupedGemmFp8` / `GroupedGemmWeightOnly` variant | routed/grouped activations, grouped quantized expert weights, scale metadata | grouped expert output | MoE 低层 expert compute | Qwen3 MoE, DeepSeek-V3 MoE, Kimi K2, Llama 4 Maverick | 优先作为 `GroupedGemmOp` 或 MoE grouped GEMM 的 tagged variant 设计 |

#### Quantized dispatch / kernel variants

| Variant | Contract 要点 |
| --- | --- |
| FP8 per-tensor W8A8 | `scale_a.shape == (1,1)` and `scale_b.shape == (1,1)` |
| FP8 block-128 W8A8 | activation / weight scale axis、block shape、accumulation dtype 必须显式 |
| INT4 weight-only | physical storage dtype、nibble order、elements per storage word、scale layout、zero-point policy 必须显式 |
| INT8 weight-only | 和 INT4 相同，但 storage/packing 简化 |
| grouped FP8 / weight-only | expert-major layout、batch offsets、per-expert scale、EP mapping 是否进入 op 需单独定 |

这里不要使用 `layout = awq_like | gptq_like | marlin_like` 这类混合枚举。AWQ/GPTQ 是量化方法或 checkpoint convention；Marlin-like 是 kernel-native packing/backend。TileOps runtime contract 应描述 physical layout，而不是让 kernel 直接理解离线算法名。

### 3.3 MoE：`moe`

MoE 的第一批目标不是创造一个巨大 `QuantizedMoeExpertsFwdOp`。上游已有 `MoeGroupedGemmNopadFwdOp`、`MoeUnpermuteFwdOp`、`FusedMoEExpertsNopadPersistent3WGFwdOp` 等边界；量化应优先落在 low-level expert GEMM 和已有 fused expert path 的 variant 上。

#### Public op contracts

| Op / contract | 输入 | 输出 | 量化语义 | 当前状态 |
| --- | --- | --- | --- | --- |
| `MoeGroupedGemmNopadFwdOp` quantized variant | permuted activations, expert weights, true sizes / offsets, scale metadata | expert GEMM output | compositional baseline 的核心 expert compute；不接 `topk_weights`，不做 unpermute | 已有 A16 op；待补 FP8 / weight-only variant |
| `GroupedGemmOp` quantized variant | padded/grouped activations and expert weights | grouped output | padded expert GEMM 或通用 grouped GEMM reuse | 已有 A16 op；待补 quantized variant |
| `MoeUnpermuteFwdOp` | expert output, `fwd_idx`, `topk_weights` | token-order output | weighted combine / unpermute；若 expert output 已是 A16，通常不需要 quantized finalize op | 已有 op；优先复用 |
| `FusedMoEExpertsNopadPersistent3WGFwdOp` quantized variant | hidden states, expert weights, routing metadata | token-order output | 后续 fused path；可把 quantized expert compute、activation、unpermute 合并 | 已有 fused A16 path；量化 variant 作为第二步 |
| shared expert path | hidden states and shared expert weights | shared expert output | shared expert 可先走独立 dense MLP / GEMM path，再由上层 block merge | 已有 shared MoE abstraction；不塞进第一批 routed expert op |

#### Quantized dispatch / kernel variants

| Variant | Contract 要点 |
| --- | --- |
| routed activation quantize | dispatch 后 token layout 已改变；是否量化由 GEMM consumer 和 benchmark 决定 |
| FP8 expert grouped GEMM | per-expert / block scale、accumulation policy、output dtype |
| INT4 / INT8 weight-only expert GEMM | packed expert weight layout、scale/zero metadata、expert-major storage |
| fused expert pipeline | 只在 compositional baseline 站稳后推进；不要第一版就混入 shared expert、EP、unpermute 全部语义 |

EP 相关 contract 需要在 tracking issue 中明确：第一版是 single-GPU / local-expert-only，还是接收 `expert_map`、local/global expert id 和 EP placement metadata。

### 3.4 Attention：`attention`

上游 `GroupedQueryAttentionPrefillFwdOp` 已经通过 dtype、scale 和 `backend="fp8"` 表达 FP8 Tensor Core prefill；`GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp` 已经有 FP8 cache storage-only/on-the-fly dequant 路径。因此第一批缺口应集中在 decode KV cache contract，而不是重新命名已有 prefill op。

#### Public op contracts

| Op / contract | 输入 | 输出 | 量化语义 | 当前状态 |
| --- | --- | --- | --- | --- |
| `GroupedQueryAttentionPrefillFwdOp` FP8 variant | packed `q/k/v: fp8`, `q/k/v_scale`, cu-seqlens | A16 output | dense FP8-input prefill compute；backend dispatch 到 FP8 Tensor Core kernel | 已有 manifest / op；不新增 FP8 专属 public op |
| `GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp` FP8 cache variant | A16 `q/k_new/v_new`, fp8 `k_pages/v_pages`, `k_scale/v_scale`, block table | A16 output and cache update | paged prefill 读取 FP8 cache；storage-only/on-the-fly dequant path | 已有 manifest / op；fused RoPE + FP8 append 仍是后续 variant |
| `GroupedQueryAttentionDecodeWithKVCacheFwdOp` FP8 cache variant | `q`, dense `k/v fp8 cache`, `cache_seqlens`, `k_scale/v_scale` | output | dense decode cache read；需要明确 variable length / padding 语义 | 缺 |
| `GroupedQueryAttentionDecodePagedWithKVCacheFwdOp` FP8 cache variant | `q`, paged `k/v fp8 pages`, `block_table`, `cache_seqlens`, `k_scale/v_scale` | output | paged decode cache read；scale indexing 必须进入 shape rules | 缺 |
| `GroupedQueryAttentionKVCacheAppendFwdOp` | new `k/v`, destination pages, slot mapping/cache positions, scale policy, scale metadata | updated pages + scale metadata | 只有真正写入 persistent cache pages 时才新增 | 可作为后续；纯 `k/v -> fp8 + scale` 仍可复用 `FP8QuantOp` |

#### Quantized dispatch / kernel variants

| Variant | Contract 要点 |
| --- | --- |
| materialized dequant fallback | reference/debug；不要默认进入 public serving op |
| fused dequant + A16 compute | decode hot path 的现实起点；fp8 cache 在 tile 内应用 scale，不写完整 A16 K/V |
| FP8 Tensor Core attention | Q/K/V scale placement、softmax dtype、accumulation policy、q 是否上游量化都必须显式 |
| paged scale indexing | `per_tensor`, `per_head`, `per_request_head`, `per_page_head`, `per_block_head` 等 shape rules |
| fused RoPE + FP8 append | 已有 prefill paged path 的 follow-up kernel variant，不和 decode contract 混写 |

### 3.5 Normalization / Activation：`normalization` and activation families

这一类只有在下游 quantized GEMM/MoE 已经稳定消费时才值得进入 public op。否则它们会和 `FP8QuantOp` 重复。

#### Public op contracts

| Op / contract | 输入 | 输出 | 量化语义 | 当前状态 |
| --- | --- | --- | --- | --- |
| `RMSNormQuantizeFwdOp` | `x`, `rms_weight`, scale policy | `x_fp8`, scale metadata | norm output 直接进入 W8A8 GEMM 时使用；普通 A16 norm 仍由现有 RMSNorm op 负责 | 候选；输出 tuple 应固定，不使用 optional output |
| `FusedAddRMSNormQuantizeFwdOp` | residual, hidden states, norm weight, scale policy | updated residual, `x_fp8`, scale metadata | residual add + norm + quantize fused boundary | 候选；必须和 non-residual norm 分开 |
| activation-specific quantize fusion | activation inputs | quantized activation + scale metadata | 只有 `SiluAndMulQuantizeFwdOp`、`GeluAndMulQuantizeFwdOp` 这类具体 activation fusion 才有意义 | 先不新增泛泛的 `ActivationQuantizeFwdOp` |

#### Quantized dispatch / kernel variants

| Variant | Contract 要点 |
| --- | --- |
| norm + online amax + quantize | consumer 是 W8A8 GEMM 时才进入 |
| fused add norm + quantize | block boundary 需要 updated residual 时才进入 |
| activation + quantize | 绑定具体 activation，不做无语义的 generic activation quantize |

### 3.6 Linear Attention / Sequence Modeling：`linear_attn` / `sequence_modeling`

GDN/KDA/MLA state/cache 的量化需求真实存在，但当前 public state contract 尚未稳定。第三章只做 contract inventory，不提前冻结 op 名。

| Future contract area | 需要先明确的事项 | 相关模型 |
| --- | --- | --- |
| Gated DeltaNet recurrent state quantization | state tensor 组成、prefill/decode state 是否同构、量化是否独立执行还是融合进 recurrence | Qwen3.5 hybrid models |
| KDA recurrent / prefill-cache quantization | state layout、update 语义、public implementation 和 checkpoint metadata | Kimi3 |
| MLA latent-cache quantization | latent cache 组成、scale granularity、decode backend contract | DeepSeek MLA, Kimi K2 MLA |

这类方向进入后续 tracking issue。现在不使用 `GatedDeltaNetStateQuantizeFwdOp`、`KDAStateQuantizeFwdOp`、`MLAStateOrLatentCacheQuantizeFwdOp` 作为 public op 名，避免在 state contract 尚未确定前过早冻结 API。

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
layout: dense, packed, paged, expert-major, token-major, kernel-native layout id
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
| `FP8QuantOp` and possible dequant/reference helpers | `quantization` helper track | 负责 narrow scale-aware cast / dequant，不代表完整推理算子 |
| Consumer-specific pack / layout conversion | owning consumer family first | 只有被 Linear / MoE / KV cache 的同一 physical layout 明确复用时，才抽成独立 helper |
| `GemmFp8Op` extensions | `gemm` | 已有 FP8 GEMM public op；继续补 accumulation policy、scale algebra、layout/backend coverage |
| `GemmWeightOnlyOp` or `GemmWeightOnlyW4A16Op` | `gemm` | packed INT4/INT8 weight 是不同 physical input contract，可以新增 |
| `GroupedGemmOp` quantized variants | `gemm` | 可被 MoE 复用的 grouped expert compute；优先作为 tagged variant 设计 |
| `MoeGroupedGemmNopadFwdOp` quantized variants | `moe` | compositional MoE expert compute 的第一批承载点 |
| `MoeUnpermuteFwdOp` | `moe` | weighted combine / unpermute 通常复用，不因上游量化而改名 |
| `FusedMoEExpertsNopadPersistent3WGFwdOp` quantized variant | `moe` | compositional baseline 稳定后的 fused target |
| `GroupedQueryAttentionPrefillFwdOp` FP8 backend | `attention` | 已有 canonical prefill op 的 dtype/backend variant |
| `GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp` FP8 cache variant | `attention` | 已有 paged prefill FP8 cache storage-only path |
| `GroupedQueryAttentionDecodeWithKVCacheFwdOp` / `GroupedQueryAttentionDecodePagedWithKVCacheFwdOp` FP8 cache variants | `attention` | 第一批 decode KV-cache 缺口 |
| Quantized MLA / KDA / linear attention state | owning attention / linear-attention / sequence-modeling family | 等 state/cache contract 稳定后按原算子 family 扩展；现在只做 inventory |

### 4.4 不进入范围

| 类别 | 当前归类 | 不纳入原因 |
| --- | --- | --- |
| 普通 fp8 cast / helper | `quantization` helper | 只是 narrow helper，不表达完整推理算子 |
| 普通 bf16/fp16 attention | `attention` | 没有 quantized tensor / scale metadata，仍是普通 attention |
| 普通 MoE routing / permute | `moe` | routing 本身不是量化；只有 quantized expert compute / activation dispatch 才进入 |
| 离线量化算法本身 | external tooling | AWQ/GPTQ calibration、QAT training、compressed-tensors conversion 不是 TileOps kernel |
| engine-level memory policy | runtime / serving engine | KV cache block allocation、TP/EP placement、checkpoint loader policy 不是 TileOps kernel |

## 5. 第一批核心目标：按现有 Family 落地

第一批不追求覆盖所有量化算法，而是沿着第 3 章的候选算子目录，补齐最能支撑现代推理路径的关键缺口。每条线都落到现有 family 中。

### 5.0 优先级总表

| 优先级 | Family | 第一批目标 | 为什么先做 |
| --- | --- | --- | --- |
| P0 | `quantization` helpers | 收紧 `FP8QuantOp` scale algebra、provided-scale / online-absmax path、reference dequant utility | 其他 family 都依赖统一 scale 语义 |
| P0 | `gemm` | 扩展已有 `GemmFp8Op` contract；新增 weight-only GEMM physical layout design | GEMM 是 Qwen/Llama/Kimi/DeepSeek 量化推理最直接的热路径 |
| P1 | `gemm` / `moe` | `GroupedGemmOp` / `MoeGroupedGemmNopadFwdOp` 的 FP8 和 weight-only expert variants | MoE 是 Qwen3、DeepSeek-V3、Kimi K2 / Kimi K2 Thinking、Llama 4 Maverick 的共同热点 |
| P2 | `attention` | 在已有 FP8 prefill / paged prefill FP8 cache 基础上补 decode FP8 KV-cache variants 和 scale metadata | 长上下文 decode 的内存收益明确，但 paged decode contract 需要和 serving engine 对齐 |
| P2 | `linear_attn` / `sequence_modeling` | quantized MLA/KDA/linear-attention state path | 模型需求明确，但 checkpoint / runtime contract 还需要逐个确认 |

### 5.1 `quantization` helpers：Scale-Aware Quantize / Dequantize

目标是把已有 `FP8QuantOp` 的 scale 语义收紧，而不是另造一组重叠 helper。第一版重点是：

```text
scale algebra:
  统一 scale / descale 的含义。建议公开 contract 中的 scale 表示 dequant scale，
  即 x_hat = x_fp8 * scale。

provided-scale path:
  x, scale -> x_fp8

online-absmax path:
  x -> amax -> scale -> x_fp8

reference dequant:
  x_fp8, scale -> x_a16，用于 correctness/debug/fallback。
```

第一版覆盖：

```text
per-tensor
per-head
per-token-per-128-channel
per-128x128 block, when consumed by GemmFp8Op
```

consumer-specific pack / transpose / swizzle 先留在 GEMM、MoE 或 KV cache owning family 中。只有多个 consumer 共享同一种 physical layout 时，才抽独立 pack helper。

### 5.2 `gemm`：FP8 GEMM and Weight-Only GEMM

这是推理量化的主算子。第一批分两条线：

```text
1. Extend existing GemmFp8Op.
2. Add a real weight-only GEMM contract, because packed INT4/INT8 weight
   changes the physical input contract.
```

`GemmFp8Op` 扩展项：

```text
scale algebra
scale axis semantics
accumulation dtype
accumulation policy
output dtype
E4M3 / E5M2 coverage
transpose / layout coverage
```

`GemmWeightOnlyOp` 候选 contract：

```text
GemmWeightOnlyOp(
    x:                 Tensor[M, K],          # fp16/bf16
    w_storage:         Tensor[physical_shape], # uint8/int32 packed storage
    w_scale:           Tensor[scale_shape],
    w_zero:            Optional[Tensor[zero_shape]],
    logical_weight_bits: enum(4, 8),
    signedness:        enum("signed", "unsigned"),
    group_size:        int,
    pack_order:        explicit layout id,
    kernel_layout_version: str,
    out_dtype:         same_as(x),
) -> y: Tensor[M, N]
```

这里不把 AWQ/GPTQ 的离线校准与量化算法、QAT 训练、checkpoint conversion 或 loader policy 放进 TileOps。TileOps 只接收已经转换到 kernel-native physical layout 的权重、scale、zero metadata，并负责 serving-time unpack / scale application / matmul。

### 5.3 `moe`：Quantized Expert Compute

MoE 是 Qwen3、DeepSeek-V3、Kimi K2 / Kimi K2 Thinking 和 Llama 4 Maverick 的共同热点。第一批应固定 compositional expert compute，而不是先写一个全包式 `QuantizedMoeExpertsFwdOp`。

第一批目标：

```text
permute / routing metadata
  -> optional routed activation quantize
  -> quantized MoeGroupedGemmNopadFwdOp or GroupedGemmOp variant
  -> activation
  -> quantized MoeGroupedGemmNopadFwdOp or GroupedGemmOp variant
  -> existing MoeUnpermuteFwdOp weighted combine
```

需要覆盖的 workload：

```text
Qwen3-30B-A3B / 235B-A22B style E=128, top_k=8
DeepSeek-V3 style E=256, top_k=8
Kimi K2 style E=384, top_k=8, shared expert handled outside first routed op
Llama 4 Maverick style E=128 MoE
```

第一版先写清：

```text
expert weight physical layout
per-expert / per-block scale metadata
activation quantization point after routing
expert-major vs token-major storage
whether EP metadata is out-of-scope or appears as expert_map
```

`FusedMoEExpertsNopadPersistent3WGFwdOp` 的 quantized variant 是第二步：等 compositional baseline 的 correctness、layout 和 benchmark 稳定后，再决定哪些边界值得 fusion。

### 5.4 `attention`：FP8 KV Cache and Decode Attention

长上下文 decode 的核心压力来自 KV cache。上游已经有两类相关能力：

```text
GroupedQueryAttentionPrefillFwdOp:
  已有 FP8 Q/K/V + q/k/v_scale + backend="fp8" 的 prefill dispatch。

GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp:
  已有 paged prefill 读取 FP8 cache pages 的 storage-only/on-the-fly dequant path。
```

第一批缺口集中在 decode：

```text
GroupedQueryAttentionDecodeWithKVCacheFwdOp FP8 cache variant:
  q + dense fp8 K/V cache + cache_seqlens + scale metadata -> o

GroupedQueryAttentionDecodePagedWithKVCacheFwdOp FP8 cache variant:
  q + fp8 K/V pages + block_table + cache_seqlens + scale metadata -> o
```

如果要做真实 cache append，而不是普通 `K/V -> fp8 + scale` helper，则需要单独定义：

```text
GroupedQueryAttentionKVCacheAppendFwdOp(
    k_new,
    v_new,
    k_pages,
    v_pages,
    slot_mapping or cache_positions,
    scale_policy,
    scale_metadata,
) -> updated cache / metadata
```

第一版不要把 paged allocator 和 serving-engine block policy 塞进 attention kernel；TileOps 只负责可测的 quantized KV read / scale application / attention compute。materialized dequant path 先作为 reference/debug/fallback，不默认进入 public serving op。

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

新增本文档，明确量化能力在现有 family 中的归属、模型需求、缺口和共享 metadata / layout contract。同步检查上游已有 `FP8QuantOp`、`GemmFp8Op`、`GroupedQueryAttentionPrefillFwdOp` FP8 backend、`GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp` FP8 cache path、MoE grouped expert abstraction 应该补哪些 manifest 条目，以及是否需要调整 source metadata。

### Phase 1：Helper Primitive Contract

在 `quantization` helper 方向先收紧已有 `FP8QuantOp`：

```text
FP8QuantOp scale algebra
provided-scale quantize path
online-absmax quantize path
reference dequant utility, if needed by tests
```

先收敛：

```text
scale shape rules
dtype_combos
grouping modes
rounding / clamp policy
nonfinite policy
reference implementation
```

### Phase 2：`gemm` Quantized Linear

在已有 `GemmOp` / `GemmFp8Op` / `GroupedGemmOp` 基础上推进：

```text
GemmFp8Op contract extensions
GemmWeightOnlyOp or GemmWeightOnlyW4A16Op
GroupedGemmOp FP8 / weight-only variants, if MoE needs reusable grouped compute
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

在现有 MoE family 上扩展，不另起模型专属 op。第一批以 low-level grouped expert compute 为主：

```text
MoeGroupedGemmNopadFwdOp FP8 / weight-only variant
GroupedGemmOp quantized expert variant, if padded grouped path is needed
MoeUnpermuteFwdOp reuse
FusedMoEExpertsNopadPersistent3WGFwdOp quantized variant, after compositional baseline
```

关键 gate：

```text
top_k=8
E=128/256/384
H=3072/7168
expert intermediate size from Qwen / DeepSeek / Kimi / Llama workloads
per-expert scale metadata
packed routed expert weights
shared expert precision policy, likely outside first routed expert op
```

### Phase 4：`attention` FP8 KV Cache

先基于已有 FP8 prefill / paged prefill FP8 cache path 补 decode：

```text
GroupedQueryAttentionDecodeWithKVCacheFwdOp FP8 cache variant
GroupedQueryAttentionDecodePagedWithKVCacheFwdOp FP8 cache variant
```

再评估：

```text
real KV cache append op, only if TileOps owns persistent page write
per-attention-head scale, only when backend and calibration contract are explicit
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
- Qwen3.5-35B-A3B model card, Gated DeltaNet / Gated Attention hybrid architecture: https://huggingface.co/Qwen/Qwen3.5-35B-A3B
- Llama 4 Scout / Maverick model card and quantization notes: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E
- Llama 4 Maverick FP8 model card: https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
- Meta Llama 4 release blog: https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- DeepSeek-V3 Technical Report, FP8 training and fine-grained quantization: https://arxiv.org/html/2412.19437v1
- DeepSeek-V3 FP8 weight file documentation: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/README_WEIGHTS.md
- MoonshotAI Kimi K2 repository: https://github.com/moonshotai/kimi-k2
- Kimi K3 technical blog, KDA / QAT / prefill-cache context: https://www.kimi.com/blog/kimi-k3
- Kimi K2 Thinking model card, native INT4 quantization: https://huggingface.co/moonshotai/Kimi-K2-Thinking
- vLLM quantization documentation: https://docs.vllm.ai/en/latest/features/quantization/
- vLLM quantized KV cache documentation: https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
- vLLM FP8 KV-cache and attention quantization blog: https://vllm.ai/blog/2026-04-22-fp8-kvcache
- AWQ paper: https://arxiv.org/abs/2306.00978
