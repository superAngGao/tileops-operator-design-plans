# Quantized Inference 算子补齐计划

## 1. 计划定位

这份文档用于讨论 TileOps 是否需要系统补齐推理量化相关算子，以及第一批工作如何落地。它不是 manifest spec，也不冻结任何新 op 的公开 API；真正的 op contract 仍以后续 tracking issue、manifest PR 和实现 PR 为准。

这里的基本判断是：TileOps 当前已经有若干 FP8 相关能力，但还没有形成覆盖主流开源模型推理路径的 quantized inference family。现有能力更像局部 kernel：

```text
FP8QuantOp
FP8LightingIndexerOp
GroupedQueryAttentionPrefillFP8TensorCoreFwdOp
fp8 elementwise benchmarks
MoE prepare/finalize abstraction 中的 quantization hook
```

而主流推理需求正在变成一组完整路径：

```text
quantize / pack / scale metadata
  -> quantized GEMM / MoE experts / attention KV path
  -> fused dequant / activation / residual / output
  -> paged KV cache quantization and decode reuse
```

这份计划的目标不是把所有量化算法都搬进 TileOps，而是把推理热路径中的稳定 kernel boundary 抽出来，补齐 manifest、reference、test、benchmark 和 model-shaped workloads。

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

## 3. TileOps 当前能力与缺口

### 3.1 已有能力

当前仓库中已经能看到这些量化相关能力：

| 能力 | 位置 | 状态判断 |
| --- | --- | --- |
| FP8 activation quantization | `tileops/ops/fp8_quant.py`, `tileops/kernels/fp8_quant.py` | 有专用 op，但 shape / scale contract 仍偏局部 |
| FP8 LightingIndexer | `tileops/ops/fp8_lighting_indexer.py` | attention indexing 相关局部能力 |
| FP8 GQA prefill tensor-core path | `GroupedQueryAttentionPrefillFP8TensorCoreFwdOp` in `tileops/manifest/attention.yaml` | 已进入 attention manifest，是目前最接近 serving hot path 的 FP8 kernel |
| MoE prepare/finalize abstraction | `tileops/ops/moe/abc.py` | 已预留 quantization / EP dispatch 位置，但默认实现仍是 pass-through |
| FP8 elementwise / selection benchmarks | `benchmarks/ops/bench_elementwise_fp8.py`, `benchmarks/ops/bench_independent_elementwise.py` | 可作为 dtype coverage，不是完整推理量化路径 |

### 3.2 主要缺口

| 缺口 | 为什么重要 | 当前 TileOps 风险 |
| --- | --- | --- |
| 通用量化 family / manifest | `FP8QuantOp` 是局部 op，不能表达 W8A8、W4A16、KV cache、MoE expert 等完整语义 | 量化能力分散在 attention / indexing / ad hoc kernel 中 |
| Weight-only INT4/INT8 GEMM | Qwen AWQ/GPTQ、Llama on-the-fly INT4、Kimi INT4 MoE 都依赖 weight-only serving path | 没有 AWQ/GPTQ packed layout、group scales、fused dequant GEMM |
| FP8 W8A8 GEMM / Linear | Llama 4 Maverick FP8、DeepSeek-V3 FP8 framework、Qwen FP8 都需要 scale-aware GEMM | 只有 FP8 GQA prefill 和局部 FP8 quant，缺 Linear / MoE experts 主路径 |
| Quantized MoE experts | Qwen3 MoE、DeepSeekMoE、Kimi K2/K3 都是 MoE 主路径 | 现有 MoE manifest 是 bf16/fp16 hidden and weights；缺 quantized routed/shared experts |
| KV cache quantization | 长上下文 decode 中 KV cache 是核心内存瓶颈；vLLM 已支持 FP8 KV cache per-tensor/per-head scale | 缺 paged KV cache quant/dequant、FP8 decode attention with quantized KV、scale metadata contract |
| scale / pack / layout helper | INT4/FP8 性能依赖 packed layout、scale shape、swizzle、TMA/WGMMA friendly memory format | 如果不先定义 metadata 和 layout，GEMM/MoE/attention op 会各自发明不兼容格式 |

## 4. Family 边界

本计划建议新增或补齐一个 `quantization` / `quantized_inference` 方向，但它不应该吸收所有低精度 dtype 使用场景。

### 4.1 进入范围

进入本计划的 op 必须满足至少一条：

```text
1. op 负责生成、消费或转换 quantized tensor 与 scale / zero-point / group metadata；
2. op 的主计算是 quantized GEMM / attention / MoE expert / KV cache read-write；
3. op 的性能或正确性依赖 packed layout、group scale、online quantization 或 fused dequant。
```

典型成员：

```text
FP8Quantize / FP8Dequantize
GroupQuantize / GroupDequantize
WeightOnlyLinear / QuantizedLinear
QuantizedMoeExperts
FP8KVCacheQuantize / FP8KVCacheAttention
QuantizedTensorPack / LayoutConvert
```

### 4.2 不进入范围

| 类别 | 当前归类 | 不纳入原因 |
| --- | --- | --- |
| 普通 fp8 elementwise | `elementwise` | 只是 dtype coverage，不表达推理量化 contract |
| 普通 bf16/fp16 attention | `attention` | 没有 quantized tensor / scale metadata |
| 普通 MoE routing / permute | `moe` | routing 本身不是量化；只有 quantized expert compute / activation dispatch 才进入 |
| 离线量化算法本身 | external tooling | AWQ/GPTQ calibration、QAT training、compressed-tensors conversion 不是 TileOps kernel |
| engine-level memory policy | runtime / serving engine | KV cache block allocation、TP/EP placement、checkpoint loader policy 不是 TileOps kernel |

## 5. 第一批核心目标

第一批不追求覆盖所有量化算法，而是补齐最能支撑现代推理路径的四条线。

### 5.1 Scale-Aware FP8 Quantize / Dequantize

目标是把现有 `FP8QuantOp` 从局部形状扩展为 manifest 可描述的基础 op。

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

这条线直接服务 DeepSeek-style fine-grained FP8、vLLM FP8 KV cache，以及现有 FP8 GQA / FP8Quant 的统一化。

### 5.2 Quantized Linear / GEMM

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

### 5.3 Quantized MoE Experts

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

### 5.4 FP8 KV Cache Quantization and Decode Attention

长上下文 decode 的核心压力来自 KV cache。vLLM 已经把 FP8 KV cache 作为 serving 功能，支持 per-tensor 和 per-head scale。

候选接口：

```text
FP8KVCacheQuantizeFwdOp(
    k:      Tensor[B, S, H_kv, D],
    v:      Tensor[B, S, H_kv, D],
    mode:   enum("per_tensor", "per_head"),
) -> (k_fp8, v_fp8, k_scale, v_scale)

GQADecodeWithFP8KVCacheFwdOp(
    q:          Tensor[B, H, D],
    k_cache:    Tensor[paged_or_dense_fp8_layout],
    v_cache:    Tensor[paged_or_dense_fp8_layout],
    k_scale:    Tensor[scale_shape],
    v_scale:    Tensor[scale_shape],
    block_table: Optional[Tensor],
) -> o: Tensor[B, H, D]
```

第一版可以先做 dense reference / non-paged smoke，再进入 paged cache contract。不要把 paged allocator 和 block table policy 塞进 quantization op；TileOps 只负责可测的 quantized KV read / dequant / attention compute。

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

### Phase 0：Inventory and Scope

新增本文档，明确 quantized inference 的 family 边界、模型需求和缺口。同步检查现有 FP8Quant、FP8 GQA、MoE abstraction 是否应该补 manifest 条目或迁移 family。

### Phase 1：Quantization Primitive Manifest

新增或补齐基础 op：

```text
FP8QuantizeFwdOp
FP8DequantizeFwdOp
QuantizedTensorPackFwdOp, if needed
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

### Phase 2：Quantized Linear

新增：

```text
FP8LinearW8A8FwdOp
WeightOnlyLinearW4A16FwdOp
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

### Phase 3：Quantized MoE

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

### Phase 4：FP8 KV Cache

先做：

```text
FP8KVCacheQuantizeFwdOp
GQADecodeWithFP8KVCacheFwdOp
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
- Kimi K2 Thinking model card, native INT4 quantization: https://huggingface.co/moonshotai/Kimi-K2-Thinking
- vLLM quantization documentation: https://docs.vllm.ai/en/latest/features/quantization/
- vLLM quantized KV cache documentation: https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
- AWQ paper: https://arxiv.org/abs/2306.00978
