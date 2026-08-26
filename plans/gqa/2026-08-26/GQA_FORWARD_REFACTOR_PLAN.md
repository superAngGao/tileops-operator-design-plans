# GQA 前向算子重构计划

本文定义 GQA 前向 Op 的长期边界。核心原则是：**按存储拓扑拆 Op，prefill/decode 作为同一拓扑下的 kernel specialization；Op 只使用一层 `get_or_build_kernel` dispatch/cache。**

## 一、Public Op 边界

### 1.1 三种输入拓扑

| 拓扑 | 核心输入 | Public Op 决策 | 主要场景 |
| --- | --- | --- | --- |
| Dense | `q: [B,Sq,H,D]`，`k/v: [B,Skv,Hkv,D]` | `GroupedQueryAttentionDenseFwdOp`；同一 ABI 覆盖 dense prefill 与 contiguous short-Q/decode | Batch=1 低延迟、规则 batch、连续 KV |
| Paged | packed Q、只读 K/V pages、page table、query/KV lengths | `GroupedQueryAttentionPagedFwdOp`；同一 ABI 覆盖 paged prefill/extend/decode | continuous batching、prefix cache、chunked prefill |
| Varlen | packed Q/K/V、`cu_seqlens_q/kv`、最大长度 | 保留设计与内部实现价值，暂缓作为 release-facing Op 发布 | 训练、离线 ragged batch、无 paged cache 的动态 batching |

Dense 不通过构造 `cu_seqlens` 复用 Varlen kernel。Paged Op 不分配 page、不 append、不修改 KV cache；runtime 在调用前完成 slot 分配和 KV 写入。

### 1.2 小变体

`causal/non-causal` · full/sliding-window · default/custom `sm_scale` · no-softcap/softcap · FP16/BF16/native-FP8 · no-RoPE/NeoX/interleaved RoPE · general/persistent/WS/split-K specialization；这些均不产生新的 public Op。

### 1.3 主流算子库与推理框架

| 项目 | 代表接口 | Dense / Varlen / Paged 是否统一 | Prefill / Decode 是否统一 |
| --- | --- | --- | --- |
| [vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backend.py) | per-layer `Attention(query, key, value, kv_cache, attn_metadata)` | **模型层统一，backend 内区分**：模型不直接选择 layout，backend 消费对应 metadata/KV layout | **统一**：同一 Attention 接口根据 metadata 处理 prefill/decode |
| [SGLang](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/radix_attention.py) | `RadixAttention(q, k, v, forward_batch, ...)` | **模型层统一，backend 内区分**：`ForwardBatch` 携带当前存储与批次 metadata | **模型层统一**：backend 内部分为 `forward_extend` / `forward_decode` |
| [TokenSpeed](https://github.com/lightseekorg/tokenspeed) | runtime attention layer；`mha_prefill`、`mha_extend_with_kvcache`、`mha_decode_with_kvcache` | **kernel API 不统一**：uncached packed 与 paged cache 是不同函数族 | **kernel API 不统一**：prefill、paged extend、paged decode 分开；runtime backend 负责路由 |
| [FlashInfer](https://docs.flashinfer.ai/api/attention.html) | Ragged/Paged wrappers；Paged `BatchAttention` | **不统一**：Dense/Ragged/Paged 使用不同函数或 wrapper | **部分统一**：传统 Prefill/Decode wrapper 分开；`BatchAttention` 可统一 Paged mixed prefill/decode |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention/blob/main/README.md#flashattention-2) | `flash_attn_func`、`flash_attn_varlen_func`、`flash_attn_with_kvcache` | **部分统一**：Dense、Varlen、KV-cache 分开；KV-cache 内以 `block_table` 区分 contiguous/paged | **KV-cache 内统一**：同一入口覆盖 append/prefill/decode；无 cache 的 Dense/Varlen 仍是独立入口 |

共同趋势是：模型层接口可以统一，但底层 Op 的稳定边界首先由 Dense、Paged、Varlen 等存储拓扑决定。TileOps 因此保留 Dense/Paged 两个 release-facing Op，并在各自内部选择 prefill/decode kernel。

## 二、接口与生命周期

### 2.1 参数归属

| 生命周期 | 参数 |
| --- | --- |
| Op 构造期 | `is_causal`、`sm_scale`、`softcap`、window、输出 `dtype`、`pos_encoding_mode`、`rotary_dim`、`rope_layout`、`target` |
| 每次 forward | Q/K/V 或 KV pages、page table、sequence metadata、FP8 scales、RoPE tables |
| 从当前输入临时推导 | B、Sq、Skv、H、Hkv、D、输入 dtype、decode/prefill region、resolved scale 与 kernel selection facts |

per-call facts 只用于本次验证、cache key 和 kernel 构造，不写回 Op。动态 page table 不保存在长期复用的 Op 中；需要 CUDA Graph 固定地址或 metadata 预处理时，由推理 runtime/独立 plan 对象管理。

### 2.2 Dense ABI

```python
op = GroupedQueryAttentionDenseFwdOp(
    is_causal=True,
    sm_scale=None,
    softcap=None,
    window_size_left=-1,
    window_size_right=-1,
    dtype=None,
    pos_encoding_mode="none",
    rotary_dim=None,
    rope_layout="neox",
    target=None,
)

o = op(
    q, k, v,
    q_scale=None,
    k_scale=None,
    v_scale=None,
    rope_cos=None,
    rope_sin=None,
)
```

`Sq == 1` 可以命中 decode specialization，但不改变 public ABI。RoPE tables 和 scale tensors 由调用方持有并在 forward 时传入。

### 2.3 Paged ABI

```python
op = GroupedQueryAttentionPagedFwdOp(
    is_causal=True,
    sm_scale=None,
    softcap=None,
    window_size_left=-1,
    window_size_right=-1,
    dtype=None,
    target=None,
)

o = op(
    q,
    k_pages,
    v_pages,
    page_table,
    cache_seqlens,
    qo_indptr,
    q_scale=None,
    k_scale=None,
    v_scale=None,
)
```

`qo_indptr` 描述每个 request 的 query 范围；普通 decode 是每个 request 一个 query token 的特例。框架负责为当前 layer 传入对应 KV cache，并为当前 cache group 传入对应 page table。

## 三、Kernel ownership 与 dispatch

### 3.1 职责

- Op 验证 public ABI，并按 manifest 顺序规范化 optional inputs。
- `kernel_map` 只保存 `role -> Kernel class`，不保存优先级。
- `Kernel.applies/refusal` 与 `select_kernel_key` 在 cache miss 时选择唯一 role；重叠或无覆盖应 fail closed。
- `Op.get_or_build_kernel` 是唯一的 Op 运行时 specialization cache，直接保存 concrete Kernel/callable。
- 不再增加 backend-callable 内部的第二张 shape cache。
- TileLang/JIT cache 独立负责底层编译产物复用。

### 3.2 单层 dispatch（遵循 [TileOps #1975](https://github.com/tile-ai/TileOPs/pull/1975)）

```python
def forward(self, q, k, v, q_scale=None, k_scale=None, v_scale=None,
            rope_cos=None, rope_sin=None):
    self._validate_forward_inputs(
        q, k, v, q_scale, k_scale, v_scale, rope_cos, rope_sin
    )
    inputs = self._canonicalize_inputs(
        q, k, v, q_scale, k_scale, v_scale, rope_cos, rope_sin
    )
    call = dense_selection_facts(self, q, k)
    key = DenseKernelKey.from_call(call)

    def build():
        role = select_kernel_key(DENSE_FWD_KEYS, call)
        return self.kernel_map[role](...)

    kernel = self.get_or_build_kernel(
        "gqa_dense",
        inputs,
        key=key,
        build=build,
    )
    return kernel(*inputs)
```

缓存结构只有一层：

```text
GroupedQueryAttentionDenseFwdOp
  └─ _kernel_roles["gqa_dense"][DenseKernelKey]
       └─ concrete Kernel instance
```

`DenseKernelKey`/`PagedKernelKey` 只包含选择或构造 concrete Kernel 必需的稳定 facts，例如 tensor shape、dtype、page size、target/device identity 与编译期配置；不包含 tensor 内容、page indices 或 sequence-length values。构造期语义已固定在 Op 实例中，不需要重复复制进 key。

## 四、实施顺序

1. **Dense Op 边界**：由 #1975 定义；保持单层 cache，并按独立 PR 逐个迁移 general prefill、causal/square、sliding-window、native FP8、WS 与 decode kernels。
2. **Paged Op 边界**：定义只读 paged ABI，再迁移 paged prefill/extend/decode kernels；page allocation、append 和 cache-group 管理由 runtime 负责。
3. **兼容清理**：新 Op 覆盖相应 correctness、benchmark 和 kernel-map 路径后，再删除旧的按功能拆分 Op、adapter 与无 dispatch 调用方的 kernel。
4. **Varlen**：暂缓 release；只有出现明确框架接入或训练/离线 workload 后再推进 public manifest。

每个 kernel 迁移 PR 必须证明：公开语义不变、目标 workload 正确、现有性能不回退；kernel 参数或流水差异不得扩张为新的 public Op。
