# GQA 前向算子设计

目标：用少量稳定的 public Op 覆盖主要推理场景，并允许内部按输入特征选择不同 kernel。

## 1. 设计原则

### 1.1 功能覆盖

Public Op 按 KV 存储拓扑划分，不按 mask、dtype 或 kernel 实现划分。

| 拓扑 | Public Op | 输入 | 场景 |
| --- | --- | --- | --- |
| Dense | `GroupedQueryAttentionDenseFwdOp` | BSHD Q/K/V | 连续 KV；统一 prefill/decode |
| Paged | `GroupedQueryAttentionPagedFwdOp` | packed Q、K/V pages、page table、长度信息 | continuous batching、prefix cache、chunked prefill；统一 prefill/extend/decode |
| Varlen | `GroupedQueryAttentionVarlenFwdOp`（暂缓发布） | packed Q/K/V、`cu_seqlens_q/kv` | 训练或无 paged cache 的 ragged batch |

以下能力属于 Op 配置或 kernel specialization，不产生新 Op：

`causal/non-causal` · full/sliding-window · custom `sm_scale` · softcap · FP16/BF16/native FP8 · NeoX/interleaved RoPE · general/persistent/WS/split-K

Paged Op 只读 KV cache；page 分配、KV append 和 cache mutation 由推理 runtime 完成。

### 1.2 框架适配

| 项目 | Dense / Varlen / Paged | Prefill / Decode | 代表接口 |
| --- | --- | --- | --- |
| [vLLM](https://github.com/vllm-project/vllm) | 模型层统一，backend 内区分 | 统一 | `Attention(q, k, v, kv_cache, attn_metadata)` |
| [SGLang](https://github.com/sgl-project/sglang) | 模型层统一，backend 内区分 | backend 内路由 | `RadixAttention(q, k, v, forward_batch, ...)` |
| [TokenSpeed](https://github.com/lightseekorg/tokenspeed) | kernel API 分开 | kernel API 分开 | `mha_prefill`、`mha_extend_with_kvcache`、`mha_decode_with_kvcache` |
| [FlashInfer](https://docs.flashinfer.ai/api/attention.html) | Dense/Ragged/Paged 分开 | Paged 新接口可统一 | Ragged/Paged wrappers、Paged `BatchAttention` |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | Dense/Varlen/KV-cache 分开 | KV-cache 接口统一 | `flash_attn_func`、`flash_attn_varlen_func`、`flash_attn_with_kvcache` |

结论：TileOps 底层按存储拓扑拆 Op，每个 Op 内统一 prefill/decode。框架私有对象（如 `attn_metadata`、`ForwardBatch`）不进入 TileOps ABI。

### 1.3 内部契约

| 阶段 | 内容 |
| --- | --- |
| Op 构造 | mask/window、`sm_scale`、softcap、输出 dtype、RoPE 模式/layout/rotary dim、target |
| forward 输入 | Q/K/V 或 KV pages、page table、sequence metadata、FP8 scales、RoPE tables |
| forward 推导 | B、Sq、Skv、H、Hkv、D、输入 dtype、prefill/decode 区域、kernel selection facts |

构造期语义固定在 Op 实例中；shape 和动态 metadata 每次从输入解析，不写回 Op。page table 由 runtime 持有。

## 2. Public ABI

### 2.1 Dense

```python
GroupedQueryAttentionDenseFwdOp(
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
)(
    q, k, v,
    q_scale=None, k_scale=None, v_scale=None,
    rope_cos=None, rope_sin=None,
)
```

Q/K/V 使用 BSHD。`Sq == 1` 可选择 decode kernel，但不改变接口。

### 2.2 Paged

```python
GroupedQueryAttentionPagedFwdOp(
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
)(
    q, k_pages, v_pages,
    page_table, cache_seqlens, cu_seqlens_q,
    q_scale=None, k_scale=None, v_scale=None,
    rope_cos=None, rope_sin=None,
)
```

`cu_seqlens_q` 描述 packed Q 中各 request 的范围；每个 request 一个 query token 即 decode。调用方传入当前 layer 的 KV cache 和当前 cache group 的 page table。

## 3. Kernel dispatch

遵循 [TileOps #1975](https://github.com/tile-ai/TileOPs/pull/1975)，Op 只维护一层 specialization cache：

```text
forward
  -> 规范化输入并生成 selection facts / KernelKey
  -> get_or_build_kernel(role, key, build)
       -> hit：返回 concrete Kernel 实例
       -> miss：选择唯一 kernel role，构造并缓存实例
  -> 调用 kernel
```

```python
call = dense_selection_facts(self, q, k)
key = DenseKernelKey.from_call(call)

def build():
    role = select_kernel_key(DENSE_FWD_KEYS, call)
    return self.kernel_map[role](call)

kernel = self.get_or_build_kernel("gqa_dense", inputs, key=key, build=build)
return kernel(*inputs)
```

- `kernel_map` 保存 `role -> Kernel class`。
- selector 必须得到唯一 role；无匹配或多重匹配均报错。
- `KernelKey` 只包含构造 kernel 所需的 shape、dtype 和编译期 facts，不包含 tensor 内容、page indices 或实际 sequence-length values。
- TileLang/JIT cache 独立复用编译产物。

## 4. 迁移顺序

1. 分 PR 向 Dense Op 迁移 general、causal/square、sliding-window、native FP8、WS 和 decode kernels。
2. 建立 Paged Op，再迁移 paged prefill/extend/decode kernels。
3. 新 Op 完成 correctness、benchmark 和 kernel-map 覆盖后，删除旧 Op、adapter 与无调用方 kernel。
4. Varlen 暂缓发布，出现明确框架或训练 workload 后再推进。

每个迁移 PR 必须保持公开语义不变，并验证 correctness 与目标 workload 性能不回退。
