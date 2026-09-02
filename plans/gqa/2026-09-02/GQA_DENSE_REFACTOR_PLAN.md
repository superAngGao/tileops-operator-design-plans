# GQA Dense Op 重构计划

本文只讨论连续 BSHD 存储下的 GQA/MHA forward。Varlen、Paged 和 backward 使用不同的数据拓扑，不属于 Dense Op 的吸收范围。

## 1. 新 Dense Op 的覆盖范围与重构前 Op

`GroupedQueryAttentionDenseFwdOp` 是连续 Q/K/V 的统一 public Op：

```python
GroupedQueryAttentionDenseFwdOp(
    is_causal=True,
    window_size_left=-1,
    window_size_right=-1,
    sm_scale=None,
    softcap=None,
    pos_encoding_mode="none",
    rotary_dim=None,
    rope_layout="neox",
    dtype=None,
    target=None,
)(
    q, k, v,
    q_scale=None, k_scale=None, v_scale=None,
    rope_cos=None, rope_sin=None,
)
```

Q/K/V 采用 BSHD：`q=[B,Sq,H,D]`，`k/v=[B,Skv,Hkv,D]`。同一接口覆盖：

- prefill 与 decode：`Sq` 来自本次 forward；`Sq == 1` 可选择 decode specialization；
- GQA 与 MHA：`Hkv < H` 是 GQA，`Hkv == H` 是 MHA；
- square 与 rectangular attention；
- causal、non-causal 和 sliding-window mask；
- custom scale、softcap、RoPE；
- FP16、BF16 和 native FP8。

重构前，与 Dense 拓扑相关的 public Op 和 dispatch 如下：

| 旧 Op | 语义/布局 | 原 dispatch kernel | 归属 |
| --- | --- | --- | --- |
| `GroupedQueryAttentionFwdOp` | BSHD square GQA prefill | `GQAFwdWsPersistentCausalKernel`、`GQAPrefillFwdWsPersistentCausalKernel`、`GQAPrefillFwdKernel` | 被 Dense Op 完全吸收；旧 Op 删除 |
| `MultiHeadAttentionFwdOp` | BSHD square MHA prefill | 自身无 kernel，委托 `GroupedQueryAttentionFwdOp` | 作为 `Hkv == H` 被 Dense Op 吸收；旧 Op 删除 |
| `GroupedQueryAttentionSlidingWindowFwdOp` | BSHD square sliding-window | `GQASlidingWindowFwdWgmmaPipelinedKernel` | window 功能和有效优化迁入 Dense kernel family；旧 Op 与独立 kernel 最终删除 |
| `GroupedQueryAttentionDecodeWithKVCacheFwdOp` | 连续 KV 的 GQA decode | `GQADecodeBs1Kernel`、`GQADecodeKernel` | 被 Dense Op 吸收 |
| `MultiHeadAttentionDecodeWithKVCacheFwdOp` | 连续 KV 的 MHA decode | `MHADecodeKernel` | 作为 `Hkv == H` 被 Dense Op 吸收 |
| `GroupedQueryAttentionPrefillFwdOp` | THD packed umbrella | dense、varlen、window、FP8 六类 kernel | 只把 uniform-dense/FP8-dense 分支迁入 Dense；其余归 Varlen |
| `GroupedQueryAttentionPrefillVarlenFwdOp` | THD ragged | `GQAPrefillVarlenFwdKernel` | 归未来 Varlen Op，不归 Dense |
| `GroupedQueryAttentionSlidingWindowVarlenFwdOp` | THD ragged window | `GQASlidingWindowVarlenFwdWgmmaPipelinedKernel` | 归未来 Varlen Op，不归 Dense |
| Paged prefill/decode Ops | Paged KV | Paged kernel families | 归未来 Paged Op，不归 Dense |
| GQA/MHA backward Ops | backward | backward kernels | 保留，不在本轮范围 |

旧 packed-prefill umbrella 的六个分支为：

```text
GroupedQueryAttentionPrefillFwdOp
├─ native FP8 uniform       -> GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel
├─ varlen sliding window    -> GQASlidingWindowVarlenFwdWgmmaPipelinedKernel
├─ dense square causal      -> GQAFwdWsPersistentCausalKernel
├─ dense causal main        -> GQAPrefillFwdWsPersistentCausalKernel
├─ dense general fallback   -> GQAPrefillFwdKernel
└─ ordinary varlen          -> GQAPrefillVarlenFwdKernel
```

其中 dense causal main 已迁移并重命名为 `GQADenseFwdWsPersistentCausalKernel`。Packed Op 在过渡期通过适配调用它；这不改变新 Dense kernel 的原生 BSHD ABI。

## 2. 新 Dense Op 的 dispatch 与 kernel family

Dense Op 遵循单层 specialization cache：target/backend 已在上一级确定；forward 从当前 Q/K/V 解析 shape 和 dtype，结合构造期语义与 target facts 形成 key；cache miss 时选择并构造一个 concrete kernel instance。

```text
forward(q, k, v, ...)
  -> 校验并规范化 optional tensors
  -> 生成 Dense call facts / specialization key
  -> get_or_build_kernel(role, inputs, key, build)
       -> hit: 返回 concrete Kernel 实例
       -> miss: 选择唯一 role，构造并缓存实例
  -> kernel(q, k, v, ...)
```

最终 kernel family 计划为：

下列 role 名称表示目标结构；当前第一阶段仍使用 `gqa_dense_causal_fwd_kernel -> GQADenseFwdWsPersistentCausalKernel`。

```text
GroupedQueryAttentionDenseFwdOp
├─ gqa_dense_prefill_fwd_kernel
│  └─ 由 GQADenseFwdWsPersistentCausalKernel 演进
│     └─ causal / non-causal / window / RoPE / scale / softcap
├─ gqa_dense_square_fwd_kernel
│  └─ 由 GQAFwdWsPersistentCausalKernel 迁移
├─ gqa_dense_fp8_fwd_kernel
│  └─ 由 GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel 迁移
├─ gqa_dense_decode_bs1_fwd_kernel
│  └─ 由 GQADecodeBs1Kernel 迁移
├─ gqa_dense_decode_fwd_kernel
│  └─ 由 GQADecodeKernel 迁移
└─ MHA decode specialization（仅在 A/B 证明必要时）
   └─ 由 MHADecodeKernel 迁移
```

| Dense 区域 | Dispatch 策略 | Kernel 处理 |
| --- | --- | --- |
| Causal、FP16/BF16、D=128 prefill | 主路径 | `GQADenseFwdWsPersistentCausalKernel`；已迁移 |
| Square causal 且满足 persistent 调度条件 | 优先于主路径 | 迁移 `GQAFwdWsPersistentCausalKernel`。同机数据约 `0.0387–0.0388 ms`，主路径约 `0.0420 ms`，不能删除 |
| Sliding window | 由 `window_size_left/right` 选择编译期 specialization | 将旧 window kernel 的 mask、边界计算和有效优化吸收到 Dense kernel family；不长期保留独立 window 主体 |
| Non-causal | Dense 16-bit specialization | 扩展 Dense family，不复活无调用方的旧 non-causal persistent class |
| RoPE | 由 `pos_encoding_mode`、`rotary_dim`、`rope_layout` 选择 specialization | 融入对应 Dense kernel；RoPE 不形成独立 public Op |
| Native FP8 | 由输入 dtype 和 scale tensors 选择 | 迁移现有 FP8 Tensor-Core kernel，保留独立流水 |
| `Sq == 1` GQA decode | 优先 decode specialization | 迁移 `GQADecodeBs1Kernel` / `GQADecodeKernel` |
| `Sq == 1` 且 `Hkv == H` | 比较 MHA/GQA decode candidates | `MHADecodeKernel` 仅在同 workload A/B 更快时保留 |

Scale 和 softcap 已由当前 causal main kernel 支持，不需要独立 kernel role。Window 是 Dense Op 的功能，同样必须被吸收；“删除 window kernel”仅指合并完成后删除原独立类和文件，不是删除 window 能力。

Kernel 清理边界：

- 可直接删除的无 dispatch kernel：`GQAFwdWgmmaPipelinedKernel`、`GQAFwdWsPersistentKernel`；
- 不迁入 H200 Dense family：`GQAPrefillFwdKernel`，待旧 Packed dense 分支退出后删除；
- 合并后删除独立实现：`GQASlidingWindowFwdWgmmaPipelinedKernel`；
- A/B 后决定：`MHADecodeKernel`；若 GQA decode kernel 在 `Hkv == H` 下不慢，则删除；
- Varlen/Paged kernels 不删除，只从 Dense 计划中排除。

只有 manifest 声明的 non-causal、window、RoPE、FP8、prefill 和 decode 区域全部具备 correctness implementation 后，`GroupedQueryAttentionDenseFwdOp` 才能从 `spec-only` 改为 `implemented`。
