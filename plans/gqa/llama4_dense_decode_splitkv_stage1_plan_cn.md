# Llama 4 Dense GQA Decode Split-KV Stage 1 Plan

日期：2026-06-08

本文记录 Llama 4 形状下，H200 dense single-token GQA decode kernel 的阶段一实现计划。阶段一目标不是补齐完整 serving-facing decode 接口，而是先把 small-batch，尤其 `batch = 1` 的 dense contiguous decode 性能锚点做稳。

本计划承接 `decode.md` 和 `decode-plan.md`，但阶段目标更窄：先实现并验证 split-KV baseline，再讨论 paged KV、heterogeneous length、CUDA Graph bucket、RoPE、softcap 和 append decode。

## 1. 阶段一目标

目标：

- H200 上调好 Llama 4 形状的 dense single-token GQA decode。
- 采用 FA3-like PackGQA 策略。
- split-KV 是必选项，不是后续优化项。
- 首版只做 contiguous KV cache。
- 首版只做 homogeneous `seqlen_kv`，即 batch 内所有 slot 有相同有效 KV length。
- 公开 OP 接口暂不扩张，先以实验 kernel / benchmark kernel 形态推进。

首版张量契约：

```text
q: [B, Hq, D]
k: [B, S, Hkv, D]
v: [B, S, Hkv, D]
o: [B, Hq, D]
```

语义：

```text
single-token dense full-prefix decode
read-only KV cache
external RoPE
no softcap
no sliding window
no append
```

## 2. Llama 4 目标形状

Llama 4 Scout / Maverick 的 attention 主形状为：

```text
Hq  = 40
Hkv = 8
D   = 128
GQA ratio = Hq / Hkv = 5
```

对应：

```text
Q: [B, 1, 40, 128]
K: [B, S,  8, 128]
V: [B, S,  8, 128]
```

参考来源：

- Llama 4 Scout config mirror: https://huggingface.co/FILM6912/Llama-4-Scout-17B-16E-Instruct/raw/main/config.json
- Llama 4 Maverick config mirror: https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct/raw/main/config.json

注意：本阶段只取 attention kernel 相关形状，不讨论 MoE、layer routing、long-context runtime 或完整 model serving。

## 3. 为什么 split-KV 是必选项

PackGQA 后，调度粒度从 query head 变成 KV head：

```text
base work tiles = B * Hkv
```

对 Llama 4：

```text
B=1  -> 8 base tiles
B=2  -> 16 base tiles
B=4  -> 32 base tiles
B=8  -> 64 base tiles
B=16 -> 128 base tiles
```

H200 有 132 个 SM。没有 split-KV 时，`B=1/2/4/8` 都无法充分填满 SM，`B=1` 尤其严重。因此阶段一必须直接按 split-KV 设计：

```text
work tiles = B * Hkv * num_splits
```

对 `B=1, Hkv=8`，`num_splits ~= 14-16` 时：

```text
work tiles ~= 112-128
```

这才接近 H200 的 SM 数量。

## 4. FA3 策略观察

本地 FA3 Hopper 源码中的相关策略：

- `get_pack_gqa` 会在 small `seqlen_q` / GQA 场景启用 PackGQA。
- split 数量由 `num_splits_heuristic` 估算，不只看 KV length，也看 `B * effective_heads` 对 SM 的填充效率。
- 对 `q_len = 1` 的 GQA decode，PackGQA 后 effective heads 基本是 `Hkv`。
- `Split && !Varlen` 时 FA3 通常使用 single-tile style scheduler，而不是 persistent scheduler。
- `Varlen` / dynamic length 场景才进入更复杂的 varlen dynamic persistent scheduler。

对应源码位置：

```text
hopper/heuristics.h
hopper/flash_api.cpp
hopper/flash_fwd_launch_template.h
hopper/tile_scheduler.hpp
```

阶段一先效仿 FA3 的外层策略：

```text
PackGQA + fixed split-KV + separate combine
```

不在第一版复刻完整 FA3 scheduler。

## 5. Kernel 分解

阶段一建议实现两个 kernel：

### 5.1 Partial Attention Kernel

Grid 逻辑：

```text
[B, Hkv, num_splits]
```

每个 CTA 负责：

```text
one batch slot
one KV head
one KV split range
the 5 Q heads mapped to this KV head
```

输出 partial workspace：

```text
partial_lse: [B, Hq, num_splits]       # fp32
partial_o:   [B, Hq, num_splits, D]    # fp32 or fp16/bf16, benchmark 后决定
```

建议首版 `partial_o` 用 fp32，先让 merge 数值稳定；如果 workspace traffic 明显成为瓶颈，再测 fp16/bf16 partial output。

每个 split 内执行：

```text
load packed Q heads for one KV head
stream K/V over split-local KV blocks
QK
online softmax update
PV
write partial_lse and partial_o
```

空 split 或超出 `S` 的 split 写 identity stats：

```text
partial_lse = -inf
partial_o = 0
```

这样 fixed `num_splits` 下，launch topology 和 workspace shape 始终固定。

### 5.2 Combine Kernel

Grid 逻辑：

```text
[B, Hq]
```

对每个 output row 合并 `num_splits` 个 partial：

```text
m = max_i(partial_lse_i)
lse = log(sum_i exp(partial_lse_i - m)) + m
o = sum_i exp(partial_lse_i - lse) * partial_o_i
```

公开结果仍保持 output-only：

```text
o: [B, Hq, D]
```

`partial_lse` 和 `partial_o` 是内部 workspace，不作为用户接口。

## 6. 初始 split policy

首轮 benchmark 不直接固定 heuristic，而是扫参获得实际曲线。建议候选：

```text
B=1:
  splits = 1, 4, 8, 12, 14, 15, 16, 20

B=2:
  splits = 1, 4, 8, 12

B=4:
  splits = 1, 2, 4, 8

B=8:
  splits = 1, 2, 4

B=16:
  splits = 1, 2
```

预期趋势：

```text
B 越小，最佳 split 越大。
B=1, S>=4K 时最佳 split 应接近 14-16。
B=16 时 no-split / split=1 应成为主路径。
```

第一版 heuristic 可以先写成静态表或简单规则：

```text
if B >= 16: splits = 1
elif B == 8: splits = 2
elif B == 4: splits = 4
elif B == 2: splits = 8
else: splits = 15
```

最终规则以 H200 benchmark 为准。

## 7. Benchmark 矩阵

核心 benchmark：

```text
Hq = 40
Hkv = 8
D = 128
dtype = fp16 first, bf16 second
```

Batch：

```text
B = 1, 2, 4, 8, 16
```

KV length：

```text
S = 1K, 2K, 4K, 8K, 32K, 128K
```

扩展 long-context：

```text
S = 1M
```

对比对象：

- no-split TileOps baseline
- split-KV TileOps candidate
- FA3 `flash_attn_with_kvcache` if available
- PyTorch SDPA only as correctness reference, not performance baseline

必须记录：

```text
elapsed time
effective TFLOP/s
estimated KV bandwidth
best split
partial workspace bytes
combine time percentage
```

## 8. 正确性验收

首版 correctness 覆盖：

- `B in {1, 2, 4, 8, 16}`
- `S in {1K, 4K, 32K}`
- `dtype in {fp16, bf16}`
- split 与 no-split 对齐
- split 与 PyTorch SDPA materialized reference 对齐
- empty split identity stats 不产生 NaN
- GQA head mapping 正确：

```text
kv_head = q_head // 5
```

建议容忍度：

```text
fp16: atol=1e-2, rtol=1e-2
bf16: atol=2e-2, rtol=2e-2
```

具体阈值以现有 TileOps attention 测试习惯为准。

## 9. 非目标

阶段一不做：

- paged KV cache
- batch 内 heterogeneous `real_seqlen_kv`
- varlen packed KV
- CUDA Graph capture 测试
- fused RoPE
- softcap
- sliding window
- append current K/V
- FP8 KV cache
- FP8 Tensor Core attention
-完整 FA3 dynamic persistent scheduler

这些进入后续阶段。

## 10. 后续阶段接口方向

阶段一完成后，下一步再把 kernel 接到 release-facing decode family：

```text
contiguous:
  q: [B, Hq, D]
  k/v: [B, S_cap, Hkv, D]
  real_seqlen_kv: [B]

paged:
  q: [B, Hq, D]
  k_pages/v_pages: [P_tokens, Hkv, D]
  real_seqlen_kv: [B]
  block_table: [B, max_pages_per_req]
```

后续必须保持：

```text
runtime dynamic values do not change kernel compile key
fixed bucket shape can reuse compiled kernel
split policy can become fixed per bucket
```

## 11. 当前决策总结

已定：

- 目标模型形状先用 Llama 4，而不是 Qwen3.5。
- 阶段一先做 `Hq=40, Hkv=8, D=128`。
- small batch 是性能重点，`B=1` 是核心压力测试。
- PackGQA 策略合理，应效仿 FA3。
- split-KV 必须实现。
- 首版只做 contiguous homogeneous decode。

待 benchmark 决定：

- `partial_o` 用 fp32 还是 output dtype。
- `block_N` 取 128、176，还是 TileOps 更容易表达的其它值。
- `B=1` 最佳 split 是否为 14、15、16 或 20。
- combine kernel 是否需要进一步 fuse / optimize。
- 何时引入 WS producer-consumer 形态。

