# FA3 Decode Dispatch Strategy Notes

日期：2026-06-08

本文记录本地 FlashAttention-3 Hopper forward/decode 路径的 dispatch 策略观察，重点关注 single-token GQA decode 在不同 KV length、batch size、head 数目下如何选择 PackGQA、split-KV 和 scheduler。

本文是对本地源码的调研笔记，不是 FA3 官方 API 文档。当前本地源码：

```text
/home/ga/flash-attention
HEAD: ac9b5f1 basics working (#2070)
```

相关源码：

```text
hopper/flash_api.cpp
hopper/flash_fwd_launch_template.h
hopper/heuristics.h
hopper/tile_size.h
hopper/tile_scheduler.hpp
hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp
```

## 1. Decode 在 FA3 里不是独立 kernel family

FA3 的 decode 主要通过 forward path 承载，例如 Python 侧 `flash_attn_with_kvcache` 最终进入 Hopper fwd launcher。它不是完全独立的一套 `decode_kernel`，而是通过参数组合触发 decode-like 行为：

```text
seqlen_q = 1
KV cache input
optional page_table
optional seqlens / seqused
optional k_new/v_new append
num_splits heuristic or user override
```

对 single-token full-prefix decode：

```text
q: [B, 1, Hq, D]
k/v cache: [B, S, Hkv, D] or paged layout
```

如果没有 sliding window / attention chunk，FA3 会把 `seqlen_q = 1` 的 causal full-prefix 特例视为 non-causal，因为当前 query 反正可以看见全部已有 KV。

源码对应：

```text
hopper/flash_api.cpp:
  if (seqlen_q == 1 && window_size_left == -1 && window_size_right == -1 && attention_chunk == 0) {
      ...
      is_causal = false;
  }
```

## 2. PackGQA 策略

FA3 的 PackGQA 目标是把同一个 KV head 下的多个 query heads 合到 M 维里计算，从而减少重复读取 K/V。

逻辑形态：

```text
no PackGQA:
  Q logical shape uses Hq as head axis
  base work tiles ~= B * Hq

PackGQA:
  Q is viewed as ((qhead_per_khead, seqlen_q), D, Hkv, B)
  base work tiles ~= B * Hkv
```

对于 GQA：

```text
qhead_per_khead = Hq / Hkv
```

FA3 的 PackGQA 启发式：

```text
if arch < 90:
    PackGQA = true
elif paged_KV_non_TMA:
    PackGQA = true
elif num_splits > 1:
    PackGQA = true
elif Hq == Hkv:
    PackGQA = false
else:
    PackGQA = should_pack_gqa(...)
```

`should_pack_gqa` 比较的是 M 维 tile 利用率：

```text
nopack_eff = seqlen_q / round_up(seqlen_q, blockM)
pack_eff   = seqlen_q * qhead_per_khead / round_up(seqlen_q * qhead_per_khead, blockM)

return nopack_eff < 0.9 * pack_eff
```

对 single-token GQA decode，`seqlen_q = 1`，PackGQA 通常成立。更重要的是，只要 `num_splits > 1`，FA3 会强制 PackGQA，以减少编译组合和重复 K/V 读取。

因此 Llama 4：

```text
Hq = 40
Hkv = 8
qhead_per_khead = 5
```

PackGQA 后：

```text
base work tiles = B * 8
```

## 3. Tile Size 策略

SM90 tile size 由 `tile_size_fwd_sm90` 决定。对本阶段最关心的 Llama 4 fp16/bf16、`D=128`：

### 3.1 Contiguous full-prefix decode

`D=128`、element size = 2、non-causal、non-local、non-paged-nonTMA：

```text
blockM = 128
blockN = 176
MmaPV_is_RS = true
IntraWGOverlap = true
```

这对应 contiguous KV 的常见 dense decode。

### 3.2 PagedKV non-TMA / causal / local

如果是 paged KV 但不能使用 paged-KV TMA，或者 causal/local 约束使 `use_blockN_128` 为真，则：

```text
blockM = 128
blockN = 128
```

对 `q_len = 1` 的 paged GQA decode，FA3 的 paged-KV TMA 条件通常不满足：

```text
page_size % blockN == 0
and seqlen_q * (Hq / Hkv) > blockM
```

Llama 4 single-token：

```text
seqlen_q * qhead_per_khead = 1 * 5 = 5
blockM = 128
```

所以通常不会走 paged-KV TMA，而是 paged non-TMA path。

## 4. split-KV 策略

FA3 的 `num_splits` 由 `get_num_splits` 决定，除非用户显式传入 `num_splits > 0`。

核心变量：

```text
num_n_blocks = ceil(seqlen_k_loaded / blockN)
num_m_blocks = ceil(seqlen_q * qhead_per_khead / blockM)
total_mblocks = batch_factor * Hkv * num_m_blocks
```

对 non-varlen GQA decode：

```text
batch_factor = B
effective_heads = Hkv    # because Split forces PackGQA
```

对 varlen dynamic split：

```text
batch_factor is treated as 1 for upper-bound heuristic
per-batch split can later be computed by prepare_varlen_num_blocks
```

FA3 split heuristic 的核心思想：

1. 如果 base tiles 已经接近填满 SM，则不 split。
2. 如果 KV blocks 太少，也不 split。
3. 否则扫 `num_splits = 1..max_splits`，计算 SM wave efficiency。
4. 找到最佳 efficiency 后，选择达到最佳值 85% 的最小 split。
5. 避免过多 split 带来 partial workspace 和 combine overhead。

伪代码：

```text
if total_mblocks >= 0.8 * num_sms:
    if one_kv_head_size > assumed_l2
       and num_m_blocks >= num_sms * 2
       and not causal_or_local:
        split for L2 footprint
    else:
        return 1

if num_n_blocks <= 4:
    return 1

for s in 1..min(max_splits, num_sms, num_n_blocks):
    waves = total_mblocks * s / num_sms
    efficiency = waves / ceil(waves)

return smallest s whose efficiency >= 0.85 * best_efficiency
```

注意：这个策略不是单纯 “KV 越长 split 越多”。KV length 决定 `num_n_blocks` 和 split 上限，但 batch/head 数决定 base tiles 是否已经能填满 SM。

## 5. Scheduler 策略

FA3 scheduler 选择在 `flash_fwd_launch_template.h` 中决定。

主要分支：

```text
if Varlen:
    SchedulerPersistent = VarlenDynamicPersistentTileScheduler
elif not causal and not local:
    SchedulerPersistent = StaticPersistentTileScheduler
else:
    SchedulerPersistent = DynamicPersistentTileScheduler

SchedulerSingleTile = SingleTileScheduler
```

最终是否使用 persistent：

```text
SM90:
  UsePersistentScheduler = !(Split && !Varlen)

SM80:
  UsePersistentScheduler = (causal && !Varlen) || (Varlen && Split)
```

因此 SM90 上：

```text
Split && !Varlen:
  use SingleTileScheduler

Split && Varlen:
  use VarlenDynamicPersistentTileScheduler

No split, noncausal:
  use StaticPersistentTileScheduler

No split, causal/local:
  use DynamicPersistentTileScheduler
```

这对 TileOps 阶段一很重要：如果我们先做 homogeneous contiguous split-KV，可以先用固定 grid 的 single-tile style scheduler，不必第一版就复刻 FA3 varlen dynamic persistent scheduler。

## 6. Varlen 与 Multi-Batch 的关系

FA3 里的 `Varlen` 是 template flag，不等于 `B > 1`。

以下情况会触发 Varlen：

```text
cu_seqlens_q
cu_seqlens_k
seqused_q
seqused_k
leftpad_k
```

也就是说：

```text
B > 1 + homogeneous seqlen:
  not Varlen

B > 1 + per-batch dynamic seqlen:
  Varlen or seqused path
```

FA3 varlen path 还会运行 `prepare_varlen_num_blocks`，为 dynamic persistent scheduler 准备：

```text
num_m_blocks per batch
num_splits_dynamic per batch
batch sorting metadata
num_nheads_in_l2 per batch
tile_count_semaphore
```

阶段一暂不进入这套复杂路径。

## 7. Llama 4 上的 FA3-like split 估算

目标形状：

```text
Hq = 40
Hkv = 8
D = 128
qhead_per_khead = 5
SM = 132  # H200
```

contiguous full-prefix fp16/bf16 下近似：

```text
blockM = 128
blockN = 176
num_m_blocks = ceil(1 * 5 / 128) = 1
base_tiles = B * 8
```

估算 split：

| Batch | base tiles | expected split | work tiles |
| ---: | ---: | ---: | ---: |
| 1 | 8 | 14-15 | 112-120 |
| 2 | 16 | 8 | 128 |
| 4 | 32 | 4 | 128 |
| 8 | 64 | 2 | 128 |
| 16 | 128 | 1 | 128 |

按 KV length 细化，`B=1` 大致为：

| KV length | expected split |
| ---: | ---: |
| 512 | 1 |
| 1K | 6 |
| 2K | 11 |
| 4K | 14 |
| 8K+ | 15 |

这里的 `8K+` 并不表示更长 KV 一定需要更多 split。对 `B=1,Hkv=8` 来说，split 15 已经把 work tiles 推到 120，达到 heuristic 认为足够好的 SM wave efficiency；继续增加 split 会增加 partial workspace 和 combine 成本。

## 8. 对 TileOps 阶段一的启发

TileOps 阶段一应效仿 FA3 的外层策略：

```text
PackGQA
fixed split-KV
partial attention kernel
separate combine kernel
homogeneous contiguous KV
single-tile style scheduler
```

不需要第一版实现：

```text
varlen dynamic persistent scheduler
paged-KV TMA
L2-aware varlen batch sorting
dynamic per-batch split
append KV
```

最重要的性能假设：

```text
small batch decode 性能来自 split-KV 填满 SM
PackGQA 降低 head-grid 并行度，但减少重复 K/V 读取
split-KV 负责把并行度补回来
```

因此首轮 benchmark 应验证：

```text
B=1 时 split 14/15/16 是否最优
B 增大时最佳 split 是否按 8/4/2/1 下降
combine overhead 是否低于 split 带来的 SM occupancy 收益
PackGQA 是否优于 no-pack ablation
```

## 9. 与阶段一计划的对应关系

本策略文档对应：

```text
llama4_dense_decode_splitkv_stage1_plan_cn.md
```

阶段一计划定义我们要实现的 TileOps kernel；本文定义 FA3 的对照策略与调参预期。

