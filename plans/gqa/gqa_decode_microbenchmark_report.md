# GQA Decode Microbenchmark 实验报告

日期：2026-06-09

## 目标

本报告用于持续记录 GQA decode kernel dispatch 建模所需的 microbenchmark 数据。
当前阶段聚焦：

```text
small batch
single-token decode
dense KV
contiguous KV
fp16
D = 128
```

覆盖两个主 workload：

```text
Llama 4:          Hq = 40, Hkv = 8, group = 5
Qwen3.5-35B-A3B: Hq = 16, Hkv = 2, group = 8
```

实验目标不是证明单一 kernel 最优，而是测出不同参数区间的策略边界：

```text
no-split vs split-KV
TileOps split vs TileOps WS
TileOps split vs FA3
TileOps split vs FlashInfer TC
best num_split vs S / scenario
```

## 实验入口

实验 worktree：

```text
/home/ga/TileOPs-llama4-decode-splitkv
```

新增 microbenchmark 脚本：

```text
benchmarks/ops/attention/bench_gqa_decode_policy_microbench.py
```

脚本输出 JSONL，每条记录包含：

```text
type, status,
scenario, B, S, Hq, Hkv, group, D, dtype, layout, backend,
block_N, num_split, chunk_len, scheduler, use_ws,
latency_ms, tflops, bandwidth_tbs, max_diff,
skip_reason, error_log
```

## 运行方式

在 nightly Docker / GPU1 中运行。TileLang JIT 建议保留临时目录清理变量：

```bash
docker run --rm --gpus '"device=1"' --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e TILELANG_CLEANUP_TEMP_FILES=1 \
  -e PYTHONPATH=/workspace/TileOPs_live \
  -v /home/ga/TileOPs-llama4-decode-splitkv:/workspace/TileOPs_live \
  -w /workspace/TileOPs_live \
  tileops-runner-sshd:nightly-tl019-fullstack-no-tileops-ldfix-registered-tmux \
  /bin/bash -lc '
    python -m benchmarks.ops.attention.bench_gqa_decode_policy_microbench \
      --scenarios llama4_g5_hkv8,qwen35_g8_hkv2 \
      --b-list 1 \
      --s-list 4096,8192 \
      --splits 1,8,15 \
      --block-n-list 128 \
      --backends tileops_split,tileops_ws,fa3,flashinfer_tc \
      --output results/gqa_decode_policy_microbench.jsonl
  '
```

后续完整 sweep：

```text
B in {1, 2}
S in {4K, 8K, 16K, 32K, 64K, 128K}
num_split in {1, 2, 4, 8, 12, 15, 16, 24, 32}
block_N in {64, 128, 256}
backend in {tileops_split, tileops_ws, fa3, flashinfer_tc}
scenario in {llama4_g5_hkv8, qwen35_g8_hkv2}
```

## 已有观测

### Llama 4 group=5

已有手动测量：

```text
S=4K:  FlashInfer TC 快于 TileOps split/WS
S=8K:  FlashInfer TC 接近 FA3，并快于 TileOps split/WS
S=32K: TileOps split-KV 快于 FlashInfer TC
S=64K: TileOps split-KV 明显快于 FlashInfer TC
S=128K: 待补测
```

S64K 记录：

```text
FlashInfer TC: 0.1284 ms
FA3 split15:   0.1059 ms
TileOps split: 0.0814 ms
TileOps WS:    0.0856 ms
```

### Qwen3.5 group=8

已有手动测量：

```text
Hq=16, Hkv=2, group=8

S=4K:  TileOps split 0.0128 ms, TileOps WS 0.0125 ms
S=8K:  TileOps split 0.0186 ms, TileOps WS 0.0184 ms
S=32K: TileOps split 0.0352 ms, TileOps WS 0.0359 ms
```

初步看，WS 在短 S 上略有机会，但还没有稳定优势，需要系统 sweep。

## 下一步

1. 先跑 smoke sweep，确认 JSONL 输出和 backend 可用性。
2. 跑 Llama4 `B=1, S={4K,8K,16K,32K,64K,128K}` 的 split 曲线。
3. 跑 Qwen3.5 `B=1, S={4K,8K,16K,32K,64K,128K}` 的 split 曲线。
4. 对每个 scenario 找 `best num_split` 和 `WS crossover`。
5. 把结果整理成第一版 dispatch policy 表。

## 实验记录

### 2026-06-09 Iteration 1: smoke harness

建立 microbenchmark 脚本和本报告，并用 nightly Docker / GPU1 跑第一轮 smoke。

运行参数：

```text
scenario in {llama4_g5_hkv8, qwen35_g8_hkv2}
B = 1
S = 4K
num_split in {1, 8}
block_N = 128
backend in {tileops_split, tileops_ws, fa3, flashinfer_tc}
n_warmup = 2
n_repeat = 10
n_trials = 1
```

结果文件：

```text
/home/ga/TileOPs-llama4-decode-splitkv/results/gqa_decode_policy_microbench_iter001_smoke.jsonl
```

环境：

```text
GPU: NVIDIA H200
Torch: 2.10.0+cu128
Driver: 575.57.08
```

注意：本轮是 harness smoke，计时 repeat/trial 较短，不作为正式性能结论。

关键结果：

```text
Llama4 group=5, S=4K:
    split=1 tileops_split: 0.043309 ms
    split=1 fa3:           0.049209 ms
    split=1 flashinfer_tc: unsupported, group_size=5
    split=8 tileops_split: 0.014218 ms
    split=8 tileops_ws:    0.014893 ms
    split=8 fa3:           0.015584 ms

Qwen3.5 group=8, S=4K:
    split=1 tileops_split: 0.045239 ms
    split=1 fa3:           0.048777 ms
    split=1 flashinfer_tc: 0.022608 ms
    split=8 tileops_split: 0.012835 ms
    split=8 tileops_ws:    0.012352 ms
    split=8 fa3:           0.014413 ms
```

本轮结论：

```text
1. JSONL schema 已能记录 result / skip。
2. FlashInfer group=5 unsupported 已不再中断 sweep。
3. Qwen3.5 group=8 的 FlashInfer TC backend 可用。
4. TileOps split / WS / FA3 均能在两个 scenario 上输出结果。
5. 下一轮可以开始正式 Level 0 sweep。
```

### 2026-06-09 Iteration 2: S4K/S8K short Level 0 sweep

运行参数：

```text
scenario in {llama4_g5_hkv8, qwen35_g8_hkv2}
B = 1
S in {4K, 8K}
num_split in {1, 4, 8, 15}
block_N = 128
backend in {tileops_split, tileops_ws, fa3, flashinfer_tc}
n_warmup = 10
n_repeat = 50
n_trials = 3
```

结果文件：

```text
/home/ga/TileOPs-llama4-decode-splitkv/results/gqa_decode_policy_microbench_iter002_s4k_s8k.jsonl
```

记录数量：

```text
environment: 1
ok result:   46
skip:        18
```

skip 分类：

```text
FlashInfer group_size=5 unsupported: 2
FlashInfer 不参与 num_split sweep: 12
TileOps WS num_split=1 unsupported: 4
```

各场景当前 top results：

```text
Llama4 group=5, S=4K:
    tileops_split split=8:  0.013994 ms
    tileops_ws    split=8:  0.014828 ms
    fa3           split=15: 0.015140 ms
    tileops_split split=15: 0.015588 ms

Llama4 group=5, S=8K:
    tileops_split split=8:  0.019834 ms
    fa3           split=15: 0.020354 ms
    fa3           split=8:  0.021694 ms
    tileops_split split=15: 0.022964 ms

Qwen3.5 group=8, S=4K:
    tileops_ws    split=15: 0.011215 ms
    tileops_split split=15: 0.012105 ms
    tileops_split split=8:  0.012466 ms
    tileops_ws    split=8:  0.012561 ms

Qwen3.5 group=8, S=8K:
    fa3           split=15: 0.016959 ms
    tileops_ws    split=15: 0.018235 ms
    tileops_split split=15: 0.018404 ms
    tileops_ws    split=8:  0.018608 ms
```

本轮观察：

```text
1. Llama4 group=5 在 S=4K/8K 上，TileOps split=8 是当前最优点。
2. Llama4 group=5 当前 WS 全部慢于 non-WS。
3. Qwen3.5 group=8 在 S=4K 上，WS split=15 明显优于 non-WS 和 FA3。
4. Qwen3.5 group=8 在 S=8K 上，FA3 split=15 暂时领先。
5. split=1 no-split 在两个 scenario 上都明显不是短 S 的最优策略。
```

下一轮：

```text
1. 补 S={16K,32K,64K,128K}。
2. 对 Llama4 重点扫 num_split={8,12,15,16,24,32}。
3. 对 Qwen3.5 保留 WS/non-WS 对照，确认 WS crossover 是否只存在短 S。
4. 开始整理第一版 best-backend / best-split 表。
```

### 2026-06-09 Iteration 3: S16K/S32K Level 0 sweep

运行参数：

```text
scenario in {llama4_g5_hkv8, qwen35_g8_hkv2}
B = 1
S in {16K, 32K}
num_split in {1, 4, 8, 12, 15, 16, 24}
block_N = 128
backend in {tileops_split, tileops_ws, fa3, flashinfer_tc}
n_warmup = 10
n_repeat = 50
n_trials = 3
```

结果文件：

```text
/home/ga/TileOPs-llama4-decode-splitkv/results/gqa_decode_policy_microbench_iter003_s16k_s32k.jsonl
```

记录数量：

```text
environment: 1
ok result:   82
skip:        30
```

skip 分类：

```text
FlashInfer group_size=5 unsupported: 2
FlashInfer 不参与 num_split sweep: 24
TileOps WS num_split=1 unsupported: 4
```

各场景当前 top results：

```text
Llama4 group=5, S=16K:
    tileops_split split=16: 0.029191 ms
    tileops_ws    split=16: 0.029894 ms
    fa3           split=12: 0.030049 ms
    fa3           split=16: 0.030520 ms
    tileops_split split=8:  0.031016 ms

Llama4 group=5, S=32K:
    tileops_split split=16: 0.046391 ms
    tileops_ws    split=16: 0.047413 ms
    tileops_split split=15: 0.047733 ms
    tileops_ws    split=15: 0.048268 ms
    fa3           split=12: 0.048500 ms

Qwen3.5 group=8, S=16K:
    fa3           split=24: 0.018353 ms
    tileops_split split=16: 0.018669 ms
    tileops_ws    split=16: 0.019645 ms
    fa3           split=16: 0.020888 ms
    fa3           split=15: 0.022447 ms
    flashinfer_tc split=1:  0.024724 ms

Qwen3.5 group=8, S=32K:
    fa3           split=24: 0.026022 ms
    tileops_split split=16: 0.030344 ms
    fa3           split=16: 0.031270 ms
    tileops_ws    split=16: 0.032843 ms
    tileops_split split=15: 0.032930 ms
```

本轮观察：

```text
1. Llama4 group=5 在 S=16K/32K 上，TileOps split=16 成为当前最优点。
2. Llama4 group=5 的 WS 仍然略慢于 non-WS，但 split=16 区间差距已经缩小。
3. Qwen3.5 group=8 在 S=16K/32K 上，FA3 split=24 目前领先。
4. Qwen3.5 group=8 的 TileOps split=16 是最接近的 TileOps 点。
5. best num_split 非单调：Llama4 从 S=4K/8K 的 split=8 转到 S=16K/32K 的 split=16；过大的 split=24 会退化。
```

下一轮：

```text
1. 跑 S={64K,128K}。
2. Llama4 重点保留 split in {8, 15, 16, 24, 32}。
3. Qwen3.5 重点保留 split in {16, 24, 32}，并保留 FlashInfer TC split=1 对照。
4. 观察长上下文下 TileOps 是否重新反超 FA3 / FlashInfer。
```

### 2026-06-09 Iteration 4: S64K/S128K Level 0 sweep

运行参数：

```text
scenario in {llama4_g5_hkv8, qwen35_g8_hkv2}
B = 1
S in {64K, 128K}
num_split in {1, 8, 15, 16, 24, 32}
block_N = 128
backend in {tileops_split, tileops_ws, fa3, flashinfer_tc}
n_warmup = 10
n_repeat = 50
n_trials = 3
```

结果文件：

```text
/home/ga/TileOPs-llama4-decode-splitkv/results/gqa_decode_policy_microbench_iter004_s64k_s128k.jsonl
```

记录数量：

```text
environment: 1
ok result:   70
skip:        26
```

各场景当前 top results：

```text
Llama4 group=5, S=64K:
    tileops_split split=16: 0.077572 ms
    tileops_split split=32: 0.081125 ms
    tileops_ws    split=16: 0.081219 ms
    tileops_split split=15: 0.081313 ms
    fa3           split=16: 0.103605 ms

Llama4 group=5, S=128K:
    tileops_split split=16: 0.135570 ms
    tileops_split split=32: 0.138637 ms
    tileops_split split=15: 0.142298 ms
    tileops_ws    split=16: 0.150724 ms
    fa3           split=16: 0.217072 ms

Qwen3.5 group=8, S=64K:
    tileops_split split=32: 0.031005 ms
    fa3           split=32: 0.035544 ms
    tileops_ws    split=32: 0.038426 ms
    fa3           split=24: 0.040676 ms
    tileops_split split=24: 0.047950 ms

Qwen3.5 group=8, S=128K:
    tileops_split split=32: 0.053475 ms
    fa3           split=32: 0.062277 ms
    tileops_ws    split=32: 0.063729 ms
    fa3           split=24: 0.084549 ms
    tileops_split split=24: 0.088048 ms
```

本轮观察：

```text
1. Llama4 group=5 在 S=64K/128K 上继续由 TileOps split 领先。
2. Llama4 的最佳 split 稳定在 16；split=32 接近但没有超过 split=16。
3. Llama4 的 WS 继续慢于 non-WS。
4. Qwen3.5 group=8 在 S=64K/128K 上由 TileOps split=32 明显领先。
5. Qwen3.5 的 long-context crossover 大致发生在 S=32K 到 64K 之间。
6. Qwen3.5 的 WS 在长 S 上不如 non-WS。
```

### 2026-06-09 Iteration 5: Qwen3.5 focused split sweep

运行参数：

```text
scenario = qwen35_g8_hkv2
B = 1
S in {4K, 32K, 64K}
num_split in {1, 12, 15, 16, 20, 24, 28, 32}
block_N = 128
backend in {tileops_split, tileops_ws, fa3, flashinfer_tc}
n_warmup = 10
n_repeat = 50
n_trials = 3
```

结果文件：

```text
/home/ga/TileOPs-llama4-decode-splitkv/results/gqa_decode_policy_microbench_iter005_qwen_focus.jsonl
```

记录数量：

```text
environment: 1
ok result:   72
skip:        24
```

各 S 当前 top results：

```text
Qwen3.5 group=8, S=4K:
    tileops_ws    split=32: 0.007609 ms
    tileops_split split=32: 0.008479 ms
    tileops_ws    split=16: 0.008886 ms
    tileops_split split=16: 0.009485 ms
    tileops_ws    split=15: 0.010804 ms
    tileops_split split=15: 0.012107 ms
    fa3           split=24: 0.012205 ms
    fa3           split=28: 0.012220 ms

Qwen3.5 group=8, S=32K:
    tileops_split split=32: 0.019880 ms
    tileops_ws    split=32: 0.023159 ms
    fa3           split=32: 0.023604 ms
    fa3           split=28: 0.024610 ms
    fa3           split=24: 0.025774 ms
    tileops_split split=28: 0.025986 ms

Qwen3.5 group=8, S=64K:
    tileops_split split=32: 0.031151 ms
    fa3           split=32: 0.035837 ms
    fa3           split=28: 0.037645 ms
    tileops_ws    split=32: 0.038405 ms
    fa3           split=24: 0.040720 ms
    fa3           split=20: 0.044408 ms
```

本轮观察：

```text
1. Qwen3.5 S=4K 的最佳点从 Iteration 2 的 WS split=15 更新为 WS split=32。
2. S=4K split=32 对应 chunk_len=128，速度明显高于上一轮，需要追加 correctness spot-check，避免把异常点直接纳入 dispatch。
3. Qwen3.5 S=32K 的最佳点从 Iteration 3 的 FA3 split=24 更新为 TileOps split=32。
4. Qwen3.5 S=64K 继续由 TileOps split=32 领先。
5. WS 目前只在 S=4K 明确领先；S=32K/64K 上 non-WS split=32 更优。
```

### 2026-06-09 Iteration 6: Llama4 focused split boundary sweep

运行参数：

```text
scenario = llama4_g5_hkv8
B = 1
S in {8K, 16K}
num_split in {1, 6, 8, 10, 12, 14, 15, 16, 20, 24, 32}
block_N = 128
backend in {tileops_split, tileops_ws, fa3, flashinfer_tc}
n_warmup = 10
n_repeat = 50
n_trials = 3
```

结果文件：

```text
/home/ga/TileOPs-llama4-decode-splitkv/results/gqa_decode_policy_microbench_iter006_llama_boundary.jsonl
```

记录数量：

```text
environment: 1
ok result:   64
skip:        24
```

各 S 当前 top results：

```text
Llama4 group=5, S=8K:
    tileops_split split=16: 0.018377 ms
    tileops_ws    split=16: 0.018808 ms
    tileops_split split=8:  0.019761 ms
    fa3           split=16: 0.019969 ms
    fa3           split=14: 0.020137 ms
    fa3           split=12: 0.020220 ms

Llama4 group=5, S=16K:
    tileops_split split=16: 0.028454 ms
    tileops_ws    split=16: 0.029570 ms
    fa3           split=12: 0.029915 ms
    fa3           split=14: 0.029918 ms
    fa3           split=15: 0.030218 ms
    fa3           split=16: 0.030324 ms
```

本轮观察：

```text
1. Llama4 S=8K 的最佳点从 Iteration 2 的 split=8 更新为 TileOps split=16。
2. Llama4 S=16K 继续由 TileOps split=16 领先。
3. Llama4 split=16 对应的 chunk_len 在 S=8K/16K 分别为 512/1024，表现明显强于过小或过大的 chunk。
4. WS 在 split=16 时非常接近 non-WS，但仍未超过 non-WS。
5. FA3 在 split=12-16 区间比较平稳，是重要 baseline，但当前没有超过 TileOps split=16。
```

### 2026-06-09 Iteration 7: Qwen3.5 S4K split=32 correctness spot-check

运行参数：

```text
scenario = qwen35_g8_hkv2
B = 1
S = 4K
num_split = 32
block_N = 128
backend in {tileops_split, tileops_ws, fa3}
check = true
n_warmup = 2
n_repeat = 10
n_trials = 1
```

结果文件：

```text
/home/ga/TileOPs-llama4-decode-splitkv/results/gqa_decode_policy_microbench_iter007_qwen_s4k_split32_check.jsonl
```

结果：

```text
Qwen3.5 group=8, S=4K, split=32:
    tileops_split max_diff = 6.103515625e-05
    tileops_ws    max_diff = 6.103515625e-05
    fa3           max_diff = 3.0517578125e-05
```

本轮观察：

```text
1. Qwen3.5 S=4K split=32 的 TileOps split / WS correctness spot-check 通过。
2. 本轮 repeat/trial 较短，不更新正式性能数字。
3. Qwen3.5 S=4K WS split=32 可以进入 dispatch 候选，但仍需正式 repeat/trial 复测确认稳定性。
```

### 2026-06-09 Iteration 8: Qwen3.5 S8K-S32K crossover focused sweep

运行参数：

```text
scenario = qwen35_g8_hkv2
B = 1
S in {8K, 16K, 24K, 32K}
num_split in {1, 12, 15, 16, 20, 24, 28, 32}
block_N = 128
backend in {tileops_split, tileops_ws, fa3, flashinfer_tc}
n_warmup = 10
n_repeat = 50
n_trials = 3
```

结果文件：

```text
/home/ga/TileOPs-llama4-decode-splitkv/results/gqa_decode_policy_microbench_iter008_qwen_crossover.jsonl
```

记录数量：

```text
environment: 1
ok result:   96
skip:        32
```

各 S 当前 top results：

```text
Qwen3.5 group=8, S=8K:
    tileops_ws    split=32: 0.010728 ms
    tileops_split split=32: 0.010779 ms
    tileops_split split=16: 0.012743 ms
    tileops_ws    split=16: 0.012861 ms
    fa3           split=32: 0.014714 ms
    fa3           split=24: 0.014727 ms

Qwen3.5 group=8, S=16K:
    tileops_split split=32: 0.014258 ms
    tileops_ws    split=32: 0.014927 ms
    fa3           split=32: 0.017398 ms
    fa3           split=28: 0.018317 ms
    fa3           split=24: 0.018332 ms
    tileops_split split=16: 0.018492 ms

Qwen3.5 group=8, S=24K:
    tileops_split split=32: 0.017434 ms
    tileops_split split=24: 0.019030 ms
    tileops_ws    split=32: 0.019457 ms
    fa3           split=32: 0.020882 ms
    fa3           split=28: 0.020918 ms
    tileops_ws    split=24: 0.021131 ms

Qwen3.5 group=8, S=32K:
    tileops_split split=32: 0.019906 ms
    tileops_ws    split=32: 0.023202 ms
    fa3           split=32: 0.023595 ms
    fa3           split=28: 0.024586 ms
    fa3           split=24: 0.025923 ms
    tileops_split split=28: 0.025945 ms
```

本轮观察：

```text
1. Qwen3.5 的 FA3 -> TileOps crossover 比 Iteration 3/4 粗扫判断得更早；S=8K 起 TileOps split=32 已经领先。
2. S=8K 时 WS split=32 略快于 non-WS，差距很小。
3. S=16K/24K/32K 时 non-WS split=32 最快，WS 明显落后。
4. Qwen3.5 的强候选策略可以先简化为：S=4K/8K 试 WS split=32，S>=16K 用 TileOps split=32。
5. 需要继续对 Qwen3.5 split=32 在 S={8K,16K,24K,32K} 做 correctness spot-check。
```

### 2026-06-09 Iteration 9: Qwen3.5 split=32 correctness spot-check

运行参数：

```text
scenario = qwen35_g8_hkv2
B = 1
S in {8K, 16K, 24K, 32K}
num_split = 32
block_N = 128
backend in {tileops_split, tileops_ws, fa3}
check = true
n_warmup = 1
n_repeat = 5
n_trials = 1
```

结果文件：

```text
/home/ga/TileOPs-llama4-decode-splitkv/results/gqa_decode_policy_microbench_iter009_qwen_split32_check.jsonl
```

结果：

```text
Qwen3.5 group=8, split=32:
    S=8K:
        tileops_split max_diff = 3.0517578125e-05
        tileops_ws    max_diff = 3.0517578125e-05
        fa3           max_diff = 3.0517578125e-05

    S=16K:
        tileops_split max_diff = 3.0517578125e-05
        tileops_ws    max_diff = 1.52587890625e-05
        fa3           max_diff = 3.0517578125e-05

    S=24K:
        tileops_split max_diff = 1.52587890625e-05
        tileops_ws    max_diff = 1.52587890625e-05
        fa3           max_diff = 1.52587890625e-05

    S=32K:
        tileops_split max_diff = 1.52587890625e-05
        tileops_ws    max_diff = 1.52587890625e-05
        fa3           max_diff = 1.52587890625e-05
```

本轮观察：

```text
1. Qwen3.5 S={8K,16K,24K,32K} 的 split=32 correctness spot-check 通过。
2. 本轮 repeat/trial 较短，不更新正式性能数字。
3. 结合 Iteration 7，Qwen3.5 S={4K,8K,16K,24K,32K} 的 split=32 候选都有 correctness 记录。
```

### 2026-06-09 Iteration 10: Llama4 S4K split focused sweep

运行参数：

```text
scenario = llama4_g5_hkv8
B = 1
S = 4K
num_split in {8, 12, 16, 24, 32}
block_N = 128
backend in {tileops_split, tileops_ws, fa3}
n_warmup = 10
n_repeat = 50
n_trials = 3
```

结果文件：

```text
/home/ga/TileOPs-llama4-decode-splitkv/results/gqa_decode_policy_microbench_iter010_llama_s4k_focus.jsonl
```

各 split 当前 top results：

```text
Llama4 group=5, S=4K:
    tileops_ws    split=16: 0.013194 ms
    tileops_split split=16: 0.013278 ms
    tileops_split split=8:  0.013873 ms
    tileops_ws    split=32: 0.014091 ms
    tileops_ws    split=8:  0.014788 ms
    fa3           split=12: 0.014959 ms
```

本轮观察：

```text
1. Llama4 S=4K 的最佳观测点从 TileOps split=8 更新为 TileOps WS split=16。
2. WS split=16 与 non-WS split=16 差距很小，约 0.6%，需要 correctness / stability 复测后再决定是否在 dispatch 中启用 WS。
3. split=16 对 S=4K/8K/16K 都是强候选，Llama4 的第一版规则可以先统一为 split=16。
4. split=24/32 在 S=4K 明显不如 split=16，过小 chunk 会带来额外开销。
```

### 2026-06-09 Iteration 11: Llama4 S4K split=16 correctness spot-check

运行参数：

```text
scenario = llama4_g5_hkv8
B = 1
S = 4K
num_split = 16
block_N = 128
backend in {tileops_split, tileops_ws, fa3}
check = true
n_warmup = 1
n_repeat = 5
n_trials = 1
```

结果文件：

```text
/home/ga/TileOPs-llama4-decode-splitkv/results/gqa_decode_policy_microbench_iter011_llama_s4k_split16_check.jsonl
```

结果：

```text
Llama4 group=5, S=4K, split=16:
    tileops_split max_diff = 6.103515625e-05
    tileops_ws    max_diff = 6.103515625e-05
    fa3           max_diff = 1.220703125e-04
```

本轮观察：

```text
1. Llama4 S=4K split=16 的 TileOps split / WS correctness spot-check 通过。
2. WS split=16 的性能小胜仍需要稳定性复测；在 dispatch 里可以先保守选择 non-WS split=16，或把 WS 作为短 S candidate。
```

### 2026-06-09 Iteration 12: upstream GQA decode comparison

运行方式：

```text
upstream worktree: /home/ga/TileOPs
experiment worktree: /home/ga/TileOPs-llama4-decode-splitkv
GPU: H200, GPU1
B = 1
S in {4K, 8K, 16K, 32K, 64K, 128K}
block_N = 128
upstream: tileops_split, split in {16, 32}
experiment: tileops_split/tileops_ws, split in {16, 32}
```

结果文件：

```text
/home/ga/TileOPs/results/gqa_decode_policy_microbench_upstream_compare_iter001.jsonl
/home/ga/TileOPs-llama4-decode-splitkv/results/gqa_decode_policy_microbench_exp_compare_iter012.jsonl
```

对比结论：

```text
Llama4 group=5:
    upstream split=16 已经非常强。
    experiment best 相比 upstream best 没有领先，约为 0.94x-0.99x。
    当前不能声称 Llama4 kernel 本体领先 upstream。

Qwen3.5 group=8:
    如果对比 upstream default split=16，experiment dispatch 到 split=32 后有明显收益：
        S=4K:    1.19x
        S=8K:    1.17x
        S=16K:   1.33x
        S=32K:   1.57x
        S=64K:   1.77x
        S=128K:  1.91x

    如果 upstream 也手动使用 split=32，experiment non-WS 与 upstream 基本持平：
        S=4K:    0.99x
        S=8K:    1.00x
        S=16K:   0.995x
        S=32K:   1.005x
        S=64K:   1.002x
        S=128K:  1.000x
```

本轮观察：

```text
1. 目前主要收益来自 dispatch 参数选择，尤其 Qwen3.5 的 split=32，而不是 non-WS kernel 本体超过 upstream。
2. WS 在 Qwen3.5 S=4K 有约 9% 的增益，相比 upstream split=32；在 S>=8K 没有稳定优势。
3. Llama4 需要谨慎：实验 worktree 的 split path 使用 fp32 partial workspace 后，性能相对 upstream split path 略慢。
4. 下一阶段应把目标表述为“建立更好的 GQA decode dispatch 策略”，而不是“当前 kernel 已全面超过 upstream”。
```

### 2026-06-09 Iteration 13: upstream correctness spot-check

运行方式：

```text
upstream worktree: /home/ga/TileOPs
B = 1
scenario in {llama4_g5_hkv8, qwen35_g8_hkv2}
S in {4K, 8K, 32K}
split in {16, 32}
backend = tileops_split
check = true
reference = torch SDPA math
```

结果文件：

```text
/home/ga/TileOPs/results/gqa_decode_policy_microbench_upstream_check_iter002.jsonl
```

结果：

```text
Llama4 group=5:
    S=4K,  split=16: max_diff = 1.8310546875e-04
    S=4K,  split=32: max_diff = 1.2207031250e-04
    S=8K,  split=16: max_diff = 1.3732910156e-04
    S=8K,  split=32: max_diff = 1.2207031250e-04
    S=32K, split=16: max_diff = 6.1035156250e-05
    S=32K, split=32: max_diff = 5.3405761719e-05

Qwen3.5 group=8:
    S=4K,  split=16: max_diff = 1.7166137695e-04
    S=4K,  split=32: max_diff = 1.2207031250e-04
    S=8K,  split=16: max_diff = 9.1552734375e-05
    S=8K,  split=32: max_diff = 1.2207031250e-04
    S=32K, split=16: max_diff = 5.7220458984e-05
    S=32K, split=32: max_diff = 5.3405761719e-05
```

本轮观察：

```text
1. upstream GQA decode 在抽样场景下 correctness 通过。
2. 最大 max_diff 为 1.83e-4，属于 fp16 decode 与 torch math reference 对比的可接受范围。
3. 因此 Iteration 12 的 upstream 性能对比可以视为有效，不是错误输出导致的虚高性能。
```

### 2026-06-09 Iteration 14: fast-partial correctness spot-check

背景：Iteration 12/13 后确认 upstream non-WS split path 本身正确且很强。本地 non-WS 相比 upstream 的主要性能差异来自实验版 fp32 partial workspace / combine。为避免本地 non-WS 和 WS 被 fp32 partial 带宽成本拖慢，本轮把本地 `GQADecodeKernel` 的 split partial 输出恢复为 fp16 fast path；WS path 也保持 fp16 partial，与 upstream-style combine 对齐。

运行参数：

```text
scenario in {llama4_g5_hkv8, qwen35_g8_hkv2}
B = 1
S in {4K, 8K, 32K}
num_split in {16, 32}
block_N = 128
backend in {tileops_split, tileops_ws}
check = true
n_warmup = 1
n_repeat = 5
n_trials = 1
```

结果文件：

```text
/home/ga/TileOPs-llama4-decode-splitkv/results/gqa_decode_policy_microbench_iter014_fastpartial_check.jsonl
```

correctness 结果：

```text
ok result rows: 24
max_diff range: 4.76837158203125e-05 ~ 1.983642578125e-04
max_diff max:   1.983642578125e-04
```

本轮观察：

```text
1. fast-partial 本地 non-WS / WS correctness spot-check 全部通过。
2. 最大误差约 2e-4，与 upstream fp16 partial spot-check 同一量级。
3. 因此可以把 fp16 partial 作为性能默认路径；fp32 partial 更适合作为后续 debug / stability candidate，而不是默认性能路径。
```

### 2026-06-09 Iteration 15: fast-partial formal perf sweep

运行参数：

```text
scenario in {llama4_g5_hkv8, qwen35_g8_hkv2}
B = 1
S in {4K, 8K, 16K, 32K, 64K, 128K}
num_split in {16, 32}
block_N = 128
backend in {tileops_split, tileops_ws}
n_warmup = 10
n_repeat = 50
n_trials = 3
```

结果文件：

```text
/home/ga/TileOPs-llama4-decode-splitkv/results/gqa_decode_policy_microbench_iter015_fastpartial_perf.jsonl
```

Llama4 group=5 当前代码结果：

```text
S=4K:
    tileops_split split=16: 0.012345 ms
    tileops_ws    split=16: 0.013181 ms
    tileops_split split=32: 0.014602 ms
    tileops_ws    split=32: 0.014093 ms
    best: tileops_split split=16

S=8K:
    tileops_split split=16: 0.017786 ms
    tileops_ws    split=16: 0.018723 ms
    tileops_split split=32: 0.021333 ms
    tileops_ws    split=32: 0.021591 ms
    best: tileops_split split=16

S=16K:
    tileops_split split=16: 0.028127 ms
    tileops_ws    split=16: 0.029523 ms
    tileops_split split=32: 0.030871 ms
    tileops_ws    split=32: 0.031539 ms
    best: tileops_split split=16

S=32K:
    tileops_split split=16: 0.046014 ms
    tileops_ws    split=16: 0.047348 ms
    tileops_split split=32: 0.048366 ms
    tileops_ws    split=32: 0.049299 ms
    best: tileops_split split=16

S=64K:
    tileops_split split=16: 0.076802 ms
    tileops_ws    split=16: 0.081263 ms
    tileops_split split=32: 0.079194 ms
    tileops_ws    split=32: 0.083385 ms
    best: tileops_split split=16

S=128K:
    tileops_split split=16: 0.136209 ms
    tileops_ws    split=16: 0.152092 ms
    tileops_split split=32: 0.139590 ms
    tileops_ws    split=32: 0.153591 ms
    best: tileops_split split=16
```

Qwen3.5 group=8 当前代码结果：

```text
S=4K:
    tileops_split split=16: 0.010056 ms
    tileops_ws    split=16: 0.010086 ms
    tileops_split split=32: 0.009308 ms
    tileops_ws    split=32: 0.008563 ms
    best: tileops_ws split=32

S=8K:
    tileops_split split=16: 0.012931 ms
    tileops_ws    split=16: 0.013157 ms
    tileops_split split=32: 0.010959 ms
    tileops_ws    split=32: 0.011253 ms
    best: tileops_split split=32

S=16K:
    tileops_split split=16: 0.018750 ms
    tileops_ws    split=16: 0.019416 ms
    tileops_split split=32: 0.013953 ms
    tileops_ws    split=32: 0.014981 ms
    best: tileops_split split=32

S=32K:
    tileops_split split=16: 0.030396 ms
    tileops_ws    split=16: 0.032768 ms
    tileops_split split=32: 0.019572 ms
    tileops_ws    split=32: 0.023179 ms
    best: tileops_split split=32

S=64K:
    tileops_split split=16: 0.054302 ms
    tileops_ws    split=16: 0.059093 ms
    tileops_split split=32: 0.030584 ms
    tileops_ws    split=32: 0.038783 ms
    best: tileops_split split=32

S=128K:
    tileops_split split=16: 0.101387 ms
    tileops_ws    split=16: 0.104696 ms
    tileops_split split=32: 0.052765 ms
    tileops_ws    split=32: 0.063883 ms
    best: tileops_split split=32
```

相对 Iteration 12 实验版 fp32 partial 的变化：

```text
Llama4 tileops_split:
    split=16: S=4K 1.09x faster, S=8K 1.05x, S=16K 1.03x, S>=32K about 1.01x-1.02x
    split=32: S=4K 1.13x faster, S=8K 1.09x, S=16K 1.06x, S>=32K about 1.01x-1.04x

Qwen3.5 tileops_split:
    split=16/32: mostly unchanged, about 0.99x-1.02x

tileops_ws:
    effectively unchanged versus Iteration 12, because WS was already on the fp16 partial fast path.
```

本轮观察：

```text
1. 本地 non-WS 已回到 upstream-style fp16 partial fast path，Llama4 的 fp32 partial 额外开销被消除。
2. Llama4 最佳策略更明确：S=4K-128K 都选择 tileops_split split=16；WS 不再是 4K 默认候选。
3. Qwen3.5 最佳策略也更明确：S=4K 选择 tileops_ws split=32；S>=8K 选择 tileops_split split=32。
4. Qwen3.5 的主要收益仍来自 dispatch 到 split=32；WS 只在 4K 有稳定额外收益。
5. 当前可以把默认性能路径表述为：non-WS 与 upstream kernel 本体基本对齐，新增收益来自 split policy + very-short-S WS candidate。
6. fp16 partial fast path 下，num_split=15 的 combine reduce_max 在 TileLang lowering 中会触发 layout mismatch；当前默认 dispatch 不使用 split=15，后续如需支持任意 split，应增加 combine fallback 或 padding 策略。
```

## 当前全局 best table

以 Iteration 15 fast-partial 当前代码版本为准，当前 `B=1, block_N=128` 的推荐点：

```text
Llama4 group=5:
    S=4K:    tileops_split split=16  0.012345 ms
    S=8K:    tileops_split split=16  0.017786 ms
    S=16K:   tileops_split split=16  0.028127 ms
    S=32K:   tileops_split split=16  0.046014 ms
    S=64K:   tileops_split split=16  0.076802 ms
    S=128K:  tileops_split split=16  0.136209 ms

Qwen3.5 group=8:
    S=4K:    tileops_ws    split=32  0.008563 ms
    S=8K:    tileops_split split=32  0.010959 ms
    S=16K:   tileops_split split=32  0.013953 ms
    S=32K:   tileops_split split=32  0.019572 ms
    S=64K:   tileops_split split=32  0.030584 ms
    S=128K:  tileops_split split=32  0.052765 ms
```

第一版 dispatch 观察：

```text
Llama4 group=5:
    4K-128K:    TileOps split=16
    WS:         当前 formal sweep 不启用；短 S 没有稳定超过 non-WS
    FlashInfer: group_size=5 unsupported

Qwen3.5 group=8:
    4K:         TileOps WS split=32
    >=8K:       TileOps split=32
    WS:         只在 4K 默认启用；>=8K 暂不启用
```

下一轮：

```text
1. 把 fast-partial 代码 diff 拆成清晰 patch：non-WS upstream-style fp16 partial restore + WS use_ws path。
2. 增加/整理最小 correctness 测试，覆盖 Llama4 split=16 和 Qwen3.5 split=32。
3. 基于当前表实现第一版 dispatch policy：Llama4 -> split=16；Qwen3.5 -> split=32，且仅 S=4K 启用 WS。
4. 如需保留 fp32 partial，把它降级为显式 debug/stability config，而非默认路径。
5. 评估 split=15 等非整齐 split 的 combine fallback，决定是否纳入可调参数空间。
```
