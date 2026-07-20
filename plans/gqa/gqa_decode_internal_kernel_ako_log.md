# GQA Decode Internal Kernel AKO Log

Date: 2026-06-09

## Scope

This log records AKO-style optimization attempts inside the GQA decode kernels.

Dispatch policy is intentionally out of scope for this round. The baseline dispatch choices found
earlier are useful only as fixed comparison points:

```text
Llama4 group=5:   compare against split=16
Qwen3.5 group=8: compare against split=32
```

The goal here is different:

```text
Given a fixed backend / split point, can a kernel-internal variant beat the current kernel?
```

## Baseline

Protected kernel commit:

```text
2d4b7b7 Add GQA decode WS fast partial kernel
```

Baseline properties:

```text
non-WS split path:
    upstream-style fp16 partial output
    upstream-style combine

WS split path:
    producer / consumer WS kernel
    fp16 partial output
    upstream-style combine
```

Latest correctness / perf references:

```text
correctness:
    results/gqa_decode_policy_microbench_iter014_fastpartial_check.jsonl
    max_diff max = 1.983642578125e-04

performance:
    results/gqa_decode_policy_microbench_iter015_fastpartial_perf.jsonl
```

## Candidate Directions

This round allows kernel-internal directions only:

```text
2. combine kernel optimization
3. WS pipeline tuning
4. group-aware / small-M kernel variant
```

Dispatch changes are not considered a valid improvement in this log.

## AKO Protocol

Each variant must record:

```text
variant id
hypothesis
changed files
fixed comparison point
correctness result
performance result
decision: keep / reject / defer
notes / traps
```

Correctness gate:

```text
Compare against torch SDPA math reference.
Accept fp16 decode max_diff around upstream fast-partial range, roughly <= 2e-4 for this sweep.
```

Performance gate:

```text
Compare against commit 2d4b7b7 on the same GPU / Docker / timing protocol.
Do not compare against upstream default split=16 when evaluating kernel-internal improvement.
```

## Variant v001: group-aware block_H probe

Hypothesis:

```text
GQA decode effective M is the query group size.

Current benchmark config uses block_H=64 while the useful rows are:
    Llama4: group=5
    Qwen3.5: group=8

A smaller block_H may reduce wasted QK / PV work and register/shared-memory pressure.
```

Planned probe:

```text
Llama4 group=5:
    S in {4K, 8K, 32K}
    split=16
    block_H candidates in {8, 16, 32, 64}
    backend in {tileops_split}

Qwen3.5 group=8:
    S in {4K, 8K, 32K}
    split=32
    block_H candidates in {8, 16, 32, 64}
    backend in {tileops_split, tileops_ws}
```

Status:

```text
defer
```

Result:

```text
results/gqa_decode_internal_ako_v001_blockh_probe.jsonl

rows:    36
ok:       9
failed:  27
```

Findings:

```text
block_H=8:
    non-WS: M must be divisible by 16
    WS: WGMMA lowering constraints not satisfied for M=8

block_H=16/32:
    non-WS: TileLang fragment layout conflict between acc_s and acc_s_cast
    WS: WGMMA lowering constraints not satisfied for M=16/32

block_H=64:
    correctness passed and matches the existing baseline
```

Decision:

```text
defer
```

Reason:

```text
The current split / WS kernels cannot become group-aware small-M kernels by only shrinking
block_H. This direction requires a new kernel family or different fragment/layout strategy.
Do not change the protected kernel for v001.
```

## Variant v002: non-WS split_length shared-mask

Hypothesis:

```text
The non-WS split kernel copies split_length to shared memory but still reads split_length[sid]
from the original tensor in the score mask loop.

Using split_length_shared[sid] should remove repeated global-memory reads from the inner mask
loop and align non-WS with the WS path.
```

Changed file:

```text
tileops/kernels/attention/gqa_decode.py
```

Fixed comparison points:

```text
Llama4 group=5:
    split=16, backend=tileops_split

Qwen3.5 group=8:
    split=32, backend=tileops_split
```

Status:

```text
keep
```

Correctness result:

```text
results/gqa_decode_internal_ako_v002_shared_mask.jsonl

rows:         6
status:       all ok
max_diff max: 1.678466796875e-04
```

Formal performance result:

```text
results/gqa_decode_internal_ako_v002_shared_mask_perf.jsonl

protocol:
    n_warmup = 10
    n_repeat = 50
    n_trials = 3
```

Compared with `results/gqa_decode_policy_microbench_iter015_fastpartial_perf.jsonl`:

```text
All non-WS points, old/new speedup:

Llama4 group=5:
    S=4K,   split=16: 1.010x
    S=4K,   split=32: 1.016x
    S=8K,   split=16: 1.008x
    S=8K,   split=32: 1.011x
    S=16K,  split=16: 1.013x
    S=16K,  split=32: 1.012x
    S=32K,  split=16: 1.008x
    S=32K,  split=32: 1.012x
    S=64K,  split=16: 1.008x
    S=64K,  split=32: 1.005x
    S=128K, split=16: 1.000x
    S=128K, split=32: 1.003x

Qwen3.5 group=8:
    S=4K,   split=16: 1.015x
    S=4K,   split=32: 1.007x
    S=8K,   split=16: 1.040x
    S=8K,   split=32: 1.017x
    S=16K,  split=16: 1.061x
    S=16K,  split=32: 1.021x
    S=32K,  split=16: 1.081x
    S=32K,  split=32: 1.029x
    S=64K,  split=16: 1.106x
    S=64K,  split=32: 1.027x
    S=128K, split=16: 1.111x
    S=128K, split=32: 1.042x
```

Recommended fixed-point speedup:

```text
Llama4 split=16:
    avg 1.008x
    min 1.000x
    max 1.013x

Qwen3.5 split=32:
    avg 1.024x
    min 1.007x
    max 1.042x
```

Decision:

```text
keep
```

Reason:

```text
This is a true kernel-internal improvement: dispatch, split, block_N, dtype, and timing protocol
are unchanged. The only data-path change is replacing repeated global split_length reads in the
non-WS score mask with the already-copied shared value.

The improvement is small for Llama4 and larger for Qwen3.5, especially long S. No correctness
regression was observed.
```

## Variant v003: combine glse_vec reuse

Hypothesis:

```text
Both non-WS and WS combine first load glse[bz, by, k] into a fragment glse_vec, then use
glse[bz, by, k] again in the serial logsum and weighted-output loops.

Reusing glse_vec[k] should remove repeated global glse reads in combine.
The expected gain should be larger when num_split is 32 or when combine is more visible.
```

Changed file:

```text
tileops/kernels/attention/gqa_decode.py
```

Fixed comparison points:

```text
Llama4 group=5:
    split=16, backend=tileops_split

Qwen3.5 group=8:
    split=32, backend in {tileops_split, tileops_ws}
```

Status:

```text
keep
```

Correctness result:

```text
results/gqa_decode_internal_ako_v003_glse_vec_probe.jsonl

rows:         9
status:       all ok
max_diff max: 1.8310546875e-04
```

Formal performance result:

```text
results/gqa_decode_internal_ako_v003_glse_vec_perf.jsonl

protocol:
    n_warmup = 10
    n_repeat = 50
    n_trials = 3
```

Compared with v002 for non-WS:

```text
Llama4 group=5, tileops_split:
    mostly 1.000x-1.006x over v002
    slight noise at S=64K, about 0.999x

Qwen3.5 group=8, tileops_split:
    split=32 points improve by about 1.002x-1.009x over v002
```

Compared with Iteration 15 protected fast-partial baseline:

```text
Recommended fixed points:

Llama4 group=5, tileops_split split=16:
    S=4K:    1.014x
    S=8K:    1.014x
    S=16K:   1.017x
    S=32K:   1.010x
    S=64K:   1.007x
    S=128K:  1.003x
    avg:     1.011x

Qwen3.5 group=8, tileops_split split=32:
    S=4K:    1.014x
    S=8K:    1.026x
    S=16K:   1.025x
    S=32K:   1.038x
    S=64K:   1.033x
    S=128K:  1.044x
    avg:     1.030x

Qwen3.5 group=8, tileops_ws split=32:
    S=4K:    1.005x
    S=8K:    1.007x
    S=16K:   1.004x
    S=32K:   1.005x
    S=64K:   1.003x
    S=128K:  1.003x
    avg:     1.005x
```

Decision:

```text
keep
```

Reason:

```text
This is also a true kernel-internal combine data-path improvement. Dispatch, split, block_N,
backend, dtype, and timing protocol are unchanged.

The gain is small but consistent enough to keep. It stacks with v002 for non-WS and also gives
a small WS improvement because both combine macros had the same repeated global glse read.
```

## Variant v004: non-WS direct partial store

Hypothesis:

```text
The non-WS split kernel currently stores acc_o through an intermediate O_shared buffer:

    acc_o -> O_shared -> Output_partial

The WS split kernel already writes acc_o directly into Output_partial.
Changing non-WS to direct partial store should remove one shared-memory round trip and reduce
epilogue overhead without changing dispatch, split, combine, or workspace dtype.
```

Changed file:

```text
tileops/kernels/attention/gqa_decode.py
```

Fixed comparison points:

```text
Llama4 group=5:
    split=16, backend=tileops_split

Qwen3.5 group=8:
    split=32, backend=tileops_split
```

Status:

```text
reject
```

Probe result:

```text
results/gqa_decode_internal_ako_v004_direct_partial_probe.jsonl

rows:         6
status:       all ok
max_diff max: 1.6021728515625e-04
```

Latency probe:

```text
Llama4 split=16:
    S=4K:   0.013375 ms
    S=8K:   0.018630 ms
    S=32K:  0.046531 ms

Qwen3.5 split=32:
    S=4K:   0.009270 ms
    S=8K:   0.010848 ms
    S=32K:  0.019038 ms
```

Decision:

```text
reject
```

Reason:

```text
Correctness passed, but direct elementwise global store is slower than the original
acc_o -> O_shared -> Output_partial T.copy sequence in the short probe.

The T.copy path likely gives better vectorized/layout-aware store code than a hand-written
T.Parallel elementwise store in this non-WS kernel.
```

## Variant v005: non-WS split_base precompute

Hypothesis:

```text
The non-WS split kernel repeats the split base expression inside the K and V copy slices:

    (seqlen_kv // (num_split * block_N) * block_N) * sid

The WS path already computes this once as split_base.
Precomputing split_base in non-WS may simplify index codegen in the main loop without changing
the algorithm or memory layout.
```

Changed file:

```text
tileops/kernels/attention/gqa_decode.py
```

Fixed comparison points:

```text
Llama4 group=5:
    split=16, backend=tileops_split

Qwen3.5 group=8:
    split=32, backend=tileops_split
```

Status:

```text
reject
```

Probe result:

```text
results/gqa_decode_internal_ako_v005_split_base_probe.jsonl

rows:         6
status:       all ok
max_diff max: 1.4495849609375e-04
```

Formal performance result:

```text
results/gqa_decode_internal_ako_v005_split_base_perf.jsonl
```

Compared with v003:

```text
Llama4 group=5, tileops_split split=16:
    avg 0.999x
    min 0.996x
    max 1.003x

Qwen3.5 group=8, tileops_split split=32:
    avg 1.000x
    min 0.996x
    max 1.004x
```

Decision:

```text
reject
```

Reason:

```text
Correctness passed, but the performance change is noise-level and not consistently positive on
the fixed comparison points. Keeping this would add code churn without a reliable kernel-internal
gain.
```

## Variant v006: combine threads=256 probe

Hypothesis:

```text
Both non-WS and WS combine kernels use threads=128.
For D=128 and split={16,32}, increasing combine threads to 256 may improve the reduce / output
accumulation schedule, especially for split=32.
```

Changed file:

```text
tileops/kernels/attention/gqa_decode.py
```

Fixed comparison points:

```text
Llama4 group=5:
    split=16, backend=tileops_split

Qwen3.5 group=8:
    split=32, backend in {tileops_split, tileops_ws}
```

Status:

```text
reject
```

Probe result:

```text
results/gqa_decode_internal_ako_v006_combine_threads_probe.jsonl

rows:         9
status:       all ok
max_diff max: 1.8310546875e-04
```

Latency probe:

```text
Llama4 split=16:
    S=4K:   0.012362 ms
    S=8K:   0.017742 ms
    S=32K:  0.045674 ms

Qwen3.5 split=32:
    non-WS:
        S=4K:   0.009349 ms
        S=8K:   0.010910 ms
        S=32K:  0.019483 ms
    WS:
        S=4K:   0.008718 ms
        S=8K:   0.011397 ms
        S=32K:  0.023462 ms
```

Decision:

```text
reject
```

Reason:

```text
Correctness passed, but combine threads=256 is consistently slower in the probe, including the
important Qwen3.5 S=4K WS point. The existing threads=128 combine remains the better default.
```

## Variant v007: combine denominator-form weights

Hypothesis:

```text
The combine kernel currently computes:

    lse_logsum = log2(sum(exp2(glse - lse_max))) + lse_max
    w = exp2(glse - lse_logsum)

For the weighted output, the same weight can be computed as:

    denom = sum(exp2(glse - lse_max))
    w = exp2(glse - lse_max) / denom

This removes one log2 operation from combine and may simplify the dependency chain.
```

Changed file:

```text
tileops/kernels/attention/gqa_decode.py
```

Fixed comparison points:

```text
Llama4 group=5:
    split=16, backend=tileops_split

Qwen3.5 group=8:
    split=32, backend in {tileops_split, tileops_ws}
```

Status:

```text
keep
```

Correctness probe:

```text
results/gqa_decode_internal_ako_v007_denom_combine_probe.jsonl

rows:         9
status:       all ok
max_diff max: 1.52587890625e-04
```

Formal perf sweep:

```text
results/gqa_decode_internal_ako_v007_denom_combine_perf.jsonl
baseline: results/gqa_decode_internal_ako_v003_glse_vec_perf.jsonl
```

Recommended dispatch points, v003 -> v007:

```text
Llama4 group=5, split=16, non-WS:
    S=4K:    0.01217984 -> 0.01217534 ms, 1.0004x
    S=8K:    0.01754746 -> 0.01761662 ms, 0.9961x
    S=16K:   0.02765816 -> 0.02771982 ms, 0.9978x
    S=32K:   0.04554104 -> 0.04547514 ms, 1.0014x
    S=64K:   0.07623234 -> 0.07609726 ms, 1.0018x
    S=128K:  0.13584818 -> 0.13607930 ms, 0.9983x
    geo mean: 0.9993x

Qwen3.5 group=8:
    S=4K,   split=32, WS:      0.00851126 -> 0.00845194 ms, 1.0070x
    S=8K,   split=32, non-WS:  0.01068028 -> 0.01056120 ms, 1.0113x
    S=16K,  split=32, non-WS:  0.01361406 -> 0.01348812 ms, 1.0093x
    S=32K,  split=32, non-WS:  0.01885316 -> 0.01881982 ms, 1.0018x
    S=64K,  split=32, non-WS:  0.02961980 -> 0.02943356 ms, 1.0063x
    S=128K, split=32, non-WS:  0.05055554 -> 0.05050804 ms, 1.0009x
    Qwen non-WS split=32 geo mean: 1.0061x

All recommended points geo mean: 1.0027x
```

Recheck:

```text
results/gqa_decode_internal_ako_v007_denom_combine_recheck.jsonl

The first formal sweep had one anomalous Llama4 S=8K WS split=16 row:
    0.01871108 -> 0.02420796 ms, 0.7729x

Focused recheck did not reproduce that regression:
    Llama4 S=8K WS split=16: 0.01878320 ms
```

Decision:

```text
keep
```

Reason:

```text
The denominator form removes one log2 from combine and gives a small but stable benefit on the
Qwen3.5 split=32 path. It is nearly neutral on the Llama4 split=16 non-WS recommendation line.

This is a real internal-kernel cleanup, but the magnitude is small: roughly +0.27% geo mean over
the current recommended points, with the clearest local benefit on Qwen3.5 non-WS split=32
(about +0.61% geo mean). Record it as a minor AKO internal optimization, not as the main AKO
win for this case.
```

## Variant v008: cache denominator combine weights

Hypothesis:

```text
Variant v007 computes denom with:

    exp2(glse - lse_max)

and then computes the same exp2 again when applying each split weight. Cache these denominator
terms in a local fragment:

    weight_vec[k] = exp2(glse_vec[k] - lse_max)
    denom += weight_vec[k]
    w = weight_vec[k] / denom

This trades up to num_split extra local values for removing one exp2 per split in combine.
It may help split=32 Qwen3.5 if the combine math is still visible, but could lose if register
pressure becomes the limiting factor.
```

Changed file:

```text
tileops/kernels/attention/gqa_decode.py
```

Status:

```text
reject
```

Correctness probe:

```text
results/gqa_decode_internal_ako_v008_cache_weight_probe.jsonl

rows:         24
status:       all ok
max_diff max: 1.9073486328125e-04
```

Probe comparison against v007 probe:

```text
Llama4 group=5, split=16, non-WS:
    S=4K:   0.0123265 -> 0.0122942 ms, 1.0026x
    S=8K:   0.0175728 -> 0.0177788 ms, 0.9884x
    S=32K:  0.0455343 -> 0.0454724 ms, 1.0014x

Qwen3.5 group=8, split=32:
    S=4K,  non-WS:  0.0090577 -> 0.0089918 ms, 1.0073x
    S=4K,  WS:      0.0083857 -> 0.0085120 ms, 0.9852x
    S=8K,  non-WS:  0.0106879 -> 0.0105988 ms, 1.0084x
    S=8K,  WS:      0.0110943 -> 0.0111104 ms, 0.9986x
    S=32K, non-WS:  0.0190237 -> 0.0191104 ms, 0.9955x
    S=32K, WS:      0.0232527 -> 0.0230402 ms, 1.0092x
```

Decision:

```text
reject
```

Reason:

```text
The cached weight fragment removes repeated exp2 instructions, but the added local fragment does
not produce a stable win. It regresses the important Qwen3.5 S=4K split=32 WS point and does not
improve the Qwen3.5 S=32K non-WS recommendation point.

Keep v007's denominator form without caching the per-split weights.
```

## Variant v009: fragment scalar denominator in combine

Hypothesis:

```text
Variant v007 keeps the denominator as:

    denom = T.alloc_local([1], accum_dtype)

while lse_max is already an accum_dtype fragment scalar. Try:

    denom = T.alloc_fragment([1], accum_dtype)

This is a narrow lowering probe: if TileLang treats the fragment scalar more directly in registers,
combine may get a tiny benefit without adding a full num_split fragment as v008 did.
```

Changed file:

```text
tileops/kernels/attention/gqa_decode.py
```

Status:

```text
reject
```

Correctness probe:

```text
results/gqa_decode_internal_ako_v009_fragment_denom_probe.jsonl

rows:         24
status:       all ok
max_diff max: 1.8310546875e-04
```

Focused perf:

```text
results/gqa_decode_internal_ako_v009_fragment_denom_perf_focus.jsonl

recommended points geo mean vs v007: 1.0004x
```

Formal perf sweep:

```text
results/gqa_decode_internal_ako_v009_fragment_denom_perf.jsonl
baseline: results/gqa_decode_internal_ako_v007_denom_combine_perf.jsonl
```

Recommended dispatch points, v007 -> v009:

```text
Llama4 group=5, split=16, non-WS:
    S=4K:    0.01217534 -> 0.01222852 ms, 0.9957x
    S=8K:    0.01761662 -> 0.01761148 ms, 1.0003x
    S=16K:   0.02771982 -> 0.02768194 ms, 1.0014x
    S=32K:   0.04547514 -> 0.04557192 ms, 0.9979x
    S=64K:   0.07609726 -> 0.07612010 ms, 0.9997x
    S=128K:  0.13607930 -> 0.13597802 ms, 1.0007x
    geo mean: 0.9993x

Qwen3.5 group=8:
    S=4K,   split=32, WS:      0.00845194 -> 0.00840772 ms, 1.0053x
    S=8K,   split=32, non-WS:  0.01056120 -> 0.01060412 ms, 0.9960x
    S=16K,  split=32, non-WS:  0.01348812 -> 0.01362050 ms, 0.9903x
    S=32K,  split=32, non-WS:  0.01881982 -> 0.01883074 ms, 0.9994x
    S=64K,  split=32, non-WS:  0.02943356 -> 0.02946298 ms, 0.9990x
    S=128K, split=32, non-WS:  0.05050804 -> 0.05053428 ms, 0.9995x
    Qwen non-WS split=32 geo mean: 0.9984x

All recommended points geo mean: 0.9987x
```

Decision:

```text
reject
```

Reason:

```text
The fragment scalar denominator improves the Qwen3.5 S=4K split=32 WS point, but the full sweep
shows a small net regression over the recommended points, especially Qwen3.5 S=16K split=32
non-WS. The v007 alloc_local denominator remains the better global default.
```
