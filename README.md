# TileOps Operator Design Plans

This repository collects discussion-stage design plans for TileOps operator families and release tracks. The documents here are not manifest specs and do not freeze public APIs. They are intended to help reviewers align on operator boundaries, model-driven requirements, release phases, and follow-up tracking issues before implementation PRs.

## Plans

| Plan | Scope | Status |
| --- | --- | --- |
| [Cross-Layer 算子族发布计划](plans/cross-layer-ops-release-plan-cn.md) | Cross-layer operator taxonomy, MHC manifest alignment, Block AttnRes path, and adjacent depth/cache mechanisms | Discussion draft |
| [Quantized Inference 算子补齐计划](plans/quantized-inference-release-plan-cn.md) | Quantized inference gaps for modern open-source LLM serving, including FP8, INT4, MoE experts, and KV cache quantization | Discussion draft |
| [GQA / FP8 Attention planning bundle](plans/gqa/) | GQA prefill/decode plans, FP8 GQA Hopper notes, issue drafts, assets, and contribution reports migrated from `tileops-gqa-plan` | Planning archive |

## How To Read

These plans separate three layers:

1. **Model mechanisms**: what recent models use.
2. **Operator boundaries**: what can become a stable TileOps op with a tensor signature, correctness reference, and benchmark workload.
3. **Release phases**: what should be discussed, specified, implemented, and promoted through manifest status.

The plans should not be read as a promise that every mechanism named in the document becomes a TileOps operator. A mechanism enters a TileOps manifest only after its kernel boundary, dtype/layout contract, correctness policy, benchmark workload, and source metadata are reviewed.

## Relationship To TileOps

The canonical implementation, manifest entries, tests, and benchmarks belong in the TileOps repository. This repository is a public discussion package for design planning and review.

## Migration Notes

- `plans/gqa/` was migrated from `superAngGao/tileops-gqa-plan` and intentionally preserves its working documents, issue drafts, figures, and reports.
