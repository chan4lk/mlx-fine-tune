# Learnings: get-asr-working-local

Build learnings, spec gaps, and patterns discovered.

**Categories:** spec_gap | design_gap | pattern | best_practice | agent_issue

---

## [L1] best_practice — mlx_vlm 0.6.3 load_model() silently drops the strict kwar...

**When:** 2026-06-17 17:47 UTC
**Category:** best_practice
**Priority:** high
**Status:** pending

### Detail
mlx_vlm 0.6.3 load_model() silently drops the strict kwarg; Gemma 4 E4B needs strict=False in from_pretrained because kv-shared layers 24-41 have quantized weights in the checkpoint that the model architecture omits. Fix: utils.py line 658 model.load_weights(..., strict=kwargs.get('strict', True))

### Action
Pass strict=False in FastVisionModel.from_pretrained for Gemma 4 E4B; patch mlx_vlm/utils.py until upstream is fixed

---
