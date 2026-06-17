# Design: Get ASR Working Locally

**Change:** get-asr-working-local
**Created:** 2026-06-17

## Technical Approach

Three targeted edits, no new files, no new dependencies:

1. **Fix the usage comment** in `audio_asr.py` — one-line change at line 19.
2. **Update README.md** — add an ASR subsection to the existing examples section.
3. **Smoke-test** — run `uv run python audio_asr.py` locally to confirm AC1.

## Architecture

No architectural changes. The script is self-contained: it generates synthetic audio via `tempfile`, trains for 30 steps, and saves adapters to `gemma4_audio_asr_lora/`.

## File Changes Map

| File | Action | Description |
|------|--------|-------------|
| `audio_asr.py` | Modify | Fix usage comment on line 19 (wrong path) |
| `README.md` | Modify | Add ASR setup section: model size, run command, real dataset pointer |

## Data Model Changes

None.

## API Changes

None.

## Key Decisions

- **Do not add `soundfile` to `pyproject.toml`** — it's already bundled by `mlx-tune[audio]`; adding it explicitly would be misleading and could cause version conflicts.
- **E4B as default** — the script already uses E4B; README will note E2B as a lower-memory option.
- **No smoke-test automation** — the model download (~2 GB) makes CI impractical; manual verification is sufficient.

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Model download fails during verify | Low | Medium | Wrap in try/except (already done); document in README |
| `mlx-tune` API changes in future version | Low | Low | Pin `>=0.5.1` already in pyproject.toml |
