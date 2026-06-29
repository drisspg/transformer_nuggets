# Scratch Space

- Use `agent_space/` (git-ignored, at repo root) for temporary scripts, scratch files, and throwaway experiments.
- Do not commit files from this directory.

## Cursor Cloud specific instructions

This is a Python research library (`transformer_nuggets`, PyTorch + Triton). There is no long-running service/app — work is via the library, examples, tests, and CLI scripts.

- Python deps live in a virtualenv at `.venv` (created by the update script). Run tools with `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/ruff` (or activate it). Install command mirrors CI: `pip install -e ".[dev]"`.
- The cloud VM is CPU-only (no GPU / no `nvidia-smi`). `torch.cuda.is_available()` is `False`. Many tests and most `examples/` (fp8, mx, cute DSL, memory_viz, etc.) require CUDA and will auto-skip or fail to run; this is expected, not an environment problem.
- Lint: `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .` (these mirror the `prek`/pre-commit + CI checks).
- Tests: `.venv/bin/pytest`. Default config (`pyproject.toml`) deselects `slow` tests; run them with `pytest -m slow`. On CPU expect a large number of `skipped` (GPU) tests.
- Known pre-existing breakage (not env-related): `test/test_perfetto.py` fails to collect because it imports `chrome_trace_to_track_event_trace` / `default_track_event_path`, which no longer exist in `transformer_nuggets/utils/perfetto.py`. Run the rest of the suite with `pytest --ignore=test/test_perfetto.py` until that test is fixed upstream.
