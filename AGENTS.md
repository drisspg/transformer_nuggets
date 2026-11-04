# Scratch Space

- Use `agent_space/` (git-ignored, at repo root) for temporary scripts, scratch files, and throwaway experiments.
- Do not commit files from this directory.

# Project Skills

- `.agents/skills/analyzing-pytorch-rooflines/SKILL.md` — eager/AOT roofline capture,
  formula registration, trace interpretation, ranking, replay, and NCU follow-ups.

# CuTeDSL Conventions

- Standalone compiled kernels use `transformer_nuggets.cute.compile_tvm_ffi`, which owns
  the typed TVM-FFI option, fake environment stream, and outer artifact name.
- Compile from fake compact tensors whenever the runtime ABI is compact. Use
  `make_fake_strided_tensor` only for intentionally strided runtime contracts.
- Validate runtime compactness and alignment before relying on a fake tensor's
  `assumed_align` promise.
- Use `set_name_prefix(...)` on `@cute.kernel` wrappers before launch; do not pass the
  removed `_name_prefix=` keyword.

# warpdrive Testing Tiers

Pick the smallest tier that covers the change; run bigger tiers only when asked or before
landing core synchronization/scheduling changes.

- Every change (~30 s on a GB300): `pytest -n 12 test/test_cute_warpdrive*.py`. Fuzz tests are
  skipped by default (`addopts` excludes the `fuzz` marker).
- Quick fuzz (~2-3 min, on request or before landing layer changes):
  `pytest -n 12 -m fuzz test/test_cute_warpdrive*.py`.
- Long campaigns, sanitizer and CUTracer runs, SASS/perf A/Bs: only when the user asks.
  Use `scripts/fuzz_warpdrive.py {runtime,static,mutate}` under `gpu-run`.
- Before pushing, run CI's full pre-commit suite: `prek run --all-files`. It pins ruff 0.14.9,
  which flags rules the dev ruff does not (e.g. E741), and runs pyupgrade, which ruff alone
  does not cover.
