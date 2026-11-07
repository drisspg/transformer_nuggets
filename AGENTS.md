# Scratch Space

- Use `agent_space/` (git-ignored, at repo root) for temporary scripts, scratch files, and throwaway experiments.
- Do not commit files from this directory.

# Worktree Environments

- Every worktree gets its own `.venv` built with `uv venv` + `uv pip install -e` from that
  worktree (uv hard-links wheels from its cache, so it takes seconds). Never symlink, copy,
  or `cp -al` another env: console scripts (`pytest`, `prek`) keep the source env's
  interpreter in their shebang, so they silently run the other env's Python and packages.
- The same applies to side-by-side envs for other CuTeDSL versions: build each with
  `uv venv`, and invoke tools as `<env>/bin/python -m pytest` rather than `<env>/bin/pytest`.
- Verify from outside the repo: `cd /tmp && <env>/bin/python -c "import transformer_nuggets,
  cutlass; print(transformer_nuggets.__file__, cutlass.__version__)"`.

# Project Skills

- `.agents/skills/analyzing-pytorch-rooflines/SKILL.md` — eager/AOT roofline capture,
  formula registration, trace interpretation, ranking, replay, and NCU follow-ups.

# Authoring warpdrive kernels

- Read `transformer_nuggets/cute/warpdrive/SKILL.md` when authoring or porting a kernel.
  The skill, authoring guide, safety contract and examples live with the library and must
  match the checkout being used.
- Protocol semantics live in the topic guides under `transformer_nuggets/cute/warpdrive/docs/`.
  Fuzz report labels live in `test/warpdrive_fuzz/CASES.md`, not a public error-code API.

# CuTeDSL Conventions

- Standalone compiled kernels use `transformer_nuggets.cute.compile_tvm_ffi`, which owns
  the typed TVM-FFI option, fake environment stream, and outer artifact name.
- Compile from fake compact tensors whenever the runtime ABI is compact. Use
  `make_fake_strided_tensor` only for intentionally strided runtime contracts.
- Validate runtime compactness and alignment before relying on a fake tensor's
  `assumed_align` promise.
- Use `set_name_prefix(...)` on `@cute.kernel` wrappers before launch; do not pass the
  removed `_name_prefix=` keyword.

# Adding CuTeDSL Kernels

Compile kernels through a module-level function decorated with `@jit_cache`
(`transformer_nuggets/cute/jit_cache/`, see its README). The warpdrive examples and the warpdrive
tests do; `transformer_nuggets/cute/cache.py` (`compile_tvm_ffi_and_cache`, in memory only)
remains for the older non-warpdrive kernels. A compiled kernel is stored on disk and reused
until one of its key inputs changes:

- the compile function's arguments (static, pickleable values or objects with
  `__cute_cache_key__`, as warpdrive kernels have) and the compile target;
- the Python, cutlass, tvm_ffi, torch, and CUDA versions and the codegen environment variables
  listed in `transformer_nuggets/cute/jit_cache/_key.py` (`CUTE_DSL_ARCH` reaches it through the
  target);
- the source of the compile function's module and of every `transformer_nuggets` module it
  imports, directly or transitively (`_key.module_closure`). Unrelated modules do not invalidate
  it.

Imports are found statically, so keep the code a kernel traces reachable by import:

- Import kernel helpers, constants, and ops with ordinary `import`/`from` statements (local and
  relative imports are fine). A module named as a string literal, as in
  `importlib.import_module("transformer_nuggets.x.y")`, also counts; computed module names do
  not.
- `import pkg.sub.module` does not add `pkg/sub/__init__.py`; keep package `__init__` files to
  plain re-exports rather than state a kernel depends on.
- Pass `extra_sources=(...)` to `jit_cache` only for inputs import analysis cannot see: files
  read while tracing, or code outside `transformer_nuggets` other than the versioned
  dependencies above. A class passed as a compile argument is keyed by its module and name
  only, so if its module is not imported by the compile module, list that module in
  `extra_sources` (warpdrive kernels key their class modules' sources themselves).
- Do not add `extra_sources` for modules the compile module already imports; they are hashed.
- Pass a warpdrive kernel's static choices to its constructor: the key is the constructor call,
  and a public attribute assigned afterwards (other than `cluster`, work streams, and the IKET,
  tracing and alias settings) raises.

When testing specialization reuse, assert on `compile_fn.cache_info()` (distinct keys in
`currsize`, launches in `hits + misses`; a disk load is a hit) instead of disabling the disk
cache with `CUTE_DSL_NO_CACHE`, which forces cold compiles. A cached warpdrive kernel restores
its trace results (`describe()`, TMA/alias reports, warnings); only a test that asserts on
tracing itself (IKET ranges, monkeypatched trace hooks) takes the `traced_compiles` fixture and
calls `compile_fn.cache_clear()`.

The disk cache is `$TRANSFORMER_NUGGETS_CUTE_CACHE_DIR`, default
`~/.cache/transformer_nuggets/cute`; it is content-addressed and shared by every worktree.
`rm -rf` it to clear it.

# warpdrive Testing Tiers

Pick the smallest tier that covers the change; run bigger tiers only when asked or before
landing core synchronization/scheduling changes.

- Every change (~45 s cold, ~22 s with a warm jit_cache on a GB300):
  `pytest -n 12 test/test_cute_warpdrive*.py`. Fuzz tests are skipped by default (`addopts`
  excludes the `fuzz` marker).
- Quick fuzz (~2-3 min, on request or before landing layer changes):
  `pytest -n 12 -m fuzz test/test_cute_warpdrive*.py`.
- Long campaigns, sanitizer and CUTracer runs, SASS/perf A/Bs: only when the user asks.
  Use `scripts/fuzz_warpdrive.py {runtime,static,mutate}` under `gpu-run`.
  `CUTE_DSL_ARCH=sm_100a python scripts/warpdrive_sass.py dump OUTDIR` records the seven standard configs; `compare BEFORE AFTER` gates on PTX and compiler identity and prints SASS diffs (`--strict-sass` also requires identical SASS). ptxas can assign registers differently from identical PTX (seen on the small decode configs), so a SASS-only diff with identical PTX is not a code change.
- Before pushing, run CI's full pre-commit suite: `prek run --all-files`. It pins ruff 0.14.9,
  which flags rules the dev ruff does not (e.g. E741), and runs pyupgrade, which ruff alone
  does not cover.
