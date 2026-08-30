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
