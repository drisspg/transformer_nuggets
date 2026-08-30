"""Tests for the canonical CuTeDSL compile and fake-tensor helpers."""

from __future__ import annotations

import pytest
import torch

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")
cute_utils = pytest.importorskip("transformer_nuggets.cute.utils")
cute_cache = pytest.importorskip("transformer_nuggets.cute.cache")

Float16 = cutlass.Float16
compile_tvm_ffi = cute_utils.compile_tvm_ffi
make_fake_compact_tensor = cute_utils.make_fake_compact_tensor
make_fake_strided_tensor = cute_utils.make_fake_strided_tensor
requires_int64_abi = cute_utils.requires_int64_abi
tensor_supports_contiguous_dim = cute_utils.tensor_supports_contiguous_dim


def test_compile_tvm_ffi_enforces_compile_contract():
    """Reject unstable names and runtime Torch tensors at the compile boundary."""
    with pytest.raises(TypeError, match=r"get_name\(\) or name="):
        compile_tvm_ffi(object())
    with pytest.raises(ValueError, match="lowercase"):
        compile_tvm_ffi(object(), name="Bad-Name")
    with pytest.raises(TypeError, match="fake CuTe tensors"):
        compile_tvm_ffi(object(), {"nested": torch.empty(0)}, name="valid_name")


def test_compile_tvm_ffi_adds_fake_stream_and_typed_option(monkeypatch):
    """Own the typed TVM-FFI option, environment stream, and artifact name centrally."""
    fake_stream = object()
    fake_tensor = object()
    observed = {}

    class FakeCompile:
        def __getitem__(self, option):
            observed["option"] = option
            return self

        def __call__(self, entrypoint, *args, **kwargs):
            observed.update(entrypoint=entrypoint, args=args, kwargs=kwargs)
            return "compiled"

    class EntryPoint:
        @staticmethod
        def get_name() -> str:
            return "stable_kernel_name"

        def set_name_prefix(self, name: str) -> None:
            observed["name_prefix"] = name

    monkeypatch.setattr(cute, "compile", FakeCompile())
    monkeypatch.setattr(
        cute.runtime,
        "make_fake_stream",
        lambda *, use_tvm_ffi_env_stream: fake_stream if use_tvm_ffi_env_stream else None,
    )
    entrypoint = EntryPoint()

    assert compile_tvm_ffi(entrypoint, fake_tensor) == "compiled"
    assert observed == {
        "option": cute.EnableTVMFFI,
        "entrypoint": entrypoint,
        "args": (fake_tensor, fake_stream),
        "kwargs": {},
        "name_prefix": "stable_kernel_name",
    }


def test_make_fake_strided_tensor_keeps_only_one_static_stride():
    """Encode dynamic outer strides while preserving one contiguous tensor mode."""
    fake = make_fake_strided_tensor(
        Float16,
        (cute.sym_int(), 3, 128),
        stride_divisibility=8,
        use_int64_strides=False,
    )
    assert fake.stride[-1] == 1
    assert fake._assumed_align == 16
    assert "div=8" in str(fake)


def test_auto_cache_key_distinguishes_fake_tensor_abis():
    """Keep stride order and alignment promises in automatic compile-cache identities."""
    row_major_align4 = make_fake_compact_tensor(
        Float16,
        (8, 16),
        stride_order=(1, 0),
        assumed_align=4,
    )
    row_major_align32 = make_fake_compact_tensor(
        Float16,
        (8, 16),
        stride_order=(1, 0),
        assumed_align=32,
    )
    column_major_align4 = make_fake_compact_tensor(
        Float16,
        (8, 16),
        stride_order=(0, 1),
        assumed_align=4,
    )
    base_key = cute_cache._generate_cache_key(row_major_align4)
    assert base_key != cute_cache._generate_cache_key(row_major_align32)
    assert base_key != cute_cache._generate_cache_key(column_major_align4)


def test_tensor_supports_contiguous_dim_tracks_slice_alignment():
    """Separate logical contiguity from the alignment promised to codegen."""
    compact = torch.empty(2, 3, 128)
    assert tensor_supports_contiguous_dim(compact, alignment_bytes=16)

    misaligned_storage = torch.empty(compact.numel() + 1)
    misaligned = misaligned_storage[1:].view_as(compact)
    assert misaligned.is_contiguous()
    assert tensor_supports_contiguous_dim(misaligned, alignment_bytes=4)
    assert not tensor_supports_contiguous_dim(misaligned, alignment_bytes=16)

    outer_strided = torch.empty(2, 3, 2, 128)[:, :, 0, :]
    assert not outer_strided.is_contiguous()
    assert tensor_supports_contiguous_dim(outer_strided, alignment_bytes=16)

    last_dim_strided = torch.empty(2, 3, 256)[..., ::2]
    assert not tensor_supports_contiguous_dim(last_dim_strided, alignment_bytes=4)


def test_requires_int64_abi_checks_unreachable_singleton_stride():
    """Treat every ABI-visible stride as significant even when its mode has size one."""
    compact = torch.empty(1, 1)
    oversized = compact.as_strided((1, 1), (2**31, 1))
    assert not requires_int64_abi(compact)
    assert requires_int64_abi(oversized)
