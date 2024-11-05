import torch
import triton
import triton.language as tl
import triton.tools.experimental_descriptor
from triton.language.extra.cuda._experimental_tma import experimental_device_tensormap_create2d

# Autotuner does not work with TMA. Use manual config.
configs = {
    torch.float8_e4m3fn: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8,
        "num_stages": 4,
        "num_warps": 8,
    },
    torch.float16: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 3,
        "num_warps": 8,
    },
}


def validate_matmul_inputs(
    a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor
) -> bool:
    """
    Validate inputs for matrix multiplication with scaling.

    Args:
        a (torch.Tensor): First input matrix
        b (torch.Tensor): Second input matrix
        a_scale (torch.Tensor): Scaling factor for a
        b_scale (torch.Tensor): Scaling factor for b

    Returns:
        bool: True if inputs are valid, raises AssertionError otherwise
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"

    ROW_WISE_SCALING = a_scale.numel() != 1
    if ROW_WISE_SCALING:
        assert a_scale.dim() == 2, f"a_scale must be a 2D tensor but got {a_scale.dim()}"
        assert b_scale.dim() == 2, f"b_scale must be a 2D tensor but got {b_scale.dim()}"
        assert a_scale.shape[0] == a.shape[0], (
            f"a_scale must have same number of rows as a, got {a_scale.shape[0]} vs {a.shape[0]}"
        )
        assert a_scale.shape[1] == 1, f"a_scale must have 1 column, got {a_scale.shape[1]}"
        assert b_scale.shape[1] == b.shape[1], (
            f"b_scale must have same number of columns as b, got {b_scale.shape[0]} vs {b.shape[1]}"
        )
        assert b_scale.shape[0] == 1, f"b_scale must have 1 column, got {b_scale.shape[1]}"
    else:
        assert a_scale.numel() == 1, (
            f"a_scale must be a scalar for per-tensor scaling, got shape {a_scale.shape}"
        )
        assert b_scale.numel() == 1, (
            f"b_scale must be a scalar for per-tensor scaling, got shape {b_scale.shape}"
        )

    return ROW_WISE_SCALING


def is_row_major(stride):
    assert len(stride) == 2, "is_row_major only supports 2D tensors"
    return stride[0] > stride[1] and stride[1] == 1


def is_col_major(stride):
    assert len(stride) == 2, "is_col_major only supports 2D tensors"
    return stride[1] > stride[0] and stride[0] == 1


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


@triton.jit
def load_scales(a_scale_ptr, b_scale_ptr, ROW_WISE_SCALING: tl.constexpr):
    if ROW_WISE_SCALING:
        # For row-wise scaling, we'll return the pointers
        return a_scale_ptr, b_scale_ptr
    else:
        # For per-tensor scaling, we'll load the scalar values
        a_scale = tl.load(a_scale_ptr)
        b_scale = tl.load(b_scale_ptr)
        return a_scale, b_scale


@triton.jit
def apply_scaling(
    accumulator,
    a_scale,
    b_scale,
    ROW_WISE_SCALING: tl.constexpr,
    offs_cm,
    offs_cn,
    M,
    N,
    stride_a_scale_m,
    stride_b_scale_n,
):
    if ROW_WISE_SCALING:
        # For row-wise scaling, we need to load the scales for each row/column
        a_scales = tl.load(
            a_scale + (offs_cm * stride_a_scale_m),
            mask=offs_cm < M,
            other=0.0,
        )
        b_scales = tl.load(
            b_scale + (offs_cn * stride_b_scale_n),
            mask=offs_cn < N,
            other=0.0,
        )
        acc_scale = a_scales[:, None] * b_scales[None, :]
    else:
        # For per-tensor scaling, we can directly use the loaded scalar values
        acc_scale = a_scale * b_scale

    return accumulator * acc_scale


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_a_scale_m,
    stride_b_scale_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    ROW_WISE_SCALING: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    pid_m = 0
    pid_n = 0
    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    a_scale, b_scale = load_scales(a_scale_ptr, b_scale_ptr, ROW_WISE_SCALING)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            start_m = pid_m * BLOCK_SIZE_M
            start_n = pid_n * BLOCK_SIZE_N
            offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
            offs_am = tl.where(offs_am < M, offs_am, 0)
            offs_bn = tl.where(offs_bn < N, offs_bn, 0)
            offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
            offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)

        if ki == k_tiles - 1:
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            # Apply inverse scaling
            accumulator = apply_scaling(
                accumulator,
                a_scale,
                b_scale,
                ROW_WISE_SCALING,
                offs_cm,
                offs_cn,
                M,
                N,
                stride_a_scale_m,
                stride_b_scale_n,
            )
            c = accumulator.to(c_ptr.dtype.element_ty)
            tl.store(c_ptrs, c, mask=c_mask)
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def matmul_persistent(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    # Check constraints.
    ROW_WISE_SCALING = validate_matmul_inputs(a, b, a_scale, b_scale)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=output_dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])),
    )
    matmul_kernel_persistent[grid](
        a,
        a_scale,
        b,
        b_scale,
        c,  #
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        a_scale.stride(0) if ROW_WISE_SCALING else 0,
        b_scale.stride(1) if ROW_WISE_SCALING else 0,
        BLOCK_SIZE_M=configs[dtype]["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=configs[dtype]["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=configs[dtype]["BLOCK_SIZE_K"],
        GROUP_SIZE_M=configs[dtype]["GROUP_SIZE_M"],
        NUM_SMS=NUM_SMS,
        num_stages=configs[dtype]["num_stages"],
        num_warps=configs[dtype]["num_warps"],
        ROW_WISE_SCALING=ROW_WISE_SCALING,
    )
    return c


@triton.jit
def matmul_kernel_tma_persistent(
    a_desc_ptr,
    a_scale_ptr,
    b_desc_ptr,
    b_scale_ptr,
    c_desc_ptr,
    M,
    N,
    K,
    stride_a_scale_m,
    stride_b_scale_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    output_dtype: tl.constexpr,
    ROW_WISE_SCALING: tl.constexpr,
):
    tl.inline_asm_elementwise(
        "fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg",
        "=r, l",
        [a_desc_ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    tl.inline_asm_elementwise(
        "fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg",
        "=r, l",
        [b_desc_ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    tl.inline_asm_elementwise(
        "fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg",
        "=r, l",
        [c_desc_ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    dtype = tl.float8e4nv
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    a_scale, b_scale = load_scales(a_scale_ptr, b_scale_ptr, ROW_WISE_SCALING)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

        offs_k = ki * BLOCK_SIZE_K

        a = tl._experimental_descriptor_load(
            a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype
        )
        b = tl._experimental_descriptor_load(
            b_desc_ptr, [offs_bn, offs_k], [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype
        )
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            offs_cm = offs_am + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = offs_bn + tl.arange(0, BLOCK_SIZE_N)
            # Apply scaling
            accumulator = apply_scaling(
                accumulator,
                a_scale,
                b_scale,
                ROW_WISE_SCALING,
                offs_cm,
                offs_cn,
                M,
                N,
                stride_a_scale_m,
                stride_b_scale_n,
            )
            c = accumulator.to(output_dtype)

            tl._experimental_descriptor_store(c_desc_ptr, c, [offs_am, offs_bn])
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def matmul_tma_persistent(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    # Check constraints.
    assert is_row_major(a.stride()), "a must be row major"
    assert is_col_major(b.stride()), "b must be col major"
    ROW_WISE_SCALING = validate_matmul_inputs(a, b, a_scale, b_scale)

    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=output_dtype)
    desc_a = triton.tools.experimental_descriptor.create_2d_tma_descriptor(
        a.data_ptr(),
        M,
        K,
        configs[dtype]["BLOCK_SIZE_M"],
        configs[dtype]["BLOCK_SIZE_K"],
        a.element_size(),
    )
    desc_b = triton.tools.experimental_descriptor.create_2d_tma_descriptor(
        b.data_ptr(),
        N,
        K,
        configs[dtype]["BLOCK_SIZE_N"],
        configs[dtype]["BLOCK_SIZE_K"],
        b.element_size(),
    )
    desc_c = triton.tools.experimental_descriptor.create_2d_tma_descriptor(
        c.data_ptr(),
        M,
        N,
        configs[dtype]["BLOCK_SIZE_M"],
        configs[dtype]["BLOCK_SIZE_N"],
        c.element_size(),
    )
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    triton_out_dtype = tl.float8e4nv if output_dtype == torch.float8_e4m3fn else tl.bfloat16

    grid = lambda META: (
        min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])),
    )
    matmul_kernel_tma_persistent[grid](
        desc_a,
        a_scale,
        desc_b,
        b_scale,
        desc_c,
        M,
        N,
        K,
        a_scale.stride(0) if ROW_WISE_SCALING else 0,
        b_scale.stride(1) if ROW_WISE_SCALING else 0,
        BLOCK_SIZE_M=configs[dtype]["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=configs[dtype]["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=configs[dtype]["BLOCK_SIZE_K"],
        GROUP_SIZE_M=configs[dtype]["GROUP_SIZE_M"],
        NUM_SMS=NUM_SMS,
        num_stages=configs[dtype]["num_stages"],
        num_warps=configs[dtype]["num_warps"],
        output_dtype=triton_out_dtype,
        ROW_WISE_SCALING=ROW_WISE_SCALING,
    )
    return c


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_device_tma_persistent(
    workspace_ptr,
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_a_scale_m,
    stride_b_scale_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
    ROW_WISE_SCALING: tl.constexpr,
):
    # Matmul using TMA and device-side descriptor creation
    dtype = tl.float8e4nv
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    TMA_SIZE: tl.constexpr = 128
    workspace_base = workspace_ptr + start_pid * 3 * TMA_SIZE
    a_desc_ptr = workspace_base
    b_desc_ptr = workspace_base + TMA_SIZE
    c_desc_ptr = workspace_base + 2 * TMA_SIZE

    experimental_device_tensormap_create2d(
        desc_ptr=a_desc_ptr,
        global_address=a_ptr,
        load_size=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        global_size=[M, K],
        element_ty=a_ptr.dtype.element_ty,
    )
    experimental_device_tensormap_create2d(
        desc_ptr=b_desc_ptr,
        global_address=b_ptr,
        load_size=[BLOCK_SIZE_N, BLOCK_SIZE_K],
        global_size=[N, K],
        element_ty=b_ptr.dtype.element_ty,
    )
    experimental_device_tensormap_create2d(
        desc_ptr=c_desc_ptr,
        global_address=c_ptr,
        load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        global_size=[M, N],
        element_ty=c_ptr.dtype.element_ty,
    )
    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(a_desc_ptr)
    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(b_desc_ptr)
    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    a_scale, b_scale = load_scales(a_scale_ptr, b_scale_ptr, ROW_WISE_SCALING)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

        offs_k = ki * BLOCK_SIZE_K

        a = tl._experimental_descriptor_load(
            a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype
        )
        b = tl._experimental_descriptor_load(
            b_desc_ptr, [offs_bn, offs_k], [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype
        )
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            # Apply inverse scaling
            offs_cm = offs_am + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = offs_bn + tl.arange(0, BLOCK_SIZE_N)
            # Apply scaling
            accumulator = apply_scaling(
                accumulator,
                a_scale,
                b_scale,
                ROW_WISE_SCALING,
                offs_cm,
                offs_cn,
                M,
                N,
                stride_a_scale_m,
                stride_b_scale_n,
            )
            c = accumulator.to(OUTPUT_DTYPE)

            tl._experimental_descriptor_store(c_desc_ptr, c, [offs_am, offs_bn])

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def matmul_device_tma_persistent(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    assert is_row_major(a.stride()), "a must be row major"
    assert is_col_major(b.stride()), "b must be col major"
    ROW_WISE_SCALING = validate_matmul_inputs(a, b, a_scale, b_scale)

    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=output_dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    tma_size = 128
    workspace = torch.empty(NUM_SMS * 3 * tma_size, dtype=torch.uint8, device="cuda")
    triton_out_dtype = tl.float8e4nv if output_dtype == torch.float8_e4m3fn else tl.bfloat16

    grid = lambda META: (
        min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])),
    )
    matmul_kernel_device_tma_persistent[grid](
        workspace,
        a,
        a_scale,
        b,
        b_scale,
        c,
        M,
        N,
        K,
        a_scale.stride(0) if ROW_WISE_SCALING else 0,
        b_scale.stride(1) if ROW_WISE_SCALING else 0,
        BLOCK_SIZE_M=configs[dtype]["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=configs[dtype]["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=configs[dtype]["BLOCK_SIZE_K"],
        GROUP_SIZE_M=configs[dtype]["GROUP_SIZE_M"],
        NUM_SMS=NUM_SMS,
        num_stages=configs[dtype]["num_stages"],
        num_warps=configs[dtype]["num_warps"],
        OUTPUT_DTYPE=triton_out_dtype,
        ROW_WISE_SCALING=ROW_WISE_SCALING,
    )
    return c
