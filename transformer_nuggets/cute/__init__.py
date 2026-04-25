from transformer_nuggets.cute.cache import (
    compile_and_cache,
    auto_compile_and_cache,
    clear_cute_cache,
    get_cache_stats,
    set_cache_size,
)
from transformer_nuggets.cute.base import CuteOp
from transformer_nuggets.cute.element_wise import ElementwiseOp, elementwise_op
from transformer_nuggets.cute.utils import visualize_tv_layout
from transformer_nuggets.cute import profiler


_SYMMETRIC_MEMORY_EXPORTS = {
    "compile_symmetric_memory_all_reduce",
    "init_torchrun_process_group",
    "run_symmetric_memory_all_reduce_example",
    "symmetric_memory_all_reduce",
    "symmetric_memory_peer_tensors",
}


def __getattr__(name):
    if name in _SYMMETRIC_MEMORY_EXPORTS:
        from transformer_nuggets.cute import symmetric_memory

        return getattr(symmetric_memory, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
