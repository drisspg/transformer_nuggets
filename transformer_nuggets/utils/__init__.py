from transformer_nuggets.utils.benchmark import (
    DEFAULT_CUPTI_MONITOR_PM_METRICS,
    CudaBenchmarkStats,
    CuptiMonitorConfig,
    ProfileConfig,
    attach_oom_observer,
    benchmark_cuda_function_in_microseconds,
    benchmark_cuda_function_in_microseconds_triton,
    benchmark_cuda_function_stats,
    benchmark_torch_function_in_microseconds,
    cuda_memory_usage,
    locked_clocks,
    max_memory_usage,
    profile_function,
    profiler,
    save_memory_snapshot,
    supported_cupti_monitor_metrics,
)
from transformer_nuggets.utils.memory_viz import generate_memory_comparison_html
from transformer_nuggets.utils.merge_traces import merge_traces
from transformer_nuggets.utils.perfetto import (
    TraceFormat,
    add_cuda_graph_annotation_boxes,
    default_trace_path,
    default_track_event_path,
    perfetto_trace_path,
    read_trace,
    split_overlapping_slices,
    write_perfetto_trace,
    write_trace,
    write_track_event_trace,
)
from transformer_nuggets.utils.tracing import LoggingMode, NanInfDetect
from transformer_nuggets.utils.track_event import chrome_trace_to_track_event_trace
from transformer_nuggets.utils.triton import print_sass
# from transformer_nuggets.utils.model_extraction import extract_attention_data
