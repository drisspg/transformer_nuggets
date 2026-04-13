import json
import pickle
from pathlib import Path

import pytest

from transformer_nuggets.utils.memory_viz import (
    _extract_frames,
    _is_cpython_c_frame,
    _shorten_path,
    generate_memory_comparison_html,
    generate_memory_html,
    process_snapshot,
)


DATA_DIR = Path(__file__).parent / "data"
SNAPSHOT_PATH = DATA_DIR / "mini_snapshot.pickle"


@pytest.fixture
def snapshot():
    with open(SNAPSHOT_PATH, "rb") as f:
        return pickle.load(f)


def _make_event(action, addr, size, time_us=0, filename="test.py", name="func", line=1):
    return {
        "action": action,
        "addr": addr,
        "size": size,
        "stream": 0,
        "time_us": time_us,
        "compile_context": "N/A",
        "user_metadata": "",
        "frames": [{"filename": filename, "name": name, "line": line}],
    }


def _make_snapshot(events):
    return {"device_traces": [events], "segments": [], "allocator_settings": {}}


def _extract_bootstrap(html):
    marker = '<script id="memory-viz-bootstrap" type="application/json">'
    start = html.index(marker) + len(marker)
    end = html.index("</script>", start)
    return json.loads(html[start:end])


def _extract_named_bootstrap(html, script_id):
    marker = f'<script id="{script_id}" type="application/json">'
    start = html.index(marker) + len(marker)
    end = html.index("</script>", start)
    return json.loads(html[start:end])


class TestExtractFrames:
    def test_filters_unwind_frames(self):
        frames = [
            {"filename": "??", "line": 0, "name": "torch::unwind::unwind()"},
            {"filename": "foo.py", "line": 10, "name": "bar"},
        ]
        assert _extract_frames(frames) == ["foo.py:10 bar"]

    def test_filters_cpython_c_frames(self):
        frames = [
            {
                "filename": "/usr/local/src/conda/python-3.12/Objects/call.c",
                "line": 0,
                "name": "_PyObject_MakeTPCall",
            },
            {"filename": "my_script.py", "line": 5, "name": "main"},
        ]
        assert _extract_frames(frames) == ["my_script.py:5 main"]

    def test_keeps_cpp_frames_without_filename(self):
        frames = [
            {"filename": "", "line": 0, "name": "at::native::matmul(at::Tensor const&)"},
        ]
        result = _extract_frames(frames)
        assert len(result) == 1
        assert "matmul" in result[0]

    def test_shortens_site_packages_path(self):
        frames = [
            {
                "filename": "/home/user/.conda/envs/dev/lib/python3.12/site-packages/torch/nn/linear.py",
                "line": 42,
                "name": "forward",
            },
        ]
        result = _extract_frames(frames)
        assert result == ["torch/nn/linear.py:42 forward"]


class TestHelpers:
    def test_shorten_path_site_packages(self):
        assert _shorten_path("/foo/site-packages/torch/nn.py") == "torch/nn.py"

    def test_shorten_path_lib_python(self):
        assert _shorten_path("/foo/lib/python3.12/collections.py") == "3.12/collections.py"

    def test_shorten_path_no_match(self):
        assert _shorten_path("/home/user/my_script.py") == "/home/user/my_script.py"

    def test_is_cpython_c_frame(self):
        assert _is_cpython_c_frame("/usr/local/src/conda/python-3.12/call.c", "_PyObject_Call")
        assert _is_cpython_c_frame("eval.c", "_PyEval_EvalFrameDefault")
        assert not _is_cpython_c_frame("my_module.py", "forward")


class TestProcessSnapshot:
    def test_returns_correct_tuple_shape(self, snapshot):
        timeline, allocs, frames, stacks, max_ts, hwm_ts = process_snapshot(snapshot)
        assert len(timeline) > 0
        assert len(allocs) > 0
        assert len(frames) > 0
        assert len(stacks) > 0
        assert max_ts > 0

    def test_stacks_reference_valid_frame_indices(self, snapshot):
        _, _, frames, stacks, *_ = process_snapshot(snapshot)
        for stack in stacks:
            for fi in stack:
                assert 0 <= fi < len(frames)

    def test_timeline_fields(self, snapshot):
        timeline, *_ = process_snapshot(snapshot)
        entry = timeline[0]
        assert set(entry.keys()) == {"t", "a", "r", "h", "act", "s", "si"}

    def test_alloc_poly_fields(self, snapshot):
        _, allocs, *_ = process_snapshot(snapshot)
        poly = allocs[0]
        assert "si" in poly
        assert "s" in poly
        assert "ts" in poly
        assert "offsets" in poly
        assert len(poly["ts"]) == len(poly["offsets"])
        assert len(poly["ts"]) >= 2

    def test_hwm_is_max_allocated(self, snapshot):
        timeline, *_ = process_snapshot(snapshot)
        hwm = max(e["h"] for e in timeline)
        max_allocated = max(e["a"] for e in timeline)
        assert hwm == max_allocated

    def test_allocated_never_negative(self, snapshot):
        timeline, *_ = process_snapshot(snapshot)
        assert all(e["a"] >= 0 for e in timeline)

    def test_stack_indices_valid(self, snapshot):
        _, allocs, _, stacks, *_ = process_snapshot(snapshot)
        for a in allocs:
            assert 0 <= a["si"] < len(stacks)

    def test_empty_device_returns_empty(self, snapshot):
        result = process_snapshot(snapshot, device=99)
        assert result == ([], [], [], [], 0, 0)

    def test_polygon_offsets_non_negative(self, snapshot):
        _, allocs, *_ = process_snapshot(snapshot)
        for poly in allocs:
            assert all(o >= 0 for o in poly["offsets"])


class TestGenerateHTML:
    def test_produces_valid_html(self, snapshot):
        html = generate_memory_html(snapshot, title="Test")
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_no_remaining_placeholders(self, snapshot):
        html = generate_memory_html(snapshot, title="Test")
        for placeholder in [
            "__TITLE__",
            "__DOCUMENT_TITLE__",
            "__VISIBLE_TITLE__",
            "__BOOTSTRAP__",
            "__TITLE_LEFT__",
            "__TITLE_RIGHT__",
        ]:
            assert placeholder not in html

    def test_title_appears_in_html(self, snapshot):
        html = generate_memory_html(snapshot, title="My Custom Title")
        assert "My Custom Title" in html
        assert "<h1>My Custom Title</h1>" in html

    def test_title_and_bootstrap_are_escaped(self, snapshot):
        title = '</script><script>alert("x")</script>'
        html = generate_memory_html(snapshot, title=title)
        assert (
            "<title>&lt;/script&gt;&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;</title>"
            in html
        )
        assert "<h1>&lt;/script&gt;&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;</h1>" in html
        assert title not in html
        assert r"\u003c/script>\u003cscript>alert(\"x\")\u003c/script>" in html

    def test_d3_loaded(self, snapshot):
        html = generate_memory_html(snapshot, title="Test")
        assert "d3.v7" in html

    def test_hwm_timestep_in_meta(self, snapshot):
        html = generate_memory_html(snapshot, title="Test")
        assert "hwm_timestep" in html

    def test_search_input_present(self, snapshot):
        html = generate_memory_html(snapshot, title="Test")
        assert "search-input" in html

    def test_minimap_present(self, snapshot):
        html = generate_memory_html(snapshot, title="Test")
        assert "minimap" in html
        assert "const chartUpdateHooks = [];" in html
        assert "chartUpdateHooks.push(updateMinimap);" in html

    def test_leaks_view_present_in_registry(self, snapshot):
        html = generate_memory_html(snapshot, title="Test")
        assert "const detailViews = [" in html
        assert "id: 'leaks'" in html
        assert "label: 'Leaks'" in html

    def test_show_leaks_function_present(self, snapshot):
        html = generate_memory_html(snapshot, title="Test")
        assert "function showLeaks()" in html

    def test_leak_bar_css_present(self, snapshot):
        html = generate_memory_html(snapshot, title="Test")
        assert "leak-bar" in html

    def test_precomputes_derived_view_data(self, snapshot):
        html = generate_memory_html(snapshot, title="Test")
        assert "function buildDerivedData()" in html
        assert "const derivedData = buildDerivedData();" in html
        assert "stackSummaries" in html
        assert "peakAllocIndices" in html
        assert "leakGroups" in html

    def test_uses_requested_device(self):
        snapshot = {
            "device_traces": [
                [],
                [_make_event("alloc", 0x2000, 256, time_us=5, filename="rank1.py", name="alloc")],
            ],
            "segments": [
                {
                    "device": 1,
                    "stream": 0,
                    "address": 0x1000,
                    "blocks": [
                        {
                            "state": "active_allocated",
                            "size": 128,
                            "addr": 0x1000,
                            "frames": [{"filename": "rank1.py", "name": "seed", "line": 1}],
                        }
                    ],
                }
            ],
            "allocator_settings": {},
        }
        bootstrap = _extract_bootstrap(generate_memory_html(snapshot, device=1, title="Rank 1"))
        assert bootstrap["meta"]["device"] == 1
        assert bootstrap["meta"]["num_events"] == 2
        assert bootstrap["meta"]["num_allocs"] == 2


class TestGenerateComparisonHTML:
    def test_includes_trace_toggle_controls(self, snapshot):
        html = generate_memory_comparison_html(
            snapshot,
            snapshot,
            title_left="Trace A",
            title_right="Trace B",
        )
        assert 'id="trace-toggle-group"' in html
        assert 'data-trace-side="left"' in html
        assert 'data-trace-side="right"' in html
        assert "Trace A" in html
        assert "Trace B" in html

    def test_uses_fixed_layout_instead_of_splitter(self, snapshot):
        html = generate_memory_comparison_html(snapshot, snapshot)
        assert "function applyPaneLayout()" in html
        assert "window.addEventListener('resize', applyPaneLayout);" in html
        assert 'id="splitter"' not in html

    def test_uses_requested_devices_per_side(self):
        snapshot_left = {
            "device_traces": [
                [_make_event("alloc", 0x1000, 128, time_us=1, filename="left.py", name="left")]
            ],
            "segments": [
                {
                    "device": 0,
                    "stream": 0,
                    "address": 0x1000,
                    "blocks": [
                        {
                            "state": "active_allocated",
                            "size": 64,
                            "addr": 0x1000,
                            "frames": [{"filename": "left.py", "name": "seed", "line": 1}],
                        }
                    ],
                }
            ],
            "allocator_settings": {},
        }
        snapshot_right = {
            "device_traces": [
                [],
                [_make_event("alloc", 0x2000, 256, time_us=2, filename="right.py", name="right")],
            ],
            "segments": [
                {
                    "device": 1,
                    "stream": 0,
                    "address": 0x2000,
                    "blocks": [
                        {
                            "state": "active_allocated",
                            "size": 128,
                            "addr": 0x2000,
                            "frames": [{"filename": "right.py", "name": "seed", "line": 1}],
                        }
                    ],
                }
            ],
            "allocator_settings": {},
        }
        html = generate_memory_comparison_html(
            snapshot_left,
            snapshot_right,
            device_left=0,
            device_right=1,
            title_left="Rank 0",
            title_right="Rank 1",
        )
        bootstrap_left = _extract_named_bootstrap(html, "bootstrap-left")
        bootstrap_right = _extract_named_bootstrap(html, "bootstrap-right")
        assert bootstrap_left["meta"]["device"] == 0
        assert bootstrap_right["meta"]["device"] == 1
        assert bootstrap_left["meta"]["num_allocs"] == 2
        assert bootstrap_right["meta"]["num_allocs"] == 2


class TestNeverFreedAllocations:
    def test_never_freed_end_at_max_ts(self):
        events = [
            _make_event("alloc", 0x1000, 1024, time_us=1),
            _make_event("alloc", 0x2000, 2048, time_us=2),
        ]
        _, allocs, _, _, max_ts, _ = process_snapshot(_make_snapshot(events))
        assert len(allocs) == 2
        for poly in allocs:
            assert poly["ts"][-1] == max_ts

    def test_freed_alloc_ends_before_max_ts(self):
        events = [
            _make_event("alloc", 0x1000, 1024, time_us=1),
            _make_event("alloc", 0x2000, 2048, time_us=2),
            _make_event("free_completed", 0x1000, 1024, time_us=3),
        ]
        _, allocs, _, _, max_ts, _ = process_snapshot(_make_snapshot(events))
        freed = [a for a in allocs if a["ts"][-1] < max_ts]
        alive = [a for a in allocs if a["ts"][-1] == max_ts]
        assert len(freed) == 1
        assert len(alive) == 1
        assert freed[0]["s"] == 1024
        assert alive[0]["s"] == 2048

    def test_real_snapshot_has_both(self, snapshot):
        _, allocs, _, _, max_ts, _ = process_snapshot(snapshot)
        freed = [a for a in allocs if a["ts"][-1] < max_ts]
        alive = [a for a in allocs if a["ts"][-1] == max_ts]
        assert len(freed) > 0
        assert len(alive) > 0

    def test_html_uses_dedicated_persistent_palette(self, snapshot):
        html = generate_memory_html(snapshot, title="Test")
        assert "const PERSISTENT_ALPHAS = [0.55, 0.62, 0.70];" in html


class TestStackDeduplication:
    def test_identical_frames_share_stack_index(self):
        events = [
            _make_event("alloc", 0x1000, 1024, filename="a.py", name="foo", line=10),
            _make_event("alloc", 0x2000, 2048, filename="a.py", name="foo", line=10),
        ]
        _, allocs, frames, stacks, *_ = process_snapshot(_make_snapshot(events))
        assert allocs[0]["si"] == allocs[1]["si"]
        assert len(stacks) == 1

    def test_different_frames_get_different_stacks(self):
        events = [
            _make_event("alloc", 0x1000, 1024, filename="a.py", name="foo", line=10),
            _make_event("alloc", 0x2000, 2048, filename="b.py", name="bar", line=20),
        ]
        _, allocs, _, stacks, *_ = process_snapshot(_make_snapshot(events))
        assert allocs[0]["si"] != allocs[1]["si"]
        assert len(stacks) == 2

    def test_real_snapshot_deduplicates(self, snapshot):
        _, allocs, _, stacks, *_ = process_snapshot(snapshot)
        used_stacks = {a["si"] for a in allocs}
        assert len(used_stacks) < len(allocs)


class TestFreeShiftsAbove:
    def test_freeing_bottom_shifts_above_down(self):
        events = [
            _make_event("alloc", 0x1000, 100, time_us=1, name="bottom"),
            _make_event("alloc", 0x2000, 200, time_us=2, name="top"),
            _make_event("free_completed", 0x1000, 100, time_us=3),
        ]
        _, allocs, *_ = process_snapshot(_make_snapshot(events))
        bottom = next(a for a in allocs if a["s"] == 100)
        top = next(a for a in allocs if a["s"] == 200)
        assert bottom["offsets"][0] == 0
        assert top["offsets"][0] == 100
        assert top["offsets"][-1] == 0

    def test_freeing_middle_shifts_only_above(self):
        events = [
            _make_event("alloc", 0x1000, 100, time_us=1, name="a"),
            _make_event("alloc", 0x2000, 200, time_us=2, name="b"),
            _make_event("alloc", 0x3000, 300, time_us=3, name="c"),
            _make_event("free_completed", 0x2000, 200, time_us=4),
        ]
        _, allocs, *_ = process_snapshot(_make_snapshot(events))
        a = next(p for p in allocs if p["s"] == 100)
        c = next(p for p in allocs if p["s"] == 300)
        assert a["offsets"][-1] == 0
        assert c["offsets"][-1] == c["offsets"][0] - 200


class TestSegmentEvents:
    def test_segment_events_dont_create_polys(self):
        events = [
            _make_event("segment_alloc", 0xA000, 4096, time_us=1),
            _make_event("alloc", 0x1000, 1024, time_us=2),
            _make_event("segment_free", 0xA000, 4096, time_us=3),
        ]
        timeline, allocs, *_ = process_snapshot(_make_snapshot(events))
        assert len(allocs) == 1
        assert allocs[0]["s"] == 1024

    def test_segment_events_affect_reserved(self):
        events = [
            _make_event("segment_alloc", 0xA000, 4096, time_us=1),
            _make_event("alloc", 0x1000, 1024, time_us=2),
            _make_event("segment_free", 0xA000, 4096, time_us=3),
        ]
        timeline, *_ = process_snapshot(_make_snapshot(events))
        reserved_values = [e["r"] for e in timeline]
        assert reserved_values[0] == 4096
        assert reserved_values[-1] == 0


class TestTimelineConsistency:
    def test_max_at_time_matches_timeline(self, snapshot):
        timeline, *_ = process_snapshot(snapshot)
        max_at_time = [e["a"] for e in timeline]
        assert len(max_at_time) == len(timeline)
        assert all(m >= 0 for m in max_at_time)

    def test_hwm_monotonically_increases(self, snapshot):
        timeline, *_ = process_snapshot(snapshot)
        hwm_values = [e["h"] for e in timeline]
        for i in range(1, len(hwm_values)):
            assert hwm_values[i] >= hwm_values[i - 1]

    def test_allocated_matches_alloc_minus_free(self):
        events = [
            _make_event("alloc", 0x1000, 100, time_us=1),
            _make_event("alloc", 0x2000, 200, time_us=2),
            _make_event("free_completed", 0x1000, 100, time_us=3),
        ]
        timeline, *_ = process_snapshot(_make_snapshot(events))
        assert timeline[0]["a"] == 100
        assert timeline[1]["a"] == 300
        assert timeline[2]["a"] == 200


def _find_leaks(allocs: list[dict], max_ts: int, early_pct: float = 0.05) -> list[int]:
    """Reimplement the JS showLeaks() never-freed detection in Python for testing."""
    early_threshold = max_ts * early_pct
    return [
        i
        for i in range(len(allocs))
        if allocs[i]["ts"][-1] >= max_ts and allocs[i]["ts"][0] > early_threshold
    ]


class TestLeakDetection:
    def test_all_freed_no_leaks(self):
        events = [
            _make_event("alloc", 0x1000, 100, time_us=1, name="a"),
            _make_event("alloc", 0x2000, 200, time_us=2, name="b"),
            _make_event("free_completed", 0x1000, 100, time_us=3),
            _make_event("free_completed", 0x2000, 200, time_us=4),
        ]
        _, allocs, _, _, max_ts, _ = process_snapshot(_make_snapshot(events))
        assert len(_find_leaks(allocs, max_ts)) == 0

    def test_early_alloc_filtered_out(self):
        events = [
            _make_event("alloc", 0x1000, 100, time_us=1, name="model_param"),
        ]
        _, allocs, _, _, max_ts, _ = process_snapshot(_make_snapshot(events))
        assert len(_find_leaks(allocs, max_ts)) == 0

    def test_late_never_freed_is_candidate(self):
        events = []
        for i in range(20):
            events.append(
                _make_event("alloc", 0x1000 + i * 0x100, 100, time_us=i + 1, name="setup")
            )
            events.append(_make_event("free_completed", 0x1000 + i * 0x100, 100, time_us=i + 100))
        events.append(_make_event("alloc", 0x9000, 512, time_us=500, name="leaked"))
        _, allocs, _, _, max_ts, _ = process_snapshot(_make_snapshot(events))
        candidates = _find_leaks(allocs, max_ts)
        assert len(candidates) == 1
        assert allocs[candidates[0]]["s"] == 512

    def test_multiple_leaks_from_same_site_all_detected(self):
        events = []
        for i in range(10):
            events.append(
                _make_event("alloc", 0x1000 + i * 0x100, 100, time_us=i + 1, name="churn")
            )
            events.append(_make_event("free_completed", 0x1000 + i * 0x100, 100, time_us=i + 50))
        for i in range(3):
            events.append(
                _make_event("alloc", 0x9000 + i * 0x100, 200, time_us=60 + i, name="leaky_append")
            )
        _, allocs, _, _, max_ts, _ = process_snapshot(_make_snapshot(events))
        candidates = _find_leaks(allocs, max_ts)
        assert len(candidates) == 3
        for c in candidates:
            assert allocs[c]["s"] == 200

    def test_mixed_early_and_late_never_freed(self):
        events = [
            _make_event("alloc", 0x1000, 100, time_us=1, name="param"),
        ]
        for i in range(20):
            events.append(_make_event("alloc", 0x2000 + i * 0x100, 50, time_us=10 + i, name="tmp"))
            events.append(
                _make_event("free_completed", 0x2000 + i * 0x100, 50, time_us=10 + i + 1)
            )
        events.append(_make_event("alloc", 0x9000, 300, time_us=500, name="leaked"))
        _, allocs, _, _, max_ts, _ = process_snapshot(_make_snapshot(events))
        candidates = _find_leaks(allocs, max_ts)
        assert len(candidates) == 1
        assert allocs[candidates[0]]["s"] == 300
