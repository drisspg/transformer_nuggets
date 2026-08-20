import html as html_lib
import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from transformer_nuggets.utils import memory_viz
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


def _make_event(action, addr, size, time_us=0, filename="test.py", name="func", line=1, **extra):
    event = {
        "action": action,
        "addr": addr,
        "size": size,
        "stream": 0,
        "time_us": time_us,
        "compile_context": "N/A",
        "user_metadata": "",
        "frames": [{"filename": filename, "name": name, "line": line}],
    }
    event.update(extra)
    return event


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


def _chrome_binary() -> str | None:
    cached = [
        *sorted(Path.home().glob(".agent-browser/browsers/chrome-*/chrome"), reverse=True),
        *sorted(
            Path.home().glob(".local/share/pyppeteer/local-chromium/*/chrome-linux/chrome"),
            reverse=True,
        ),
    ]
    candidates = [
        os.environ.get("CHROME_BIN"),
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        *(str(path) for path in cached),
    ]
    return next(
        (candidate for candidate in candidates if candidate and Path(candidate).is_file()), None
    )


def _run_browser_check(tmp_path: Path, document: str, test_script: str) -> dict:
    chrome = _chrome_binary()
    if chrome is None:
        pytest.skip("Chrome/Chromium is required for memory visualizer browser tests")
    collector = """
<script>
window.__memoryVizErrors = [];
window.addEventListener('error', event => window.__memoryVizErrors.push(String(event.error || event.message)));
window.addEventListener('unhandledrejection', event => window.__memoryVizErrors.push(String(event.reason)));
</script>
"""
    finish = """
function finishMemoryVizTest(result) {
  const output = document.createElement('pre');
  output.id = 'memory-viz-test-result';
  output.textContent = JSON.stringify({...result, errors: window.__memoryVizErrors});
  document.body.appendChild(output);
}
"""
    instrumented = document.replace("<head>", f"<head>{collector}", 1).replace(
        "</body>", f"<script>{finish}\n{test_script}</script></body>", 1
    )
    path = tmp_path / "memory_viz.html"
    path.write_text(instrumented)
    result = subprocess.run(
        [
            chrome,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--allow-file-access-from-files",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=1500",
            "--dump-dom",
            path.as_uri(),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    marker = '<pre id="memory-viz-test-result">'
    start = result.stdout.index(marker) + len(marker)
    end = result.stdout.index("</pre>", start)
    return json.loads(html_lib.unescape(result.stdout[start:end]))


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
        timeline, allocs, frames, stacks, max_ts, _hwm_ts = process_snapshot(snapshot)
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
        assert set(entry.keys()) == {
            "step",
            "t",
            "a",
            "r",
            "h",
            "act",
            "s",
            "si",
            "addr",
            "pool",
            "device_free",
            "metadata",
        }

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


class TestCurrentSnapshotSemantics:
    def test_live_snapshot_block_is_not_replayed_twice(self):
        snapshot = {
            "device_traces": [[_make_event("alloc", 0x1100, 100)]],
            "segments": [
                {
                    "device": 0,
                    "address": 0x1000,
                    "total_size": 4096,
                    "stream": 0,
                    "segment_pool_id": (0, 0),
                    "blocks": [
                        {
                            "state": "active_allocated",
                            "address": 0x1100,
                            "size": 128,
                            "requested_size": 100,
                            "frames": [],
                        }
                    ],
                }
            ],
            "allocator_settings": {},
        }

        timeline, allocs, *_ = process_snapshot(snapshot)

        assert len(allocs) == 1
        assert timeline[-1]["a"] == 100
        assert max(event["h"] for event in timeline) == 100

    def test_snapshot_only_blocks_use_address_and_requested_size(self):
        snapshot = {
            "device_traces": [[]],
            "segments": [
                {
                    "device": 0,
                    "address": 0x1000,
                    "total_size": 4096,
                    "stream": 0,
                    "segment_pool_id": (0, 0),
                    "blocks": [
                        {
                            "state": "active_allocated",
                            "address": 0x1100,
                            "size": 128,
                            "requested_size": 100,
                            "frames": [],
                        },
                        {
                            "state": "active_allocated",
                            "address": 0x1200,
                            "size": 64,
                            "requested_size": 50,
                            "frames": [],
                        },
                    ],
                }
            ],
            "allocator_settings": {},
        }

        timeline, allocs, *_ = process_snapshot(snapshot)

        assert {alloc["addr"] for alloc in allocs} == {"0x1100", "0x1200"}
        assert {alloc["s"] for alloc in allocs} == {100, 50}
        assert timeline[0]["a"] == 150

    def test_snapshot_blocks_without_addresses_use_segment_offsets(self):
        snapshot = {
            "device_traces": [[]],
            "segments": [
                {
                    "device": 0,
                    "address": 0x1000,
                    "total_size": 4096,
                    "stream": 0,
                    "segment_pool_id": (0, 0),
                    "blocks": [
                        {
                            "state": "active_allocated",
                            "size": 1000,
                            "requested_size": 900,
                            "frames": [],
                        },
                        {
                            "state": "active_allocated",
                            "size": 1500,
                            "requested_size": 1400,
                            "frames": [],
                        },
                    ],
                }
            ],
            "allocator_settings": {},
        }

        timeline, allocs, *_ = process_snapshot(snapshot)

        assert [alloc["addr"] for alloc in allocs] == ["0x1000", "0x13e8"]
        assert timeline[0]["a"] == 2300

    @pytest.mark.parametrize("state", ["active_pending_free", "active_awaiting_free"])
    def test_snapshot_pending_free_blocks_remain_visible(self, state):
        snapshot = {
            "device_traces": [[]],
            "segments": [
                {
                    "device": 0,
                    "address": 0x1000,
                    "total_size": 4096,
                    "stream": 0,
                    "segment_pool_id": (0, 0),
                    "blocks": [
                        {
                            "state": state,
                            "address": 0x1100,
                            "size": 128,
                            "requested_size": 100,
                            "frames": [],
                        }
                    ],
                }
            ],
            "allocator_settings": {},
        }

        timeline, allocs, *_ = process_snapshot(snapshot)

        assert timeline[0]["a"] == 100
        assert allocs[0]["free_requested"] is True

    def test_free_requested_waits_for_free_completed(self):
        events = [
            _make_event("alloc", 0x1000, 100),
            _make_event("free_requested", 0x1000, 100),
            _make_event("free_completed", 0x1000, 100),
        ]

        timeline, allocs, *_ = process_snapshot(_make_snapshot(events))

        assert [event["a"] for event in timeline] == [100, 100, 0]
        assert allocs[0]["free_requested"] is True

    def test_unmatched_free_reconstructs_initial_allocation(self):
        timeline, allocs, *_ = process_snapshot(
            _make_snapshot([_make_event("free_completed", 0x1000, 100)])
        )

        assert len(allocs) == 1
        assert allocs[0]["origin"] == "unmatched_free"
        assert [event["a"] for event in timeline] == [100, 0]

    def test_address_reuse_resets_annotations(self):
        events = [
            _make_event("alloc", 0x1000, 100),
            _make_event("annotate", 0x1000, 0, user_metadata="first"),
            _make_event("free_completed", 0x1000, 100),
            _make_event("alloc", 0x1000, 200),
        ]

        _, allocs, *_ = process_snapshot(_make_snapshot(events))

        assert allocs[0]["annotations"] == ["first"]
        assert allocs[1]["annotations"] == []

    def test_annotation_survives_evicted_alloc_event(self):
        snapshot = {
            "device_traces": [[_make_event("annotate", 0x1100, 0, user_metadata="weight")]],
            "segments": [
                {
                    "device": 0,
                    "address": 0x1000,
                    "total_size": 4096,
                    "stream": 0,
                    "segment_pool_id": (0, 0),
                    "blocks": [
                        {
                            "state": "active_allocated",
                            "address": 0x1100,
                            "size": 128,
                            "requested_size": 100,
                            "frames": [],
                        }
                    ],
                }
            ],
            "allocator_settings": {"trace_alloc_overflowed": True},
        }

        _, allocs, *_ = process_snapshot(snapshot)

        assert allocs[0]["annotations"] == ["weight"]

    def test_segment_map_and_unmap_update_reserved_memory(self):
        events = [
            _make_event("segment_map", 0xA000, 4096),
            _make_event("segment_unmap", 0xA000, 2048),
        ]

        timeline, *_ = process_snapshot(_make_snapshot(events))

        assert [event["r"] for event in timeline] == [4096, 2048]

    def test_private_pool_and_diagnostics_are_serialized(self):
        events = [
            _make_event("segment_alloc", 0x1000, 4096, pool_id=(1, 0)),
            _make_event("alloc", 0x1100, 100, pool_id=(1, 0)),
            _make_event("annotate", 0x1100, 0, user_metadata="graph output"),
            _make_event("snapshot", 0, 0),
            _make_event("oom", 0, 1024, device_free=512),
        ]
        snapshot = {
            "device_traces": [events],
            "segments": [
                {
                    "device": 0,
                    "address": 0x1000,
                    "total_size": 4096,
                    "allocated_size": 100,
                    "active_size": 100,
                    "stream": 0,
                    "segment_pool_id": (1, 0),
                    "segment_type": "small",
                    "blocks": [
                        {
                            "state": "active_allocated",
                            "address": 0x1100,
                            "size": 128,
                            "requested_size": 100,
                            "frames": [],
                        },
                        {
                            "state": "inactive",
                            "address": 0x1180,
                            "size": 3968,
                            "requested_size": 0,
                            "frames": [],
                        },
                    ],
                }
            ],
            "allocator_settings": {"trace_alloc_overflowed": False},
        }

        bootstrap = _extract_bootstrap(generate_memory_html(snapshot))

        assert bootstrap["allocs"][0]["pool"] == [1, 0]
        assert bootstrap["allocs"][0]["annotations"] == ["graph output"]
        assert bootstrap["private_pools"][0]["reserved_bytes"] == 4096
        assert bootstrap["private_pools"][0]["inactive_bytes"] == 3968
        assert {event["act"] for event in bootstrap["events"]} >= {"snapshot", "oom"}
        assert bootstrap["allocator_settings"]["trace_alloc_overflowed"] is False

    def test_fx_metadata_is_preserved(self):
        event = _make_event("alloc", 0x1000, 100)
        event["category"] = "activations"
        event["frames"][0].update(
            fx_node_op="call_function",
            fx_node_name="linear",
            fx_original_trace="model.py:10",
        )

        bootstrap = _extract_bootstrap(generate_memory_html(_make_snapshot([event])))

        assert bootstrap["allocs"][0]["category"] == "activations"
        assert bootstrap["allocs"][0]["fx"][0]["fx_node_name"] == "linear"

    @pytest.mark.slow
    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA Graph snapshot test requires CUDA"
    )
    def test_cuda_graph_snapshot_is_not_double_counted(self):
        torch.cuda.empty_cache()
        torch.cuda.memory._record_memory_history(
            max_entries=100_000,
            stacks="all",
            clear_history=True,
        )
        try:
            x = torch.randn(1024, 1024, device="cuda")
            output = torch.empty_like(x)
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                torch.sin(x, out=output)
            torch.cuda.current_stream().wait_stream(warmup_stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                torch.sin(x, out=output)
                captured_tmp = x * 2
                output.add_(captured_tmp)
            graph.replay()
            torch.cuda.synchronize()

            snapshot = torch.cuda.memory._snapshot()
            timeline, *_ = process_snapshot(snapshot, torch.cuda.current_device())
            expected_active = sum(
                block.get("requested_size", block["size"])
                for segment in snapshot["segments"]
                if segment["device"] == torch.cuda.current_device()
                for block in segment["blocks"]
                if block["state"]
                in {"active_allocated", "active_pending_free", "active_awaiting_free"}
            )
            expected_reserved = sum(
                segment["total_size"]
                for segment in snapshot["segments"]
                if segment["device"] == torch.cuda.current_device()
            )
            bootstrap = _extract_bootstrap(
                generate_memory_html(snapshot, device=torch.cuda.current_device())
            )

            assert timeline[-1]["a"] == expected_active
            assert timeline[-1]["r"] == expected_reserved == torch.cuda.memory_reserved()
            assert bootstrap["private_pools"]
            assert bootstrap["meta"]["private_pool_reserved_bytes"] > 0
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)


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
            "__D3_SOURCE__",
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
        assert "https://d3js.org v7.9.0" in html
        assert '<script src="https://d3js.org' not in html

    def test_negative_device_generates_empty_data(self, snapshot):
        bootstrap = _extract_bootstrap(generate_memory_html(snapshot, device=-1))
        assert bootstrap["allocs"] == []
        assert bootstrap["segments"] == []
        assert bootstrap["private_pools"] == []

    def test_empty_snapshot_has_zero_safe_javascript(self):
        html = generate_memory_html(_make_snapshot([]), title="Empty")
        bootstrap = _extract_bootstrap(html)
        assert bootstrap["timeline"] == [0]
        assert bootstrap["reserved_timeline"] == [0]
        assert "const MAX_TS = Math.max(1, META.max_timestep);" in html
        assert "const hitBucketSize = MAX_TS / NUM_HIT_BUCKETS;" in html
        assert "...RESERVED_TIMELINE.slice" not in html

    def test_panels_and_controls_start_compact(self, snapshot):
        html = generate_memory_html(snapshot, title="Test")
        assert '<div id="controls" class="collapsed">' in html
        assert '<div id="detail-panel" class="collapsed">' in html
        assert 'id="controls-toggle"' in html
        assert 'id="panel-toggle"' in html
        assert 'id="chart-legend"' in html
        assert "memory retained for replay" in html
        assert "allocator event order" in html
        assert "setDetailPanelCollapsed(false);" in html

    def test_exposes_default_allocator_diagnostics(self, snapshot):
        html = generate_memory_html(snapshot, title="Test")
        for feature in [
            "reserved-toggle",
            "showPools",
            "showEvents",
            "showSegments",
            "showSettings",
            "reserved_timeline",
            "private_pools",
            "allocator_settings",
        ]:
            assert feature in html

    def test_dynamic_text_is_escaped_before_inner_html(self, snapshot):
        html = generate_memory_html(snapshot, title="Test")
        assert "function escapeHtml(value)" in html
        assert "${escapeHtml(primary)}" in html
        assert "${escapeHtml(d.ctx || 'None')}" in html
        assert "${escapeHtml(g.frame)}" in html

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
        assert "setTimeout(applyPaneLayout, 200);" in html
        assert 'id="splitter"' not in html

    def test_comparison_includes_allocator_diagnostics_and_safe_zoom(self, snapshot):
        html = generate_memory_comparison_html(snapshot, snapshot)
        assert '<div id="controls" class="collapsed">' in html
        assert '<div id="detail-panel" class="collapsed">' in html
        assert 'id="chart-legend"' in html
        assert "memory retained for replay" in html
        assert "setDetailPanelCollapsed(false);" in html
        assert "const MAX_TS = Math.max(1, META.max_timestep);" in html
        assert "setReservedVisible" in html
        assert "id: 'pools'" in html
        assert "id: 'events'" in html
        assert "id: 'segments'" in html
        assert "id: 'settings'" in html
        assert "Math.max(1, TIMELINE.length - 1)" in html

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
                            "requested_size": 64,
                            "address": 0x1000,
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
                            "requested_size": 128,
                            "address": 0x2000,
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
        assert bootstrap_left["meta"]["num_allocs"] == 1
        assert bootstrap_right["meta"]["num_allocs"] == 1


class TestBrowserRuntime:
    def test_single_view_handles_empty_data_resize_and_untrusted_frames(self, tmp_path):
        payload = '</span><img id="pwned">'
        snapshot = _make_snapshot(
            [_make_event("alloc", 0x1000, 100, filename=f"/tmp/{payload}.py")]
        )
        result = _run_browser_check(
            tmp_path,
            generate_memory_html(snapshot),
            f"""
renderStack(0, 'malicious');
const escapedFrame = detailBody.textContent.includes({json.dumps(payload)});
const defaultCollapsed = detailPanel.classList.contains('collapsed') && controls.classList.contains('collapsed');
controlsToggle.click();
const controlsOpened = !controls.classList.contains('collapsed');
legendToggle.click();
const legendOpened = chartLegend.classList.contains('open') && legendToggle.getAttribute('aria-expanded') === 'true';
document.getElementById('reserved-toggle').click();
for (const tab of ['pools', 'events', 'segments', 'settings']) {{
  document.querySelector(`.detail-tab[data-tab="${{tab}}"]`).click();
}}
const overlay = document.querySelector('#chart-container rect[pointer-events="all"]');
overlay.dispatchEvent(new MouseEvent('mousemove', {{bubbles: true, clientX: 100, clientY: 100}}));
window.dispatchEvent(new Event('resize'));
document.getElementById('panel-toggle').click();
setTimeout(() => {{
  const invalid = [...document.querySelectorAll('*')].some(element =>
    [...element.attributes].some(attr => /NaN|Infinity/.test(attr.value))
  );
  const canvas = document.getElementById('alloc-canvas');
  finishMemoryVizTest({{
    invalid,
    escaped: escapedFrame,
    defaultCollapsed,
    controlsOpened,
    legendOpened,
    injected: Boolean(document.getElementById('pwned')),
    canvasAligned: Math.abs(parseFloat(canvas.style.width) - container.getBoundingClientRect().width) < 1,
  }});
}}, 400);
""",
        )

        assert result == {
            "invalid": False,
            "escaped": True,
            "defaultCollapsed": True,
            "controlsOpened": True,
            "legendOpened": True,
            "injected": False,
            "canvasAligned": True,
            "errors": [],
        }

    def test_comparison_links_empty_and_nonempty_panes_without_nan(self, tmp_path):
        payload = '</span><img id="pwned">'
        left = _make_snapshot([])
        right = _make_snapshot([_make_event("alloc", 0x1000, 100, filename=f"/tmp/{payload}.py")])
        result = _run_browser_check(
            tmp_path,
            generate_memory_comparison_html(left, right),
            f"""
renderStack('right', 0, 'malicious');
const escapedFrame = detailBody.textContent.includes({json.dumps(payload)});
const defaultCollapsed = detailPanel.classList.contains('collapsed') && controls.classList.contains('collapsed');
controlsToggle.click();
const controlsOpened = !controls.classList.contains('collapsed');
legendToggle.click();
const legendOpened = chartLegend.classList.contains('open') && legendToggle.getAttribute('aria-expanded') === 'true';
document.getElementById('reserved-toggle').click();
for (const tab of ['pools', 'events', 'segments', 'settings']) {{
  document.querySelector(`.detail-tab[data-tab="${{tab}}"]`).click();
}}
for (const overlay of document.querySelectorAll('.chart-pane rect[pointer-events="all"]')) {{
  overlay.dispatchEvent(new MouseEvent('mousemove', {{bubbles: true, clientX: 80, clientY: 80}}));
  overlay.dispatchEvent(new WheelEvent('wheel', {{bubbles: true, deltaY: -100, clientX: 80, clientY: 80}}));
}}
document.getElementById('panel-toggle').click();
window.dispatchEvent(new Event('resize'));
setTimeout(() => {{
  const invalid = [...document.querySelectorAll('*')].some(element =>
    [...element.attributes].some(attr => /NaN|Infinity/.test(attr.value))
  );
  const canvases = [...document.querySelectorAll('.chart-pane canvas')];
  finishMemoryVizTest({{
    invalid,
    escaped: escapedFrame,
    defaultCollapsed,
    controlsOpened,
    legendOpened,
    injected: Boolean(document.getElementById('pwned')),
    canvasAligned: canvases.every(canvas =>
      Math.abs(parseFloat(canvas.style.width) - canvas.parentElement.getBoundingClientRect().width) < 1
    ),
  }});
}}, 400);
""",
        )

        assert result == {
            "invalid": False,
            "escaped": True,
            "defaultCollapsed": True,
            "controlsOpened": True,
            "legendOpened": True,
            "injected": False,
            "canvasAligned": True,
            "errors": [],
        }


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
    def test_shared_frame_objects_use_identity_fast_paths(self, monkeypatch):
        raw_frames = [{"filename": "a.py", "name": "foo", "line": 10}]
        events = [
            _make_event("alloc", 0x1000, 100),
            _make_event("free_completed", 0x1000, 100),
        ]
        for event in events:
            event["frames"] = raw_frames

        extract_calls = 0
        fx_calls = 0
        original_extract = memory_viz._extract_frames
        original_fx = memory_viz._fx_metadata

        def extract(frames):
            nonlocal extract_calls
            extract_calls += 1
            return original_extract(frames)

        def fx(frames):
            nonlocal fx_calls
            fx_calls += 1
            return original_fx(frames)

        monkeypatch.setattr(memory_viz, "_extract_frames", extract)
        monkeypatch.setattr(memory_viz, "_fx_metadata", fx)

        process_snapshot(_make_snapshot(events))

        assert extract_calls == 1
        assert fx_calls == 1

    def test_identical_frames_share_stack_index(self):
        events = [
            _make_event("alloc", 0x1000, 1024, filename="a.py", name="foo", line=10),
            _make_event("alloc", 0x2000, 2048, filename="a.py", name="foo", line=10),
        ]
        _, allocs, _, stacks, *_ = process_snapshot(_make_snapshot(events))
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
        _, allocs, *_ = process_snapshot(snapshot)
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
        _, allocs, *_ = process_snapshot(_make_snapshot(events))
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
