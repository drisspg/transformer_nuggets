import pickle
from pathlib import Path

import pytest

from transformer_nuggets.utils.memory_viz import (
    _extract_frames,
    _is_cpython_c_frame,
    _shorten_path,
    generate_memory_html,
    process_snapshot,
)

DATA_DIR = Path(__file__).parent / "data"
SNAPSHOT_PATH = DATA_DIR / "mini_snapshot.pickle"


@pytest.fixture
def snapshot():
    with open(SNAPSHOT_PATH, "rb") as f:
        return pickle.load(f)


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
        timeline, allocs, stacks, categories, max_ts = process_snapshot(snapshot)
        assert len(timeline) > 0
        assert len(allocs) > 0
        assert len(stacks) > 0
        assert len(categories) == len(stacks)
        assert max_ts > 0

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
        timeline, allocs, stacks, *_ = process_snapshot(snapshot)
        for e in timeline:
            assert 0 <= e["si"] < len(stacks)
        for a in allocs:
            assert 0 <= a["si"] < len(stacks)

    def test_empty_device_returns_empty(self, snapshot):
        result = process_snapshot(snapshot, device=99)
        assert result == ([], [], [], [], 0)

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
        for placeholder in ["__TITLE__", "__TIMELINE__", "__ALLOCS__", "__STACKS__", "__CATEGORIES__", "__META__"]:
            assert placeholder not in html

    def test_title_appears_in_html(self, snapshot):
        html = generate_memory_html(snapshot, title="My Custom Title")
        assert "My Custom Title" in html

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
