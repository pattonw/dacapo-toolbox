import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from funlib.geometry import Coordinate, Roi
from funlib.persistence import Array

pytest.importorskip("matplotlib")

from dacapo_toolbox.vis.preview import cube, graph_cube


def test_cube_writes_an_image(tmp_path):
    raw = Array(
        np.random.default_rng(0).integers(0, 255, (8, 8, 8), dtype=np.uint8),
        offset=Coordinate(0, 0, 0),
        voxel_size=Coordinate(4, 2, 2),
    )
    labels = Array(
        (np.arange(8 * 8 * 8).reshape(8, 8, 8) % 5).astype(np.uint32),
        offset=Coordinate(0, 0, 0),
        voxel_size=Coordinate(4, 2, 2),
    )
    out = tmp_path / "cube.png"
    cube(
        {"raw": raw, "labels": labels},
        {"raw": "raw", "labels": "labels"},
        str(out),
        "test",
        dpi=40,
    )
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.parametrize("color_by", [None, "skeleton_id"])
def test_graph_cube_writes_an_image(tmp_path, color_by):
    graph = nx.Graph()
    rng = np.random.default_rng(0)
    for i in range(300):
        graph.add_node(
            i,
            position=tuple(rng.uniform(0, 100, 3)),
            skeleton_id=str(i % 7),
        )
    for i in range(298):
        if i % 7 == (i + 1) % 7 - 1:
            graph.add_edge(i, i + 1)
    graph.add_node(999)  # no position: skipped, not an error
    out = tmp_path / f"graph_{color_by}.png"
    graph_cube(
        graph,
        Roi((0, 0, 0), (100, 100, 100)),
        str(out),
        "skeletons",
        color_by=color_by,
        max_nodes=100,
        dpi=40,
    )
    assert out.exists() and out.stat().st_size > 0
    # whole groups are kept when subsampling by color_by
    ax = graph_cube(
        graph,
        Roi((0, 0, 0), (100, 100, 100)),
        str(out),
        "x",
        color_by=color_by,
        max_nodes=100,
        ax=plt.figure().add_subplot(111, projection="3d"),
    )
    n_drawn = ax.collections[0].get_offsets().shape[0]
    # 7 groups of 42 or 43 nodes: two whole groups fit the budget of 100
    assert n_drawn in (85, 86) if color_by is not None else n_drawn == 100
    # overwrite=False leaves the file alone
    before = out.stat().st_mtime_ns
    graph_cube(graph, Roi((0, 0, 0), (100, 100, 100)), str(out), "x", overwrite=False)
    assert out.stat().st_mtime_ns == before
