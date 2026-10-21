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
    ax = graph_cube(
        graph,
        Roi((0, 0, 0), (100, 100, 100)),
        str(out),
        "x",
        color_by=color_by,
        max_nodes=100,
        ax=plt.figure().add_subplot(111, projection="3d"),
    )
    if color_by is None:
        assert ax.collections[0].get_offsets().shape[0] == 100
    # overwrite=False leaves the file alone
    before = out.stat().st_mtime_ns
    graph_cube(graph, Roi((0, 0, 0), (100, 100, 100)), str(out), "x", overwrite=False)
    assert out.stat().st_mtime_ns == before


def test_graph_cube_thins_every_skeleton():
    # seven chains of 40 nodes with a budget of 100: every chain is drawn,
    # thinned to every third node plus its ends, and stays connected
    graph = nx.Graph()
    for s in range(7):
        for i in range(40):
            graph.add_node((s, i), position=(s * 10.0, i * 2.0, 0.0), skeleton_id=s)
            if i:
                graph.add_edge((s, i - 1), (s, i))
    ax = graph_cube(
        graph,
        Roi((0, 0, 0), (100, 100, 100)),
        "unused.png",
        "x",
        color_by="skeleton_id",
        max_nodes=100,
        ax=plt.figure().add_subplot(111, projection="3d"),
    )
    points = ax.collections[0].get_offsets()
    assert len(points) == 7 * 14
    segments = ax.collections[1]._segments3d
    assert len(segments) == 7 * 13
