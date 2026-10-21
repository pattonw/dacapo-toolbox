import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from funlib.geometry import Coordinate, Roi
from funlib.persistence import Array

pytest.importorskip("matplotlib")

from dacapo_toolbox.vis.preview import (
    _mesh_strides,
    cube,
    draw_label_meshes,
    get_cmap,
    graph_cube,
    label_meshes,
)


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


def test_cube_draws_label_meshes(tmp_path):
    # two boxes in a mostly empty volume: one mesh each, inside the roi
    labels = np.zeros((10, 20, 20), dtype=np.uint64)
    labels[2:5, 3:8, 3:8] = 7
    labels[5:9, 10:18, 12:19] = 2**40
    arr = Array(labels, offset=Coordinate(40, 0, 0), voxel_size=Coordinate(4, 2, 2))
    meshes = label_meshes(arr)
    assert sorted(label for label, _, _ in meshes) == [7, 2**40]
    for _, verts, faces in meshes:
        assert len(faces)
        assert (verts >= np.array(arr.roi.begin) - 4).all()
        assert (verts <= np.array(arr.roi.end) + 4).all()
    # a budget smaller than the array strides it, finest axes first
    assert _mesh_strides(arr, 1000) == (1, 2, 2)
    out = tmp_path / "mesh.png"
    cube({"labels": arr}, {"labels": "mesh"}, str(out), "mesh", dpi=40)
    assert out.exists() and out.stat().st_size > 0


def test_mesh_box_is_the_label_roi(monkeypatch):
    # labels smaller than the raw: the box marks the labels, the axes the union
    import dacapo_toolbox.vis.preview as preview

    raw = Array(np.zeros((20, 40, 40), np.uint8), voxel_size=Coordinate(4, 2, 2))
    labels = np.zeros((10, 20, 20), dtype=np.uint32)
    labels[2:8, 5:15, 5:15] = 3
    arr = Array(labels, offset=Coordinate(20, 20, 20), voxel_size=Coordinate(4, 2, 2))
    drawn = []
    original = preview._draw_cuboid
    monkeypatch.setattr(
        preview,
        "_draw_cuboid",
        lambda ax, roi, alpha, frame=None: (
            drawn.append(roi),
            original(ax, roi, alpha, frame=frame),
        ),
    )
    ax = plt.figure().add_subplot(111, projection="3d")
    draw_label_meshes(ax, arr, raw.roi.union(arr.roi), get_cmap())
    assert drawn == [arr.roi]
    assert ax.get_xlim() == (0, 80) and ax.get_zlim() == (0, 80)
