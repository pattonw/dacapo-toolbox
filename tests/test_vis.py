import networkx as nx
import numpy as np
import pytest
from funlib.geometry import Coordinate, Roi
from funlib.persistence import Array

plt = pytest.importorskip("matplotlib.pyplot", reason="needs the `vis` extra")

from dacapo_toolbox.vis.preview import (  # noqa: E402
    cube,
    draw_label_meshes,
    get_cmap,
    gif_2d,
    graph_cube,
    label_meshes,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
GIF_SIGNATURE = b"GIF89a"
ROI = Roi((0, 0, 0), (100, 100, 100))


@pytest.fixture
def ax():
    """A 3D axes on its own figure, closed after the test."""
    fig = plt.figure()
    yield fig.add_subplot(111, projection="3d")
    plt.close(fig)


@pytest.fixture
def two_boxes():
    """Labels 7 and 2**40 as two boxes in a mostly empty volume, the second
    touching the far end of the roi."""
    labels = np.zeros((10, 20, 20), dtype=np.uint64)
    labels[2:5, 3:8, 3:8] = 7
    labels[5:10, 10:20, 12:20] = 2**40
    return Array(labels, offset=Coordinate(40, 0, 0), voxel_size=Coordinate(4, 2, 2))


def point_xs(ax):
    """Plot x coordinate (the last position component) of every drawn node.
    Read before the canvas is drawn, while the offsets are in data units."""
    return sorted(ax.collections[0].get_offsets()[:, 0].tolist())


def segment_count(ax):
    """Number of drawn edges. matplotlib fills in a 3D line collection's
    segments when it projects them, so draw first."""
    ax.figure.canvas.draw()
    return len(ax.collections[1].get_segments())


def mesh_bounds(meshes):
    return {
        label: (tuple(verts.min(axis=0)), tuple(verts.max(axis=0)))
        for label, verts, _ in meshes
    }


def volume(array_type):
    """An 8x8x8 array of the kind each array type draws, at voxel size (4, 2, 2)."""
    rng = np.random.default_rng(0)
    if array_type == "raw":
        data = rng.integers(0, 255, (8, 8, 8), dtype=np.uint8)
    elif array_type in ("labels", "mesh"):
        data = (np.arange(8 * 8 * 8).reshape(8, 8, 8) // 128).astype(np.uint32)
    elif array_type == "affs":
        data = rng.random((3, 8, 8, 8), dtype=np.float32)
    else:
        data = rng.random((6, 8, 8, 8), dtype=np.float32)
    return Array(data, offset=Coordinate(0, 0, 0), voxel_size=Coordinate(4, 2, 2))


@pytest.mark.parametrize("array_type", ["labels", "mesh", "affs", "pca"])
def test_cube_writes_a_png(tmp_path, array_type):
    out = tmp_path / "cube.png"
    cube(
        {"raw": volume("raw"), "other": volume(array_type)},
        {"raw": "raw", "other": array_type},
        str(out),
        "test",
        dpi=40,
    )
    assert out.read_bytes()[:8] == PNG_SIGNATURE


@pytest.mark.parametrize("array_type", ["labels", "affs", "pca"])
def test_gif_2d_writes_a_gif(tmp_path, array_type):
    out = tmp_path / "sections.gif"
    gif_2d(
        {"raw": volume("raw"), "other": volume(array_type)},
        {"raw": "raw", "other": array_type},
        str(out),
        "test",
        dpi=20,
    )
    assert out.read_bytes()[:6] == GIF_SIGNATURE


@pytest.mark.parametrize("preview", [cube, gif_2d])
def test_previews_leave_an_existing_file_when_overwrite_is_false(tmp_path, preview):
    out = tmp_path / "preview"
    out.write_bytes(b"already here")
    preview({"raw": volume("raw")}, {"raw": "raw"}, str(out), "test", overwrite=False)
    assert out.read_bytes() == b"already here"


def test_graph_cube_writes_a_png(tmp_path):
    graph = nx.path_graph(10)
    for i in graph.nodes:
        graph.nodes[i]["position"] = (10.0 * i, 50.0, 50.0)
    out = tmp_path / "graph.png"
    graph_cube(graph, ROI, str(out), "skeleton", dpi=40)
    assert out.read_bytes()[:8] == PNG_SIGNATURE


def test_graph_cube_overwrite_false_leaves_the_file(tmp_path):
    graph = nx.Graph()
    graph.add_node(0, position=(50.0, 50.0, 50.0))
    out = tmp_path / "graph.png"
    out.write_bytes(b"already here")
    graph_cube(graph, ROI, str(out), "point", overwrite=False)
    assert out.read_bytes() == b"already here"


@pytest.mark.parametrize("color_by", [None, "skeleton_id"])
def test_graph_cube_thins_every_skeleton(ax, color_by):
    # seven chains of 40 nodes with a budget of 100: every chain is drawn,
    # thinned to every third node plus its end, and stays connected
    graph = nx.Graph()
    for s in range(7):
        for i in range(40):
            graph.add_node((s, i), position=(s * 10.0, i * 2.0, 0.0), skeleton_id=s)
            if i:
                graph.add_edge((s, i - 1), (s, i))
    graph_cube(graph, ROI, "unused.png", "x", color_by=color_by, max_nodes=100, ax=ax)
    assert len(point_xs(ax)) == 7 * 14
    assert segment_count(ax) == 7 * 13


def test_graph_cube_points_are_sampled_to_max_nodes(ax):
    graph = nx.Graph()
    for i in range(300):
        graph.add_node(i, position=(float(i % 100), float(i // 100), 0.0))
    graph_cube(graph, ROI, "unused.png", "x", max_nodes=100, ax=ax)
    assert len(point_xs(ax)) == 100


def test_graph_cube_keeps_leaves_and_branch_points(ax):
    # a star of 30 arms of length 2 with a budget of 10: the centre (x 50), a
    # branch point, and every arm end (x 52), a leaf, are drawn; the nodes
    # between them (x 51) are thinned away
    graph = nx.Graph()
    graph.add_node("c", position=(50.0, 50.0, 50.0))
    for a in range(30):
        graph.add_node((a, 1), position=(50.0, 50.0 + a, 51.0))
        graph.add_node((a, 2), position=(50.0, 50.0 + a, 52.0))
        graph.add_edge("c", (a, 1))
        graph.add_edge((a, 1), (a, 2))
    graph_cube(graph, ROI, "unused.png", "x", max_nodes=10, ax=ax)
    assert point_xs(ax) == [50.0] + [52.0] * 30


def test_graph_cube_splits_paths_at_nodes_without_position(ax):
    # a chain along x whose middle node (20) has no position, thinned to every
    # fourth node: each half is drawn from its first node to its last, and the
    # ten segments are the five within each half, none across the gap
    graph = nx.path_graph(41)
    for i in graph.nodes:
        if i != 20:
            graph.nodes[i]["position"] = (0.0, 0.0, float(i))
    graph_cube(graph, ROI, "unused.png", "x", max_nodes=10, ax=ax)
    assert point_xs(ax) == [0, 4, 8, 12, 16, 19, 21, 25, 29, 33, 37, 40]
    assert segment_count(ax) == 10


def test_label_meshes_span_their_label_voxels(two_boxes):
    # each mesh spans exactly its label's voxels in world units (z, y, x)
    assert mesh_bounds(label_meshes(two_boxes)) == {
        7: ((48, 6, 6), (60, 16, 16)),
        2**40: ((60, 20, 24), (80, 40, 40)),
    }


def test_label_meshes_stride_to_the_voxel_budget(two_boxes):
    # 4000 voxels against a budget of 1000: the fine axes (y, x) are read at
    # every second voxel, so voxels 4 and 6 stand for label 7 (3 to 7) and its
    # mesh runs from 4 * 2 - 1 to 7 * 2 + 1; likewise for the second label
    assert mesh_bounds(label_meshes(two_boxes, max_voxels=1000)) == {
        7: ((48, 7, 7), (60, 15, 15)),
        2**40: ((60, 19, 23), (80, 39, 39)),
    }


def test_draw_label_meshes_boxes_the_label_roi_inside_the_frame(ax):
    # labels smaller than the raw: what is drawn extends to the box around the
    # label roi, and the axes span the frame (here the union with the raw)
    raw = Array(np.zeros((20, 40, 40), np.uint8), voxel_size=Coordinate(4, 2, 2))
    labels = np.zeros((10, 20, 20), dtype=np.uint32)
    labels[2:8, 5:15, 5:15] = 3
    arr = Array(labels, offset=Coordinate(20, 20, 20), voxel_size=Coordinate(4, 2, 2))
    draw_label_meshes(ax, arr, raw.roi.union(arr.roi), get_cmap())
    # the plot's x, y, z are the roi's dims 2, 1, 0
    assert tuple(ax.xy_dataLim.extents) == (20, 20, 60, 60)
    assert tuple(ax.zz_dataLim.intervalx) == (20, 60)
    assert (ax.get_xlim(), ax.get_ylim(), ax.get_zlim()) == ((0, 80), (0, 80), (0, 80))
