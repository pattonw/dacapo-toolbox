from typing import Any

import numpy as np
import pytest
import torch
from funlib.geometry import Coordinate
from funlib.persistence import open_ds, prepare_ds

pytest.importorskip("volara")
pytest.importorskip("volara_ml")

from dacapo_toolbox.postprocessing import blockwise_predict_mutex  # noqa: E402

NEIGHBORHOOD = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
SHAPE = (32, 32, 32)
BLOCKSIZE = (16, 16, 16)
# Two bright cubes on a dark background, each straddling block boundaries and
# separated by a four-voxel dark gap along z.
CUBE_A = (slice(2, 14), slice(2, 30), slice(2, 30))
CUBE_B = (slice(18, 30), slice(2, 30), slice(2, 30))


def shrink(cube, by=1):
    return tuple(slice(s.start + by, s.stop - by) for s in cube)


def write_raw(store):
    raw = np.zeros(SHAPE, dtype=np.uint8)
    raw[CUBE_A] = 255
    raw[CUBE_B] = 255
    array = prepare_ds(
        store,
        shape=SHAPE,
        offset=Coordinate(0, 0, 0),
        voxel_size=Coordinate(1, 1, 1),
        axis_names=["z", "y", "x"],
        units=["nm", "nm", "nm"],
        dtype=np.uint8,
        mode="w",
    )
    array[:] = raw
    return raw


def write_affinity_model(path, gain=20.0):
    """
    A 3x3x3 convolution whose channel i reads the voxel and its neighbour at
    NEIGHBORHOOD[i]: both bright gives sigmoid(gain / 4), anything else at most
    sigmoid(-gain / 4). The input is raw / 255, so bright is 1.0.
    """
    conv = torch.nn.Conv3d(1, len(NEIGHBORHOOD), kernel_size=3)
    assert conv.bias is not None
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.fill_(-0.75 * gain)
        for channel, offset in enumerate(NEIGHBORHOOD):
            conv.weight[channel, 0, 1, 1, 1] = gain / 2
            conv.weight[channel, 0, 1 + offset[0], 1 + offset[1], 1 + offset[2]] = (
                gain / 2
            )
    torch.save(torch.nn.Sequential(conv, torch.nn.Sigmoid()), path)


@pytest.mark.parametrize(
    "edge_scores, extract_frag_bias",
    [
        (None, None),
        (
            [
                ("z", [Coordinate(1, 0, 0)], -0.5),
                ("xy", [Coordinate(0, 1, 0), Coordinate(0, 0, 1)], -0.5),
            ],
            [-0.5, -0.5, -0.5],
        ),
    ],
    ids=["default-edges", "named-edges"],
)
def test_blockwise_predict_mutex_segments_two_cubes(
    tmp_path, monkeypatch, edge_scores, extract_frag_bias
):
    # The prediction workers inherit the environment; keep them on the CPU so
    # the result does not depend on which GPU, if any, the host has.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    raw = write_raw(tmp_path / "data.zarr/raw")
    write_affinity_model(tmp_path / "model.pt")

    blockwise_predict_mutex(
        raw_store=tmp_path / "data.zarr/raw",
        labels_store=tmp_path / "data.zarr/labels",
        neighborhood=NEIGHBORHOOD,
        blocksize=BLOCKSIZE,
        model_path=tmp_path / "model.pt",
        in_channels=1,
        model_context=(1, 1, 1),
        edge_scores=edge_scores,
        extract_frag_bias=extract_frag_bias,
    )

    labels = open_ds(tmp_path / "data.zarr/labels")[:]
    assert labels.shape == raw.shape

    # Each cube interior is one segment, and the two segments differ.
    inner_a = np.unique(labels[shrink(CUBE_A)])
    inner_b = np.unique(labels[shrink(CUBE_B)])
    assert inner_a.size == 1 and inner_a[0] != 0
    assert inner_b.size == 1 and inner_b[0] != 0
    assert inner_a[0] != inner_b[0]

    # Background two or more voxels from any cube belongs to no segment.
    assert not labels[15:17].any()  # the gap between the cubes
    assert not labels[0].any() and not labels[-1].any()
    assert not labels[:, 0].any() and not labels[:, -1].any()
    assert not labels[:, :, 0].any() and not labels[:, :, -1].any()


def test_blockwise_predict_mutex_rejects_malformed_edge_scores(tmp_path):
    malformed: Any = [("too", "many", "elements", -0.5)]
    with pytest.raises(ValueError, match="Invalid edge definition"):
        blockwise_predict_mutex(
            raw_store=tmp_path / "raw",
            labels_store=tmp_path / "labels",
            neighborhood=NEIGHBORHOOD,
            blocksize=BLOCKSIZE,
            model_path=tmp_path / "model.pt",
            in_channels=1,
            model_context=(1, 1, 1),
            edge_scores=malformed,
        )
