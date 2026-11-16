import random
from itertools import islice

import networkx as nx
import numpy as np
import pytest
import torch
from funlib.persistence import Array

from dacapo_toolbox.dataset import (
    DefectAugmentConfig,
    DeformAugmentConfig,
    MaskedSampling,
    PointSampling,
    SimpleAugmentConfig,
    iterable_dataset,
)


@pytest.fixture(autouse=True)
def seeded():
    """The dataset seeds every batch request from `random`."""
    random.seed(0)


@pytest.fixture
def raw():
    """A 32x32 array at voxel size 1 whose value at (y, x) is 32 * y + x, so
    a crop shows where it was read from and in which orientation."""
    return Array(
        np.arange(32 * 32, dtype=np.float32).reshape(32, 32), voxel_size=(1, 1)
    )


def constant(value, shape=(16, 16), voxel_size=(1, 1)):
    return Array(np.full(shape, value, dtype=np.uint8), voxel_size=voxel_size)


@pytest.mark.parametrize("dims, size, crop", [(2, 640, 320), (3, 64, 16)])
def test_batches_have_the_requested_shapes(dims, size, crop):
    # labels at twice the voxel size of the raw, mirrored and deformed: every
    # batch has the requested shape, in voxels of its own array
    dataset = iterable_dataset(
        {
            "raw": Array(np.zeros((size,) * dims, np.float32), voxel_size=(1,) * dims),
            "labels": Array(
                np.zeros((size // 2,) * dims, np.uint32), voxel_size=(2,) * dims
            ),
        },
        shapes={"raw": (crop,) * dims, "labels": (crop // 2,) * dims},
        simple_augment_config=SimpleAugmentConfig(p=1.0, transpose_only=[]),
        deform_augment_config=DeformAugmentConfig(
            p=1.0,
            control_point_spacing=(10,) * dims,
            jitter_sigma=(5.0,) * dims,
            scale_interval=(0.5, 2.0),
            rotate=True,
            spatial_dims=dims,
        ),
    )
    for batch in islice(dataset, 8):
        assert batch["raw"].shape == (crop,) * dims
        assert batch["labels"].shape == (crop // 2,) * dims


@pytest.mark.parametrize(
    "augments, step_along_rows, step_along_columns",
    [
        ({}, 32, 1),
        (
            {
                "simple_augment_config": SimpleAugmentConfig(
                    p=1.0,
                    mirror_only=[],
                    transpose_only=[0, 1],
                    transpose_probs={(1, 0): 1.0},
                )
            },
            1,
            32,
        ),
        (
            {
                "simple_augment_config": SimpleAugmentConfig(
                    p=1.0, mirror_only=[1], mirror_probs=[1.0], transpose_only=[]
                )
            },
            32,
            -1,
        ),
        # a deformation with neither jitter, scaling nor rotation is the identity
        ({"deform_augment_config": DeformAugmentConfig(p=1.0)}, 32, 1),
    ],
    ids=["unaugmented", "transposed", "mirrored", "default-deform"],
)
def test_batches_are_crops_of_the_array(
    raw, augments, step_along_rows, step_along_columns
):
    # trimmed by half the crop, so no crop reaches into the padding
    dataset = iterable_dataset({"raw": raw}, shapes={"raw": (8, 8)}, trim=4, **augments)
    for batch in islice(dataset, 4):
        assert (batch["raw"].diff(dim=0) == step_along_rows).all()
        assert (batch["raw"].diff(dim=1) == step_along_columns).all()


@pytest.mark.parametrize("strategy", ["integral_mask", "reject"])
def test_masked_sampling_draws_crops_inside_the_mask(raw, strategy):
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    dataset = iterable_dataset(
        {"raw": raw, "mask": Array(mask, voxel_size=(1, 1))},
        shapes={"raw": (8, 8), "mask": (8, 8)},
        sampling_strategies=MaskedSampling("mask", min_masked=1.0, strategy=strategy),
    )
    for batch in islice(dataset, 8):
        assert batch["mask"].sum() == 64


def test_each_crop_is_sampled_with_its_own_strategy():
    # the first crop is drawn inside its mask; the second, whose mask is
    # empty, asks for no masked voxels and is drawn anywhere
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    dataset = iterable_dataset(
        {
            "raw": [constant(1, (32, 32)), constant(2, (32, 32))],
            "mask": [Array(mask, voxel_size=(1, 1)), constant(0, (32, 32))],
        },
        shapes={"raw": (8, 8), "mask": (8, 8)},
        sampling_strategies=[
            MaskedSampling("mask", min_masked=1.0),
            MaskedSampling("mask", min_masked=0.0),
        ],
        trim=4,
    )
    masked_voxels = {1: 64, 2: 0}
    crops_drawn = set()
    for batch in islice(dataset, 16):
        crop = int(batch["raw"][0, 0])
        assert batch["mask"].sum() == masked_voxels[crop]
        crops_drawn.add(crop)
    assert crops_drawn == {1, 2}


def test_defect_augment_blanks_sections_of_its_array():
    # 8 batches of 32 sections of ones: sections along the first axis go
    # missing (all zero) in the raw, and the labels stay as they are
    dataset = iterable_dataset(
        {
            "raw": Array(np.ones((64, 16, 16), np.float32), voxel_size=(1, 1, 1)),
            "labels": Array(np.full((64, 16, 16), 2, np.uint32), voxel_size=(1, 1, 1)),
        },
        shapes={"raw": (32, 8, 8), "labels": (32, 8, 8)},
        trim=(16, 4, 4),
        defect_augment_config=DefectAugmentConfig("raw", p=1.0),
    )
    sections = set()
    for batch in islice(dataset, 8):
        sections.update(batch["raw"].flatten(1).unique().tolist())
        assert (batch["labels"] == 2).all()
    assert sections == {0.0, 1.0}


def test_point_sampling_centres_the_crop_on_a_point(raw):
    points = nx.Graph()
    points.add_node(1, position=(20.0, 12.0))
    dataset = iterable_dataset(
        {"raw": raw, "points": points},
        shapes={"raw": (8, 8), "points": (8, 8)},
        sampling_strategies=PointSampling("points"),
    )
    batch = next(iter(dataset))
    # the point comes back at the centre of the crop, in the crop's own
    # coordinates, and the raw voxel there is the point's: 32 * 20 + 12
    assert tuple(batch["points"].nodes[1]["position"]) == (3, 3)
    assert batch["raw"][3, 3] == 652


def test_weights_set_how_often_each_crop_is_drawn():
    dataset = iterable_dataset(
        {"raw": [constant(1), constant(2)]},
        shapes={"raw": (4, 4)},
        weights=[0.0, 1.0],
        trim=(2, 2),
    )
    for batch in islice(dataset, 8):
        assert (batch["raw"] == 2).all()


@pytest.mark.parametrize(
    "signature, transform, expected",
    [
        ("raw", lambda raw: raw * 3, {"raw": 3, "labels": 2}),
        (("raw", "tripled"), lambda raw: raw * 3, {"raw": 1, "tripled": 3}),
        ((("raw", "labels"), "sum"), torch.add, {"raw": 1, "labels": 2, "sum": 3}),
        (
            ("labels", ("plus_one", "plus_two")),
            lambda labels: (labels + 1, labels + 2),
            {"labels": 2, "plus_one": 3, "plus_two": 4},
        ),
    ],
    ids=["in-place", "one-to-one", "many-to-one", "one-to-many"],
)
def test_transforms_write_their_outputs_into_the_batch(signature, transform, expected):
    dataset = iterable_dataset(
        {"raw": constant(1), "labels": constant(2)},
        shapes={"raw": (4, 4), "labels": (4, 4)},
        trim=(2, 2),
        transforms={signature: transform},
    )
    batch = next(iter(dataset))
    for key, value in expected.items():
        assert (batch[key] == value).all()
