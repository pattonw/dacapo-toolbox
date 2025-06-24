from dacapo_toolbox.dataset import (
    iterable_dataset,
    SimpleAugmentConfig,
    DeformAugmentConfig,
    MaskedSampling,
    PointSampling,
)
from dacapo_toolbox.utils import points_to_graph
from dacapo_toolbox.transforms.affs import Affs, AffsMask
from funlib.persistence import Array
from skimage import data
from torchvision.transforms import v2 as transforms
import logging
import numpy as np
from skimage.measure import label
import networkx as nx
from itertools import product

logging.basicConfig(level=logging.DEBUG)
# logging.getLogger("gunpowder.nodes.random_location").setLevel(logging.DEBUG)

side_length = 2048

# two different datasets with vastly different blob sizes
blobs_a = data.binary_blobs(length=side_length, blob_size_fraction=20/side_length, n_dim=2)
blobs_a_gt = label(blobs_a, connectivity=2)
blobs_b = data.binary_blobs(length=side_length, blob_size_fraction=100/side_length, n_dim=2)
blobs_b_gt = label(blobs_b, connectivity=2)
mask = np.ones((side_length, side_length), dtype=bool)
mask[side_length // 2 : side_length] = 0

# raw and gt arrays at various voxel sizes
raw_a_s0 = Array(blobs_a[::1, ::1], offset=(0, 0), voxel_size=(1, 1))
raw_a_s1 = Array(blobs_a[::2, ::2], offset=(0, 0), voxel_size=(2, 2))
raw_b_s0 = Array(blobs_b[::1, ::1], offset=(0, 0), voxel_size=(2, 2))
raw_b_s1 = Array(blobs_b[::2, ::2], offset=(0, 0), voxel_size=(4, 4))
gt_a_s0 = Array(blobs_a_gt[::2, ::2], offset=(0, 0), voxel_size=(2, 2))
gt_a_s1 = Array(blobs_a_gt[::4, ::4], offset=(0, 0), voxel_size=(4, 4))
gt_b_s0 = Array(blobs_b_gt[::2, ::2], offset=(0, 0), voxel_size=(4, 4))
gt_b_s1 = Array(blobs_b_gt[::4, ::4], offset=(0, 0), voxel_size=(8, 8))
mask_a = Array(mask, offset=(0, 0), voxel_size=(1, 1))
mask_b = Array(mask, offset=(0, 0), voxel_size=(2, 2))

sample_points = []

# defining the datasets
iter_ds = iterable_dataset(
    {
        "raw_s0": [raw_a_s0, raw_b_s0],
        "gt_s0": [gt_a_s0, gt_b_s0],
        "raw_s1": [raw_a_s1, raw_b_s1],
        "gt_s1": [gt_a_s1, gt_b_s1],
        "mask": [mask_a, mask_b],
        "mask_dummy": [mask_a, mask_b],
        "sample_points": [
            None,
            points_to_graph(
                np.array(
                    [
                        (side_length * 2, side_length * 2),
                        (0, side_length * 2),
                        (side_length * 2, 0),
                        (0, 0),
                    ]
                )
            ),
        ],
    },
    shapes={
        "raw_s0": (128 * 5, 128 * 5),
        "gt_s0": (64 * 5, 64 * 5),
        "raw_s1": (64 * 5, 64 * 5),
        "gt_s1": (32 * 5, 32 * 5),
        "mask": (128 * 5, 128 * 5),
        "mask_dummy": (64 * 5, 64 * 5),
        "sample_points": (128 * 5, 128 * 5),
    },
    sampling_strategies=[
        MaskedSampling("mask_dummy", 0.8),
        PointSampling("sample_points"),
    ],
    transforms={
        ("raw_s0", "noisy_s0"): transforms.Compose(
            [transforms.ConvertImageDtype(), transforms.GaussianNoise(sigma=1.0)]
        ),
        ("raw_s1", "noisy_s1"): transforms.Compose(
            [transforms.ConvertImageDtype(), transforms.GaussianNoise(sigma=0.3)]
        ),
        ("gt_s0", "affs_s0"): Affs([[4, 0], [0, 4], [4, 4]]),
        ("gt_s0", "affs_mask_s0"): AffsMask([[4, 0], [0, 4], [4, 4]]),
        ("gt_s1", "affs_s1"): Affs([[4, 0], [0, 4], [4, 4]]),
        ("gt_s1", "affs_mask_s1"): AffsMask([[4, 0], [0, 4], [4, 4]]),
    },
    simple_augment_config=SimpleAugmentConfig(
        p=1.0, mirror_probs=[1.0, 0.0], transpose_only=[]
    ),
    deform_augment_config=DeformAugmentConfig(
        p=1.0,
        control_point_spacing=(10, 10),
        jitter_sigma=(5.0, 5.0),
        scale_interval=(0.5, 2.0),
        rotate=True,
    ),
)

import matplotlib.pyplot as plt

for i, batch in enumerate(iter_ds):
    print(f"Batch {i}")
    if i >= 8:  # Limit to 4 batches for demonstration
        break
    points = batch["sample_points"]
    xs = np.array([attrs["position"][0] for attrs in points.nodes.values()])
    ys = np.array([attrs["position"][1] for attrs in points.nodes.values()])
    plt.scatter(xs, ys, c="red", s=10)

    fig, axs = plt.subplots(2, 5, figsize=(18, 8))
    axs[0, 0].imshow(batch["noisy_s0"], cmap="gray")
    axs[0, 1].imshow(batch["gt_s0"], cmap="magma")
    axs[0, 2].imshow(batch["affs_s0"].permute(1, 2, 0).float())
    axs[0, 3].imshow(batch["affs_mask_s0"].permute(1, 2, 0).float())
    axs[0, 4].imshow(batch["mask"].float(), vmin=0, vmax=1, cmap="gray")
    axs[1, 0].imshow(batch["noisy_s1"], cmap="gray")
    axs[1, 1].imshow(batch["gt_s1"], cmap="magma")
    axs[1, 2].imshow(batch["affs_s1"].permute(1, 2, 0).float())
    axs[1, 3].imshow(batch["affs_mask_s1"].permute(1, 2, 0).float())
    axs[1, 4].imshow(batch["mask"][::2, ::2].float(), vmin=0, vmax=1, cmap="gray")
    for a, b in product(range(2), range(5)):
        s = 2 ** (a + (b % 4 != 0))
        axs[a, b].scatter(ys / s, xs / s, c="red", s=10)

    axs[0, 0].set_title("Raw")
    axs[0, 1].set_title("GT")
    axs[0, 2].set_title("Affs")
    axs[0, 3].set_title("Affs Mask")
    axs[0, 4].set_title("Mask")

    plt.savefig(f"affs_batch_{i}.png")
