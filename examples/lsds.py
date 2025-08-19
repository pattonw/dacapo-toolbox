from dacapo_toolbox.dataset import (
    iterable_dataset,
    SimpleAugmentConfig,
    DeformAugmentConfig,
)
from dacapo_toolbox.transforms.affs import Affs, AffsMask
from dacapo_toolbox.transforms.distances import SignedDistanceTransform, SDTBoundaryMask
from dacapo_toolbox.transforms.lsds import LSD
from funlib.persistence import Array
from skimage import data
from torchvision.transforms import v2 as transforms
import logging
import numpy as np
from skimage.measure import label

logging.basicConfig(level=logging.DEBUG)
# logging.getLogger("gunpowder.nodes.random_location").setLevel(logging.DEBUG)

# two different datasets with vastly different blob sizes
blobs_a = data.binary_blobs(length=512, blob_size_fraction=0.05, n_dim=2)
blobs_a_gt = label(blobs_a, connectivity=2)
blobs_b = data.binary_blobs(length=512, blob_size_fraction=0.50, n_dim=2)
blobs_b_gt = label(blobs_b, connectivity=2)

# raw and gt arrays at various voxel sizes
raw_a_s0 = Array(blobs_a[::1, ::1], offset=(0, 0), voxel_size=(1, 1))
raw_a_s1 = Array(blobs_a[::2, ::2], offset=(0, 0), voxel_size=(2, 2))
raw_b_s0 = Array(blobs_b[::1, ::1], offset=(0, 0), voxel_size=(2, 2))
raw_b_s1 = Array(blobs_b[::2, ::2], offset=(0, 0), voxel_size=(4, 4))
gt_a_s0 = Array(blobs_a_gt[::2, ::2], offset=(0, 0), voxel_size=(2, 2))
gt_a_s1 = Array(blobs_a_gt[::4, ::4], offset=(0, 0), voxel_size=(4, 4))
gt_b_s0 = Array(blobs_b_gt[::2, ::2], offset=(0, 0), voxel_size=(4, 4))
gt_b_s1 = Array(blobs_b_gt[::4, ::4], offset=(0, 0), voxel_size=(8, 8))

# defining the datasets
iter_ds = iterable_dataset(
    {
        "raw_s0": [raw_a_s0, raw_b_s0],
        "gt_s0": [gt_a_s0, gt_b_s0],
        "raw_s1": [raw_a_s1, raw_b_s1],
        "gt_s1": [gt_a_s1, gt_b_s1],
    },
    shapes={
        "raw_s0": (128 * 5, 128 * 5),
        "gt_s0": (64 * 5, 64 * 5),
        "raw_s1": (64 * 5, 64 * 5),
        "gt_s1": (32 * 5, 32 * 5),
    },
    transforms={
        ("raw_s0", "noisy_s0"): transforms.Compose(
            [transforms.ConvertImageDtype(), transforms.GaussianNoise(sigma=1.0)]
        ),
        ("raw_s1", "noisy_s1"): transforms.Compose(
            [transforms.ConvertImageDtype(), transforms.GaussianNoise(sigma=0.3)]
        ),
        ("gt_s0", "lsd_s0"): LSD(sigma=10.0),
        ("gt_s1", "lsd_s1"): LSD(sigma=5.0),
    },
    sample_points=[np.array([(0, 0)]), np.array([(512, 512), (0, 512), (512, 0)])],
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

fig, axs = plt.subplots(2, 5, figsize=(15, 8))
for i, batch in enumerate(iter_ds):
    if i >= 8:  # Limit to 4 batches for demonstration
        break
    axs[0, 0].imshow(batch["noisy_s0"], cmap="gray")
    axs[0, 1].imshow(batch["lsd_s0"][[0, 1, 5]].permute(1, 2, 0).float())
    axs[0, 2].imshow(batch["lsd_s0"][[2, 3, 5]].permute(1, 2, 0).float())
    axs[0, 3].imshow(batch["lsd_s0"][4], cmap="gray")
    axs[0, 4].imshow(batch["lsd_s0"][5], cmap="gray")
    axs[1, 0].imshow(batch["noisy_s1"], cmap="gray")
    axs[1, 1].imshow(batch["lsd_s1"][[0, 1, 5]].permute(1, 2, 0).float())
    axs[1, 2].imshow(batch["lsd_s1"][[2, 3, 5]].permute(1, 2, 0).float())
    axs[1, 3].imshow(batch["lsd_s1"][4], cmap="gray")
    axs[1, 4].imshow(batch["lsd_s1"][5], cmap="gray")

    axs[0, 0].set_title("Raw")
    axs[0, 1].set_title("LSD (Offsets)")
    axs[0, 2].set_title("LSD (Variance)")
    axs[0, 3].set_title("LSD (Pearson)")
    axs[0, 4].set_title("LSD (Mass)")

    plt.pause(0.1)
    input("Press Enter to continue...")  # Pause after each batch
