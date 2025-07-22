from dacapo_toolbox.dataset import iterable_dataset
from funlib.persistence import Array
import numpy as np


def test_iterable_dataset():
    """
    test 3 arrays with different shapes, offsets, and voxel sizes.
    raw: simply an array of coordinates for each pixel
    gt: sum over channels of raw
    mask: binary mask of gt % 2

    We want to have good test coverage for all possible inputs of the iterable dataset.
    """
    raw = Array(np.random.rand(10, 1, 12, 8, 16), offset=(0, 0, 0), voxel_size=(1, 1, 1))
    gt = Array(np.random.rand(10, 1, 12, 13, 13), offset=(0, 0, 0), voxel_size=(1, 1, 1))
    mask = Array(np.random.rand(10, 1, 9, 13, 15), offset=(0, 0, 0), voxel_size=(1, 1, 1))

    iter_ds = iterable_dataset(
        {"raw": raw, "gt": gt, "mask": mask},
        shapes={"raw": (2, 2, 2), "gt": (2, 2, 2), "mask": (2, 2, 2)},
    )

    print(next(iter(iter_ds)))