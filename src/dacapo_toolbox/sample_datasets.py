from pathlib import Path
from itertools import product

import wget
import zarr
import h5py

from funlib.persistence import open_ds, prepare_ds, Array
from pathlib import Path
from funlib.geometry import Coordinate


def cremi(zarr_path: Path) -> tuple[Array, Array, Array, Array]:
    # Download some cremi data
    # immediately convert it to zarr for convenience
    if not Path("cremi.zarr").exists():
        wget.download(
            "https://cremi.org/static/data/sample_C_20160501.hdf",
            "sample_C_20160501.hdf",
        )
        wget.download(
            "https://cremi.org/static/data/sample_A_20160501.hdf",
            "sample_A_20160501.hdf",
        )
        hdf_datasets = {
            "train": "sample_C_20160501.hdf",
            "test": "sample_A_20160501.hdf",
        }
        hdf_arrays = {
            "raw": "volumes/raw",
            "labels": "volumes/labels/neuron_ids",
        }
        for mode, dataset in product(["train", "test"], ["raw", "labels"]):
            data = h5py.File(hdf_datasets[mode], "r")[hdf_arrays[dataset]][:]
            arr = prepare_ds(
                f"cremi.zarr/{mode}/{dataset}",
                data.shape,
                voxel_size=(40, 4, 4),
                units=["nm", "nm", "nm"],
                axis_names=["z", "y", "x"],
                dtype=data.dtype,
            )
            arr[:] = data

        Path("sample_A_20160501.hdf").unlink()
        Path("sample_C_20160501.hdf").unlink()

    raw_train = open_ds("cremi.zarr/train/raw")
    labels_train = open_ds("cremi.zarr/train/labels")
    raw_test = open_ds("cremi.zarr/test/raw")
    labels_test = open_ds("cremi.zarr/test/labels")
    return raw_train, labels_train, raw_test, labels_test
