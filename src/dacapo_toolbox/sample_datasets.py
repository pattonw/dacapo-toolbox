from pathlib import Path

import wget
import zarr
import h5py

from funlib.persistence import open_ds, Array
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
        zarr.save_array(
            "cremi.zarr/train/raw",
            h5py.File("sample_C_20160501.hdf", "r")["volumes/raw"][:],
        )
        zarr.save_array(
            "cremi.zarr/train/labels",
            h5py.File("sample_C_20160501.hdf", "r")["volumes/labels/neuron_ids"][:],
        )
        zarr.save_array(
            "cremi.zarr/test/raw",
            h5py.File("sample_A_20160501.hdf", "r")["volumes/raw"][:],
        )
        zarr.save_array(
            "cremi.zarr/test/labels",
            h5py.File("sample_A_20160501.hdf", "r")["volumes/labels/neuron_ids"][:],
        )

        Path("sample_A_20160501.hdf").unlink()
        Path("sample_C_20160501.hdf").unlink()

        raw_train_arr = zarr.open("cremi.zarr/train/raw", mode="a")
        labels_train_arr = zarr.open("cremi.zarr/train/labels", mode="a")
        raw_test_arr = zarr.open("cremi.zarr/test/raw", mode="a")
        labels_test_arr = zarr.open("cremi.zarr/test/labels", mode="a")
        for arr in [
            raw_train_arr,
            labels_train_arr,
            raw_test_arr,
            labels_test_arr,
        ]:
            arr.attrs["resolution"] = Coordinate((40, 4, 4))
            arr.attrs["axis_names"] = ["z", "y", "x"]
            arr.attrs["units"] = ["nm", "nm", "nm"]

    raw_train = open_ds("cremi.zarr/train/raw")
    labels_train = open_ds("cremi.zarr/train/labels")
    raw_test = open_ds("cremi.zarr/test/raw")
    labels_test = open_ds("cremi.zarr/test/labels")
    return raw_train, labels_train, raw_test, labels_test
