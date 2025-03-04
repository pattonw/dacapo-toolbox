import gunpowder as gp

import pytest
from pytest_lazy_fixtures import lf
import numpy as np

from dacapo_toolbox.datasplits.datasets.arrays import (
    ZarrArrayConfig,
    BinarizeArrayConfig,
    DummyArrayConfig,
)

import zarr
import numpy as np
from numcodecs import Zstd
import pytest
from .serde import serde_test

from dacapo_toolbox.tmp import num_channels_from_array

import pytest
from pytest_lazy_fixtures import lf
from funlib.persistence import Array


def dummy_array(_tmp_path):
    return DummyArrayConfig(name="dummy_array")


def zarr_array(tmp_path):
    zarr_array_config = ZarrArrayConfig(
        name="zarr_array",
        file_name=tmp_path / "zarr_array.zarr",
        dataset="volumes/test",
    )
    zarr_container = zarr.open(str(zarr_array_config.file_name))
    dataset = zarr_container.create_dataset(
        zarr_array_config.dataset, data=np.zeros((100, 50, 25), dtype=np.float32)
    )
    dataset.attrs["offset"] = (12, 12, 12)
    dataset.attrs["voxel_size"] = (1, 2, 4)
    dataset.attrs["axis_names"] = ["z", "y", "x"]
    return zarr_array_config


def cellmap_array(tmp_path):
    zarr_array_config = ZarrArrayConfig(
        name="zarr_array",
        file_name=tmp_path / "zarr_array.zarr",
        dataset="volumes/test",
    )
    zarr_container = zarr.open(str(zarr_array_config.file_name))
    dataset = zarr_container.create_dataset(
        zarr_array_config.dataset,
        data=np.arange(0, 100, dtype=np.uint8).reshape(10, 5, 2),
    )
    dataset.attrs["offset"] = (12, 12, 12)
    dataset.attrs["voxel_size"] = (1, 2, 4)
    dataset.attrs["axis_names"] = ["z", "y", "x"]

    cellmap_array_config = BinarizeArrayConfig(
        name="cellmap_zarr_array",
        source_array_config=zarr_array_config,
        groupings=[
            ("a", list(range(0, 10))),
            ("b", list(range(10, 70))),
            ("c", list(range(70, 90))),
        ],
    )

    return cellmap_array_config


def multiscale_zarr(tmp_path):
    zarr_metadata = {
        "multiscales": [
            {
                "axes": [
                    {"name": "z", "type": "space", "unit": "nanometer"},
                    {"name": "y", "type": "space", "unit": "nanometer"},
                    {"name": "x", "type": "space", "unit": "nanometer"},
                ],
                "coordinateTransformations": [],
                "datasets": [
                    {
                        "coordinateTransformations": [
                            {"scale": [4.2, 7.4, 5.6], "type": "scale"},
                            {"translation": [6.0, 10.0, 2.0], "type": "translation"},
                        ],
                        "path": "s0",
                    },
                    {
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 2.0, 4.0]},
                            {"type": "translation", "translation": [12.0, 12.0, 12.0]},
                        ],
                        "path": "s1",
                    },
                ],
                "name": "multiscale_dataset",
                "version": "0.4",
            }
        ],
        "omero": {
            "id": 1,
            "name": "test_image",
            "channels": [],
        },
    }
    ome_zarr_array_config = ZarrArrayConfig(
        name="ome_zarr_array",
        file_name=tmp_path / "ome_zarr_array.zarr",
        dataset="multiscale_dataset/s1",
        ome_metadata=True,
    )

    store = zarr.DirectoryStore(ome_zarr_array_config.file_name)
    multiscale_group = zarr.group(
        store=store, path="multiscale_dataset", overwrite=True
    )

    for level in [0, 1]:
        scaling = pow(2, level)
        multiscale_group.require_dataset(
            name=f"s{level}",
            shape=(100 / scaling, 80 / scaling, 60 / scaling),
            chunks=10,
            dtype=np.float32,
            compressor=Zstd(level=6),
        )

    multiscale_group.attrs.update(zarr_metadata)

    return ome_zarr_array_config


@pytest.mark.parametrize(
    "array_func",
    [
        cellmap_array,
        zarr_array,
        dummy_array,
        multiscale_zarr,
    ],
)
def test_arrays(tmp_path, array_func):
    # Create Array from config
    array_config = array_func(tmp_path)
    serde_test(array_config)

    # Create Array from config
    array: Array = array_config.array("r+")

    # Test API
    # channels/axis_names
    if "c^" in array.axis_names:
        assert num_channels_from_array(array) is not None
    else:
        assert num_channels_from_array(array) is None
    # dims/voxel_size/roi
    assert array.spatial_dims == array.voxel_size.dims
    assert array.spatial_dims == array.roi.dims
    # fetching data:
    expected_data_shape = array.roi.shape / array.voxel_size
    assert array[array.roi].shape[-array.spatial_dims :] == expected_data_shape
    # setting data:
    if array.is_writeable:
        data_slice = array[0]
        array[0] = data_slice + 1
        assert data_slice.sum() == 0
        assert (array[0] - data_slice).sum() == data_slice.size

    # Make sure the DaCapoArraySource can properly read
    # the data in `array`
    key = gp.ArrayKey("TEST")
    source_node = gp.ArraySource(key, array)

    with gp.build(source_node):
        request = gp.BatchRequest()
        request[key] = gp.ArraySpec(roi=array.roi)
        batch = source_node.request_batch(request)
        data = batch[key].data
        if data.dtype == bool:
            data = data.astype(np.uint8)
        assert (data - array[array.roi]).sum() == 0
