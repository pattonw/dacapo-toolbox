from funlib.persistence import open_ds, prepare_ds, Array
from funlib.geometry import Roi, Coordinate

from pathlib import Path

import zarr
import neuroglancer
import attr

from abc import ABC, abstractmethod
import itertools
import json
from upath import UPath as Path
from typing import Optional, Tuple

from pydantic import BaseModel


class LocalArrayIdentifier(BaseModel):
    """
    Represents a local array identifier.

    Attributes:
        container (Path): The path to the container.
        dataset (str): The dataset name.
    Method:
        __str__ : Returns the string representation of the identifier.
    """

    container: Path = attr.ib()
    dataset: str = attr.ib()


class LocalContainerIdentifier(BaseModel):
    """
    Represents a local container identifier.

    Attributes:
        container (Path): The path to the container.
    Method:
        array_identifier : Creates a local array identifier for the given dataset.

    """

    container: Path = attr.ib()

    def array_identifier(self, dataset) -> LocalArrayIdentifier:
        """
        Creates a local array identifier for the given dataset.

        Args:
            dataset: The dataset for which to create the array identifier.
        Returns:
            LocalArrayIdentifier: The local array identifier.
        Raises:
            TypeError: If the dataset is not a string.
        Examples:
            >>> container = Path('path/to/container')
            >>> container.array_identifier('dataset')
            LocalArrayIdentifier(container=Path('path/to/container'), dataset='dataset')
        """
        return LocalArrayIdentifier(self.container, dataset)


def num_channels_from_array(array: Array) -> int | None:
    if array.channel_dims == 1:
        assert array.axis_names[0] == "c^", array.axis_names
        return array.shape[0]
    elif array.channel_dims == 0:
        return None
    else:
        raise ValueError(
            "Trying to get number of channels from an array with multiple channel dimensions:",
            array.axis_names,
        )


def gp_to_funlib_array(gp_array) -> Array:
    n_dims = len(gp_array.data.shape)
    physical_dims = gp_array.spec.roi.dims
    channel_dims = n_dims - physical_dims
    axis_names = (["b^", "c^"][-channel_dims:] if channel_dims > 0 else []) + [
        "z",
        "y",
        "x",
    ][-physical_dims:]
    return Array(
        gp_array.data,
        offset=gp_array.spec.roi.offset,
        voxel_size=gp_array.spec.voxel_size,
        axis_names=axis_names,
    )


def np_to_funlib_array(np_array, offset: Coordinate, voxel_size: Coordinate) -> Array:
    n_dims = len(np_array.shape)
    physical_dims = offset.dims
    channel_dims = n_dims - physical_dims
    axis_names = (["b^", "c^"][-channel_dims:] if channel_dims > 0 else []) + [
        "z",
        "y",
        "x",
    ][-physical_dims:]
    return Array(
        np_array,
        offset=offset,
        voxel_size=voxel_size,
        axis_names=axis_names,
    )


def create_from_identifier(
    array_identifier,
    axis_names,
    roi: Roi,
    num_channels: int | None,
    voxel_size: Coordinate,
    dtype,
    mode="a+",
    write_size=None,
    name=None,
    overwrite=False,
) -> Array:
    out_path = Path(f"{array_identifier.container}/{array_identifier.dataset}")
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True)

    list_num_channels = [num_channels] if num_channels is not None else []
    return prepare_ds(
        out_path,
        shape=(*list_num_channels, *roi.shape / voxel_size),
        offset=roi.offset,
        voxel_size=voxel_size,
        axis_names=axis_names,
        dtype=dtype,
        chunk_shape=(
            (*list_num_channels, *write_size / voxel_size)
            if write_size is not None
            else None
        ),
        mode=mode if overwrite is False else "w",
    )


def open_from_identifier(array_identifier, name: str = "", mode: str = "r") -> Array:
    return open_ds(
        f"{array_identifier.container}/{array_identifier.dataset}", mode=mode
    )
