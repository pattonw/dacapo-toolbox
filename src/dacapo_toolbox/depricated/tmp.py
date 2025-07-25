from funlib.persistence import open_ds, prepare_ds, Array
from funlib.geometry import Roi, Coordinate

import attr

from upath import UPath as Path

import numpy as np
import numpy_indexed as npi


@attr.s
class LocalArrayIdentifier:
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


@attr.s
class LocalContainerIdentifier:
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


def relabel(array, return_backwards_map=False, inplace=False):
    """
    Relabel array, such that IDs are consecutive. Excludes 0.

    Args:
        array (ndarray):
                The array to relabel.
        return_backwards_map (``bool``, optional):
                If ``True``, return an ndarray that maps new labels (indices in
                the array) to old labels.
        inplace (``bool``, optional):
                Perform the replacement in-place on ``array``.
    Returns:
        A tuple ``(relabelled, n)``, where ``relabelled`` is the relabelled
        array and ``n`` the number of unique labels found.
        If ``return_backwards_map`` is ``True``, returns ``(relabelled, n,
        backwards_map)``.
    Raises:
        ValueError:
                If ``array`` is not of type ``np.ndarray``.
    Examples:
        >>> array = np.array([[1, 2, 0], [0, 2, 1]])
        >>> relabel(array)
        (array([[1, 2, 0], [0, 2, 1]]), 2)
        >>> relabel(array, return_backwards_map=True)
        (array([[1, 2, 0], [0, 2, 1]]), 2, [0, 1, 2])
    Note:
        This function is used to relabel an array, such that IDs are consecutive. Excludes 0.

    """

    if array.size == 0:
        if return_backwards_map:
            return array, 0, []
        else:
            return array, 0

    # get all labels except 0
    old_labels = np.unique(array)
    old_labels = old_labels[old_labels != 0]

    if old_labels.size == 0:
        if return_backwards_map:
            return array, 0, [0]
        else:
            return array, 0

    n = len(old_labels)
    new_labels = np.arange(1, n + 1, dtype=array.dtype)

    replaced = npi.remap(
        array.flatten(), old_labels, new_labels, inplace=inplace
    ).reshape(array.shape)

    if return_backwards_map:
        backwards_map = np.insert(old_labels, 0, 0)
        return replaced, n, backwards_map

    return replaced, n


def int_gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def gcd(a: Coordinate[int], b: Coordinate[int]) -> Coordinate[int]:
    return Coordinate(int_gcd(x, y) for x, y in zip(a, b))
