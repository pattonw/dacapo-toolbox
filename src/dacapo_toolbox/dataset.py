import random
import logging
from collections.abc import Sequence

import gunpowder as gp
from gunpowder.nodes.gp_graph_source import GraphSource as GPGraphSource
import networkx as nx
import dask.array as da
import numpy as np

from funlib.geometry import Coordinate, Roi
from funlib.persistence import Array

import torch

import time
import functools
from dataclasses import dataclass
from .tmp import gcd
from typing import Callable

logger = logging.getLogger(__name__)


def interpolatable_dtypes(dtype) -> bool:
    return dtype in [np.float32, np.float64, np.uint8, np.uint16]


def nx_to_gp_graph(
    graph: nx.Graph,
    scale: Sequence[float],
) -> gp.Graph:
    """
    Convert a NetworkX graph to a gunpowder Graph.
    """
    graph: gp.Graph = gp.Graph(
        [
            gp.Node(
                node,
                np.array(attrs.pop("position")) / np.array(scale),
                attrs=attrs,
            )
            for node, attrs in graph.nodes(data=True)
        ],  # type: ignore[arg-type]
        [gp.Edge(u, v, attrs) for u, v, attrs in graph.edges(data=True)],  # type: ignore[arg-type]
        gp.GraphSpec(Roi((None,) * len(scale), (None,) * len(scale))),
    )
    return graph


def gp_to_nx_graph(
    graph: gp.Graph,
) -> nx.Graph:
    """
    Convert a gunpowder Graph to a NetworkX graph.
    """
    g = nx.Graph()
    for node in graph.nodes:
        g.add_node(node.id, position=node.location, **node.attrs)
    for edge in graph.edges:
        g.add_edge(edge.u, edge.v, **edge.attrs)
    return g


class PipelineDataset(torch.utils.data.IterableDataset):
    """
    A torch dataset that wraps a gunpowder pipeline and provides batches of data.
    It has support for applying torchvision style transforms to the resulting batches.
    """

    def __init__(
        self,
        pipeline: gp.Pipeline,
        request: gp.BatchRequest,
        keys: list[gp.ArrayKey],
        transforms: dict[
            str | tuple[str | tuple[str, ...], str | tuple[str, ...]], Callable
        ]
        | None = None,
    ):
        self.pipeline = pipeline
        self.request = request
        self.keys = keys
        self.transforms = transforms

    def __iter__(self):
        while True:
            t1 = time.time()
            batch_request = self.request.copy()
            batch_request._random_seed = random.randint(0, 2**32 - 1)
            batch = self.pipeline.request_batch(batch_request)
            # TODO: Throw warning on incorrect byteorder!
            torch_batch = {
                str(key): torch.from_numpy(
                    batch[key]
                    .data.astype(batch[key].data.dtype.newbyteorder("="))
                    .copy()
                )
                if isinstance(key, gp.ArrayKey)
                else gp_to_nx_graph(batch[key])
                for key in self.keys
            }
            torch_batch["metadata"] = {
                str(key): (batch[key].spec.roi.offset, batch[key].spec.voxel_size)
                for key in self.keys
                if isinstance(key, gp.ArrayKey)
            }

            if self.transforms is not None:
                for transform_signature, transform_func in self.transforms.items():
                    if isinstance(transform_signature, tuple):
                        in_key, out_key = transform_signature
                    else:
                        in_key, out_key = transform_signature, transform_signature

                    if isinstance(in_key, str):
                        in_keys = [in_key]
                    elif isinstance(in_key, tuple):
                        in_keys = list(in_key)

                    for in_key in in_keys:
                        assert in_key in torch_batch, (
                            f"Can only process keys that are in the batch. Please ensure that {in_key} "
                            f"is either provided as a dataset or created as the result of a transform "
                            f"of the form ({{in_key}}, {in_key})) *before* the transform ({in_key})."
                        )
                    in_tensors = [torch_batch[in_key] for in_key in in_keys]
                    out_tensor = transform_func(*in_tensors)
                    if isinstance(out_key, str):
                        torch_batch[out_key] = out_tensor
                    else:
                        out_keys = out_key
                        out_tensors = out_tensor
                        for out_key, out_tensor in zip(out_keys, out_tensors):
                            torch_batch[out_key] = out_tensor

            t2 = time.time()
            logger.debug(f"Batch generated in {t2 - t1:.4f} seconds")
            yield torch_batch


@dataclass
class SimpleAugmentConfig:
    """
    The simple augment handles non-interpolating geometric transformations.
    This includes mirroring and transposing in n-dimensional space.
    See https://github.com/funkelab/gunpowder/blob/main/gunpowder/nodes/simple_augment.py
    for more details.

    Parameters:
        :param p: Probability of applying the augmentations.
        :param mirror_only: List of axes to mirror. If None, all axes may be mirrored.
        :param transpose_only: List of axes to transpose. If None, all axes may be transposed.
        :param mirror_probs: List of probabilities for each axis in `mirror_only`.
            If None, uses equal probability for all axes.
        :param transpose_probs: Dictionary mapping tuples of axes to probabilities for transposing.
            If None, uses equal probability for all axes.
    """

    p: float = 0.0
    mirror_only: Sequence[int] | None = None
    transpose_only: Sequence[int] | None = None
    mirror_probs: Sequence[float] | None = None
    transpose_probs: dict[tuple[int, ...], float] | Sequence[float] | None = None


@dataclass
class DeformAugmentConfig:
    """
    The deform augment handles interpolating geometric transformations.
    This includes scaling, rotation, and elastic deformations.
    See https://github.com/funkelab/gunpowder/blob/main/gunpowder/nodes/deform_augment.py
    for more details.

    Parameters:
        :param p: Probability of applying the augmentations.
        :param control_point_spacing: Spacing of the control points for the elastic deformation.
        :param jitter_sigma: Standard deviation of the Gaussian noise added to the control points.
        :param scale_interval: Interval for scaling the input data.
        :param rotate: Whether to apply random rotations.
        :param subsample: Subsampling factor for the control points.
        :param spatial_dims: Number of spatial dimensions.
        :param rotation_axes: Axes around which to rotate. If None, rotates around all axes.
    """

    p: float = 0.0
    control_point_spacing: Sequence[int] | None = None
    jitter_sigma: Sequence[float] | None = None
    scale_interval: tuple[float, float] | None = None
    rotate: bool = False
    subsample: int = 4
    spatial_dims: int = 3
    rotation_axes: Sequence[int] | None = None


@dataclass
class MaskedSampling:
    """
    Sampling strategy that uses a mask to determine which samples to include.

    :param mask_key: The key of the mask array in the dataset.
    :param min_masked: Minimum fraction of samples that must be masked in to include the
        sample. If less than this fraction is masked in, the sample is skipped.
    :param strategy: Optional strategy to apply to the mask. by default generates an integral
        mask for quick sampling at the cost of extra memory usage. If your dataset is large, you
        may want to use "reject".
    """

    mask_key: str
    min_masked: float = 1.0
    strategy: str = "integral_mask"


@dataclass
class PointSampling:
    """
    Sampling strategy that uses a set of points to determine which samples to include.

    :param sample_points_key: The key of the sample points array in the dataset.
    """

    sample_points_key: str


def iterable_dataset(
    datasets: dict[str, Array | nx.Graph | Sequence[Array] | Sequence[nx.Graph]],
    shapes: dict[str, Sequence[int]],
    weights: Sequence[float] | None = None,
    transforms: dict[
        str | tuple[str | tuple[str, ...], str | tuple[str, ...]], Callable
    ]
    | None = None,
    sampling_strategies: MaskedSampling
    | PointSampling
    | Sequence[MaskedSampling | PointSampling]
    | None = None,
    trim: int | Sequence[int] | None = None,
    simple_augment_config: SimpleAugmentConfig | None = None,
    deform_augment_config: DeformAugmentConfig | None = None,
    interpolatable: dict[str, bool] | None = None,
) -> torch.utils.data.IterableDataset:
    """
    Builds a gunpowder pipeline and wraps it in a torch IterableDataset.
    See https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for more info
    """

    # Check the validity of the inputs
    assert len(datasets) >= 1, "Expected at least one dataset, got an empty dictionary"
    assert "ROI_MASK" not in datasets, (
        "The key 'ROI_MASK' is reserved for internal use. "
        "Please use a different key for your dataset."
    )

    if interpolatable is None:
        interpolatable = {}

    # convert single arrays to lists
    datasets: dict[str, list[Array] | list[nx.Graph]] = {
        name: [ds] if isinstance(ds, (Array, nx.Graph)) else list(ds)
        for name, ds in datasets.items()
    }

    # define keys:
    keys = [
        gp.ArrayKey(name) if isinstance(dataset[0], Array) else gp.GraphKey(name)
        for name, dataset in datasets.items()
    ]
    array_keys = [key for key in keys if isinstance(key, gp.ArrayKey)]
    graph_keys = [key for key in keys if isinstance(key, gp.GraphKey)]

    roi_mask_key = gp.ArrayKey("ROI_MASK")

    # reorganize from raw: [a,b,c], gt: [a,b,c] to (raw,gt): [(a,a), (b,b), (c,c)]
    crops_datasets: list[tuple[Array | nx.Graph, ...]] = list(zip(*datasets.values()))
    crops_scale = [
        functools.reduce(
            lambda x, y: gcd(x, y),
            [array.voxel_size for array in crop_datasets if isinstance(array, Array)],
        )
        for crop_datasets in crops_datasets
    ]

    # check that for each key, all arrays have the same voxel size when scaled
    # by the crop scale
    assert all(
        functools.reduce(
            lambda x, y: x * 0 if x != y else x,
            [
                array.voxel_size / scale
                for array, scale in zip(datasets[key], crops_scale)
                if isinstance(array, Array)
            ],
        )
        == datasets[key][0].voxel_size / crops_scale[0]
        for key in datasets
        if isinstance(datasets[key][0], Array)
    )

    if isinstance(sampling_strategies, (MaskedSampling, PointSampling)):
        sampling_strategies = [sampling_strategies] * len(crops_datasets)
    elif sampling_strategies is None:
        sampling_strategies = [None] * len(crops_datasets)

    # Get source nodes
    dataset_sources = []

    # type hints since zip seems to get rid of the type info
    # crop_datasets: list[Array | nx.Graph]
    # crop_scale: Coordinate
    # sampling_strategy: MaskedSampling | PointSampling | None

    for crop_datasets, crop_scale, sampling_strategy in zip(
        crops_datasets, crops_scale, sampling_strategies
    ):
        crop_arrays = [array for array in crop_datasets if isinstance(array, Array)]

        for dataset in crop_datasets:
            assert dataset is None or isinstance(dataset, (Array, nx.Graph)), (
                f"Dataset {dataset} is not an Array or a NetworkX graph. "
                f"{type(dataset)} is not supported. Please provide a valid dataset."
            )

        # smallest roi
        crop_roi = (
            functools.reduce(
                lambda x, y: x.intersect(y), [array.roi for array in crop_arrays]
            )
            / crop_scale
        )
        crop_voxel_size = (
            functools.reduce(
                lambda x, y: Coordinate(*map(min, x, y)),
                [array.voxel_size for array in crop_arrays],
            )
            / crop_scale
        )
        crop_roi.snap_to_grid(crop_voxel_size, mode="grow")
        if trim is not None:
            if isinstance(trim, int):
                trim = [trim] * crop_roi.dims()
            trim = Coordinate(trim)
            crop_roi = crop_roi.grow(-trim * crop_voxel_size, -trim * crop_voxel_size)

        crop_graphs = [
            nx_to_gp_graph(graph, crop_scale)
            if graph is not None
            else nx_to_gp_graph(nx.Graph(), crop_scale)
            for graph in crop_datasets
            if isinstance(graph, nx.Graph) or graph is None
        ]

        crop_sources = tuple(
            gp.ArraySource(
                key,
                Array(
                    array.data,
                    offset=array.roi.offset / crop_scale,
                    voxel_size=array.voxel_size / crop_scale,
                    units=array.units,
                    axis_names=array.axis_names,
                    types=array.types,
                ),
                interpolatable=interpolatable.get(
                    str(key), interpolatable_dtypes(array.dtype)
                ),
            )
            + gp.Pad(
                key,
                None
                if not (
                    isinstance(sampling_strategy, MaskedSampling)
                    and sampling_strategy.mask_key == str(key)
                )
                else Coordinate((0,) * len(crop_scale)),
            )
            for key, array in zip(array_keys, crop_arrays)
        ) + tuple(
            GPGraphSource(key, graph) for key, graph in zip(graph_keys, crop_graphs)
        )

        crop_sources += (
            gp.ArraySource(  # a dummy array for consisntency
                roi_mask_key,
                Array(
                    da.ones(crop_roi.shape),
                    offset=crop_roi.offset,
                    voxel_size=crop_voxel_size,
                ),
            ),
        )

        dataset_source = crop_sources + gp.MergeProvider()

        if sampling_strategy is None:
            # If no sampling strategy is provided, use random sampling
            dataset_source += gp.RandomLocation()
        elif isinstance(sampling_strategy, PointSampling):
            assert gp.GraphKey(sampling_strategy.sample_points_key) in keys, (
                f"Sample points key {sampling_strategy.sample_points_key} must be one of the dataset keys: {keys}. "
                "Please ensure that the sample points are provided as part of the dataset."
            )
            dataset_source += gp.RandomLocation(
                ensure_nonempty=gp.GraphKey(sampling_strategy.sample_points_key),
                ensure_centered=True,
            )
        elif (
            isinstance(sampling_strategy, MaskedSampling)
            and sampling_strategy.strategy == "integral_mask"
        ):
            assert gp.ArrayKey(sampling_strategy.mask_key) in keys, (
                f"Mask key {sampling_strategy.mask_key} must be one of the dataset keys: {keys}. "
                "Please ensure that the mask is provided as part of the dataset."
            )
            dataset_source += gp.RandomLocation(
                min_masked=sampling_strategy.min_masked,
                mask=gp.ArrayKey(sampling_strategy.mask_key),
            )
        elif sampling_strategy == "reject":
            dataset_source += gp.RandomLocation()
            dataset_source += gp.Reject(roi_mask_key, 1.0)
        else:
            raise ValueError(
                f"Unsupported sampling strategy: {sampling_strategy}. "
                "Please use either `None`, or provide a PointSampling/MaskSampling instance."
            )

        dataset_sources.append(dataset_source)

    pipeline = tuple(dataset_sources) + gp.RandomProvider(weights)

    if deform_augment_config is not None:
        pipeline += gp.DeformAugment(
            control_point_spacing=Coordinate(
                deform_augment_config.control_point_spacing
                or (1,) * crop_scale[0].dims()
            ),
            jitter_sigma=deform_augment_config.jitter_sigma
            or (0,) * crop_scale[0].dims(),
            scale_interval=deform_augment_config.scale_interval,
            rotate=deform_augment_config.rotate,
            subsample=deform_augment_config.subsample,
            spatial_dims=deform_augment_config.spatial_dims,
            rotation_axes=deform_augment_config.rotation_axes,
            use_fast_points_transform=True,
            p=deform_augment_config.p,
        )
    if simple_augment_config is not None:
        pipeline += gp.SimpleAugment(
            mirror_only=simple_augment_config.mirror_only,
            transpose_only=simple_augment_config.transpose_only,
            mirror_probs=simple_augment_config.mirror_probs,
            transpose_probs=simple_augment_config.transpose_probs,
            p=simple_augment_config.p,
        )

    # generate request for all necessary inputs to training
    request = gp.BatchRequest()
    for key in array_keys:
        crop_scale = crops_scale[0]
        data_shape = shapes.get(str(key), None)
        assert data_shape is not None, (
            f"Shape for key {key} not provided. Please provide a shape for all keys."
        )
        request.add(
            key, Coordinate(data_shape) * datasets[str(key)][0].voxel_size / crop_scale
        )
    for key in graph_keys:
        data_shape = shapes.get(str(key), None)
        assert data_shape is not None, (
            f"Shape for key {key} not provided. Please provide a shape for all keys."
        )
        request.add(key, Coordinate(data_shape))

    # Add mask placeholder to guarantee center voxel is contained in
    # the mask, and to be used for some sampling strategies.
    request.add(
        roi_mask_key,
        crop_voxel_size,
    )

    # Build the pipeline
    gp.build(pipeline).__enter__()
    return PipelineDataset(pipeline, request=request, keys=keys, transforms=transforms)
