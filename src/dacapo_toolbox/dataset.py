import random
import logging
from collections.abc import Sequence

import gunpowder as gp
import networkx as nx
import dask.array as da
import numpy as np

from funlib.geometry import Coordinate, Roi
from funlib.persistence import Array

import torch
import time
from typing import Any
import functools
from dataclasses import dataclass
from .tmp import gcd

logger = logging.getLogger(__name__)


class PipelineDataset(torch.utils.data.IterableDataset):
    """
    A torch dataset that wraps a gunpowder pipeline and provides batches of data.
    It has support for applying torchvision style transforms to the resulting batches.
    """

    def __init__(self, pipeline, request, keys, transforms=None):
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
            torch_batch = {
                str(key): torch.from_numpy(batch[key].data.copy()) for key in self.keys
            }
            torch_batch["metadata"] = {
                str(key): (batch[key].spec.roi.offset, batch[key].spec.voxel_size)
                for key in self.keys
            }

            for transform_signature, transform_func in self.transforms.items():
                if isinstance(transform_signature, tuple):
                    in_key, out_key = transform_signature
                else:
                    in_key, out_key = transform_signature, transform_signature

                assert in_key in torch_batch, (
                    f"Can only process keys that are in the batch. Please ensure that {in_key} "
                    f"is either provided as a dataset or created as the result of a transform "
                    f"of the form ({{in_key}}, {in_key})) *before* the transform ({in_key})."
                )

                in_tensor = torch_batch[in_key]
                out_tensor = transform_func(in_tensor)
                assert tuple(in_tensor.shape) == tuple(
                    out_tensor.shape[-len(in_tensor.shape) :]
                ), (
                    f"Transform {transform_signature} changed the shape of the "
                    f"tensor: {in_tensor.shape} -> {out_tensor.shape}"
                )
                torch_batch[out_key] = transform_func(torch_batch[in_key])

            t2 = time.time()
            logger.warn(f"Batch generated in {t2 - t1:.4f} seconds, ")
            yield torch_batch


@dataclass
class SimpleAugmentConfig:
    """
    The simple augment handles non-interpolating geometric transformations.
    This includes mirroring and transposing in n-dimensional space.

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
    p: float = 0.0
    control_point_spacing: Sequence[int] | None = None
    jitter_sigma: Sequence[float] | None = None
    scale_interval: tuple[float, float] | None = None
    rotate: bool = False
    subsample: int = 4
    spatial_dims: int = 3
    rotation_axes: Sequence[int] | None = None


# # predictor acting on the batch
# # or predictor in the dataset already
# dataloader = torch.utils.data.DataLoader(dataset, num_workers=10)
# for batch in dataloader:
#     gt, raw = batch["gt"], batch["raw"]
#     gt_metadata = batch["metadata"]["gt"]
#     target = predictor(gt, gt_metadata)

# # Most transform the gt into the target
# # affs_transform(gt) -> affs
# # weights_transform(labels) -> weights
# # lsd_transform(gt) -> lsds
# # distance_transform(gt) -> distances

# # Transforms the target into the input
# # n2v_transform(raw) -> masked_raw

# "Array | nplike"
# arrays = {
#     "raw": [np.ones(10, 10, 10), np.random(10, 10, 10)],
#     "gt": [np.zeros(10, 10, 10), np.ones(10, 10, 10)],
# }
# shapes = {"raw": (12, 12, 12), "gt": (10, 10, 10)}
# transforms = {"raw": Normalize()}

# input_arrays ["raw", "gold_standard", "mask", "em", "liconn", "weights", "distances"]
# computed_arrays ["affs", "weights", "lsds", "distances"]
# "affs" = AffsTransform("gt")

# transforms = {
#     "raw": Compose([Normalize(), GaussianNoise()]),  # 1
#     ("gt","affs"): AffsTransform(neighborhood=[(1, 0, 0), (0, 1, 0), (0, 0, 1)]),  # 2
#     ("gt", "lsd"): LsdTransform(),  # 2
#     ("gt", "weights"): WeightsTransform(),  # 2
#     "gt": BinaryErosion(),  # 1
#     "weights": GaussianNoise(),  # 3
#     ("weights", "weights2"): GaussianNoise(),  # 4
# }  # errors

# transforms = {
#     "raw": Compose([Normalize(), GaussianNoise()]),  # 1
#     "gt": BinaryErosion(),  # 1
#     ("gt","affs"): AffsTransform(neighborhood=[(1, 0, 0), (0, 1, 0), (0, 0, 1)]),  # 2
#     ("gt", "lsd"): LsdTransform(),  # 2
#     ("gt", "weights"): WeightsTransform(),  # 2
#     "weights": GaussianNoise(),  # 3
#     ("weights", "weights2"): GaussianNoise(),  # 4
# }  # fine

# transforms = {
#     "raw_s0": Normalize(),
#     "raw_s1": Normalize(),
#     ("gt_s0", "affs_s0"): AffsTransform(neighborhood=[(1, 0, 0), (0, 1, 0), (0, 0, 1)])
# }

# # assert count(key) <= 1, "Cannot do multiple transforms on the same key"


def iterable_dataset(
    datasets: dict[str, Array | Sequence[Array]],
    shapes: dict[str, Sequence[int]],
    sample_points: np.ndarray | Sequence[np.ndarray | None] | None = None,
    weights: Sequence[float] | None = None,
    transforms: dict[str, Any | Sequence[Any]] | None = None,
    sampling_strategy: str | Sequence[str] = "random",
    trim: Sequence[int] | None = None,
    simple_augment_config: SimpleAugmentConfig | None = None,
    deform_augment_config: DeformAugmentConfig | None = None,
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

    # convert single arrays to lists
    datasets: dict[str, list[Array]] = {
        name: [arrays] if isinstance(arrays, Array) else list(arrays)
        for name, arrays in datasets.items()
    }

    # define keys:
    names = list(datasets.keys())
    keys = [gp.ArrayKey(name) for name in names]

    sample_points_key = gp.GraphKey("SAMPLE_POINTS")
    if sample_points is None:
        # no sample points for any crop
        sample_points = [None] * len(datasets[names[0]])
    elif isinstance(sample_points, nx.Graph):
        # sample points for a single crop
        sample_points = [sample_points]
    else:
        # sample points should be provided for each crop
        assert len(sample_points) == len(datasets[names[0]]), (
            "Sample points must be provided for each crop (group of datasets). "
            f"Got {len(sample_points)} sample points for {len(datasets[names[0]])} crops."
        )

    if isinstance(sampling_strategy, str):
        # single sampling strategy for all crops
        sampling_strategy = [sampling_strategy] * len(datasets[names[0]])
    else:
        # sampling strategy should be provided for each crop
        assert len(sampling_strategy) == len(datasets[names[0]]), (
            "Sampling strategy must be provided for each crop (group of datasets). "
            f"Got {len(sampling_strategy)} strategies for {len(datasets[names[0]])} crops."
        )

    roi_mask_key = gp.ArrayKey("ROI_MASK")

    # reorganize from raw: [a,b,c], gt: [a,b,c] to (raw,gt): [(a,a), (b,b), (c,c)]
    crop_datasets: list[tuple[Array, ...]] = list(zip(*datasets.values()))
    crop_scales = [
        functools.reduce(
            lambda x, y: gcd(x, y), [array.voxel_size for array in crop_arrays]
        )
        for crop_arrays in crop_datasets
    ]

    # check that for each key, all arrays have the same voxel size when scaled
    # by the crop scale
    assert all(
        functools.reduce(
            lambda x, y: x * 0 if x != y else x,
            [
                array.voxel_size / scale
                for array, scale in zip(datasets[key], crop_scales)
            ],
        )
        == datasets[key][0].voxel_size / crop_scales[0]
        for key in datasets
    )

    # Get source nodes
    dataset_sources = []
    for crop_arrays, crop_scale, points, sampling_strat in zip(
        crop_datasets, crop_scales, sample_points, sampling_strategy
    ):
        # type hints since zip seems to get rid of the type info
        crop_arrays: list[Array]
        crop_scale: Coordinate
        points: np.ndarray | None
        sampling_strat: str

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
            crop_roi = crop_roi.grow(-trim * crop_voxel_size, -trim * crop_voxel_size)

        array_sources = tuple(
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
            )
            + gp.Pad(key, None)
            for key, array in zip(keys, crop_arrays)
        )

        if points is not None:
            graph = gp.Graph(
                [gp.Node(i, loc) for i, loc in enumerate(points)],
                [],
                gp.GraphSpec(
                    Roi((None,) * len(crop_voxel_size), (None,) * len(crop_voxel_size))
                ),
            )
            crop_sources = (
                *array_sources,
                gp.gp_graph_source.GraphSource(sample_points_key, graph),
                gp.ArraySource(  # a dummy array for consisntency
                    roi_mask_key,
                    Array(
                        da.ones(crop_roi.shape + (crop_voxel_size * 100)),
                        offset=crop_roi.offset - (crop_voxel_size * 50),
                        voxel_size=crop_voxel_size,
                    ),
                ),
            )
        else:
            crop_sources = (
                *array_sources,
                gp.gp_graph_source.GraphSource(
                    sample_points_key,
                    gp.Graph(  # A dummy graph for consistency
                        [],
                        [],
                        gp.GraphSpec(
                            Roi(
                                (None,) * len(crop_voxel_size),
                                (None,) * len(crop_voxel_size),
                            )
                        ),
                    ),
                ),
                gp.ArraySource(
                    roi_mask_key,
                    Array(
                        da.ones(crop_roi.shape),
                        offset=crop_roi.offset,
                        voxel_size=crop_voxel_size,
                    ),
                ),
            )

        dataset_source = crop_sources + gp.MergeProvider()

        if points is not None:
            dataset_source += gp.RandomLocation(
                ensure_nonempty=sample_points_key,
                ensure_centered=sample_points_key,
            )
        elif sampling_strat == "integral_mask":
            dataset_source += gp.RandomLocation(
                min_masked=1,
                mask=roi_mask_key,
            )
        elif sampling_strat == "reject":
            dataset_source += gp.RandomLocation()
            dataset_source += gp.Reject(roi_mask_key, 1.0)
        elif sampling_strat is None or sampling_strat == "random":
            dataset_source += gp.RandomLocation()

        dataset_sources.append(dataset_source)

    pipeline = tuple(dataset_sources) + gp.RandomProvider(weights)

    if deform_augment_config is not None:
        pipeline += gp.DeformAugment(
            control_point_spacing=Coordinate(
                deform_augment_config.control_point_spacing
            ),
            jitter_sigma=deform_augment_config.jitter_sigma,
            scale_interval=deform_augment_config.scale_interval,
            rotate=deform_augment_config.rotate,
            subsample=deform_augment_config.subsample,
            spatial_dims=deform_augment_config.spatial_dims,
            rotation_axes=deform_augment_config.rotation_axes,
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
    for key, array_shape in zip(keys, shapes.values()):
        crop_scale = crop_scales[0]
        request.add(
            key, Coordinate(array_shape) * datasets[str(key)][0].voxel_size / crop_scale
        )

    # Add mask placeholder to guarantee center voxel is contained in
    # the mask, and to be used for some sampling strategies.
    request.add(
        roi_mask_key,
        crop_voxel_size,
    )

    # Build the pipeline
    gp.build(pipeline).__enter__()
    return PipelineDataset(pipeline, request=request, keys=keys, transforms=transforms)
