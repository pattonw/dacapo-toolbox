import attr
from collections.abc import Sequence

from .task_config import TaskConfig

from .predictors import AffinitiesPredictor
from .losses import AffinitiesLoss
from .post_processors import WatershedPostProcessor
from .evaluators import InstanceEvaluator

from funlib.geometry import Coordinate

from typing import List


@attr.s
class AffinitiesTaskConfig(TaskConfig):
    """
    This is a Affinities task config used for generating and
    evaluating voxel affinities for instance segmentations.

    Attributes:
        neighborhood: A list of Coordinate objects.
        lsds: Whether or not to train lsds along with your affinities.
        num_lsd_voxels: The number of voxels to use for the lsd center of mass calculation.
        downsample_lsds: The factor by which to downsample the lsds.
        lsds_to_affs_weight_ratio: If training with lsds, set how much they should be weighted compared to affs.
        affs_weight_clipmin: The minimum value for affinities weights.
        affs_weight_clipmax: The maximum value for affinities weights.
        lsd_weight_clipmin: The minimum value for lsds weights.
        lsd_weight_clipmax: The maximum value for lsds weights.
        background_as_object: Whether to treat the background as a separate object.
    """

    neighborhood: List[Sequence[int]] = attr.ib(
        metadata={
            "help_text": "The neighborhood upon which to calculate affinities. "
            "This is provided as a list of offsets, where each offset is a list of "
            "ints defining the offset in each axis in voxels."
        }
    )
    lsds: bool = attr.ib(
        default=False,
        metadata={
            "help_text": "Whether or not to train lsds along with your affinities. "
            "It has been shown that lsds as an auxiliary task can help affinity predictions."
        },
    )
    num_lsd_voxels: int = attr.ib(
        default=10,
        metadata={
            "help_text": "The number of voxels to use for the lsd center of mass calculation."
        },
    )
    downsample_lsds: int = attr.ib(
        default=1,
        metadata={
            "help_text": "The factor by which to downsample the lsds. "
            "This can be useful to reduce the computational cost of training."
        },
    )
    lsds_to_affs_weight_ratio: float = attr.ib(
        default=1,
        metadata={
            "help_text": "If training with lsds, set how much they should be weighted compared to affs."
        },
    )
    affs_weight_clipmin: float = attr.ib(
        default=0.05,
        metadata={"help_text": "The minimum value for affinities weights."},
    )
    affs_weight_clipmax: float = attr.ib(
        default=0.95,
        metadata={"help_text": "The maximum value for affinities weights."},
    )
    lsd_weight_clipmin: float = attr.ib(
        default=0.05,
        metadata={"help_text": "The minimum value for lsds weights."},
    )
    lsd_weight_clipmax: float = attr.ib(
        default=0.95,
        metadata={"help_text": "The maximum value for lsds weights."},
    )
    background_as_object: bool = attr.ib(
        default=False,
        metadata={
            "help_text": (
                "Whether to treat the background as a separate object. "
                "If set to false background should get an affinity near 0. If "
                "set to true, the background should also have high affinity with other background."
            )
        },
    )

    @property
    def predictor(self):
        neighborhood = [Coordinate(offset) for offset in self.neighborhood]

        return AffinitiesPredictor(
            neighborhood=neighborhood,
            lsds=self.lsds,
            num_voxels=self.num_lsd_voxels,
            downsample_lsds=self.downsample_lsds,
            affs_weight_clipmin=self.affs_weight_clipmin,
            affs_weight_clipmax=self.affs_weight_clipmax,
            lsd_weight_clipmin=self.lsd_weight_clipmin,
            lsd_weight_clipmax=self.lsd_weight_clipmax,
            background_as_object=self.background_as_object,
        )

    @property
    def loss(self):
        return AffinitiesLoss(len(self.neighborhood), self.lsds_to_affs_weight_ratio)

    @property
    def post_processor(self):
        neighborhood = [Coordinate(offset) for offset in self.neighborhood]
        return WatershedPostProcessor(offsets=neighborhood)

    @property
    def evaluator(self):
        return InstanceEvaluator()

    @property
    def channels(self) -> list[str]:
        return [f"aff_{'.'.join(map(str, n))}" for n in self.neighborhood]
