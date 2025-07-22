import attr

from .task_config import TaskConfig
from .predictors import DistancePredictor
from .losses import MSELoss
from .post_processors import ThresholdPostProcessor
from .evaluators import BinarySegmentationEvaluator

from typing import List


@attr.s
class DistanceTaskConfig(TaskConfig):
    """This is a Distance task config used for generating and
    evaluating signed distance transforms as a way of generating
    segmentations.

    The advantage of generating distance transforms over regular
    affinities is you can get a denser signal, i.e. 1 misclassified
    pixel in an affinity prediction could merge 2 otherwise very
    distinct objects, this cannot happen with distances.

    Attributes:
        channels: A list of channel names.
        clip_distance: Maximum distance to consider for false positive/negatives.
        tol_distance: Tolerance distance for counting false positives/negatives
        scale_factor: The amount by which to scale distances before applying a tanh normalization.
        mask_distances: Whether or not to mask out regions where the true distance to
                       object boundary cannot be known. This is anywhere that the distance to crop boundary
                       is less than the distance to object boundary.
        clipmin: The minimum value for distance weights.
        clipmax: The maximum value for distance weights.
    """

    channels: List[str] = attr.ib(metadata={"help_text": "A list of channel names."})
    clip_distance: float = attr.ib(
        metadata={
            "help_text": "Maximum distance to consider for false positive/negatives."
        },
    )
    tol_distance: float = attr.ib(
        metadata={
            "help_text": "Tolerance distance for counting false positives/negatives"
        },
    )
    scale_factor: float = attr.ib(
        default=1,
        metadata={
            "help_text": "The amount by which to scale distances before applying "
            "a tanh normalization."
        },
    )
    mask_distances: bool = attr.ib(
        default=True,
        metadata={
            "help_text": "Whether or not to mask out regions where the true distance to "
            "object boundary cannot be known. This is anywhere that the distance to crop boundary "
            "is less than the distance to object boundary."
        },
    )
    clipmin: float = attr.ib(
        default=0.05,
        metadata={"help_text": "The minimum value for distance weights."},
    )
    clipmax: float = attr.ib(
        default=0.95,
        metadata={"help_text": "The maximum value for distance weights."},
    )

    @property
    def predictor(self):
        return DistancePredictor(
            channels=self.channels,
            scale_factor=self.scale_factor,
            mask_distances=self.mask_distances,
            clipmin=self.clipmin,
            clipmax=self.clipmax,
        )

    @property
    def loss(self):
        return MSELoss()

    @property
    def post_processor(self):
        return ThresholdPostProcessor()

    @property
    def evaluator(self):
        return BinarySegmentationEvaluator(
            clip_distance=self.clip_distance,
            tol_distance=self.tol_distance,
            channels=self.channels,
        )
