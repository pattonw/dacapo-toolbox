import attr

from .task_config import TaskConfig

from .predictors import InnerDistancePredictor
from .losses import MSELoss
from .post_processors import ThresholdPostProcessor
from .evaluators import BinarySegmentationEvaluator

from typing import List


@attr.s
class InnerDistanceTaskConfig(TaskConfig):
    """
    This is a Distance task config used for generating and
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
    Notes:
        This is a subclass of TaskConfig.

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

    @property
    def predictor(self):
        return InnerDistancePredictor(
            channels=self.channels,
            scale_factor=self.scale_factor,
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
