import attr

from .task_config import TaskConfig
from .predictors import DummyPredictor
from .losses import DummyLoss
from .post_processors import DummyPostProcessor
from .evaluators import DummyEvaluator

from typing import Tuple


@attr.s
class DummyTaskConfig(TaskConfig):
    """A class for creating a dummy task configuration object.

    This class extends the TaskConfig class and initializes dummy task configuration
    with default attributes. It is mainly used for testing aspects
    of the application without the need of creating real task configurations.

    Attributes:
        task_type (cls): The type of task. Here, set to DummyTask.
        embedding_dims (int): A dummy attribute represented as an integer.
        detection_threshold (float): Another dummy attribute represented as a float.
    """

    embedding_dims: int = attr.ib(metadata={"help_text": "Dummy attribute."})

    detection_threshold: float = attr.ib(metadata={"help_text": "Dummy attribute."})

    @property
    def predictor(self):
        return DummyPredictor(self.embedding_dims)

    @property
    def loss(self):
        return DummyLoss()

    @property
    def post_processor(self):
        return DummyPostProcessor(self.detection_threshold)

    @property
    def evaluator(self):
        return DummyEvaluator()

    @property
    def channels(self) -> list[str]:
        return [f"e{x}" for x in range(self.embedding_dims)]
