import attr

from .task_config import TaskConfig

from .predictors import OneHotPredictor
from .losses import DummyLoss
from .post_processors import ArgmaxPostProcessor
from .evaluators import DummyEvaluator

from typing import List


@attr.s
class OneHotTaskConfig(TaskConfig):
    """
    Class that derives from the TaskConfig to perform one hot prediction tasks.

    Attributes:
        task_type: the type of task, in this case, OneHotTask.
        classes: a List of classes which starts from id 0.

    """

    classes: List[str] = attr.ib(
        metadata={"help_text": "The classes corresponding with each id starting from 0"}
    )
    kernel_size: int | None = attr.ib(
        default=None,
    )

    @property
    def predictor(self):
        return OneHotPredictor(classes=self.classes, kernel_size=self.kernel_size)

    @property
    def loss(self):
        return DummyLoss()

    @property
    def post_processor(self):
        return ArgmaxPostProcessor()

    @property
    def evaluator(self):
        return DummyEvaluator()
