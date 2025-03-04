from .dataset_config import DatasetConfig
from .arrays import ArrayConfig

from funlib.geometry import Coordinate

import attr


@attr.s
class RawGTDatasetConfig(DatasetConfig):
    raw: ArrayConfig = attr.ib(
        metadata={"help_text": "The raw dataset. This is the input data for training."}
    )
    gt: ArrayConfig | None = attr.ib(
        metadata={
            "help_text": "The ground truth data. This is the ground truth data for training."
        },
        default=None,
    )
    mask: ArrayConfig | None = attr.ib(
        metadata={
            "help_text": "The mask data. This controls what data is ignored during training."
        },
        default=None,
    )
    sample_points: list[Coordinate] | None = attr.ib(
        metadata={"help_text": "The list of sample points in the dataset."},
        default=None,
    )
