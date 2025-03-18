from .dataset_config import DatasetConfig
from .arrays import ArrayConfig

from funlib.geometry import Coordinate
from funlib.persistence import Array

import attr


@attr.s
class RawGTDatasetConfig(DatasetConfig):
    raw_config: ArrayConfig = attr.ib(
        metadata={"help_text": "The raw dataset. This is the input data for training."}
    )
    gt_config: ArrayConfig | None = attr.ib(
        metadata={
            "help_text": "The ground truth data. This is the ground truth data for training."
        },
        default=None,
    )
    mask_config: ArrayConfig | None = attr.ib(
        metadata={
            "help_text": "The mask data. This controls what data is ignored during training."
        },
        default=None,
    )
    sample_points: list[Coordinate] | None = attr.ib(
        metadata={"help_text": "The list of sample points in the dataset."},
        default=None,
    )
    weight: int = attr.ib(
        metadata={"help_text": "The weight of the dataset."},
        default=1,
    )


    @property
    def raw(self) -> Array:
        return self.raw_config.array("r")
    
    @property
    def gt(self) -> Array | None:
        return self.gt_config.array("r") if self.gt_config is not None else None
    
    @property
    def mask(self) -> Array | None:
        return self.mask_config.array("r") if self.mask_config is not None else None