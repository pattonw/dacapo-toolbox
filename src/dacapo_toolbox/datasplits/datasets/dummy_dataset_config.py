from .dataset_config import DatasetConfig
from .arrays import ArrayConfig, DummyArrayConfig

import attr


@attr.s
class DummyDatasetConfig(DatasetConfig):
    """
    A dummy configuration class for test datasets.

    Attributes:
        dataset_type : Clearly mentions the type of dataset
        raw_config : This attribute holds the configurations related to dataset arrays.
    Methods:
        verify: A dummy verification method for testing purposes, always returns False and a message.
    Notes:
        This class is used to create a configuration object for the dummy dataset.
    """

    raw_config: ArrayConfig = attr.ib(DummyArrayConfig(name="dummy_array"))
