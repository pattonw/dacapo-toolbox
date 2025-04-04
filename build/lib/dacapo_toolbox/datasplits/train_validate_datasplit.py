from .datasplit import DataSplit
from .datasets import DatasetConfig

from typing import List
import warnings


class TrainValidateDataSplit(DataSplit):
    """
    A DataSplit that contains a list of training and validation datasets. This
    class is used to split the data into training and validation datasets. The
    training and validation datasets are used to train and validate the model
    respectively.

    Attributes:
        train : list
            The list of training datasets.
        validate : list
            The list of validation datasets.
    Methods:
        __init__(datasplit_config)
            Initializes the TrainValidateDataSplit class with specified config to
            split the data into training and validation datasets.
    Notes:
        This class is used to split the data into training and validation datasets.
    """

    train: List[DatasetConfig]
    validate: List[DatasetConfig]

    def __init__(self, datasplit_config):
        """
        Initializes the TrainValidateDataSplit class with specified config to
        split the data into training and validation datasets.

        Args:
            datasplit_config : obj
                The configuration to initialize the TrainValidateDataSplit class.
        Raises:
            Exception
                If the model setup cannot be loaded, an Exception is thrown which
                is logged and handled by training the model without head matching.
        Examples:
            >>> train_validate_datasplit = TrainValidateDataSplit(datasplit_config)
        Notes:
            This function is called by the TrainValidateDataSplit class to initialize
            the TrainValidateDataSplit class with specified config to split the data
            into training and validation datasets.
        """
        super().__init__()
        warnings.warn(
            "TrainValidateDataSplit is deprecated. Use SimpleDataSplitConfig instead.",
            DeprecationWarning,
        )

        self.train = [train_config for train_config in datasplit_config.train_configs]
        self.validate = [
            validate_config for validate_config in datasplit_config.validate_configs
        ]
