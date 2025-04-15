import attr

from abc import ABC, abstractmethod
from collections.abc import Sequence

from dacapo_toolbox.datasplits.datasets import DatasetConfig
from dacapo_toolbox.tasks import TaskConfig

import torch


@attr.s
class TrainerConfig(ABC):
    """
    A class that defines how the pipeline should be built. I.e. do you want to use gunpowder
    or pytorch lightning or something else?
    Regardless of what tool is used, you should be able to return an iterator that yields
    torch tensors that can be used to train a model. There must also be a simple integer
    attribute that defines the number of workers to use in the case of parallelization
    with a default of None for no multiprocessing (1 may mean a single subprocess worker
    providing the data)
    """

    name: str = attr.ib(
        metadata={
            "help_text": "A unique name for this trainer. This will be saved so you "
            "and others can find and reuse this trainer. Keep it short "
            "and avoid special characters."
        }
    )

    @abstractmethod
    def iterable_dataset(
        self,
        datasets: list[DatasetConfig],
        input_size: Sequence[int],
        output_size: Sequence[int],
        task: TaskConfig | None = None,
    ) -> torch.utils.data.IterableDataset:
        """
        Returns an pytorch compatible IterableDataset.
        See https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for more info
        """
        pass
