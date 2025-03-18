import attr
import neuroglancer
from typing import Tuple

from .datasets import DatasetConfig

import itertools
import logging

logger = logging.getLogger(__name__)


@attr.s
class DataSplitConfig:
    """
    A class used to create a DataSplit configuration object.

    Attributes:
        name : str
            A name for the datasplit. This name will be saved so it can be found
            and reused easily. It is recommended to keep it short and avoid special
            characters.
    Methods:
        verify() -> Tuple[bool, str]:
            Validates if it is a valid data split configuration.
    Notes:
        This class is used to create a DataSplit configuration object.
    """

    name: str = attr.ib(
        metadata={
            "help_text": "A unique name for this datasplit. This will be saved so "
            "you and others can find and reuse this datasplit. Keep it short "
            "and avoid special characters."
        }
    )

    def verify(self) -> Tuple[bool, str]:
        """
        Validates if the current configuration is a valid data split configuration.

        Returns:
            Tuple[bool, str]
                True if the configuration is valid,
                False otherwise along with respective validation error message.
        Raises:
            NotImplementedError
                If the method is not implemented in the derived class.
        Examples:
            >>> datasplit_config = DataSplitConfig(name="datasplit")
            >>> datasplit_config.verify()
            (True, "No validation for this DataSplit")
        Notes:
            This method is used to validate the configuration of DataSplit.
        """
        return True, "No validation for this DataSplit"

    @property
    def train(self) -> list[DatasetConfig]:
        pass

    @property
    def validate(self) -> list[DatasetConfig]:
        pass

    def _neuroglancer(self, embedded=False, bind_address="0.0.0.0", bind_port=0):
        """
        A method to visualize the data in Neuroglancer. It creates a Neuroglancer viewer and adds the layers of the training and validation datasets to it.

        Args:
            embedded : bool
                A boolean flag to indicate if the Neuroglancer viewer is to be embedded in the notebook.
            bind_address : str
                Bind address for Neuroglancer webserver
            bind_port : int
                Bind port for Neuroglancer webserver
        Returns:
            viewer : obj
                The Neuroglancer viewer object.
        Raises:
            Exception
                If the model setup cannot be loaded, an Exception is thrown which is logged and handled by training the model without head matching.
        Examples:
            >>> viewer = datasplit._neuroglancer(embedded=True)
        Notes:
            This function is called by the DataSplit class to visualize the data in Neuroglancer.
            It creates a Neuroglancer viewer and adds the layers of the training and validation datasets to it.
            Neuroglancer is a powerful tool for visualizing large-scale volumetric data.
        """
        neuroglancer.set_server_bind_address(
            bind_address=bind_address, bind_port=bind_port
        )
        viewer = neuroglancer.Viewer()
        with viewer.txn() as s:
            train_layers = {}
            for i, dataset in enumerate(self.train):
                train_layers.update(
                    dataset._neuroglancer_layers(
                        # exclude_layers=set(train_layers.keys())
                    )
                )

            validate_layers = {}
            if self.validate is not None:
                for i, dataset in enumerate(self.validate):
                    validate_layers.update(
                        dataset._neuroglancer_layers(
                            # exclude_layers=set(validate_layers.keys())
                        )
                    )

            for k, layer in itertools.chain(
                train_layers.items(), validate_layers.items()
            ):
                s.layers[k] = layer

            s.layout = neuroglancer.row_layout(
                [
                    neuroglancer.LayerGroupViewer(layers=list(train_layers.keys())),
                    neuroglancer.LayerGroupViewer(layers=list(validate_layers.keys())),
                ]
            )
        logger.info(f"Neuroglancer link: {viewer}")
        if embedded:
            from IPython.display import IFrame

            return IFrame(viewer.get_viewer_url(), width=800, height=600)
        return viewer
