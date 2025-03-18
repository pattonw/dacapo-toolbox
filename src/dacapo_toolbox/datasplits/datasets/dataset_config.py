import attr

from funlib.geometry import Coordinate
from funlib.persistence import Array

from abc import ABC, abstractmethod


@attr.s
class DatasetConfig(ABC):
    """
    A class used to define configuration for datasets. This provides the
    framework to create a Dataset instance.

    Attributes:
        name: str (eg: "sample_dataset").
            A unique identifier to name the dataset.
            It aids in easy identification and reusability of this dataset.
            Advised to keep it short and refrain from using special characters.

        weight: int (default=1).
            A numeric value that indicates how frequently this dataset should be
            sampled in comparison to others. Higher the weight, more frequently it
            gets sampled.
    Methods:
        verify:
            Checks and validates the dataset configuration. The specific rules for
            validation need to be defined by the user.
    Notes:
        This class is used to create a configuration object for datasets.
    """

    name: str = attr.ib(
        metadata={
            "help_text": "A unique name for this dataset. This will be saved so you "
            "and others can find and reuse this dataset. Keep it short "
            "and avoid special characters."
        }
    )

    weight: int
    sample_points: list[Coordinate] | None

    @property
    @abstractmethod
    def raw(self) -> Array:
        pass

    @property
    def gt(self) -> Array | None:
        pass

    @property
    def mask(self) -> Array | None:
        pass

    def _neuroglancer_layers(self, prefix="", exclude_layers=None):
        """
        Generates neuroglancer layers for raw, gt and mask if they can be viewed by neuroglance, excluding those in
        the exclude_layers.

        Args:
            prefix (str, optional): A prefix to be added to the layer names.
            exclude_layers (set, optional): A set of layer names to exclude.
        Returns:
            dict: A dictionary containing layer names as keys and corresponding neuroglancer layer as values.
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        Examples:
            >>> dataset = Dataset("dataset")
            >>> dataset._neuroglancer_layers()
            {"raw": neuroglancer_layer}
        Notes:
            This method is used to generate neuroglancer layers for raw, gt and mask if they can be viewed by neuroglance.
        """
        layers = {}
        exclude_layers = exclude_layers if exclude_layers is not None else set()
        if (
            self.raw._can_neuroglance()
            and self.raw._source_name() not in exclude_layers
        ):
            layers[self.raw._source_name()] = self.raw._neuroglancer_layer()
        if self.gt is not None and self.gt._can_neuroglance():
            new_layers = self.gt._neuroglancer_layer()
            if isinstance(new_layers, list):
                names = self.gt._source_name()
                for name, layer in zip(names, new_layers):
                    if name not in exclude_layers:
                        layers[name] = layer
            elif self.gt._source_name() not in exclude_layers:
                layers[self.gt._source_name()] = new_layers
        if self.mask is not None and self.mask._can_neuroglance():
            new_layers = self.mask._neuroglancer_layer()
            if isinstance(new_layers, list):
                names = self.mask._source_name()
                for name, layer in zip(names, new_layers):
                    if name not in exclude_layers:
                        layers[name] = layer
            elif self.gt._source_name() not in exclude_layers:
                layers["mask_" + self.mask._source_name()] = new_layers
        return layers
