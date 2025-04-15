import attr
import re

import torch
from .architecture import ArchitectureConfig

from funlib.geometry import Coordinate

from bioimageio.core import load_description
from bioimageio.core.backends.pytorch_backend import PytorchModelAdapter
from bioimageio.spec import InvalidDescr
from bioimageio.spec.model.v0_5 import (
    ModelDescr,
    OutputTensorDescr,
    InputTensorDescr,
)

from pathlib import Path
import zipfile
import numpy as np


@attr.s
class ModelZooConfig(ArchitectureConfig):
    """
    A thin wrapper allowing users to pass in any model zoo model.

    Support is currently limited to models saved with the `PytorchStateDictWeightsDescr`.
    See more info here: https://bioimage-io.github.io/spec-bioimage-io/bioimageio/spec/model/v0_5.html#PytorchStateDictWeightsDescr

    We try to support all model_id formats that are supported by the `bioimageio.core` `load_description` function.
    However there seem to be some cases that fail. You may receive an `InvalidDescr` error when trying to load a model from
    a downloaded rdf file. In this case downloading the zipped model, or using the models name e.g. "affable-shark"
    should work.
    """

    model_id: str = attr.ib(
        metadata={
            "help_text": "The model id from the model zoo to use. Can be any of:\n"
            '\t1) Url to a model zoo model (e.g. "https://.../rdf.yaml")\n'
            '\t2) Local path to a model zoo model (e.g. "some/local/rdf.yaml")\n'
            '\t3) Local path to a zipped model (e.g. "some/local/package.zip")\n'
            "\t4) Specific versioned model (e.g. {model_name}/{version})\n"
            "\t5) More options available, see: https://github.com/bioimage-io/spec-bioimage-io/tree/main"
        }
    )
    trainable_layers: str | None = attr.ib(
        default=None, metadata={"help_text": "Regex pattern for trainable layers"}
    )

    _model_description: ModelDescr | None = None
    _model_adapter: PytorchModelAdapter | None = None

    def module(self) -> torch.nn.Module:
        module = self.model_adapter._model
        for name, param in module.named_parameters():
            if self.trainable_layers is not None and re.match(
                self.trainable_layers, name
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False
        return module

    @property
    def model_adapter(self) -> PytorchModelAdapter:
        if self._model_adapter is None:
            assert self.model_description.weights.pytorch_state_dict is not None, (
                "We only support loading bioimageio models with a pytorch state dict"
            )

            return PytorchModelAdapter(
                model_description=self.model_description,
                devices=None,
            )
        return self._model_adapter

    @property
    def model_description(self) -> ModelDescr:
        if self._model_description is None:
            descr = load_description(self.model_id)
            assert isinstance(descr, ModelDescr)
            self._model_description = descr
            if isinstance(self._model_description, InvalidDescr):
                raise Exception("Invalid model description")
        assert self._model_description is not None
        return self._model_description

    @property
    def input_desc(self) -> InputTensorDescr:
        assert len(self.model_description.inputs) == 1, (
            f"Only models with one input are supported, found {self.model_description.inputs}"
        )
        return self.model_description.inputs[0]

    @property
    def output_desc(self) -> OutputTensorDescr:
        assert len(self.model_description.outputs) == 1, (
            f"Only models with one output are supported, found {self.model_description.outputs}"
        )
        return self.model_description.outputs[0]

    @property
    def input_shape(self):
        shape = [
            axis.size.min
            for axis in self.input_desc.axes
            if axis.type not in ["batch", "channel", "index"]
        ]
        return Coordinate(shape)

    @property
    def num_in_channels(self) -> int:
        channel_axes = [axis for axis in self.input_desc.axes if axis.type == "channel"]
        assert len(channel_axes) == 1, (
            f"Only models with one input channel axis are supported, found {channel_axes}"
        )
        return channel_axes[0].size

    @property
    def num_out_channels(self) -> int:
        channel_axes = [
            axis for axis in self.output_desc.axes if axis.type == "channel"
        ]
        assert len(channel_axes) == 1, (
            f"Only models with one output channel axis are supported, found {channel_axes}"
        )
        return channel_axes[0].size

    def scale(self, input_voxel_size: Coordinate) -> Coordinate:
        input_axes = [
            axis
            for axis in self.input_desc.axes
            if axis.type not in ["batch", "channel", "index"]
        ]
        output_axes = [
            axis
            for axis in self.output_desc.axes
            if axis.type not in ["batch", "channel", "index"]
        ]
        assert all(
            [
                in_axis.id == out_axis.id
                for in_axis, out_axis in zip(input_axes, output_axes)
            ]
        )
        scale = np.array(
            [
                in_axis.scale / out_axis.scale
                for in_axis, out_axis in zip(input_axes, output_axes)
            ]
        )
        output_voxel_size = Coordinate(np.array(input_voxel_size) / scale)
        return output_voxel_size
