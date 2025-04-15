from bioimageio.core import test_model
from bioimageio.spec import save_bioimageio_package
from bioimageio.spec.model.v0_5 import (
    ModelDescr,
    WeightsDescr,
    PytorchStateDictWeightsDescr,
    Author,
    CiteEntry,
    LicenseId,
    HttpUrl,
    ArchitectureFromLibraryDescr,
    OutputTensorDescr,
    InputTensorDescr,
    BatchAxis,
    ChannelAxis,
    SpaceInputAxis,
    SpaceOutputAxis,
    Identifier,
    AxisId,
    TensorId,
    SizeReference,
    FileDescr,
    Doi,
    IntervalOrRatioDataDescr,
    ParameterizedSize,
    Version,
)

from collections.abc import Sequence

from dacapo_toolbox.architectures import ArchitectureConfig
from dacapo_toolbox.tasks import TaskConfig

import torch


import logging
import sys
import hashlib
import tempfile
from pathlib import Path

from funlib.geometry import Coordinate
import numpy as np

logger = logging.getLogger(__name__)


def save_bioimage_io_model(
    path: Path,
    architecture: ArchitectureConfig,
    weights: dict[str, torch.Tensor],
    authors: list[Author],
    cite: list[CiteEntry] | None = None,
    license: str = "MIT",
    input_test_image_path: Path | None = None,
    output_test_image_path: Path | None = None,
    in_voxel_size: Coordinate | None = None,
    test_saved_model: bool = False,
    task: TaskConfig | None = None,
):
    from dacapo_toolbox.converter import converter

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        task_obj = task.task_type(task) if task is not None else None
        if task_obj is not None:
            model = task_obj.create_model(architecture=architecture)
        else:
            model = architecture.module()

        input_axes: list[BatchAxis | ChannelAxis | SpaceInputAxis] = [
            BatchAxis(),
            ChannelAxis(
                channel_names=[
                    Identifier(f"in_c{i}") for i in range(architecture.num_in_channels)
                ]
            ),
        ]
        input_shape = architecture.input_shape
        in_voxel_size = (
            in_voxel_size
            if in_voxel_size is not None
            else Coordinate((1,) * input_shape.dims)
        )

        output_shape = architecture.compute_output_shape(input_shape)

        out_voxel_size = architecture.scale(in_voxel_size)
        if any(os == 0 for os in out_voxel_size):
            out_voxel_size = in_voxel_size
            in_voxel_size = architecture.inv_scale(out_voxel_size)

        input_axes += [
            SpaceInputAxis(
                id=AxisId(f"d{i}"),
                size=ParameterizedSize(min=s, step=s),
                scale=scale,
            )
            for i, (s, scale) in enumerate(zip(input_shape, in_voxel_size))
        ]
        data_descr = IntervalOrRatioDataDescr(type="float32")

        if input_test_image_path is None:
            input_test_image_path = tmp_path / "input_test_image.npy"
            test_image = (
                np.random.random(
                    (
                        1,
                        architecture.num_in_channels,
                        *architecture.input_shape,
                    )
                ).astype(np.float32)
                * 2
                - 1
            )
            np.save(input_test_image_path, test_image)

        input_descr = InputTensorDescr(
            id=TensorId("raw"),
            axes=input_axes,
            test_tensor=FileDescr(source=input_test_image_path),
            data=data_descr,
        )

        context_units = (input_shape * in_voxel_size) - (output_shape * out_voxel_size)

        context_out_voxels = context_units / out_voxel_size

        output_axes: list[BatchAxis | ChannelAxis | SpaceOutputAxis] = [
            BatchAxis(),
            ChannelAxis(
                channel_names=(
                    task_obj.channels
                    if task_obj is not None
                    else [
                        Identifier(f"c{i}")
                        for i in range(architecture.num_out_channels)
                    ]
                )
            ),
        ]
        output_axes += [
            SpaceOutputAxis(
                id=AxisId(f"d{i}"),
                size=SizeReference(
                    tensor_id=TensorId("raw"),
                    axis_id=AxisId(f"d{i}"),
                    offset=-c,
                ),
                scale=s,
            )
            for i, (c, s) in enumerate(zip(context_out_voxels, out_voxel_size))
        ]
        if output_test_image_path is None:
            output_test_image_path = tmp_path / "output_test_image.npy"
            with torch.no_grad():
                test_out_image = (
                    model.eval()(torch.from_numpy(test_image).float()).detach().numpy()
                )
            print(model)
            np.save(output_test_image_path, test_out_image)
        output_descr = OutputTensorDescr(
            id=TensorId(
                task_obj.__class__.__name__.lower().replace("task", "")
                if task_obj is not None
                else architecture.name
            ),
            axes=output_axes,
            test_tensor=FileDescr(source=output_test_image_path),
        )

        pytorch_architecture = ArchitectureFromLibraryDescr(
            callable=Identifier("from_yaml"),
            kwargs={
                "architecture_dict": converter.unstructure(architecture),
                "task_dict": converter.unstructure(task),
            },
            import_from="dacapo_toolbox.convenience",
        )

        weights_path = tmp_path / "model.pth"
        torch.save(weights, weights_path)
        if sys.version_info[1] < 11:
            raise NotImplementedError(
                "Saving to bioimageio modelzoo format is not implemented for Python versions < 3.11"
            )

        my_model_descr = ModelDescr(
            name=f"{architecture.name}" + (f"-{task.name}" if task is not None else ""),
            description="A model trained with DaCapo",
            authors=authors,
            cite=[
                CiteEntry(
                    text="paper",
                    doi=Doi("10.1234something"),
                )
            ],
            license=LicenseId(license),
            documentation=HttpUrl(
                "https://github.com/janelia-cellmap/dacapo/blob/main/README.md"
            )
            if "test" not in architecture.name
            else Path(__file__).parent.parent.parent / "README.md",
            git_repo=HttpUrl("https://github.com/janelia-cellmap/dacapo")
            if "test" not in architecture.name
            else HttpUrl("http://example.com"),
            inputs=[input_descr],
            outputs=[output_descr],
            weights=WeightsDescr(
                pytorch_state_dict=PytorchStateDictWeightsDescr(
                    source=weights_path,
                    architecture=pytorch_architecture,
                    pytorch_version=Version(torch.__version__),
                ),
            ),
        )

        if test_saved_model:
            summary = test_model(my_model_descr)
            summary.display()

        logger.info(
            "package path:",
            save_bioimageio_package(my_model_descr, output_path=path),
        )
