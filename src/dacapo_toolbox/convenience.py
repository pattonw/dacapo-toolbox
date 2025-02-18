from dacapo.experiments.datasplits import SimpleDataSplitConfig
from dacapo.experiments.architectures import ArchitectureConfig
from dacapo.experiments.tasks import AffinitiesTaskConfig, OneHotTaskConfig
from dacapo.experiments.tasks.predictors import Predictor
from dacapo.experiments.trainers import GunpowderTrainerConfig
from dacapo.experiments.trainers.gp_augments import (
    ElasticAugmentConfig,
    IntensityAugmentConfig,
    NoiseAugmentConfig,
)
from dacapo.experiments.tasks import TaskConfig
from dacapo.experiments.model import Model

from typing import Sequence
from pathlib import Path

from funlib.geometry import Coordinate

import torch
from typing import Literal

Augment = Literal["elastic"] | Literal["noise"] | Literal["intensity"]
augment_map = {
    "elastic": {
        2: ElasticAugmentConfig(
            control_point_spacing=(10, 10),
            control_point_displacement_sigma=(3.0, 3.0),
            subsample=4,
            uniform_3d_rotation=True,
            rotation_interval=(0, 1),
        ),
        3: ElasticAugmentConfig(
            control_point_spacing=(10, 10, 10),
            control_point_displacement_sigma=(3.0, 3.0, 3.0),
            subsample=4,
            uniform_3d_rotation=True,
            rotation_interval=(0, 1),
        ),
    },
    "noise": NoiseAugmentConfig(),
    "intensity": IntensityAugmentConfig(
        scale=(0.9, 1.1),
        shift=(-0.1, 0.1),
    ),
}

task_map = {
    "instance": {
        2: AffinitiesTaskConfig(
            name="affinities_2d",
            neighborhood=[
                Coordinate(1, 0),
                Coordinate(0, 1),
                Coordinate(3, 0),
                Coordinate(0, 3),
                Coordinate(9, 0),
                Coordinate(0, 9),
            ],
        ),
        3: AffinitiesTaskConfig(
            name="affinities_3d",
            neighborhood=[
                Coordinate(1, 0, 0),
                Coordinate(0, 1, 0),
                Coordinate(0, 0, 1),
                Coordinate(0, 3, 0),
                Coordinate(0, 0, 3),
                Coordinate(0, 9, 0),
                Coordinate(0, 0, 9),
            ],
        ),
    },
    "semantic": OneHotTaskConfig(name="semantic", classes=[]),
}


def dataset_from_zarr(
    zarr_container: Path,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    augments: list[Augment] | None = None,
    task: Literal["instance"] | Literal["semantic"] = "instance",
    mode: Literal["train"] | Literal["val"] = "train",
) -> torch.utils.data.IterableDataset:
    dims = len(input_shape)
    augments = (
        [augment_map[a] if a != "elastic" else augment_map[a][dims] for a in augments]
        if augments is not None
        else []
    )

    datasplit = SimpleDataSplitConfig(name="simple-dataset", path=zarr_container)
    pipeline = GunpowderTrainerConfig("gp_trainer", augments=augments)

    if task is not None:
        task = task_map[task]
        if isinstance(task, dict):
            task = task[dims]
        predictor = task.task_type(task).predictor
    else:
        predictor = None
    return pipeline.iterable_dataset(
        datasplit.train if mode == "train" else datasplit.validate,
        Coordinate(input_shape),
        Coordinate(output_shape),
        predictor=predictor,
    )


def module(
    architecture_config: ArchitectureConfig,
    task: TaskConfig | None = None,
) -> torch.nn.Module:
    if task is not None:
        predictor: Predictor = task.task_type(task).predictor
        return predictor.create_model(architecture=architecture_config)
    else:
        return Model(architecture_config, predictor)


def loss_function(task_config: TaskConfig) -> torch.nn.Module:
    task = task_config.task_type(task_config)
    return task.loss


def optimizer(module) -> torch.optim.Optimizer:
    return torch.optim.Adam(module.parameters())


def scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler._LRScheduler:
    return torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=10,
        last_epoch=-1,
    )


def train_loop(
    model: torch.nn.Module,
    dataset: torch.utils.data.IterableDataset,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_iterations: int,
    batch_size: int,
    num_workers: int = 0,
):
    model = model.to(device)
    data_loader = torch.utils.data.DataLoader(
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )
    for i, batch in enumerate(data_loader):
        optimizer.zero_grad()
        raw, target, weight = batch["raw"], batch["target"], batch["weight"]
        raw = raw.to(device)
        target = target.to(device)
        weight = weight.to(device)
        output = model(raw)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if i >= num_iterations:
            break
