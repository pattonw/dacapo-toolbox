from .architectures import ArchitectureConfig
from .tasks import TaskConfig
from .converter import converter


def from_yaml(architecture_dict: dict, task_dict: dict | None = None):
    """
    Create a model from the architecture and task config.
    """
    # Use ArchitectureConfig as the base class for dynamic subclass resolution
    architecture = converter.structure(architecture_dict, ArchitectureConfig)  # type: ignore[type-abstract]
    task = converter.structure(task_dict, TaskConfig) if task_dict is not None else None
    if task is not None:
        model = task.task_type(task).create_model(architecture=architecture)
    else:
        model = architecture.module()

    return model
