import pytest

from dacapo_toolbox.trainers import DummyTrainerConfig, GunpowderTrainerConfig
from .serde import serde_test


@pytest.fixture()
def dummy_trainer():
    yield DummyTrainerConfig(name="dummy_trainer", dummy_attr=True)


@pytest.fixture()
def gunpowder_trainer():
    yield GunpowderTrainerConfig(
        name="default_gp_trainer",
    )


@pytest.mark.parametrize(
    "trainer_config",
    [
        "dummy_trainer",
        "gunpowder_trainer",
    ],
)
def test_trainer(
    trainer_config,
):
    # Initialize the config store (uses options behind the scene)
    serde_test(trainer_config)
