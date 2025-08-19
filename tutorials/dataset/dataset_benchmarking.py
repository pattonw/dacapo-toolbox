# %% [markdown]
# # Dataset Benchmarking
#
# This notebook compares batch generation speed comparing a simple gunpowder pipeline to the
# iterable dataset alone and working with the pytorch lightning library and the huggingface
# accelerator library.

# %%
import multiprocessing as mp

mp.set_start_method("fork", force=True)
import dask

dask.config.set(scheduler="single-threaded")

from tqdm import tqdm

# %%
from dacapo_toolbox.sample_datasets import cremi
from pathlib import Path
import time

ITERATIONS = 100
BATCH_SIZE = 10
INPUT_SHAPE = (8, 80, 80)
NUM_WORKERS = 10

raw_train, labels_train, raw_test, labels_test = cremi(Path("cremi.zarr"))
raw_train.lazy_op(lambda x: x / 255.0)
labels_train.lazy_op(lambda x: x / x.max())


# %% [markdown]
# ## DaCapo-toolbox plain

# %%
from dacapo_toolbox.dataset import iterable_dataset
import torch

dataset = iterable_dataset(
    datasets={"raw": raw_train, "labels": labels_train},
    shapes={"raw": INPUT_SHAPE, "labels": INPUT_SHAPE},
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
)
batch_gen = iter(dataloader)

t1 = time.time()
for i in tqdm(range(ITERATIONS)):
    batch = next(batch_gen)
t2 = time.time()
print(f"(iterable_dataset): Time for {ITERATIONS} batches: {t2 - t1:.2f} seconds")

# %% [markdown]
# ## gunpowder

# %%
import gunpowder as gp
from funlib.persistence import Array
from funlib.geometry import Coordinate
import dask.array as da

raw_key = gp.ArrayKey("RAW")
labels_key = gp.ArrayKey("LABELS")
mask_key = gp.ArrayKey("Mask")

mask_train = Array(
    da.ones_like(labels_train.data),
    labels_train.roi.offset,
    labels_train.voxel_size,
    labels_train.axis_names,
    labels_train.units,
    labels_train.types,
)

pipeline = (
    (
        gp.ArraySource(raw_key, raw_train),
        gp.ArraySource(labels_key, labels_train),
        gp.ArraySource(mask_key, mask_train),
    )
    + gp.MergeProvider()
    + gp.Pad(raw_key, None)
    + gp.Pad(labels_key, None)
    + gp.RandomLocation(mask=mask_key, min_masked=1.0)
    + gp.PreCache(num_workers=NUM_WORKERS)
    + gp.Stack(BATCH_SIZE)
)
request = gp.BatchRequest()
request.add(raw_key, Coordinate(INPUT_SHAPE) * raw_train.voxel_size)
request.add(labels_key, Coordinate(INPUT_SHAPE) * labels_train.voxel_size)
request.add(mask_key, Coordinate(2, 2, 2) * labels_train.voxel_size)
with gp.build(pipeline):
    t1 = time.time()
    for i in tqdm(range(ITERATIONS)):
        batch = pipeline.request_batch(request)
    t2 = time.time()
print(f"(gunpowder): Time for {ITERATIONS} batches: {t2 - t1:.2f} seconds")

# %% [markdown]
# ## dacapo-toolbox with pytorch lightning

# %%
import torch
import lightning as L

# define any number of nn.Modules (or use your current ones)
model = torch.nn.Sequential(
    torch.nn.Conv3d(1, 64, 1), torch.nn.ReLU(), torch.nn.Conv3d(64, 1, 1)
)


# define the LightningModule
class LitModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch["raw"].float(), batch["labels"].float()
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        loss = torch.nn.functional.mse_loss(self.model(x), y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# init the autoencoder
lightning_dummy = LitModel(model)
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
)
trainer = L.Trainer(limit_train_batches=ITERATIONS, max_epochs=1)
t1 = time.time()
trainer.fit(model=lightning_dummy, train_dataloaders=train_loader)
t2 = time.time()
print(f"(lightning): Time for {ITERATIONS} batches: {t2 - t1:.2f} seconds")


# %% [markdown]
# ## dacapo-toolbox with huggingface accelerator
from accelerate import Accelerator

accelerator = Accelerator()
model = torch.nn.Sequential(
    torch.nn.Conv3d(1, 64, 1), torch.nn.ReLU(), torch.nn.Conv3d(64, 1, 1)
)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
training_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
)

model, optimizer, training_dataloader = accelerator.prepare(
    model, optimizer, training_dataloader
)

batch_gen = iter(training_dataloader)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t1 = time.time()
for i in tqdm(range(ITERATIONS)):
    batch = next(batch_gen)
    optimizer.zero_grad()
    inputs, targets = batch["raw"].float(), batch["labels"].float()
    inputs = inputs.unsqueeze(1)
    targets = targets.unsqueeze(1)
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
t2 = time.time()
print(f"(accelerator): Time for {ITERATIONS} batches: {t2 - t1:.2f} seconds")
