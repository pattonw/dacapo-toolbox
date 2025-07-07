# %% [markdown]
# # Cremi Example
# This tutorial demonstrates some simple pipelines using the dacapo_toolbox
# dataset on [cremi data](https://cremi.org/data/). We'll cover a fun method
# for instance segmentation using a 2.5D U-Net.

# %% [markdown]
# ## Introduction and overview
#
# In this tutorial we will cover a few basic ML tasks using the DaCapo toolbox. We will:
#
# - Prepare a dataloader for the CREMI dataset
# - Train a simple 2D U-Net for both instance and semantic segmentation
# - Visualize the results
#

# %% [markdown]
# ## Environment setup
# If you have not already done so, you will need to install DaCapo. You can do this
# by first creating a new environment and then installing the DaCapo Toolbox.
#
# I highly recommend using [uv](https://docs.astral.sh/uv/) for environment management,
# but there are many tools to choose from.
#
# ```bash
# uv init
# uv add git+https://github.com/pattonw/dacapo-toolbox.git
# ```

# %% [markdown]
# ## Data Preparation
# DaCapo works with zarr, so we will download [CREMI Sample A](https://cremi.org/static/data/sample_A%2B_20160601.hdf)
# and save it as a zarr file.

# %%
import multiprocessing as mp

mp.set_start_method("fork", force=True)
import dask

dask.config.set(scheduler="single-threaded")

import wget
from pathlib import Path
import h5py
import zarr
from tqdm import tqdm

if not Path("_static/dataset_tutorial").exists():
    Path("_static/dataset_tutorial").mkdir(parents=True, exist_ok=True)


# Download some cremi data
# immediately convert it to zarr for convenience
if not Path("cremi.zarr").exists():
    wget.download(
        "https://cremi.org/static/data/sample_C_20160501.hdf", "sample_C_20160501.hdf"
    )
    wget.download(
        "https://cremi.org/static/data/sample_A_20160501.hdf", "sample_A_20160501.hdf"
    )
    zarr.save_array(
        "cremi.zarr/train/raw",
        h5py.File("sample_C_20160501.hdf", "r")["volumes/raw"][:],
    )
    zarr.save_array(
        "cremi.zarr/train/labels",
        h5py.File("sample_C_20160501.hdf", "r")["volumes/labels/neuron_ids"][:],
    )
    zarr.save_array(
        "cremi.zarr/test/raw",
        h5py.File("sample_A_20160501.hdf", "r")["volumes/raw"][:],
    )
    zarr.save_array(
        "cremi.zarr/test/labels",
        h5py.File("sample_A_20160501.hdf", "r")["volumes/labels/neuron_ids"][:],
    )
    Path("sample_A_20160501.hdf").unlink()
    Path("sample_C_20160501.hdf").unlink()


# %% [markdown]
# ## Data Loading
# We will use the [funlib.persistence](github.com/funkelab/funlib.persistence) library to interface with zarr. This library adds support for units, voxel size, and axis names along with the ability to query our data based on a `Roi` object describing a specific rectangular piece of data. This is especially useful in a microscopy context where you regularly need to chunk your data for processing.

# %%
from funlib.persistence import open_ds, Array
from pathlib import Path
from funlib.geometry import Coordinate

voxel_size = (40, 4, 4)  # in nm
axis_names = ["z", "y", "x"]
units = ["nm", "nm", "nm"]

NUM_ITERATIONS = 300

raw_train = open_ds(
    "cremi.zarr/train/raw", voxel_size=voxel_size, axis_names=axis_names, units=units
)
labels_train = open_ds(
    "cremi.zarr/train/labels", voxel_size=voxel_size, axis_names=axis_names, units=units
)
raw_test = open_ds(
    "cremi.zarr/test/raw", voxel_size=voxel_size, axis_names=axis_names, units=units
)
labels_test = open_ds(
    "cremi.zarr/test/labels", voxel_size=voxel_size, axis_names=axis_names, units=units
)


# %% [markdown]
# Lets visualize our train and test data

# %% [markdown]
# ### Training data

# %%
from dacapo_toolbox.vis.preview import gif_2d

# create a 2D gif of the training data
gif_2d(
    arrays={"Train Raw": raw_train, "Train Labels": labels_train},
    array_types={"Train Raw": "raw", "Train Labels": "labels"},
    filename="_static/dataset_tutorial/training-data.gif",
    title="Training Data",
    fps=10,
)

# %% [markdown]
# Here we visualize the training data:
# ![training-data](_static/dataset_tutorial/training-data.gif)

# %% [markdown]
# ### Testing data

# %%
gif_2d(
    arrays={"Test Raw": raw_test, "Test Labels": labels_test},
    array_types={"Test Raw": "raw", "Test Labels": "labels"},
    filename="_static/dataset_tutorial/testing-data.gif",
    title="Testing Data",
    fps=10,
)

# %% [markdown]
# Here we visualize the test data:
# ![test-data](_static/dataset_tutorial/test-data.gif)

# %% [markdown]
# ### DaCapo
# Now that we have some data, lets look at how we can use DaCapo to interface with it for some common ML use cases.

# %% [markdown]
# ### Data Split
# We always want to be explicit when we define our data split for training and validation so that we are aware what data is being used for training and validation.

# %%
from dacapo_toolbox.dataset import (
    iterable_dataset,
    DeformAugmentConfig,
    SimpleAugmentConfig,
)

# %%
train_dataset = iterable_dataset(
    datasets={"raw": raw_train, "gt": labels_train},
    shapes={"raw": (13, 256, 256), "gt": (13, 256, 256)},
    deform_augment_config=DeformAugmentConfig(
        p=0.1,
        control_point_spacing=(2, 10, 10),
        jitter_sigma=(0.5, 2, 2),
        rotate=True,
        subsample=4,
        rotation_axes=(1, 2),
        scale_interval=(1.0, 1.0),
    ),
    simple_augment_config=SimpleAugmentConfig(
        p=1.0,
        mirror_only=(1, 2),
        transpose_only=(1, 2),
    ),
    trim=Coordinate(5, 5, 5),
)
batch_gen = iter(train_dataset)

# %%
batch = next(batch_gen)
gif_2d(
    arrays={
        "Raw": Array(batch["raw"].numpy()),
        "Labels": Array(batch["gt"].numpy() % 256),
    },
    array_types={"Raw": "raw", "Labels": "labels"},
    filename="_static/dataset_tutorial/simple-batch.gif",
    title="Simple Batch",
    fps=10,
)

# %% [markdown]
# Here we visualize the training data:
# ![simple-batch](_static/dataset_tutorial/simple-batch.gif)

# %% [markdown]
# ### Tasks
# When training for instance segmentation, it is not possible to directly predict label ids since the ids have to be unique accross the full volume which is not possible to do with the local context that a UNet operates on. So instead we need to transform our labels into some intermediate representation that is both easy to predict and easy to post process. The most common method we use is a combination of [affinities](https://arxiv.org/pdf/1706.00120) with optional [lsds](https://github.com/funkelab/lsd) for prediction plus [mutex watershed](https://arxiv.org/abs/1904.12654) for post processing.
#
# Next we will define the task that encapsulates this process.

# %%
from dacapo_toolbox.transforms.affs import Affs, AffsMask
from dacapo_toolbox.transforms.weight_balancing import BalanceLabels
import torchvision

neighborhood = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0, 7, 0),
    (0, 0, 7),
    (0, 23, 0),
    (0, 0, 23),
]
train_dataset = iterable_dataset(
    datasets={"raw": raw_train, "gt": labels_train},
    shapes={"raw": (13, 256, 256), "gt": (13, 256, 256)},
    transforms={
        ("gt", "affs"): Affs(neighborhood=neighborhood),
        ("gt", "affs_mask"): AffsMask(neighborhood=neighborhood),
        (("affs", "affs_mask"), "weights"): BalanceLabels((1, -1, -1, -1)),
    },
    deform_augment_config=DeformAugmentConfig(
        p=0.1,
        control_point_spacing=(2, 10, 10),
        jitter_sigma=(0.5, 2, 2),
        rotate=True,
        subsample=4,
        rotation_axes=(1, 2),
        scale_interval=(1.0, 1.0),
    ),
    simple_augment_config=SimpleAugmentConfig(
        p=1.0,
        mirror_only=(1, 2),
        transpose_only=(1, 2),
    ),
)

batch_gen = iter(train_dataset)

# %%
batch = next(batch_gen)
gif_2d(
    arrays={
        "Raw": Array(batch["raw"].numpy(), voxel_size=raw_train.voxel_size),
        "GT": Array(batch["gt"].numpy() % 256, voxel_size=raw_train.voxel_size),
        "Affs": Array(
            batch["affs"].float().numpy()[[0, 3, 4]],
            voxel_size=raw_train.voxel_size,
        ),
        "Affs Mask": Array(
            batch["affs_mask"].float().numpy()[[0, 3, 4]],
            voxel_size=raw_train.voxel_size,
        ),
    },
    array_types={
        "Raw": "raw",
        "GT": "labels",
        "Affs": "affs",
        "Affs Mask": "affs",
    },
    filename="_static/dataset_tutorial/affs-batch.gif",
    title="Affinities Batch",
    fps=10,
)

# %% [markdown]
# Here we visualize a batch with (raw, gt, target) triplets for the affinities task:
# ![affs-batch](_static/dataset_tutorial/affs-batch.gif)

# %% [markdown]
# ### Models
# Lets define our model

# %%
import tems
import torch


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cpu")

unet = tems.UNet.funlib_api(
    dims=3,
    in_channels=1,
    num_fmaps=32,
    fmap_inc_factor=4,
    downsample_factors=[(1, 2, 2), (1, 2, 2), (1, 2, 2)],
    kernel_size_down=[
        [(1, 3, 3), (1, 3, 3)],
        [(1, 3, 3), (1, 3, 3)],
        [(1, 3, 3), (1, 3, 3)],
        [(1, 3, 3), (1, 3, 3)],
    ],
    kernel_size_up=[
        [(1, 3, 3), (1, 3, 3)],
        [(1, 3, 3), (1, 3, 3)],
        [(3, 3, 3), (3, 3, 3)],
    ],
    activation="LeakyReLU",
)

module = torch.nn.Sequential(
    unet, torch.nn.Conv3d(32, len(neighborhood), kernel_size=1), torch.nn.Sigmoid()
).to(device)

# %% [markdown]
# ### Training loop
# Now we can bring everything together and train our model.

# %%
from tqdm import tqdm

# %%
import torch

extra = torch.tensor((2, 64, 64))
train_dataset = iterable_dataset(
    datasets={"raw": raw_train, "gt": labels_train},
    shapes={
        "raw": unet.min_input_shape + extra,
        "gt": unet.min_output_shape + extra,
    },
    transforms={
        "raw": torchvision.transforms.Lambda(lambda x: x[None].float() / 255.0),
        ("gt", "affs"): Affs(neighborhood=neighborhood),
        ("gt", "affs_mask"): AffsMask(neighborhood=neighborhood),
        (("affs", "affs_mask"), "weights"): BalanceLabels((1, -1, -1, -1)),
    },
    deform_augment_config=DeformAugmentConfig(
        p=0.1,
        control_point_spacing=(2, 10, 10),
        jitter_sigma=(0.5, 2, 2),
        rotate=True,
        subsample=4,
        rotation_axes=(1, 2),
        scale_interval=(1.0, 1.0),
    ),
    simple_augment_config=SimpleAugmentConfig(
        p=1.0,
        mirror_only=(1, 2),
        transpose_only=(1, 2),
    ),
)

loss_func = torch.nn.BCELoss(reduction="none")
optimizer = torch.optim.Adam(module.parameters(), lr=1e-4)
dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=3,
    num_workers=4,
)
losses = []

for iteration, batch in tqdm(enumerate(iter(dataloader))):
    raw, target, weight = (
        batch["raw"].to(device),
        batch["affs"].to(device),
        batch["weights"].to(device),
    )
    optimizer.zero_grad()

    output = module(raw)

    voxel_loss = loss_func(output, target.float())
    loss = (voxel_loss * weight).mean()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if iteration >= NUM_ITERATIONS:
        break

# %%
import matplotlib.pyplot as plt
from funlib.geometry import Coordinate

plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.savefig("_static/dataset_tutorial/affs-loss-curve.png")
plt.close()

# %%
import mwatershed as mws
from funlib.geometry import Roi
import numpy as np

input_shape = unet.min_input_shape + extra
output_shape = unet.min_output_shape + extra

# fetch a xy slice from the center of our validation volume
# We snap to grid to a multiple of the max downsampling factor of
# the unet (1, 8, 8) to ensure downsampling is always possible
roi = raw_test.roi
z_coord = Coordinate(1, 0, 0)
xy_coord = Coordinate(0, 1, 1)
center_offset = roi.center * z_coord + roi.offset * xy_coord + roi.shape // 4
center_size = raw_test.voxel_size * z_coord * 2 + (roi.shape * xy_coord) // 2
center_slice = Roi(center_offset, center_size)
center_slice = center_slice.snap_to_grid(raw_test.voxel_size * Coordinate(1, 8, 8))
center_slice.shape = (
    center_slice.shape
    - (Coordinate(unet.min_output_shape) * raw_test.voxel_size) * xy_coord
)
context = Coordinate(input_shape - output_shape) // 2 * raw_test.voxel_size

# Read the raw data
# raise ValueError(center_slice, raw_test.roi)
raw_input = raw_test.to_ndarray(center_slice.grow(context, context))
raw_output = raw_test.to_ndarray(center_slice)
gt = labels_test.to_ndarray(center_slice)

# Predict on the validation data
with torch.no_grad():
    device = torch.device("cpu")
    module = module.to(device)
    pred = (
        module(
            (torch.from_numpy(raw_input).float() / 255.0)
            .to(device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        .cpu()
        .detach()
        .numpy()
    )
# %%
pred_labels = mws.agglom(pred[0].astype(np.float64) - 0.5, offsets=neighborhood)
# %%
# Plot the results
gif_2d(
    arrays={
        "Raw": Array(raw_output, voxel_size=raw_test.voxel_size),
        "GT": Array(gt % 256, voxel_size=raw_test.voxel_size),
        "Pred Affs": Array(pred[0][[0, 3, 4]], voxel_size=raw_test.voxel_size),
        "Pred": Array(pred_labels % 256, voxel_size=raw_test.voxel_size),
    },
    array_types={
        "Raw": "raw",
        "GT": "labels",
        "Pred Affs": "affs",
        "Pred": "labels",
    },
    filename="_static/dataset_tutorial/affs-prediction.gif",
    title="Prediction",
    fps=10,
)

# %%

extra = torch.tensor((2, 64, 64))
neighborhood = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0, 7, 0),
    (0, 0, 7),
    (0, 23, 0),
    (0, 0, 23),
    (0, 59, 0),
    (0, 0, 59),
]

train_dataset = iterable_dataset(
    datasets={"raw": raw_train, "gt": labels_train},
    shapes={"raw": unet.min_input_shape + extra, "gt": unet.min_output_shape + extra},
    transforms={
        "raw": torchvision.transforms.Lambda(lambda x: x[None].float() / 255.0),
        ("gt", "affs"): Affs(neighborhood=neighborhood),
        ("gt", "affs_mask"): AffsMask(neighborhood=neighborhood),
        (("affs", "affs_mask"), "weights"): BalanceLabels((1, -1, -1, -1)),
    },
    deform_augment_config=DeformAugmentConfig(
        p=0.1,
        control_point_spacing=(2, 10, 10),
        jitter_sigma=(0.5, 2, 2),
        rotate=True,
        subsample=4,
        rotation_axes=(1, 2),
        scale_interval=(1.0, 1.0),
    ),
    simple_augment_config=SimpleAugmentConfig(
        p=1.0,
        mirror_only=(1, 2),
        transpose_only=(1, 2),
    ),
)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cpu")

emb_dim = 36
unet = tems.UNet.funlib_api(
    dims=3,
    in_channels=1,
    num_fmaps=32,
    fmap_inc_factor=4,
    downsample_factors=[(1, 2, 2), (1, 2, 2), (1, 2, 2)],
    kernel_size_down=[
        [(1, 3, 3), (1, 3, 3)],
        [(1, 3, 3), (1, 3, 3)],
        [(1, 3, 3), (1, 3, 3)],
        [(1, 3, 3), (1, 3, 3)],
    ],
    kernel_size_up=[
        [(1, 3, 3), (1, 3, 3)],
        [(1, 3, 3), (1, 3, 3)],
        [(3, 3, 3), (3, 3, 3)],
    ],
    activation="LeakyReLU",
)

# this ensures we output the appropriate number of channels,
# use the appropriate final activation etc.
module = torch.nn.Sequential(unet, torch.nn.Conv3d(32, emb_dim, kernel_size=1)).to(
    device
)
# torch.nn.init.kaiming_normal_(module[1].weight, mode="fan_out", nonlinearity="sigmoid")


class DistanceHead(torch.nn.Module):
    def __init__(self, in_channels: int = 12, incr_factor: int = 12):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(
            in_channels, in_channels * incr_factor, kernel_size=1
        )
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv3d(in_channels * incr_factor, 1, kernel_size=1)
        self.sigmoid = torch.nn.Sigmoid()

        torch.nn.init.kaiming_normal_(
            self.conv1.weight, mode="fan_out", nonlinearity="relu"
        )
        torch.nn.init.kaiming_normal_(
            self.conv2.weight, mode="fan_out", nonlinearity="sigmoid"
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.conv2(self.relu(self.conv1(x - y)))).squeeze(1)


dist_heads = [
    DistanceHead(in_channels=emb_dim).to(device) for _ in range(len(neighborhood))
]
learned_affs = Affs(neighborhood=neighborhood, dist_func=dist_heads)

loss_func = torch.nn.BCELoss(reduction="none")
mse_loss = torch.nn.MSELoss()

import itertools

optimizer = torch.optim.Adam(
    list(module.parameters())
    + list(itertools.chain(*[dist_head.parameters() for dist_head in dist_heads])),
    lr=1e-4,
)
dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=3,
    num_workers=4,
)
losses = []

for iteration, batch in tqdm(enumerate(iter(dataloader))):
    raw, target, weight = (
        batch["raw"].to(device),
        batch["affs"].to(device),
        batch["weights"].to(device),
    )

    optimizer.zero_grad()

    emb = module(raw)
    pred_affs = learned_affs(emb, concat_dim=1)

    voxel_loss = loss_func(pred_affs, target.float()) * weight
    loss = voxel_loss.mean() + 0.001 * mse_loss(
        emb, emb / torch.norm(emb, dim=1, keepdim=True)
    )
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if iteration >= NUM_ITERATIONS:
        break

# %%
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.savefig("_static/dataset_tutorial/emb-loss-curve.png")
plt.close()

# %%
import mwatershed as mws
from funlib.geometry import Roi

input_shape = unet.min_input_shape + extra
output_shape = unet.min_output_shape + extra

# fetch a xy slice from the center of our validation volume
# We snap to grid to a multiple of the max downsampling factor of
# the unet (1, 8, 8) to ensure downsampling is always possible
roi = raw_test.roi
z_coord = Coordinate(1, 0, 0)
xy_coord = Coordinate(0, 1, 1)
center_offset = roi.center * z_coord + roi.offset * xy_coord + roi.shape // 4
center_size = raw_test.voxel_size * z_coord * 2 + (roi.shape * xy_coord) // 2
center_slice = Roi(center_offset, center_size)
center_slice = center_slice.snap_to_grid(raw_test.voxel_size * Coordinate(1, 8, 8))
center_slice.shape = (
    center_slice.shape
    - (Coordinate(unet.min_output_shape) * raw_test.voxel_size) * xy_coord
)
context = Coordinate(input_shape - output_shape) // 2 * raw_test.voxel_size

# Read the raw data
# raise ValueError(center_slice, raw_test.roi)
raw_input = raw_test.to_ndarray(center_slice.grow(context, context))
raw_output = raw_test.to_ndarray(center_slice)
gt = labels_test.to_ndarray(center_slice)

# Predict on the validation data
with torch.no_grad():
    device = torch.device("cpu")
    module = module.to(device)
    learned_affs = learned_affs.to(device)
    emb = module(
        (torch.from_numpy(raw_input).float() / 255.0).unsqueeze(0).unsqueeze(0)
    )
    pred = learned_affs(emb, concat_dim=1).cpu().detach().numpy()

# %%
# PCA
from sklearn.decomposition import PCA

# select the long range affinity channels for visualization
emb = emb.cpu().detach().numpy()[0]
emb -= emb.mean()
emb /= emb.std()

spatial_shape = emb.shape[1:]
emb = emb.reshape(emb.shape[0], -1)  # flatten the spatial dimensions
# Apply PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(emb.T)
principal_components = principal_components.T.reshape(3, *spatial_shape)

principal_components -= principal_components.min()
principal_components /= principal_components.max()
# %%
pred_labels = mws.agglom(pred[0].astype(np.float64) - 0.5, offsets=neighborhood)
# %%
# Plot the results
gif_2d(
    arrays={
        "Raw": Array(raw_output, voxel_size=raw_test.voxel_size),
        "GT": Array(gt % 256, voxel_size=raw_test.voxel_size),
        "Pred Emb": Array(principal_components, voxel_size=raw_test.voxel_size),
        "Pred Affs": Array(pred[0][[0, 3, 4]], voxel_size=raw_test.voxel_size),
        "Pred": Array(pred_labels % 256, voxel_size=raw_test.voxel_size),
    },
    array_types={
        "Raw": "raw",
        "GT": "labels",
        "Pred Emb": "raw",
        "Pred Affs": "affs",
        "Pred": "labels",
    },
    filename="_static/dataset_tutorial/emb-prediction.gif",
    title="Prediction",
    fps=10,
)
