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
import numpy as np
from funlib.persistence import open_ds, Array
from pathlib import Path

voxel_size = (40, 4, 4)  # in nm
axis_names = ["z", "y", "x"]
units = ["nm", "nm", "nm"]

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
# gif_2d(
#     arrays={"Train Raw": raw_train, "Train Labels": labels_train},
#     array_types={"Train Raw": "raw", "Train Labels": "labels"},
#     filename="_static/dataset_tutorial/training-data.gif",
#     title="Training Data",
#     fps=10,
# )

# %% [markdown]
# Here we visualize the training data:
# ![training-data](_static/dataset_tutorial/training-data.gif)

# %% [markdown]
# ### Testing data

# %%
# gif_2d(
#     arrays={"Test Raw": raw_test, "Test Labels": labels_test},
#     array_types={"Test Raw": "raw", "Test Labels": "labels"},
#     filename="_static/dataset_tutorial/testing-data.gif",
#     title="Testing Data",
#     fps=10,
# )

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
    # deform_augment_config=DeformAugmentConfig(
    #     p=0.5,
    #     control_point_spacing=(40, 40, 40),
    #     jitter_sigma=(10, 10, 10),
    #     rotate=True,
    #     subsample=4,
    #     rotation_axes=(0, 1, 2),
    #     scale_interval=(0.8, 1.2),
    # ),  # TODO: Fix this!
    simple_augment_config=SimpleAugmentConfig(
        p=1.0,
        mirror_only=(1, 2),
        transpose_only=(1, 2),
    ),
)
batch_gen = iter(train_dataset)

# %%
batch = next(batch_gen)
# gif_2d(
#     arrays={
#         "Raw": Array(batch["raw"].numpy()),
#         "Labels": Array(batch["gt"].numpy() % 256),
#     },
#     array_types={"Raw": "raw", "Labels": "labels"},
#     filename="_static/dataset_tutorial/simple-batch.gif",
#     title="Simple Batch",
#     fps=10,
# )

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
    },
    simple_augment_config=SimpleAugmentConfig(
        p=1.0,
        mirror_only=(1, 2),
        transpose_only=(1, 2),
    ),
)

batch_gen = iter(train_dataset)

# %%
from funlib.persistence import Array
import matplotlib.pyplot as plt
from funlib.geometry import Coordinate
from matplotlib import animation
from matplotlib.colors import ListedColormap
import numpy as np


def get_cmap(seed: int = 1) -> ListedColormap:
    np.random.seed(seed)
    colors = [[0, 0, 0]] + [
        list(np.random.choice(range(256), size=3)) for _ in range(255)
    ]
    return ListedColormap(colors)


def gif_2d(
    arrays: dict[str, Array],
    array_types: dict[str, str],
    filename: str,
    title: str,
    fps: int = 10,
):
    for key, arr in arrays.items():
        assert arr.voxel_size.dims == 3, (
            f"Array {key} must be 3D, got {arr.voxel_size.dims}D"
        )

    z_slices = None
    for arr in arrays.values():
        arr_z_slices = arr.roi.shape[0] // arr.voxel_size[0]
        if z_slices is None:
            z_slices = arr_z_slices
        else:
            assert z_slices == arr_z_slices, (
                f"All arrays must have the same number of z slices, "
                f"got {z_slices} and {arr_z_slices}"
            )

    fig, axes = plt.subplots(1, len(arrays), figsize=(2 + 5 * len(arrays), 6))

    label_cmap = get_cmap()

    ims = []
    for ii in range(z_slices):
        slice_ims = []
        for jj, (key, arr) in enumerate(arrays.items()):
            roi = arr.roi.copy()
            roi.offset += Coordinate((ii,) + (0,) * (roi.dims - 1)) * arr.voxel_size
            roi.shape = Coordinate((arr.voxel_size[0], *roi.shape[1:]))
            # Show the raw data
            x = arr[roi].squeeze(-arr.voxel_size.dims)  # squeeze out z dim
            shape = x.shape
            scale_factor = shape[0] // 256 if shape[0] > 256 else 1
            # only show 256x256 pixels, more resolution not needed for gif
            x = x[::scale_factor, ::scale_factor]
            if array_types[key] == "labels":
                im = axes[jj].imshow(
                    x % 256,
                    vmin=0,
                    vmax=255,
                    cmap=label_cmap,
                    interpolation="none",
                    animated=ii != 0,
                )
            elif array_types[key] == "raw":
                im = axes[jj].imshow(
                    x,
                    cmap="grey",
                    animated=ii != 0,
                )
            elif array_types[key] == "affs":
                # Show the affinities
                im = axes[jj].imshow(
                    x.transpose(1, 2, 0),
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="none",
                    animated=ii != 0,
                )
            axes[jj].set_title(key)
            slice_ims.append(im)
        ims.append(slice_ims)

    ims = ims + ims[::-1]
    ani = animation.ArtistAnimation(fig, ims, blit=True, repeat_delay=1000)
    fig.suptitle(title, fontsize=16)
    ani.save(filename, writer="pillow", fps=fps)
    plt.close()


# %%
batch = next(batch_gen)
# gif_2d(
#     arrays={
#         "Raw": Array(batch["raw"].numpy(), voxel_size=raw_train.voxel_size),
#         "GT": Array(batch["gt"].numpy() % 256, voxel_size=raw_train.voxel_size),
#         "Affs": Array(
#             batch["affs"].float().numpy()[[0, 3, 4]], voxel_size=raw_train.voxel_size
#         ),
#         "Affs Mask": Array(
#             batch["affs_mask"].float().numpy()[[0, 3, 4]],
#             voxel_size=raw_train.voxel_size,
#         ),
#     },
#     array_types={
#         "Raw": "raw",
#         "GT": "labels",
#         "Affs": "affs",
#         "Affs Mask": "affs",
#     },
#     filename="_static/dataset_tutorial/affs-batch.gif",
#     title="Affinities Batch",
#     fps=10,
# )

# %% [markdown]
# Here we visualize a batch with (raw, gt, target) triplets for the affinities task:
# ![affs-batch](_static/dataset_tutorial/affs-batch.gif)

# %% [markdown]
# ### Models
# Lets define our model

# %%
import tems

input_shape = Coordinate((5, 156, 156))

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
)

print(
    f"Given input shape: {unet.min_input_shape}, the output shape will be: {unet.min_output_shape}"
)

# %% [markdown]
# ### Training loop
# Now we can bring everything together and train our model.

# %%
from tqdm import tqdm
import time
# %%
import torch

extra = torch.tensor((2, 64, 64))
train_dataset = iterable_dataset(
    datasets={"raw": raw_train, "gt": labels_train},
    shapes={"raw": unet.min_input_shape + extra, "gt": unet.min_output_shape + extra},
    transforms={
        "raw": torchvision.transforms.Lambda(lambda x: x[None].float() / 255.0),
        ("gt", "affs"): Affs(neighborhood=neighborhood),
        ("gt", "affs_mask"): AffsMask(neighborhood=neighborhood),
    },
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

# this ensures we output the appropriate number of channels,
# use the appropriate final activation etc.
module = torch.nn.Sequential(
    unet, torch.nn.Conv3d(32, 7, kernel_size=1), torch.nn.Sigmoid()
).to(device)
loss_func = torch.nn.BCELoss(reduction="none")
optimizer = torch.optim.Adam(module.parameters(), lr=1e-4)
dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=3,
    num_workers=2,
)
losses = []

for iteration, batch in tqdm(enumerate(iter(dataloader))):
    raw, target, weight = (
        batch["raw"].to(device),
        batch["affs"].to(device),
        batch["affs_mask"].to(device),
    )
    optimizer.zero_grad()
    
    t1 =time.time()
    output = module(raw)
    print("processing:", time.time() - t1)

    t2 = time.time()
    voxel_loss = loss_func(output, target.float())
    loss = (voxel_loss * weight).mean()
    loss.backward()
    optimizer.step()
    print("Optimizing:", time.time() - t2)

    losses.append(loss.item())

    if iteration >= 800:
        break

# %%
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.show()

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
pred_labels = (
    mws.agglom(pred[0].astype(np.float64) - 0.5, offsets=neighborhood)
)
# %%
# Plot the results
gif_2d(
    arrays={
        "Raw": Array(raw_output, voxel_size=raw_test.voxel_size),
        "GT": Array(gt % 256, voxel_size=raw_test.voxel_size),
        "Pred Affs": Array(
            pred[0][[0,3,4]], voxel_size=raw_test.voxel_size
        ),
        "Pred": Array(pred_labels % 256, voxel_size=raw_test.voxel_size),
    },
    array_types={"Raw": "raw", "GT": "labels", "Pred Affs": "affs", "Pred": "labels"},
    filename="_static/dataset_tutorial/prediction.gif",
    title="Prediction",
    fps=10,
)
# %%
