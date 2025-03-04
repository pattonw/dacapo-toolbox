# %% [markdown]
# # Simple Examples
# This tutorial goes through a few common ML tasks using the `cremi dataset <https://cremi.org/data/>` and a *2D U-Net*.


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
# ```bash
# uv init
# uv pip install git+https://github.com/pattonw/dacapo-toolbox.git
# ```

# %% [markdown]
# ## Data Preparation
# DaCapo works with zarr, so we will download [CREMI data](https://cremi.org/static/data/sample_A%2B_20160601.hdf)
# and save it as a zarr file.

# %%
!curl -o sample_A_20160501.hdf https://cremi.org/static/data/sample_A_20160501.hdf

# %% Create some data
import numpy as np
from funlib.geometry import Coordinate, Roi
from funlib.persistence import prepare_ds, open_ds
import h5py
from pathlib import Path

# %%
if not Path("cremi.zarr").exists():
    h = h5py.File("sample_A_20160501.hdf", "r")
    raw_data = h["volumes/raw"][:]
    labels_data = h["volumes/labels/neuron_ids"][:]
    raw = prepare_ds("cremi.zarr/train/raw", raw_data.shape, voxel_size=(40, 4, 4), dtype=raw_data.dtype, axis_names=["z","y","x"], units=["nm", "nm", "nm"])
    raw[raw.roi] = raw_data
    labels = prepare_ds("cremi.zarr/train/labels", labels_data.shape, voxel_size=(40, 4, 4), dtype=labels_data.dtype, axis_names=["z","y","x"], units=["nm", "nm", "nm"])
    labels[labels.roi] = labels_data
else:
    raw = open_ds("cremi.zarr/train/raw")
    labels = open_ds("cremi.zarr/train/labels")

# %% [markdown]
# Here we show a slice of the raw data:
# %%
# a custom label color map for showing instances
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Create a custom label color map for showing instances
np.random.seed(1)
colors = [[0, 0, 0]] + [list(np.random.choice(range(256), size=3)) for _ in range(254)]
label_cmap = ListedColormap(colors)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Show the raw data
axes[0].imshow(raw.data[30])
axes[0].set_title("Raw Data")

# Show the labels using the custom label color map
axes[1].imshow(labels.data[30], cmap=label_cmap, interpolation="none")
axes[1].set_title("Labels")

plt.show()



# %%
from dacapo_toolbox.convenience import dataset_from_zarr
import torch

dataset: torch.utils.data.IterableDataset = dataset_from_zarr(
    zarr_container="cremi.zarr",
    input_shape=(1, 132, 132),
    output_shape=(1, 132, 132),
    augments=[],
    task="instance",
    mode="train",
)

# %%

torch.utils.data.DataLoader(dataset, batch_size=5, num_workers=3)
batch = next(iter(dataset))
fig, axes = plt.subplots(1, 3, figsize=(12, 6))

# Show the raw data
axes[0].imshow(batch["raw"][0,0])
axes[0].set_title("Raw Data")

axes[1].imshow(batch["target"][(0, 5, 6), 0].permute((1,2,0)))
axes[1].set_title("Affs")

# Show the labels using the custom label color map
axes[2].imshow(batch["gt"][0], cmap=label_cmap, interpolation="none")
axes[2].set_title("Labels")

plt.show()
# %%
