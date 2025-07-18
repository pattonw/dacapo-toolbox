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

from pathlib import Path
from tqdm import tqdm

from funlib.persistence import Array
from funlib.geometry import Coordinate
from dacapo_toolbox.sample_datasets import cremi

NUM_ITERATIONS = 300
if not Path("_static/dataset_tutorial").exists():
    Path("_static/dataset_tutorial").mkdir(parents=True, exist_ok=True)

raw_train, labels_train, raw_test, labels_test = cremi(Path("cremi.zarr"))


# %% [markdown]
# Lets visualize our train and test data

# %% [markdown]
# ### Training data

# %%
from dacapo_toolbox.vis.preview import gif_2d, cube

# %%

# create a 2D gif of the training data
gif_2d(
    arrays={"Train Raw": raw_train, "Train Labels": labels_train},
    array_types={"Train Raw": "raw", "Train Labels": "labels"},
    filename="_static/dataset_tutorial/training-data.gif",
    title="Training Data",
    fps=10,
)
cube(
    arrays={"Train Raw": raw_train, "Train Labels": labels_train},
    array_types={"Train Raw": "raw", "Train Labels": "labels"},
    filename="_static/dataset_tutorial/training-data.jpg",
    title="Training Data",
)

# %% [markdown]
# Here we visualize the training data:
# ![training-data](_static/dataset_tutorial/training-data.gif)
# ![training-data-cube](_static/dataset_tutorial/training-data.jpg)

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
cube(
    arrays={"Test Raw": raw_test, "Test Labels": labels_test},
    array_types={"Test Raw": "raw", "Test Labels": "labels"},
    filename="_static/dataset_tutorial/testing-data.jpg",
    title="Testing Data",
)

# %% [markdown]
# Here we visualize the test data:
# ![test-data](_static/dataset_tutorial/test-data.gif)
# ![test-data-cube](_static/dataset_tutorial/test-data.jpg)

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
        "Raw": Array(batch["raw"].numpy(), voxel_size=raw_train.voxel_size),
        "Labels": Array(batch["gt"].numpy(), voxel_size=labels_train.voxel_size),
    },
    array_types={"Raw": "raw", "Labels": "labels"},
    filename="_static/dataset_tutorial/simple-batch.gif",
    title="Simple Batch",
    fps=10,
)
cube(
    arrays={
        "Raw": Array(batch["raw"].numpy(), voxel_size=raw_train.voxel_size),
        "Labels": Array(batch["gt"].numpy(), voxel_size=labels_train.voxel_size),
    },
    array_types={"Raw": "raw", "Labels": "labels"},
    filename="_static/dataset_tutorial/simple-batch.jpg",
    title="Simple Batch",
)

# %% [markdown]
# Here we visualize the training data:
# ![simple-batch](_static/dataset_tutorial/simple-batch.gif)
# ![simple-batch-cube](_static/dataset_tutorial/simple-batch.jpg)


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
cube(
    arrays={
        "Raw": Array(batch["raw"].numpy(), voxel_size=raw_train.voxel_size),
        "GT": Array(batch["gt"].numpy(), voxel_size=raw_train.voxel_size),
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
    filename="_static/dataset_tutorial/affs-batch.jpg",
    title="Affinities Batch",
)

# %% [markdown]
# Here we visualize a batch with (raw, gt, target) triplets for the affinities task:
# ![affs-batch](_static/dataset_tutorial/affs-batch.gif)
# ![affs-batch-cube](_static/dataset_tutorial/affs-batch.jpg)

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
plt.show()
plt.close()

# %%
import mwatershed as mws
from funlib.geometry import Roi
import numpy as np

module = module.eval()
unet = unet.eval()
test_roi = raw_test.roi
vs = raw_test.voxel_size
sf = Coordinate(unet.equivariant_step) * vs
eval_roi_shape = test_roi.shape // 4
growth = eval_roi_shape // sf * sf
eval_roi_shape = Coordinate(unet.min_output_shape) * vs + growth
eval_roi = Roi(test_roi.center - eval_roi_shape // (vs * 2) * vs, eval_roi_shape)
context = Coordinate(unet.context // 2) * vs

raw_input = raw_test.to_ndarray(eval_roi.grow(context, context))
raw_output = raw_test.to_ndarray(eval_roi)
gt = labels_test.to_ndarray(eval_roi)

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
cube(
    arrays={
        "Raw": Array(raw_output, voxel_size=raw_test.voxel_size),
        "GT": Array(gt, voxel_size=raw_test.voxel_size),
        "Pred Affs": Array(pred[0][[0, 3, 4]], voxel_size=raw_test.voxel_size),
        "Pred": Array(pred_labels, voxel_size=raw_test.voxel_size),
    },
    array_types={
        "Raw": "raw",
        "GT": "labels",
        "Pred Affs": "affs",
        "Pred": "labels",
    },
    filename="_static/dataset_tutorial/affs-prediction.jpg",
    title="Prediction",
)

# %%

from volara.datasets import (
    Raw as RawDataset,
    Affs as AffsDataset,
    Labels as LabelsDataset,
)
from volara.dbs import SQLite
from volara.lut import LUT
from volara.blockwise import ExtractFrags, AffAgglom, GraphMWS, Relabel
from volara_torch.blockwise import Predict
from volara_torch.models import TorchModel

raw_dataset = RawDataset(store="cremi.zarr/test/raw", scale_shift=(1.0 / 255.0, 0.0))
affs_dataset = AffsDataset(store="cremi.zarr/test/affs", neighborhood=neighborhood)
frags_dataset = LabelsDataset(store="cremi.zarr/test/frags")
labels_dataset = LabelsDataset(store="cremi.zarr/test/pred_labels")

frag_db = SQLite(
    path="cremi.zarr/frags.db",
    edge_attrs={"affs_xy": "float", "affs_z": "float", "affs_long_xy": "float"},
)
segment_lut = LUT(
    path="cremi.zarr/frag-lut",
)

unet = unet.eval()
scripted_unet = torch.jit.script(module)
torch.jit.save(scripted_unet, "cremi.zarr/affs_unet.pt")
torch.save(scripted_unet.state_dict(), "cremi.zarr/affs_unet_state.pt")

pred_size_growth = Coordinate(32, 256, 256)
blocksize = Coordinate(unet.min_output_shape) + pred_size_growth
affs_model = TorchModel(
    in_channels=1,
    min_input_shape=unet.min_input_shape,
    min_output_shape=unet.min_output_shape,
    min_step_shape=unet.equivariant_step,
    out_channels=len(neighborhood),
    out_range=(0.0, 1.0),
    save_path="cremi.zarr/affs_unet.pt",
    checkpoint_file="cremi.zarr/affs_unet_state.pt",
    pred_size_growth=pred_size_growth,
)

predict_affs = Predict(
    checkpoint=affs_model,
    in_data=raw_dataset,
    out_data=[affs_dataset],
    num_workers=1,
)

extract_frags = ExtractFrags(
    db=frag_db,
    affs_data=affs_dataset,
    frags_data=frags_dataset,
    block_size=blocksize,
    context=Coordinate(16, 16, 16),
    bias=[-0.5, -0.2, -0.2, -0.5, -0.5, -0.8, -0.8],
    remove_debris=32,
    num_workers=4,
)
aff_agglom = AffAgglom(
    db=frag_db,
    affs_data=affs_dataset,
    frags_data=frags_dataset,
    block_size=blocksize,
    context=Coordinate(16, 16, 16),
    scores={
        "affs_z": [Coordinate(1, 0, 0)],
        "affs_xy": [Coordinate(0, 1, 0), Coordinate(0, 0, 1)],
        "affs_long_xy": [
            Coordinate(0, 7, 0),
            Coordinate(0, 0, 7),
            Coordinate(0, 23, 0),
            Coordinate(0, 0, 23),
        ],
    },
    num_workers=4,
)
graph_mws = GraphMWS(
    db=frag_db,
    lut=segment_lut,
    weights={
        "affs_z": (1.0, -0.5),
        "affs_xy": (1.0, -0.2),
        "affs_long_xy": (1.0, -0.8),
    },
    roi=(raw_test.roi.offset, raw_test.roi.shape),
)

relabel = Relabel(
    frags_data=frags_dataset,
    seg_data=labels_dataset,
    lut=segment_lut,
    block_size=blocksize,
    num_workers=4,
)

# pipeline = predict_affs + extract_frags + aff_agglom + graph_mws + relabel
predict_affs.drop()
predict_affs.run_blockwise(multiprocessing=False)
# %%
extract_frags.drop()
extract_frags.run_blockwise(multiprocessing=True)
# %%
aff_agglom.drop()
aff_agglom.run_blockwise(multiprocessing=True)
# %%
graph_mws.drop()
graph_mws.run_blockwise(multiprocessing=False)
# %%
relabel.drop()
relabel.run_blockwise(multiprocessing=True)

# %%

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


extra = torch.tensor((2, 64, 64))

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
plt.show()
plt.close()

# %%
import mwatershed as mws
from funlib.geometry import Roi

# Read the raw data
# raise ValueError(center_slice, raw_test.roi)
module = module.eval()
unet = unet.eval()
raw_input = raw_test.to_ndarray(eval_roi.grow(context, context))
raw_output = raw_test.to_ndarray(eval_roi)
gt = labels_test.to_ndarray(eval_roi)

# Predict on the validation data
with torch.no_grad():
    device = torch.device("cpu")
    module = module.to(device)
    learned_affs = learned_affs.to(device)
    emb = module(
        (torch.from_numpy(raw_input).float() / 255.0).unsqueeze(0).unsqueeze(0)
    )
    pred = learned_affs(emb, concat_dim=1).cpu().detach().numpy()
    pred_labels = mws.agglom(pred[0].astype(np.float64) - 0.5, offsets=neighborhood)

# %%
# Plot the results
gif_2d(
    arrays={
        "Raw": Array(raw_output, voxel_size=raw_test.voxel_size),
        "GT": Array(gt, voxel_size=raw_test.voxel_size),
        "Pred Emb": Array(
            emb.cpu().detach().numpy()[0], voxel_size=raw_test.voxel_size
        ),
        "Pred Affs": Array(pred[0][[0, 3, 4]], voxel_size=raw_test.voxel_size),
        "Pred": Array(pred_labels, voxel_size=raw_test.voxel_size),
    },
    array_types={
        "Raw": "raw",
        "GT": "labels",
        "Pred Emb": "pca",
        "Pred Affs": "affs",
        "Pred": "labels",
    },
    filename="_static/dataset_tutorial/emb-prediction.gif",
    title="Prediction",
    fps=10,
)
cube(
    arrays={
        "Raw": Array(raw_output, voxel_size=raw_test.voxel_size),
        "GT": Array(gt, voxel_size=raw_test.voxel_size),
        "Pred Emb": Array(
            emb.cpu().detach().numpy()[0], voxel_size=raw_test.voxel_size
        ),
        "Pred Affs": Array(pred[0][[0, 3, 4]], voxel_size=raw_test.voxel_size),
        "Pred": Array(pred_labels, voxel_size=raw_test.voxel_size),
    },
    array_types={
        "Raw": "raw",
        "GT": "labels",
        "Pred Emb": "pca",
        "Pred Affs": "affs",
        "Pred": "labels",
    },
    filename="_static/dataset_tutorial/emb-prediction.jpg",
    title="Prediction",
)

# %%
from volara.datasets import (
    Raw as RawDataset,
    Affs as AffsDataset,
    Labels as LabelsDataset,
)
from volara.dbs import SQLite
from volara.lut import LUT
from volara.blockwise import ExtractFrags, AffAgglom, GraphMWS, Relabel
from volara_torch.blockwise import Predict
from volara_torch.models import TorchModel

raw_dataset = RawDataset(store="cremi.zarr/test/raw", scale_shift=(1.0 / 255.0, 0.0))
emb_dataset = RawDataset(store="cremi.zarr/test/emb")
affs_dataset = AffsDataset(store="cremi.zarr/test/emb_affs", neighborhood=neighborhood)
frags_dataset = LabelsDataset(store="cremi.zarr/test/emb_frags")
labels_dataset = LabelsDataset(store="cremi.zarr/test/emb_pred_labels")

emb_db = SQLite(
    path="cremi.zarr/emb.db",
    edge_attrs={"affs_xy": "float", "affs_z": "float", "affs_long_xy": "float"},
)
emb_lut = LUT(
    path="cremi.zarr/emb-lut",
)

unet = unet.eval()
scripted_unet = torch.jit.script(module)
torch.jit.save(scripted_unet, "cremi.zarr/emb_unet.pt")
torch.save(scripted_unet.state_dict(), "cremi.zarr/emb_unet_state.pt")
min_output_shape = Coordinate(unet.min_output_shape)
emb_model = TorchModel(
    in_channels=1,
    min_input_shape=unet.min_input_shape,
    min_output_shape=unet.min_output_shape,
    min_step_shape=unet.equivariant_step,
    out_channels=emb_dim,
    out_range=(0.0, 1.0),
    save_path="cremi.zarr/emb_unet.pt",
    checkpoint_file="cremi.zarr/emb_unet_state.pt",
    pred_size_growth=pred_size_growth,
)

min_output_shape = Coordinate(1, 1, 1)
min_input_shape = min_output_shape + Coordinate(1, 23, 23) * 2
torch.save(learned_affs, "cremi.zarr/learned_affs.pt")
affs_model = TorchModel(
    in_channels=emb_dim,
    min_input_shape=min_output_shape + Coordinate(1, 23, 23) * 2,
    min_output_shape=min_output_shape,
    min_step_shape=Coordinate(1, 1, 1),
    out_channels=len(neighborhood),
    out_range=(0.0, 1.0),
    save_path="cremi.zarr/learned_affs.pt",
    pred_size_growth=pred_size_growth,
)

predict_emb = Predict(
    checkpoint=emb_model,
    in_data=raw_dataset,
    out_data=[emb_dataset],
)
predict_affs = Predict(
    checkpoint=affs_model,
    in_data=emb_dataset,
    out_data=[affs_dataset],
)

extract_frags = ExtractFrags(
    db=emb_db,
    affs_data=affs_dataset,
    frags_data=frags_dataset,
    block_size=blocksize,
    context=Coordinate(16, 16, 16),
    bias=[-0.5, -0.2, -0.2, -0.5, -0.5, -0.8, -0.8],
    remove_debris=32,
)
aff_agglom = AffAgglom(
    db=emb_db,
    affs_data=affs_dataset,
    frags_data=frags_dataset,
    block_size=blocksize,
    context=Coordinate(16, 16, 16),
    scores={
        "affs_z": [Coordinate(1, 0, 0)],
        "affs_xy": [Coordinate(0, 1, 0), Coordinate(0, 0, 1)],
        "affs_long_xy": [Coordinate(0, 7, 0), Coordinate(0, 0, 7)],
    },
)
graph_mws = GraphMWS(
    db=emb_db,
    lut=emb_lut,
    weights={
        "affs_z": (1.0, -0.5),
        "affs_xy": (1.0, -0.2),
        "affs_long_xy": (1.0, -0.8),
    },
    roi=(raw_test.roi.offset, raw_test.roi.shape),
)

relabel = Relabel(
    frags_data=frags_dataset,
    seg_data=labels_dataset,
    lut=emb_lut,
    block_size=blocksize,
)

# pipeline = predict_emb + predict_affs + extract_frags + aff_agglom + graph_mws + relabel
predict_emb.drop()
predict_emb.run_blockwise(multiprocessing=False)
# %%
predict_affs.drop()
predict_affs.run_blockwise(multiprocessing=False)
# %%
extract_frags.drop()
extract_frags.run_blockwise(multiprocessing=False)
# %%
aff_agglom.drop()
aff_agglom.run_blockwise(multiprocessing=False)
# %%
graph_mws.drop()
graph_mws.run_blockwise(multiprocessing=False)
# %%
relabel.drop()
relabel.run_blockwise(multiprocessing=False)



