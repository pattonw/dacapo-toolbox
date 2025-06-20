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

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    label_cmap = get_cmap()

    ims = []
    for ii in range(z_slices):
        slice_ims = []
        for jj, (key, arr) in enumerate(arrays.items()):
            roi = arr.roi.copy()
            roi.offset += Coordinate((ii,) + (0,) * (roi.dims - 1)) * arr.voxel_size
            roi.shape = Coordinate((arr.voxel_size[0], *roi.shape[1:]))
            # Show the raw data
            x = arr[roi][0]
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
            axes[jj].set_title(key)
            slice_ims.append(im)
        ims.append(slice_ims)

    ims = ims + ims[::-1]
    ani = animation.ArtistAnimation(fig, ims, blit=True, repeat_delay=1000)
    fig.suptitle(title, fontsize=16)
    ani.save(filename, writer="pillow", fps=fps)
    plt.close()
