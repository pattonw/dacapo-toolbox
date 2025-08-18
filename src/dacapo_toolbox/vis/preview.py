import matplotlib.pyplot as plt
from funlib.geometry import Coordinate
from matplotlib import animation
from matplotlib.colors import ListedColormap
import numpy as np
from funlib.persistence import Array
from sklearn.decomposition import PCA

from pathlib import Path


from matplotlib import colors as mcolors
from matplotlib import cm

SKIP_PLOTS = True


def pca_nd(emb: Array, n_components: int = 3) -> Array:
    emb_data = emb[:]
    num_channels, *spatial_shape = emb_data.shape

    emb_data = emb_data - emb_data.mean(
        axis=tuple(range(1, len(emb_data.shape))), keepdims=True
    )  # center the data
    emb_data /= (
        emb_data.std(axis=tuple(range(1, len(emb_data.shape))), keepdims=True) + 1e-4
    )  # normalize the data

    emb_data = emb_data.reshape(num_channels, -1)  # flatten the spatial dimensions
    # Apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(emb_data.T)
    principal_components = principal_components.T.reshape(n_components, *spatial_shape)

    principal_components -= principal_components.min(
        axis=tuple(range(1, n_components + 1)), keepdims=True
    )
    principal_components /= principal_components.max(
        axis=tuple(range(1, n_components + 1)), keepdims=True
    )
    return Array(
        principal_components,
        voxel_size=emb.voxel_size,
        offset=emb.offset,
        units=emb.units,
        axis_names=emb.axis_names,
        types=emb.types,
    )


def get_cmap(seed: int = 1) -> ListedColormap:
    np.random.seed(seed)
    colors = [[0, 0, 0]] + [
        list(np.random.choice(range(256), size=3) / 255.0) for _ in range(255)
    ]
    return ListedColormap(colors)


def gif_2d(
    arrays: dict[str, Array],
    array_types: dict[str, str],
    filename: str,
    title: str,
    fps: int = 10,
    overwrite: bool = False,
):
    if Path(filename).exists() and not overwrite:
        return
    transformed_arrays = {}
    for key, arr in arrays.items():
        assert arr.voxel_size.dims == 3, (
            f"Array {key} must be 3D, got {arr.voxel_size.dims}D"
        )
        if array_types[key] == "pca":
            transformed_arrays[key] = pca_nd(arr)
        else:
            transformed_arrays[key] = arr
    arrays = transformed_arrays

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
            scale_factor = shape[-2] // 256 if shape[-2] > 256 else 1
            # only show 256x256 pixels, more resolution not needed for gif
            if len(shape) == 2:
                x = x[::scale_factor, ::scale_factor]
            elif len(shape) == 3:
                x = x[:, ::scale_factor, ::scale_factor]
            else:
                raise ValueError("Array must be 2D with or without channels")
            if array_types[key] == "labels":
                im = axes[jj].imshow(
                    x % 256,
                    vmin=0,
                    vmax=255,
                    cmap=label_cmap,
                    interpolation="none",
                    animated=ii != 0,
                )
            elif array_types[key] == "raw" or array_types[key] == "pca":
                if x.ndim == 2:
                    im = axes[jj].imshow(
                        x,
                        cmap="grey",
                        animated=ii != 0,
                    )
                elif x.ndim == 3:
                    im = axes[jj].imshow(
                        x.transpose(1, 2, 0),
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


def cube(
    arrays: dict[str, Array],
    array_types: dict[str, str],
    filename: str,
    title: str,
    elev: float = 30,
    azim: float = -60,
    light_azdeg: float = 205,
    light_altdeg: float = 20,
    overwrite: bool = False,
):
    if Path(filename).exists() and not overwrite:
        return

    lightsource = mcolors.LightSource(azdeg=light_azdeg, altdeg=light_altdeg)

    transformed_arrays = {}
    for key, arr in arrays.items():
        assert arr.voxel_size.dims == 3, (
            f"Array {key} must be 3D, got {arr.voxel_size.dims}D"
        )
        if array_types[key] == "pca":
            transformed_arrays[key] = pca_nd(arr)
        elif array_types[key] == "labels":
            normalized = Array(
                arr.data % 256 / 255.0,
                voxel_size=arr.voxel_size,
                offset=arr.offset,
                units=arr.units,
                axis_names=arr.axis_names,
                types=arr.types,
            )
            transformed_arrays[key] = normalized
        elif array_types[key] == "raw":
            normalized = Array(
                (arr.data - arr.data.min()) / (arr.data.max() - arr.data.min()),
                voxel_size=arr.voxel_size,
                offset=arr.offset,
                units=arr.units,
                axis_names=arr.axis_names,
                types=arr.types,
            )
            transformed_arrays[key] = normalized
        else:
            transformed_arrays[key] = arr
    arrays = transformed_arrays

    fig, axes = plt.subplots(
        1,
        len(arrays),
        figsize=(2 + 5 * len(arrays), 6),
        subplot_kw={"projection": "3d"},
    )

    label_cmap = get_cmap()

    def draw_cube(ax, arr: Array, cmap=None, interpolation=None):
        assert arr.voxel_size.dims == 3, (
            f"Array {arr.name} must be 3D, got {arr.voxel_size.dims}D"
        )
        kwargs = {
            "interpolation": interpolation,
            "cmap": cmap,
        }

        z, y, x = tuple(
            np.linspace(start, stop, count)
            for start, stop, count in zip(
                arr.roi.begin, arr.roi.end, arr.roi.shape // arr.voxel_size
            )
        )
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

        face_colors = (
            cmap(arr[arr.roi])
            if cmap is not None
            else arr[arr.roi].transpose(1, 2, 3, 0)
        )

        kwargs = {
            "rcount": 256,
            "ccount": 256,
            "shade": True,
            "lightsource": lightsource,
        }

        _lz, ly, _lx = np.s_[0, :, :], np.s_[:, 0, :], np.s_[:, :, 0]
        uz, _uy, ux = np.s_[-1, :, :], np.s_[:, -1, :], np.s_[:, :, -1]
        # ax.plot_surface(xx[lx], yy[lx], zz[lx], facecolors=face_colors[lx], **kwargs)
        ax.plot_surface(xx[ux], yy[ux], zz[ux], facecolors=face_colors[ux], **kwargs)
        ax.plot_surface(xx[ly], yy[ly], zz[ly], facecolors=face_colors[ly], **kwargs)
        # ax.plot_surface(xx[uy], yy[uy], zz[uy], facecolors=face_colors[uy], **kwargs)
        # ax.plot_surface(xx[lz], yy[lz], zz[lz], facecolors=face_colors[lz], **kwargs)
        ax.plot_surface(xx[uz], yy[uz], zz[uz], facecolors=face_colors[uz], **kwargs)

        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])
        ax.set_zlim(z[0], z[-1])
        ax.set_box_aspect(arr.roi.shape[::-1])

        ax.axis("off")

    for jj, (key, arr) in enumerate(arrays.items()):
        ax = axes[jj]

        if array_types[key] == "labels":
            draw_cube(ax, arr, cmap=label_cmap, interpolation="none")
        elif array_types[key] == "raw" or array_types[key] == "pca":
            if arr.data.ndim == 3:
                draw_cube(ax, arr, cmap=cm.gray)
            elif arr.data.ndim == 4:
                draw_cube(ax, arr)
        elif array_types[key] == "affs":
            # Show the affinities
            draw_cube(ax, arr, interpolation="none")

        ax.set_title(key)
        # Without this line, the default cube view is elev = 30, azim = -60.
        ax.view_init(elev=elev, azim=azim)

    fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
