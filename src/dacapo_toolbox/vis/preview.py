import matplotlib.pyplot as plt
from funlib.geometry import Coordinate, Roi
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


def pca_nd_faces(faces: list[np.ndarray], n_components: int = 3) -> list[np.ndarray]:
    flattened_faces = [fc.reshape(fc.shape[0], -1) for fc in faces]
    emb_data = np.concatenate(flattened_faces, axis=1)

    scale = emb_data.std(axis=1, keepdims=True) + 1e-4
    shift = emb_data.mean(axis=1, keepdims=True)

    emb_data = emb_data - shift  # center the data
    emb_data /= scale  # normalize the data

    # Apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(emb_data.T)

    min_val = principal_components.min(axis=0)
    max_val = principal_components.max(axis=0)

    principal_faces = []
    for fc in faces:
        in_face = (fc.reshape(fc.shape[0], -1) - shift) / scale
        principal_components = pca.transform(in_face.T).T.reshape(
            n_components, *fc.shape[1:]
        )

        principal_components -= min_val.reshape(n_components, 1, 1)
        principal_components /= (max_val - min_val).reshape(n_components, 1, 1)
        principal_faces.append(principal_components)
    return principal_faces


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
    overwrite: bool = True,
    dpi: int = 72,
    max_size: int = 256,
    optimize_gif: bool = False,
    frame_skip: int = 2,
):
    """
    Create a 2D GIF preview of the given arrays.

    Parameters
    ----------
    arrays : dict[str, Array]
        A dictionary of named arrays to visualize. Each array must be 3D,
        with optional channels.
    array_types : dict[str, str]
        A dictionary specifying the type of each array. Supported types are:
        - "raw": Grayscale raw data.
        - "labels": Integer labels, visualized with a random color map.
        - "affs": Affinity graphs, visualized as RGB images.
        - "pca": High-dimensional data, visualized using PCA to reduce to 3 channels.
    filename : str
        The output filename for the GIF.
    title : str
        The title to display on the GIF.
    fps : int, optional
        Frames per second for the GIF. Default is 10.
    overwrite : bool, optional
        Whether to overwrite the output file if it already exists. Default is True.
    dpi : int, optional
        Dots per inch for the output GIF. Default is 72.
    max_size : int, optional
        Maximum size (in pixels) for the largest dimension of the images. Default is 256.
    optimize_gif : bool, optional
        Whether to optimize the GIF for smaller file size. Default is False.
    frame_skip : int, optional
        Number of frames to skip when generating the GIF for faster creation. Default is 2.
    """
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

    z_arr_slices = [arr.roi.shape[0] // arr.voxel_size[0] for arr in arrays.values()]
    z_slices = min(z_arr_slices)
    assert z_slices == max(z_arr_slices), (
        f"All arrays must have the same number of z slices, got {z_arr_slices}"
    )

    fig, axes = plt.subplots(1, len(arrays), figsize=(2 + 3 * len(arrays), 4), dpi=dpi)

    label_cmap = get_cmap()

    ims = []
    for ii in range(0, z_slices, frame_skip):  # Skip frames for faster GIF
        slice_ims = []
        for jj, (key, arr) in enumerate(arrays.items()):
            roi = arr.roi.copy()
            roi.offset += Coordinate((ii,) + (0,) * (roi.dims - 1)) * arr.voxel_size
            roi.shape = Coordinate((arr.voxel_size[0], *roi.shape[1:]))
            # Show the raw data
            x = arr[roi].squeeze(-arr.voxel_size.dims)  # squeeze out z dim
            shape = x.shape
            # Limit resolution for GIF - smaller is faster
            scale_factor = max(shape[-2] // max_size, 1) if shape[-2] > max_size else 1
            # only show max_size pixels, more resolution not needed for gif
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

    # Use optimized writer settings for faster GIF creation
    from matplotlib.animation import PillowWriter

    if optimize_gif:
        writer = PillowWriter(fps=fps)
    else:
        writer = PillowWriter(fps=fps, metadata=dict(artist="dacapo"), bitrate=1800)

    ani.save(filename, writer=writer, dpi=dpi)
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
    overwrite: bool = True,
    rcount: int = 128,
    ccount: int = 128,
    shade: bool = True,
    dpi: int = 100,
):
    """
    Preview 3D arrays as cubes with matplotlib. Arrays do not need to be the same size
    their relative sizes and shifts will be respected.

    Parameters
    ----------
    arrays : dict[str, Array]
        A dictionary of named arrays to visualize. Each array must be 3D,
        with optional channels.
    array_types : dict[str, str]
        A dictionary specifying the type of each array. Supported types are:
        - "raw": Grayscale raw data.
        - "labels": Integer labels, visualized with a random color map.
        - "affs": Affinity graphs, visualized as RGB images.
        - "pca": High-dimensional data, visualized using PCA to reduce to 3
            channels.
    filename : str
        The output filename for the image.
    title : str
        The title to display on the image.
    elev : float, optional
        Elevation angle for the 3D view. Default is 30.
    azim : float, optional
        Azimuth angle for the 3D view. Default is -60.
    light_azdeg : float, optional
        Azimuth angle for the light source. Default is 205.
    light_altdeg : float, optional
        Altitude angle for the light source. Default is 20.
    overwrite : bool, optional
        Whether to overwrite the output file if it already exists. Default is True.
    rcount : int, optional
        Number of rows for the surface plot. Default is 128.
    ccount : int, optional
        Number of columns for the surface plot. Default is 128.
    shade : bool, optional
        Whether to shade the surface plot. Default is True.
    dpi : int, optional
        Dots per inch for the output image. Default is 100.
    """
    if Path(filename).exists() and not overwrite:
        return

    total_roi = None
    for arr in arrays.values():
        if total_roi is None:
            total_roi = arr.roi
        else:
            total_roi = total_roi.union(arr.roi)
    assert isinstance(total_roi, Roi)

    lightsource = mcolors.LightSource(azdeg=light_azdeg, altdeg=light_altdeg)

    def read_faces(
        arrays: dict[str, Array],
    ) -> dict[str, tuple[list[np.ndarray], list[np.ndarray]]]:
        faces = {}
        for name, arr in arrays.items():
            assert arr.voxel_size.dims == 3, (
                f"Array {name} must be 3D, got {arr.voxel_size.dims}D"
            )
            lower = arr.roi.offset
            upper = lower + arr.roi.shape - arr.voxel_size
            shape = arr.roi.shape
            vshape = shape / arr.voxel_size
            slice_thickness = arr.voxel_size

            z, y, x = tuple(
                range(start, stop, step)
                for start, stop, step in zip(arr.roi.begin, arr.roi.end, arr.voxel_size)
            )

            def face(high: bool, axis: int) -> Roi:
                a = Coordinate(axis == 0, axis == 1, axis == 2)
                b = Coordinate(axis != 0, axis != 1, axis != 2)
                base = upper if high else lower
                return Roi(base * a + lower * b, shape * b + slice_thickness * a)

            def face_coords(
                high: bool, axis: int
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                if axis == 0:
                    zz = np.ones((vshape[1], vshape[2])) * (
                        upper[0] if high else lower[0]
                    )
                    yy, xx = np.meshgrid(y, x, indexing="ij")
                elif axis == 1:
                    yy = np.ones((vshape[0], vshape[2])) * (
                        upper[1] if high else lower[1]
                    )
                    zz, xx = np.meshgrid(z, x, indexing="ij")
                else:
                    xx = np.ones((vshape[0], vshape[1])) * (
                        upper[2] if high else lower[2]
                    )
                    zz, yy = np.meshgrid(z, y, indexing="ij")
                return xx, yy, zz

            face_rois = [face(False, axis) for axis in range(3)] + [
                face(True, axis) for axis in range(3)
            ]
            face_coords_list = [face_coords(False, axis) for axis in range(3)] + [
                face_coords(True, axis) for axis in range(3)
            ]
            faces[name] = (
                [arr[r].squeeze() for r in face_rois],
                face_coords_list,
            )
        return faces

    faces = read_faces(arrays)

    transformed_faces = {}
    for key, (face_data, face_coords) in faces.items():
        if array_types[key] == "pca":
            transformed_faces[key] = (pca_nd_faces(face_data), face_coords)
        elif array_types[key] == "labels":
            if face_data[0].dtype != np.uint8:
                normalized = [fd % 256 / 255.0 for fd in face_data]
            else:
                normalized = [fd / 255.0 for fd in face_data]
            transformed_faces[key] = (normalized, face_coords)
        else:
            transformed_faces[key] = (face_data, face_coords)
    faces = transformed_faces

    fig, axes = plt.subplots(
        1,
        len(arrays),
        figsize=(2 + 5 * len(arrays), 6),
        subplot_kw={"projection": "3d"},
    )

    label_cmap = get_cmap()

    def draw_cube(
        ax,
        faces: tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
        roi: Roi,
        cmap=None,
        interpolation=None,
    ):
        face_colors, face_coords = faces
        xx, yy, zz = (
            [xxyyzz[0] for xxyyzz in face_coords],
            [xxyyzz[1] for xxyyzz in face_coords],
            [xxyyzz[2] for xxyyzz in face_coords],
        )
        kwargs = {
            "interpolation": interpolation,
            "cmap": cmap,
            "vmin": 0,
            "vmax": 1,
        }

        face_colors = [
            cmap(fc) if cmap is not None else fc.transpose(1, 2, 0)
            for fc in face_colors
        ]

        kwargs = {
            "rcount": rcount,
            "ccount": ccount,
            "shade": shade,
            "lightsource": lightsource,
        }

        # ax.plot_surface(xx[0], yy[0], zz[0], facecolors=face_colors[0], **kwargs)
        ax.plot_surface(xx[1], yy[1], zz[1], facecolors=face_colors[1], **kwargs)
        # ax.plot_surface(xx[2], yy[2], zz[2], facecolors=face_colors[2], **kwargs)
        ax.plot_surface(xx[3], yy[3], zz[3], facecolors=face_colors[3], **kwargs)
        # ax.plot_surface(xx[4], yy[4], zz[4], facecolors=face_colors[4], **kwargs)
        ax.plot_surface(xx[5], yy[5], zz[5], facecolors=face_colors[5], **kwargs)

        ax.set_xlim(roi.begin[2], roi.end[2])
        ax.set_ylim(roi.begin[1], roi.end[1])
        ax.set_zlim(roi.begin[0], roi.end[0])
        ax.set_box_aspect(roi.shape[::-1])

        ax.axis("off")

    for jj, (key, face_data) in enumerate(faces.items()):
        ax = axes[jj] if len(arrays) > 1 else axes
        face_colors = face_data[0]

        if array_types[key] == "labels":
            draw_cube(
                ax, face_data, roi=total_roi, cmap=label_cmap, interpolation="none"
            )
        elif array_types[key] == "raw" or array_types[key] == "pca":
            if face_colors[0].ndim == 2:
                draw_cube(ax, face_data, roi=total_roi, cmap=cm.gray)  # type: ignore
            elif face_colors[0].ndim == 3:
                draw_cube(ax, face_data, roi=total_roi)
        elif array_types[key] == "affs":
            # Show the affinities
            draw_cube(ax, face_data, roi=total_roi, interpolation="none")

        ax.set_title(key)
        # Without this line, the default cube view is elev = 30, azim = -60.
        ax.view_init(elev=elev, azim=azim)

    fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    plt.savefig(filename, bbox_inches="tight", pad_inches=0.1, dpi=dpi)
    plt.close(fig)
