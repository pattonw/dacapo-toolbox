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

    Parameters:
        arrays:
            A dictionary of named arrays to visualize. Each array must be 3D,
            with optional channels.
        array_types:
            A dictionary specifying the type of each array. Supported types are:
            - "raw": Grayscale raw data.
            - "labels": Integer labels, visualized with a random color map.
            - "affs": Affinity graphs, visualized as RGB images.
            - "pca": High-dimensional data, visualized using PCA to reduce to 3 channels.
        filename:
            The output filename for the GIF.
        title:
            The title to display on the GIF.
        fps:
            Frames per second for the GIF. Default is 10.
        overwrite:
            Whether to overwrite the output file if it already exists. Default is True.
        dpi:
            Dots per inch for the output GIF. Default is 72.
        max_size:
            Maximum size (in pixels) for the largest dimension of the images. Default is 256.
        optimize_gif:
            Whether to optimize the GIF for smaller file size. Default is False.
        frame_skip:
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
    cmap: ListedColormap | None = None,
    layout: str = "line",
    mesh_voxels: int = 2_000_000,
):
    """
    Preview 3D arrays as cubes with matplotlib. Arrays do not need to be the same size
    their relative sizes and shifts will be respected.

    Parameters:
        arrays
            A dictionary of named arrays to visualize. Each array must be 3D,
            with optional channels.
        array_types
            A dictionary specifying the type of each array. Supported types are:
            - "raw": Grayscale raw data.
            - "labels": Integer labels, visualized with a random color map.
            - "affs": Affinity graphs, visualized as RGB images.
            - "pca": High-dimensional data, visualized using PCA to reduce to 3
                channels.
            - "mesh": Integer labels drawn as one surface mesh per label
                (background 0 left out) inside a translucent box marking the
                roi, so it shows how much of the volume is labelled. The
                whole array is read, strided down to ``mesh_voxels``.
        filename:
            The output filename for the image.
        title:
            The title to display on the image.
        elev:
            Elevation angle for the 3D view. Default is 30.
        azim:
            Azimuth angle for the 3D view. Default is -60.
        light_azdeg:
            Azimuth angle for the light source. Default is 205.
        light_altdeg:
            Altitude angle for the light source. Default is 20.
        overwrite:
            Whether to overwrite the output file if it already exists. Default is True.
        rcount:
            Number of rows for the surface plot. Default is 128.
        ccount:
            Number of columns for the surface plot. Default is 128.
        shade:
            Whether to shade the surface plot. Default is True.
        dpi:
            Dots per inch for the output image. Default is 100.
        mesh_voxels:
            Voxel budget for each "mesh" array: it is read with a stride per
            axis (keeping voxels near isotropic) so at most about this many
            voxels are meshed. Default 2000000.
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
            if array_types[name] == "mesh":
                continue
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

    if layout == "line":
        fig, axes = plt.subplots(
            1,
            len(arrays),
            figsize=(2 + 5 * len(arrays), 6),
            subplot_kw={"projection": "3d"},
        )
    elif layout == "square":
        fig, axes = plt.subplots(
            int(np.ceil(len(arrays) ** 0.5)),
            int(np.ceil(len(arrays) ** 0.5)),
            figsize=(
                5 * int(np.ceil(len(arrays) ** 0.5)),
                5 * int(np.ceil(len(arrays) ** 0.5)),
            ),
            subplot_kw={"projection": "3d"},
        )

    label_cmap = get_cmap()
    if cmap is not None:
        label_cmap = ListedColormap(
            np.concatenate(
                (cmap.colors, label_cmap.colors[: -len(cmap.colors)]), axis=0
            )
        )

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

    for jj, key in enumerate(arrays):
        if layout == "square":
            ax = axes[jj // int(np.ceil(len(arrays) ** 0.5))][
                jj % int(np.ceil(len(arrays) ** 0.5))
            ]
        else:
            ax = axes[jj] if len(arrays) > 1 else axes

        if array_types[key] == "mesh":
            draw_label_meshes(
                ax,
                arrays[key],
                total_roi,
                label_cmap,
                mesh_voxels,
                lightsource=lightsource,
            )
            ax.set_title(key)
            ax.view_init(elev=elev, azim=azim)
            continue
        face_data = faces[key]
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


def _mesh_strides(arr: Array, max_voxels: int) -> tuple[int, int, int]:
    """Per-axis strides that bring ``arr`` to at most about ``max_voxels``
    voxels, coarsening the finest axes first so voxels approach isotropic."""
    voxel = np.array(arr.voxel_size, dtype=np.float64)
    shape = np.array(arr.roi.shape / arr.voxel_size, dtype=np.float64)
    if np.prod(shape) <= max_voxels:
        return (1, 1, 1)
    # the world step s at which prod(shape * voxel / max(s, voxel)) fits
    lo, hi = float(voxel.min()), float((shape * voxel).max())
    for _ in range(60):
        step = (lo + hi) / 2
        strides = np.maximum(1, np.round(step / voxel))
        if np.prod(np.ceil(shape / strides)) > max_voxels:
            lo = step
        else:
            hi = step
    strides = np.maximum(1, np.round(hi / voxel)).astype(int)
    return (int(strides[0]), int(strides[1]), int(strides[2]))


def label_meshes(
    arr: Array, max_voxels: int = 2_000_000
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """One marching-cubes surface per non-zero label of ``arr``, read with the
    strides of :func:`_mesh_strides`. Returns ``(label, vertices, faces)``
    with vertices in world units, ``(z, y, x)`` order."""
    from scipy import ndimage
    from skimage.measure import marching_cubes

    strides = _mesh_strides(arr, max_voxels)
    data = np.asarray(arr.data[tuple(slice(None, None, s) for s in strides)])
    ids, compact = np.unique(data, return_inverse=True)
    # find_objects numbers from 1, so label ids[i] becomes i + 1
    compact = compact.reshape(data.shape) + 1
    spacing = np.array(arr.voxel_size, dtype=np.float64) * strides
    offset = np.array(arr.roi.offset, dtype=np.float64)
    meshes = []
    for index, box in enumerate(ndimage.find_objects(compact), start=1):
        if box is None or ids[index - 1] == 0:
            continue
        # pad by one voxel so labels touching the box edge close their surface
        start = [max(b.start - 1, 0) for b in box]
        stop = [min(b.stop + 1, n) for b, n in zip(box, data.shape)]
        mask = compact[tuple(slice(a, b) for a, b in zip(start, stop))] == index
        mask = np.pad(mask, 1)
        verts, faces, _, _ = marching_cubes(
            mask.astype(np.uint8), 0.5, spacing=tuple(spacing)
        )
        verts += offset + (np.array(start) - 1) * spacing
        meshes.append((int(ids[index - 1]), verts, faces))
    return meshes


def draw_label_meshes(
    ax,
    arr: Array,
    roi: Roi,
    cmap: ListedColormap,
    max_voxels: int = 2_000_000,
    face_alpha: float = 0.08,
    lightsource: mcolors.LightSource | None = None,
):
    """Draw :func:`label_meshes` of ``arr`` on a 3D axes inside a translucent
    cuboid marking ``roi``, each label coloured as the "labels" cube colours
    it and shaded by its surface normals."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if lightsource is None:
        lightsource = mcolors.LightSource(azdeg=205, altdeg=20)
    triangles, colors, normals = [], [], []
    for label, verts, faces in label_meshes(arr, max_voxels):
        tri = verts[faces][:, :, ::-1]  # x, y, z axes take dims 2, 1, 0
        triangles.append(tri)
        normals.append(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))
        colors.append(np.tile(cmap(label % 256 / 255.0), (len(tri), 1)))
    if triangles:
        tri = np.concatenate(triangles)
        normal = np.concatenate(normals)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
        # matplotlib's own shading: brightness from the light source
        light = lightsource.direction
        shade = 0.45 + 0.55 * np.abs(normal @ np.asarray(light))
        color = np.concatenate(colors)
        color[:, :3] *= shade[:, None]
        ax.add_collection3d(
            Poly3DCollection(tri, facecolors=color, edgecolors="none", linewidths=0)
        )
    _draw_cuboid(ax, roi, face_alpha)


def _draw_cuboid(ax, roi: Roi, face_alpha: float):
    """Six lightly filled faces of ``roi`` with drawn edges, and the axes
    limits and aspect set to it, as :func:`graph_cube` and :func:`cube`
    lay out their views."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    b, e = roi.begin, roi.end
    corners = np.array(
        [[x, y, z] for z in (b[0], e[0]) for y in (b[1], e[1]) for x in (b[2], e[2])],
        dtype=np.float64,
    )
    faces = [
        [0, 1, 3, 2],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 3, 7, 5],
    ]
    ax.add_collection3d(
        Poly3DCollection(
            [corners[f] for f in faces],
            facecolors=(0.5, 0.5, 0.5, face_alpha),
            edgecolors=(0.2, 0.2, 0.2, 0.6),
            linewidths=0.8,
        )
    )
    ax.set_xlim(b[2], e[2])
    ax.set_ylim(b[1], e[1])
    ax.set_zlim(b[0], e[0])
    ax.set_box_aspect(roi.shape[::-1])
    ax.axis("off")


def _thin_along_edges(graph, nodes: list, stride: int) -> tuple[list, list]:
    """Keep every ``stride``-th node along each path of ``graph`` plus every
    leaf and branch point, joining each kept node to its nearest kept
    ancestor in a breadth-first tree, so thinned skeletons keep their shape."""
    undirected = graph.to_undirected(as_view=True) if graph.is_directed() else graph
    allowed = set(nodes)
    kept: list = []
    edges: list = []
    seen: set = set()
    for root in nodes:
        if root in seen:
            continue
        seen.add(root)
        kept.append(root)
        # (node, nearest kept ancestor, steps since that ancestor)
        frontier = [(root, root, 0)]
        while frontier:
            next_frontier = []
            for node, anchor, steps in frontier:
                for child in undirected.neighbors(node):
                    if child in seen or child not in allowed:
                        continue
                    seen.add(child)
                    depth = steps + 1
                    if depth >= stride or undirected.degree(child) != 2:
                        kept.append(child)
                        edges.append((anchor, child))
                        next_frontier.append((child, child, 0))
                    else:
                        next_frontier.append((child, anchor, depth))
            frontier = next_frontier
    return kept, edges


def graph_cube(
    graph,
    roi: Roi,
    filename: str,
    title: str,
    position_attribute: str = "position",
    color_by: str | None = None,
    max_nodes: int = 50_000,
    draw_edges: bool = True,
    node_size: float = 1.0,
    face_alpha: float = 0.08,
    elev: float = 30,
    azim: float = -60,
    overwrite: bool = True,
    dpi: int = 100,
    seed: int = 1,
    ax=None,
):
    """
    Preview a spatial graph (skeletons, points) as a 3D scatter inside a
    slightly opaque cuboid marking the image boundaries. The view, axis order
    and box aspect match :func:`cube`, so a graph preview sits beside an array
    preview of the same region without surprises.

    Parameters:
        graph:
            A ``networkx`` graph whose nodes carry a world-space position
            under ``position_attribute`` in ``(z, y, x)`` order, as
            ``funlib.persistence`` graph databases return them.
        roi:
            The region to draw the cuboid for, in the same world units as the
            node positions; usually the image or ground-truth ROI. Nodes
            outside it are still drawn.
        filename:
            The output filename for the image.
        title:
            The title to display on the image.
        position_attribute:
            The node attribute holding the position. Default is ``position``.
        color_by:
            A node attribute to colour nodes by (``skeleton_id``, ``label``);
            each distinct value gets one colour from the label colormap. None
            draws every node in one colour.
        max_nodes:
            When the graph has more nodes than this, a subset of about this
            size is drawn. With ``color_by`` every group is kept and thinned
            along its edges (every n-th node plus leaves and branch points,
            each joined to its nearest kept ancestor), so all skeletons are
            drawn with their shape; otherwise random nodes are drawn, with
            edges only between drawn nodes. Default 50000.
        draw_edges:
            Whether to draw edges between drawn nodes. Default is True.
        node_size:
            Marker size for the nodes. Default is 1.
        face_alpha:
            Opacity of the cuboid faces. Default is 0.08.
        elev, azim:
            View angles, as for :func:`cube`.
        overwrite:
            Whether to overwrite the output file if it already exists.
        dpi:
            Dots per inch for the output image. Default is 100.
        seed:
            Seed for the node subsample and the colour map.
        ax:
            An existing 3D axes to draw on instead of a new figure; the file is
            then not written and the axes are returned.
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    if ax is None and Path(filename).exists() and not overwrite:
        return None

    nodes = [n for n, data in graph.nodes(data=True) if position_attribute in data]
    rng = np.random.default_rng(seed)
    edges = list(graph.edges())
    if len(nodes) > max_nodes:
        if color_by is not None:
            # Thin every skeleton along its length rather than dropping whole
            # skeletons, so all of them are drawn with their shape intact.
            nodes, edges = _thin_along_edges(graph, nodes, -(-len(nodes) // max_nodes))
        else:
            chosen = rng.choice(len(nodes), size=max_nodes, replace=False)
            nodes = [nodes[i] for i in sorted(chosen)]
    index = {n: i for i, n in enumerate(nodes)}
    positions = np.array(
        [graph.nodes[n][position_attribute] for n in nodes], dtype=np.float64
    ).reshape(-1, 3)

    if color_by is not None:
        values = [graph.nodes[n].get(color_by) for n in nodes]
        distinct = {v: i for i, v in enumerate(dict.fromkeys(values))}
        cmap = get_cmap(seed)
        colors = cmap(np.array([1 + distinct[v] % 255 for v in values]))
    else:
        colors = np.tile(np.array([[0.1, 0.4, 0.8, 1.0]]), (len(nodes), 1))

    own_figure = ax is None
    if own_figure:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")

    # x, y, z axes take roi dims 2, 1, 0, as cube() does.
    if len(positions):
        ax.scatter(
            positions[:, 2],
            positions[:, 1],
            positions[:, 0],
            s=node_size,
            c=colors,
            depthshade=False,
            linewidths=0,
        )
    if draw_edges and len(positions):
        drawn = [(u, v) for u, v in edges if u in index and v in index]
        segments = [
            [positions[index[u]][::-1], positions[index[v]][::-1]] for u, v in drawn
        ]
        if segments:
            edge_colors = [colors[index[u]] for u, v in drawn]
            ax.add_collection3d(
                Line3DCollection(segments, colors=edge_colors, linewidths=0.5)
            )

    _draw_cuboid(ax, roi, face_alpha)
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)

    if not own_figure:
        return ax
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.1, dpi=dpi)
    plt.close(fig)
    return None
