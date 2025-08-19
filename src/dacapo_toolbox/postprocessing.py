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
from volara.workers import Worker

from funlib.geometry import Roi, Coordinate

import numpy as np

import os
from pathlib import Path
import tempfile
from collections.abc import Sequence


def mutex_pipeline():
    pass


def blockwise_mutex(
    model,
    # ...,
):
    pass


def blockwise_predict_mutex(
    raw_store: str | bytes | os.PathLike,
    labels_store: str | bytes | os.PathLike,
    neighborhood: list[tuple[int, int, int]],
    blocksize: Sequence[int],
    model_path: str | bytes | os.PathLike,
    in_channels: int,
    model_context: Sequence[int],
    roi: Roi | None = None,
    affs_store: str | bytes | os.PathLike | None = None,
    frags_store: str | bytes | os.PathLike | None = None,
    sqlite_db_path: str | bytes | os.PathLike | None = None,
    lut_path: str | bytes | os.PathLike | None = None,
    weights_path: str | bytes | os.PathLike | None = None,
    drop: bool = True,
    predict_worker: Worker | None = None,
    extract_frag_bias: Sequence[float] | None = None,
    remove_debris: int = 32,
    num_extract_frag_workers: int = 1,
    extract_frag_worker: Worker | None = None,
    edge_scores: Sequence[
        tuple[Sequence[Coordinate] | Coordinate, float]
        | tuple[str, Sequence[Coordinate] | Coordinate, float]
    ]
    | None = None,
    num_aff_agglom_workers: int = 1,
    aff_agglom_worker: Worker | None = None,
    graph_mws_worker: Worker | None = None,
    num_relabel_workers: int = 1,
    relabel_worker: Worker | None = None,
):
    """
    Run a blockwise pipeline to predict affinities, extract fragments, agglomerate edges,
    run global mutex watershed on the fragment graph, and relabel fragments into segments.

    This is all done in a single pipeline with many intermediate artifacts which can be
    kept or discarded as needed. All intermediate data is definable via the store parameters
    and anything left as `None` will be created in a temporary directory and discarded when
    done. Note that if the intermediate results are not saved, it will not be possible to
    restart the pipeline from an interrupted state as the intermediate data will be lost.

    Parameters
    ----------
    raw_store : str | bytes | os.PathLike
        Path to the raw data store.
    labels_store : str | bytes | os.PathLike
        Path to the labels store.
    neighborhood : list[tuple[int, int, int]]
        List of tuples defining the neighborhood for affinity prediction.
    blocksize : Coordinate
        The size of the blocks to process.
    model_path : str | bytes | os.PathLike
        Path to save the trained model.
    roi : Roi | None, optional
        Region of interest to process. If `None`, the entire dataset is processed.
    affs_store : str | bytes | os.PathLike | None, optional
        Path to save the predicted affinities. If `None`, a temporary directory is used.
    frags_store : str | bytes | os.PathLike | None, optional
        Path to save the extracted fragments. If `None`, a temporary directory is used.
    sqlite_db_path : str | bytes | os.PathLike | None, optional
        Path to save the SQLite database for fragments. If `None`, a temporary directory is used.
    lut_path : str | bytes | os.PathLike | None, optional
        Path to save the LUT for segment IDs. If `None`, a temporary directory is used.
    weights_path : str | bytes | os.PathLike | None, optional
        Path from which to load the model state dict. If `None` it is assumed that the model
        loaded from `model_path` is already initialized with the desired weights.
    drop : bool, optional
        If `True`, drop the pipeline after running it. This is useful to free up resources
        and avoid keeping the pipeline in memory. Defaults to `True`.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        if affs_store is None:
            affs_store = tmpdir / "data.zarr/affs"
        if frags_store is None:
            frags_store = tmpdir / "data.zarr/frags"
        if sqlite_db_path is None:
            sqlite_db_path = tmpdir / "fragments.db"
        if lut_path is None:
            lut_path = tmpdir / "lut"

        raw_dataset = RawDataset(store=Path(raw_store), scale_shift=(1.0 / 255.0, 0.0))
        affs_dataset = AffsDataset(store=Path(affs_store), neighborhood=neighborhood)
        frags_dataset = LabelsDataset(store=Path(frags_store))
        labels_dataset = LabelsDataset(store=Path(labels_store))

        edge_names = []
        edge_coords = []
        edge_biases = []
        if edge_scores is None:
            edge_scores = [([Coordinate(offset)], -0.5) for offset in neighborhood]
        for i, edge_def in enumerate(edge_scores):
            if len(edge_def) == 2:
                name = f"affs_{i}"
                coords: Coordinate | Sequence[Coordinate] = edge_def[0]
                bias: float = edge_def[1]
            elif len(edge_def) == 3:
                name: str = edge_def[0]
                coords: Coordinate | Sequence[Coordinate] = edge_def[1]
                bias: float = edge_def[2]
            else:
                raise ValueError(
                    f"Invalid edge definition: {edge_def}. "
                    "Expected a tuple of 2 (offsets, bias) or 3 (name, offsets, bias) elements."
                )
            edge_names.append(name)
            edge_coords.append(
                list(coords) if isinstance(coords, Coordinate) else coords
            )
            edge_biases.append(bias)

        frag_db = SQLite(
            path=Path(sqlite_db_path),
            edge_attrs={edge_name: "float" for edge_name in edge_names},
        )
        lut_path = Path(lut_path)
        if not lut_path.stem.endswith(".npz"):
            lut_path: Path = lut_path.with_suffix(".npz")
        print(lut_path)
        segment_lut = LUT(
            path=lut_path,
        )

        affs_model = TorchModel(
            in_channels=in_channels,
            min_input_shape=Coordinate(blocksize) + Coordinate(model_context) * 2,
            min_output_shape=Coordinate(blocksize),
            min_step_shape=Coordinate(model_context) * 0 + 1,
            out_channels=len(neighborhood),
            out_range=(0.0, 1.0),
            save_path=Path(model_path),
            checkpoint_file=Path(weights_path) if weights_path else None,
            pred_size_growth=None,
        )

        predict_affs = Predict(
            checkpoint=affs_model,
            in_data=raw_dataset,
            out_data=[affs_dataset],
            num_workers=1,
            worker_config=predict_worker,
            roi=roi,
        )

        neighborhood_context = abs(np.stack(neighborhood)).max(axis=0).astype(np.int32)

        if extract_frag_bias is None:
            neighborhood_log_distances = np.log(
                np.linalg.norm(np.array(neighborhood), axis=1)
            )
            log_min_dist: float = neighborhood_log_distances.min()
            log_max_dist: float = neighborhood_log_distances.max()
            min_bias: float = -0.7
            max_bias: float = -0.3

            extract_frag_bias: list[float] = np.interp(  # type: ignore
                neighborhood_log_distances,
                [log_min_dist, log_max_dist],
                [min_bias, max_bias],
            ).tolist()
        else:
            extract_frag_bias: list[float] = list(extract_frag_bias)

        extract_frags = ExtractFrags(
            db=frag_db,
            affs_data=affs_dataset,
            frags_data=frags_dataset,
            block_size=blocksize,
            context=neighborhood_context * 2,
            bias=extract_frag_bias,
            remove_debris=remove_debris,
            num_workers=num_extract_frag_workers,
            roi=roi,
            worker_config=extract_frag_worker,
        )
        aff_agglom = AffAgglom(
            db=frag_db,
            affs_data=affs_dataset,
            frags_data=frags_dataset,
            block_size=blocksize,
            context=neighborhood_context,
            scores={name: coords for name, coords in zip(edge_names, edge_coords)},
            roi=roi,
            num_workers=num_aff_agglom_workers,
            worker_config=aff_agglom_worker,
        )
        graph_mws = GraphMWS(
            db=frag_db,
            lut=segment_lut,
            weights={name: (1, bias) for name, bias in zip(edge_names, edge_biases)},
            roi=roi or raw_dataset.array("r").roi,
            worker_config=graph_mws_worker,
        )

        relabel = Relabel(
            frags_data=frags_dataset,
            seg_data=labels_dataset,
            lut=segment_lut,
            block_size=blocksize,
            roi=roi,
            num_workers=num_relabel_workers,
            worker_config=relabel_worker,
        )

        pipeline = predict_affs + extract_frags + aff_agglom + graph_mws + relabel
        pipeline.drop()
        pipeline.run_blockwise()
