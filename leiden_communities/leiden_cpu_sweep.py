#!/usr/bin/env python3
"""CPU-only Leiden sweep of saved train/test embedding snapshots.

Install: python -m pip install numpy scikit-learn igraph leidenalg plotly matplotlib threadpoolctl
Example:
 python leiden_cpu_sweep.py --input-dir /path/fold_01/projections --output-dir /path/sweep \
   --k-range 5 35 5 --resolution-range 0.2 2.0 0.2 --seed 42 --fold 1
Ranges include their stop value when reached by the step. No PyTorch, MRI loading,
GPU, encoder inference, or UMAP. Input snapshots: simclr_epoch_XXXX/{train,test}_embeddings.npz.

Reproduces the supplied Nov 21 code: normalize(rs+thick), float64 normalized cosine
kNN graph, maximum symmetric edge weight floored at 1e-6, RBConfiguration Leiden
(default 2 iterations), primary seed=seed+fold, comparison seed=primary+103.
Time ARI compares consecutive AVAILABLE checkpoints at fixed k/resolution.
First time ARI is undefined (NaN). Test is analyzed independently, never used to
select training parameters. Eligibility on test is descriptive only.

HTML meshes have k, resolution, epoch axes and ARI as color (four-dimensional
results). Each epoch is a separate horizontal mesh; use the legend to hide layers.
PNG overviews and TSV results are also saved. HTML embeds Plotly for offline use.
"""

from __future__ import annotations
import argparse
import csv
import json
import logging
import re
import sys
import time
from decimal import Decimal
from pathlib import Path


def inclusive_range(spec, integer=False):
    """Build an inclusive numeric grid without floating-point step drift.

    Parameters
    ----------
    spec : sequence of three numbers or strings
        Start, stop, and positive step. The stop is included only if reached.
    integer : bool, default=False
        Require integer grid points and return Python integers.

    Returns
    -------
    list of int or float
        Grid points in ascending order.

    Raises
    ------
    ValueError
        If the bounds, step, or requested integer values are invalid.
    """
    start, stop, step = map(Decimal, map(str, spec))
    if step <= 0 or stop < start:
        raise ValueError("Range requires positive step and stop >= start")
    vals = []
    while start <= stop:
        if integer and start != start.to_integral_value():
            raise ValueError("k values must be integers")
        vals.append(int(start) if integer else float(start))
        start += step
    return vals


def parse_args():
    """Parse and validate paths, parameter grids, seeds, and thresholds.

    Explicit value lists override the corresponding range arguments. The
    threshold options reproduce training eligibility annotations; they do not
    terminate this offline sweep or change either ARI calculation.

    Returns
    -------
    argparse.Namespace
        Validated arguments, including expanded ``ks`` and ``resolutions``.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory directly containing simclr_epoch_XXXX folders",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Use a new directory for each sweep",
    )
    p.add_argument(
        "--k-range",
        nargs=3,
        default=["5", "35", "5"],
        metavar=("START", "STOP", "STEP"),
    )
    p.add_argument(
        "--resolution-range",
        nargs=3,
        default=["0.2", "2.0", "0.2"],
        metavar=("START", "STOP", "STEP"),
    )
    p.add_argument(
        "--k-values", nargs="+", type=int, help="Explicit values override --k-range"
    )
    p.add_argument(
        "--resolution-values",
        nargs="+",
        type=float,
        help="Explicit values override --resolution-range",
    )
    p.add_argument("--seed", type=int, default=42, help="Original global training seed")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--cpu-threads", type=int, default=3)
    p.add_argument("--epoch-min", type=int, default=0)
    p.add_argument("--epoch-max", type=int)
    p.add_argument("--min-ssl-epochs", type=int, default=20)
    p.add_argument("--cluster-stability-threshold", type=float, default=0.90)
    p.add_argument("--cluster-seed-threshold", type=float, default=0.90)
    p.add_argument("--stability-patience", type=int, default=3)
    p.add_argument("--min-community-size", type=int, default=5)
    p.add_argument("--min-communities", type=int, default=2)
    p.add_argument("--min-community-coverage", type=float, default=0.70)
    a = p.parse_args()
    try:
        a.ks = sorted(set(a.k_values or inclusive_range(a.k_range, True)))
        a.resolutions = sorted(
            set(a.resolution_values or inclusive_range(a.resolution_range))
        )
        import math

        if min(a.ks) < 2 or any(not math.isfinite(x) or x <= 0 for x in a.resolutions):
            raise ValueError("k >= 2 and finite resolution > 0 required")
        if a.cpu_threads < 1 or a.fold < 1 or a.seed < 0:
            raise ValueError("cpu-threads/fold >= 1 and seed >= 0 required")
        if min(a.stability_patience, a.min_community_size, a.min_communities) < 1:
            raise ValueError("Patience and community counts/sizes must be positive")
        if not (
            0 <= a.cluster_stability_threshold <= 1
            and 0 <= a.cluster_seed_threshold <= 1
            and 0 < a.min_community_coverage <= 1
        ):
            raise ValueError("Invalid thresholds")
    except ValueError as exc:
        p.error(str(exc))
    return a


def load_snapshot(path):
    """Read aligned modality embeddings and construct the community space.

    Parameters
    ----------
    path : pathlib.Path
        NPZ file containing ``rs``, ``thick``, and unique ``subject_id`` values.

    Returns
    -------
    z : numpy.ndarray of shape (n_subjects, n_features)
        Float32, row-normalized sum of the original modality embeddings.
    ids : numpy.ndarray of shape (n_subjects,)
        Subject identifiers in their original saved row order.

    Raises
    ------
    ValueError
        If required keys, dimensions, finite values, or unique IDs are missing.

    Notes
    -----
    Original row order is preserved to match the supplied neighbor-search code.
    No encoder inference, model checkpoint loading, or UMAP is performed.
    """
    import numpy as np

    with np.load(path, allow_pickle=False) as d:
        for key in ("rs", "thick", "subject_id"):
            if key not in d:
                raise ValueError(f"{path}: missing {key}")
        rs, thick, ids = d["rs"], d["thick"], d["subject_id"].astype(str)
        if (
            rs.ndim != 2
            or rs.shape != thick.shape
            or ids.ndim != 1
            or len(ids) != len(rs)
        ):
            raise ValueError(f"{path}: incompatible modality or ID shapes")
        if (
            len(ids) < 4
            or len(set(ids)) != len(ids)
            or not np.isfinite(rs).all()
            or not np.isfinite(thick).all()
        ):
            raise ValueError(f"{path}: need >=4 finite embeddings and unique IDs")
        # Preserve original row order, including kNN tie behavior.
        z = rs + thick
        z = (z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)).astype(
            np.float32
        )
    return z, ids


def build_graph(z, k):
    # Intentionally matches supplied utils.py (including neighbors[:, 1:]).
    """Construct the weighted undirected cosine-neighbor graph on CPU.

    Parameters
    ----------
    z : array-like of shape (n_subjects, n_features)
        High-dimensional community vectors.
    k : int
        Requested neighbor count; capped implicitly at the cohort size minus one.

    Returns
    -------
    igraph.Graph
        Graph with a ``weight`` edge attribute.

    Notes
    -----
    This follows the uploaded ``leiden_subject_communities`` implementation:
    float64 normalization, ``k + 1`` neighbors, removal of the first neighbor
    column, explicit self-edge exclusion, and maximum-weight symmetrization.
    Similarity weights are floored at 1e-6. Keeping these details preserves the
    original calculation, including behavior when neighbor distances tie.
    """
    import numpy as np
    import igraph as ig
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize

    z = normalize(np.asarray(z, dtype=np.float64))
    distances, indices = (
        NearestNeighbors(n_neighbors=min(k + 1, len(z)), metric="cosine")
        .fit(z)
        .kneighbors(z)
    )
    edges = {}
    for i in range(len(z)):
        for j, distance in zip(indices[i, 1:], distances[i, 1:]):
            if i == int(j):
                continue
            edge = (min(i, int(j)), max(i, int(j)))
            edges[edge] = max(edges.get(edge, 0.0), max(1e-6, 1.0 - float(distance)))
    if not edges:
        raise ValueError("Empty nearest-neighbor graph")
    g = ig.Graph(n=len(z), edges=list(edges), directed=False)
    g.es["weight"] = list(edges.values())
    return g


def partition(graph, resolution, seed):
    """Run one Leiden partition with the original objective and seed.

    Parameters
    ----------
    graph : igraph.Graph
        Undirected graph containing ``weight`` edge values.
    resolution : float
        Positive RBConfiguration resolution parameter.
    seed : int
        Random seed for this Leiden run.

    Returns
    -------
    numpy.ndarray of shape (n_subjects,)
        Integer community labels in graph vertex order.

    Notes
    -----
    Uses the library default iteration count, as in the supplied utilities.
    """
    import numpy as np
    import leidenalg

    return np.asarray(
        leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=seed,
        ).membership,
        dtype=np.int64,
    )


def plot_results(rows, out):
    """Save separate training and test visualizations of both ARI metrics.

    Parameters
    ----------
    rows : list of dict
        Completed sweep records with split, epoch, k, resolution, and ARI values.
    out : pathlib.Path
        Destination directory for standalone HTML and PNG files.

    Notes
    -----
    The three spatial axes are k, resolution, and epoch; color encodes ARI.
    Each epoch is a horizontal mesh with independently toggleable visibility.
    Singleton parameter grids fall back to markers. Undefined first-checkpoint
    temporal ARI values are omitted rather than replaced with zero. No smoothing
    or interpolation is applied to the recorded metric values.
    """
    import numpy as np
    import plotly.graph_objects as go
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for split in ("train", "test"):
        subset = [r for r in rows if r["split"] == split]
        ks = sorted({r["k"] for r in subset})
        resolutions = sorted({r["resolution"] for r in subset})
        epochs = sorted({r["epoch"] for r in subset})
        X, Y = np.meshgrid(ks, resolutions)
        for metric in ("temporal_ari", "seed_ari"):
            fig = go.Figure()
            overview = plt.figure(figsize=(10, 8))
            ax = overview.add_subplot(111, projection="3d")
            for epoch in epochs:
                layer = [r for r in subset if r["epoch"] == epoch]
                lookup = {(r["k"], r["resolution"]): r[metric] for r in layer}
                C = np.array([[lookup[(k, res)] for k in ks] for res in resolutions])
                Z = np.full(X.shape, epoch)
                if not np.isfinite(C).any():
                    continue
                if len(ks) > 1 and len(resolutions) > 1:
                    fig.add_trace(
                        go.Surface(
                            x=X,
                            y=Y,
                            z=Z,
                            surfacecolor=C,
                            coloraxis="coloraxis",
                            opacity=0.75,
                            name=f"Epoch {epoch}",
                            showlegend=True,
                            hovertemplate="k=%{x}<br>resolution=%{y}<br>epoch=%{z}<br>ARI=%{surfacecolor:.4f}<extra></extra>",
                        )
                    )
                    ax.plot_wireframe(X, Y, Z, color="gray", alpha=0.15, linewidth=0.4)
                else:
                    fig.add_trace(
                        go.Scatter3d(
                            x=X.ravel(),
                            y=Y.ravel(),
                            z=Z.ravel(),
                            mode="markers",
                            marker=dict(color=C.ravel(), coloraxis="coloraxis", size=6),
                            name=f"Epoch {epoch}",
                            customdata=C.ravel(),
                            hovertemplate="k=%{x}<br>resolution=%{y}<br>epoch=%{z}<br>ARI=%{customdata:.4f}<extra></extra>",
                        )
                    )
                ax.scatter(
                    X.ravel(),
                    Y.ravel(),
                    Z.ravel(),
                    c=C.ravel(),
                    cmap="viridis",
                    vmin=-1,
                    vmax=1,
                    s=16,
                )
            title = f"{split}: {metric} (color = ARI)"
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title="Leiden k",
                    yaxis_title="Resolution",
                    zaxis_title="Epoch",
                ),
                coloraxis=dict(
                    colorscale="Viridis", cmin=-1, cmax=1, colorbar=dict(title="ARI")
                ),
                height=850,
                legend=dict(title="Toggle epoch layers", x=1.15),
                margin=dict(r=200),
            )
            fig.write_html(out / f"{split}_{metric}_mesh.html", include_plotlyjs=True)
            ax.set(xlabel="Leiden k", ylabel="Resolution", zlabel="Epoch", title=title)
            overview.colorbar(
                plt.cm.ScalarMappable(norm=plt.Normalize(-1, 1), cmap="viridis"),
                ax=ax,
                shrink=0.6,
                label="ARI",
            )
            overview.savefig(
                out / f"{split}_{metric}_mesh.png", dpi=180, bbox_inches="tight"
            )
            plt.close(overview)
            logging.info("Saved %s %s HTML mesh and PNG overview", split, metric)


def main():
    """Execute and persist the complete CPU-only parameter sweep.

    Process snapshots in numeric epoch order, separately for training and test.
    Reuse each graph across resolutions and both seeds. Compare the primary
    partition with the previous available checkpoint after subject-ID alignment;
    reject changing subject sets rather than silently comparing different cohorts.

    Persist each completed record immediately, retain both seed partitions for
    later analysis, and log every combination. Training-style eligibility and
    patience are descriptive annotations only: every combination is processed,
    including those after a threshold is met. Test results never select training
    settings. Finally, generate four HTML meshes and four PNG overviews.
    """
    a = parse_args()
    a.input_dir = a.input_dir.expanduser().resolve()
    a.output_dir = a.output_dir.expanduser().resolve()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    ## comment this to allow overwriting if this i necessary**
    ##if (a.output_dir / "sweep_config.json").exists():
    ##    raise ValueError(
    ##        "Output already contains a sweep; choose a new output directory"
    ##    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(a.output_dir / "sweep.log"),
        ],
    )
    import numpy as np
    from sklearn.metrics import adjusted_rand_score
    from threadpoolctl import threadpool_limits
    import importlib.metadata

    # Fail before processing if plotting or clustering dependencies are absent.
    import igraph, leidenalg, plotly, matplotlib

    snapshots = []
    for p in a.input_dir.glob("simclr_epoch_*"):
        match = re.fullmatch(r"simclr_epoch_(\d+)", p.name)
        if match and p.is_dir():
            epoch = int(match.group(1))
            if epoch >= a.epoch_min and (a.epoch_max is None or epoch <= a.epoch_max):
                snapshots.append((epoch, p))
    snapshots.sort()
    if not snapshots or len({e for e, _ in snapshots}) != len(snapshots):
        raise ValueError(
            "No snapshots found, or duplicate numeric epochs; input must be the projections directory"
        )
    for _, folder in snapshots:
        for split in ("train", "test"):
            if not (folder / f"{split}_embeddings.npz").is_file():
                raise FileNotFoundError(folder / f"{split}_embeddings.npz")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(
        primary_seed=a.seed + a.fold,
        secondary_seed=a.seed + a.fold + 103,
        epochs=[e for e, _ in snapshots],
        versions={
            m: importlib.metadata.version(m)
            for m in ["numpy", "scikit-learn", "igraph", "leidenalg", "plotly"]
        },
        temporal_comparison="previous available epoch; identical subject sets, aligned by ID",
        test_usage="independent descriptive analysis, never training selection",
    )
    (a.output_dir / "sweep_config.json").write_text(json.dumps(config, indent=2))
    total = len(snapshots) * 2 * len(a.ks) * len(a.resolutions)
    logging.info(
        "CPU-only sweep: %d combinations, %d Leiden calls; seeds %d and %d; threads=%d",
        total,
        total * 2,
        a.seed + a.fold,
        a.seed + a.fold + 103,
        a.cpu_threads,
    )
    logging.info(
        "k=%s; resolutions=%s; epochs=%s", a.ks, a.resolutions, config["epochs"]
    )
    rows, previous, streaks = [], {}, {}
    completed = 0
    fields = [
        "split",
        "epoch",
        "previous_epoch",
        "k",
        "effective_k",
        "resolution",
        "primary_seed",
        "secondary_seed",
        "n_subjects",
        "temporal_ari",
        "seed_ari",
        "n_communities",
        "min_size",
        "max_size",
        "n_supported",
        "supported_fraction",
        "eligible",
        "stable_checks",
        "patience_met",
        "community_sizes",
        "seconds",
    ]
    with threadpool_limits(limits=a.cpu_threads), (a.output_dir / "stability.tsv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for epoch, folder in snapshots:
            split_ids = {}
            for split in ("train", "test"):
                z, ids = load_snapshot(folder / f"{split}_embeddings.npz")
                split_ids[split] = set(ids)
                if split == "test" and split_ids["train"] & split_ids["test"]:
                    raise ValueError(f"Train/test subject overlap at epoch {epoch}")
                logging.info("Loaded split=%s epoch=%d shape=%s", split, epoch, z.shape)
                labels_dir = (
                    a.output_dir / "labels" / split / f"simclr_epoch_{epoch:04d}"
                )
                labels_dir.mkdir(parents=True, exist_ok=True)
                for k in a.ks:
                    logging.info(
                        "Building graph split=%s epoch=%d k=%d effective_k=%d",
                        split,
                        epoch,
                        k,
                        min(k, len(ids) - 1),
                    )
                    graph = build_graph(z, k)
                    for ri, res in enumerate(a.resolutions):
                        started = time.perf_counter()
                        key = (split, k, res)
                        logging.info(
                            "Processing %d/%d split=%s epoch=%d k=%d resolution=%g seeds=%d/%d",
                            completed + 1,
                            total,
                            split,
                            epoch,
                            k,
                            res,
                            a.seed + a.fold,
                            a.seed + a.fold + 103,
                        )
                        labels = partition(graph, res, a.seed + a.fold)
                        other = partition(graph, res, a.seed + a.fold + 103)
                        temporal = float("nan")
                        previous_epoch = ""
                        if key in previous:
                            prev_ids, prev_labels, previous_epoch = previous[key]
                            if set(prev_ids) != set(ids):
                                raise ValueError(
                                    f"{split}: subject set changed at epoch {epoch}; refusing incomparable temporal ARI"
                                )
                            lookup = {s: i for i, s in enumerate(prev_ids)}
                            aligned = prev_labels[[lookup[s] for s in ids]]
                            temporal = float(adjusted_rand_score(aligned, labels))
                        seed_ari = float(adjusted_rand_score(labels, other))
                        values, counts = np.unique(labels, return_counts=True)
                        supported = counts[counts >= a.min_community_size]
                        coverage = float(supported.sum() / len(ids))
                        eligible = bool(
                            epoch >= a.min_ssl_epochs
                            and np.isfinite(temporal)
                            and temporal >= a.cluster_stability_threshold
                            and seed_ari >= a.cluster_seed_threshold
                            and coverage >= a.min_community_coverage
                            and len(supported) >= a.min_communities
                        )
                        streaks[key] = streaks.get(key, 0) + 1 if eligible else 0
                        row = dict(
                            split=split,
                            epoch=epoch,
                            previous_epoch=previous_epoch,
                            k=k,
                            effective_k=min(k, len(ids) - 1),
                            resolution=res,
                            primary_seed=a.seed + a.fold,
                            secondary_seed=a.seed + a.fold + 103,
                            n_subjects=len(ids),
                            temporal_ari=temporal,
                            seed_ari=seed_ari,
                            n_communities=len(values),
                            min_size=int(counts.min()),
                            max_size=int(counts.max()),
                            n_supported=len(supported),
                            supported_fraction=coverage,
                            eligible=eligible,
                            stable_checks=streaks[key],
                            patience_met=streaks[key] >= a.stability_patience,
                            community_sizes=json.dumps(
                                dict(zip(map(str, values), map(int, counts)))
                            ),
                            seconds=time.perf_counter() - started,
                        )
                        np.savez_compressed(
                            labels_dir / f"k_{k}_resolution_{ri:03d}.npz",
                            subject_id=ids,
                            labels=labels,
                            secondary_labels=other,
                            k=k,
                            resolution=res,
                            epoch=epoch,
                            primary_seed=a.seed + a.fold,
                            secondary_seed=a.seed + a.fold + 103,
                        )
                        previous[key] = (ids.copy(), labels.copy(), epoch)
                        rows.append(row)
                        writer.writerow(row)
                        f.flush()
                        completed += 1
                        logging.info(
                            "Result time-ARI=%.6f seed-ARI=%.6f communities=%d supported=%.3f streak=%d elapsed=%.2fs",
                            temporal,
                            seed_ari,
                            len(values),
                            coverage,
                            streaks[key],
                            row["seconds"],
                        )
                    del graph
                del z
    plot_results(rows, a.output_dir)
    logging.info("Complete. Results: %s", a.output_dir / "stability.tsv")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception(
            "Sweep failed; completed TSV rows and labels remain available"
        )
        sys.exit(1)
