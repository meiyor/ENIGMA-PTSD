"""CPU Leiden stability sweep, mesh plots, and cluster-colored UMAP exports.

Existing-sweep usage (no Leiden recalculation):
 python leiden_cpu_sweep_mesh_with_umap.py --umap-only \
   --input-dir /path/fold_01/projections --sweep-dir /path/existing_sweep \
   --output-dir /path/existing_sweep --seed 42 --fold 1

Full sweep plus UMAP:
 python leiden_cpu_sweep_mesh_with_umap.py --input-dir /path/projections \
   --output-dir /path/new_sweep --k-range 5 35 5 \
   --resolution-range .2 2 .2 --zlim .2 1

Dependencies:
 python -m pip install numpy scikit-learn igraph leidenalg matplotlib plotly umap-learn threadpoolctl

Leiden calculations and existing mesh options are unchanged. Temporal ARI
compares consecutive available checkpoints; seed ARI uses seed+fold and +103.
There is no stopping on training eligibility thresholds in this offline sweep.

UMAP default matches the supplied training helper: stack thickness and BOLD
as separate modality observations, fit once on TRAIN per checkpoint, transform
TEST without refitting, euclidean metric, 5 neighbors, min_dist=.1, spread=3,
random initialization, seed+fold, n_jobs=1. Each subject's two modality points
share its primary Leiden color, with distinct markers. Leiden itself continues
to use normalized rs+thick high-dimensional vectors, never UMAP coordinates.

Existing training coordinate TSVs are ID-validated and reused when compatible;
legacy files cannot prove their original fit settings or input provenance.
Use --umap-refit to obtain a fully fingerprinted new train-only fit.

--umap-space community provides an optional one-point-per-subject projection
of the exact Leiden space; this is explicitly different from training UMAP.

Each epoch exports one offline HTML selector for every saved k/resolution
combination, one coordinate TSV, a provenance file, and PNGs for all combinations
(unless --umap-no-png). --umap-write-tables adds potentially large per-combination
coordinate-label tables. Train/test communities are independent; color/ID
correspondence across splits, settings or epochs must not be inferred.
The test split never fits UMAP or selects training settings. No GPU, PyTorch,
raw MRI loading, or encoder inference is used. Seeded UMAP uses one CPU job;
--cpu-threads limits numerical library threads.
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
        required=False,
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
    p.add_argument(
        "--plot-only", type=Path, help="Existing stability.tsv; regenerate plots only"
    )
    p.add_argument("--zlim", nargs=2, type=float, metavar=("MIN", "MAX"))
    p.add_argument("--clim", nargs=2, type=float, metavar=("MIN", "MAX"))
    p.add_argument(
        "--mesh-epoch", default="last", help="last, best-train, or epoch number"
    )
    p.add_argument(
        "--umap-only",
        action="store_true",
        help="Reuse saved sweep labels; do not run Leiden or mesh generation",
    )
    p.add_argument(
        "--sweep-dir",
        type=Path,
        help="Existing sweep directory containing labels/; defaults to output-dir",
    )
    p.add_argument(
        "--skip-umap", action="store_true", help="Run original sweep/mesh behavior only"
    )
    p.add_argument(
        "--umap-space",
        choices=("modalities", "community"),
        default="modalities",
        help="Default follows training UMAP: two modality points per subject",
    )
    p.add_argument("--umap-neighbors", type=int, default=5)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--umap-spread", type=float, default=3.0)
    p.add_argument("--umap-metric", default="euclidean")
    p.add_argument(
        "--umap-refit",
        action="store_true",
        help="Ignore coordinate caches; fit TRAIN once per epoch",
    )
    p.add_argument(
        "--umap-no-png",
        action="store_true",
        help="Only interactive HTML; useful for very large grids",
    )
    p.add_argument(
        "--umap-write-tables",
        action="store_true",
        help="Also save a long coordinate-label TSV per combination",
    )
    p.add_argument("--umap-dpi", type=int, default=150)
    a = p.parse_args()
    import math

    for name in ("zlim", "clim"):
        limits = getattr(a, name)
        if limits is not None and (
            not all(math.isfinite(x) for x in limits) or limits[0] >= limits[1]
        ):
            p.error(f"--{name} requires finite MIN < MAX")
    if a.umap_only and (a.plot_only is not None or a.skip_umap):
        p.error("--umap-only cannot be combined with --plot-only or --skip-umap")
    if a.umap_only and a.input_dir is None:
        p.error("--umap-only requires --input-dir pointing to projection snapshots")
    if (
        a.umap_neighbors < 2
        or a.umap_dpi < 1
        or not (math.isfinite(a.umap_min_dist) and math.isfinite(a.umap_spread))
        or not 0 <= a.umap_min_dist <= a.umap_spread
        or a.umap_spread <= 0
    ):
        p.error("Invalid UMAP neighbors, distance/spread, or DPI")
    if a.input_dir is None and a.plot_only is None:
        p.error("Supply --input-dir for a sweep or --plot-only for saved results")
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


def plot_epoch_meshes(subset, out, split, metric, lower, upper, zlim):
    """Plot all epochs against k or resolution, without averaging parameters.

    Each HTML slider selects the other, fixed parameter. PNGs contain one
    surface panel per fixed parameter, so all parameter combinations remain
    visible. All panels and both views share the supplied ARI/color limits.
    """
    import numpy as np
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import math

    epochs = sorted({int(r["epoch"]) for r in subset})
    norm = Normalize(lower, upper)
    for varying, fixed, label in [
        ("k", "resolution", "Leiden k"),
        ("resolution", "k", "Resolution"),
    ]:
        values = sorted({float(r[varying]) for r in subset})
        fixed_values = sorted({float(r[fixed]) for r in subset})
        X, Y = np.meshgrid(epochs, values)
        frames = []
        columns = min(3, len(fixed_values))
        nrows = math.ceil(len(fixed_values) / columns)
        png = plt.figure(figsize=(6 * columns, 5 * nrows))
        for index, fixed_value in enumerate(fixed_values):
            lookup = {
                (int(r["epoch"]), float(r[varying])): float(r[metric])
                for r in subset
                if float(r[fixed]) == fixed_value
            }
            Z = np.array(
                [[lookup.get((e, value), np.nan) for e in epochs] for value in values]
            )
            title = f"{split}: {metric}; fixed {fixed}={fixed_value:g}"
            if min(Z.shape) > 1:
                trace = go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    cmin=lower,
                    cmax=upper,
                    colorscale="Viridis",
                    colorbar=dict(title="ARI"),
                    connectgaps=False,
                    hovertemplate="epoch=%{x}<br>parameter=%{y}<br>ARI=%{z:.5f}<extra></extra>",
                )
            else:
                trace = go.Scatter3d(
                    x=X.ravel(),
                    y=Y.ravel(),
                    z=Z.ravel(),
                    mode="lines",
                    line=dict(
                        color=Z.ravel(),
                        colorscale="Viridis",
                        cmin=lower,
                        cmax=upper,
                        showscale=True,
                        colorbar=dict(title="ARI"),
                    ),
                )
            frames.append(
                go.Frame(name=str(fixed_value), data=[trace], layout=dict(title=title))
            )
            Z_display = np.where((Z >= zlim[0]) & (Z <= zlim[1]), Z, np.nan)
            ax = png.add_subplot(nrows, columns, index + 1, projection="3d")
            if min(Z.shape) > 1 and np.isfinite(Z).any():
                ax.plot_surface(
                    X,
                    Y,
                    np.ma.masked_invalid(Z_display),
                    cmap="viridis",
                    norm=norm,
                    edgecolor="k",
                    linewidth=0.15,
                    rcount=Z.shape[0],
                    ccount=Z.shape[1],
                )
            else:
                ax.plot(X.ravel(), Y.ravel(), Z.ravel())
            ax.set(
                xlabel="Epoch",
                ylabel=label,
                zlabel="ARI",
                zlim=zlim,
                title=f"{fixed} = {fixed_value:g}",
            )
        fig = go.Figure(data=frames[0].data, frames=frames)
        fig.update_layout(
            title=f"{split}: {metric}; fixed {fixed}={fixed_values[0]:g}",
            scene=dict(
                xaxis_title="Epoch",
                yaxis_title=label,
                zaxis_title="ARI",
                zaxis=dict(range=list(zlim)),
            ),
            height=850,
            sliders=[
                dict(
                    currentvalue=dict(prefix=f"{fixed}: "),
                    steps=[
                        dict(
                            label=f"{value:g}",
                            method="animate",
                            args=[
                                [str(value)],
                                dict(
                                    mode="immediate",
                                    frame=dict(duration=0, redraw=True),
                                    transition=dict(duration=0),
                                ),
                            ],
                        )
                        for value in fixed_values
                    ],
                )
            ],
        )
        stem = f"{split}_{metric}_epochs_{varying}"
        fig.write_html(out / f"{stem}.html", include_plotlyjs=True)
        png.suptitle(f"{split}: {metric} across all epochs", fontsize=16)
        png.subplots_adjust(
            top=0.82 if nrows == 1 else 0.92, right=0.88, hspace=0.25, wspace=0.15
        )
        bar_ax = png.add_axes([0.92, 0.2, 0.015, 0.6])
        png.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=bar_ax, label="ARI"
        )
        png.savefig(out / f"{stem}.png", dpi=160, bbox_inches="tight")
        plt.close(png)
        logging.info("Saved all-epoch %s mesh and PNG panels: %s", varying, stem)


def plot_results(rows, out, zlim=None, clim=None, mesh_epoch="last"):
    """Plot measured ARI surfaces and a contour volume without rerunning Leiden.

    Surface HTML files show k versus resolution with ARI as height and color;
    an epoch slider selects the checkpoint. PNG files show the last finite
    checkpoint. Volume HTML files retain k, resolution, and epoch as axes and
    display ARI isosurfaces. Contours interpolate between measured grid points
    for visualization only; saved metrics are never smoothed or changed.

    Color limits use the finite observed range pooled across train/test for
    each metric, making splits comparable. Constant metrics receive a small
    nonzero display range. Undefined temporal ARIs are omitted.
    """
    import numpy as np
    import plotly.graph_objects as go
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    chosen_epoch = None
    if mesh_epoch == "best-train":
        scores = {}
        for row in rows:
            value = float(row["temporal_ari"])
            if row["split"] == "train" and np.isfinite(value):
                scores.setdefault(int(row["epoch"]), []).append(value)
        if scores:
            chosen_epoch = max(scores, key=lambda e: (float(np.mean(scores[e])), e))
            logging.info(
                "Parameter mesh epoch=%d: highest mean training temporal ARI over grid",
                chosen_epoch,
            )
    elif mesh_epoch != "last":
        chosen_epoch = int(mesh_epoch)
    for metric in ("temporal_ari", "seed_ari"):
        observed = np.array([float(r[metric]) for r in rows])
        observed = observed[np.isfinite(observed)]
        if not len(observed):
            logging.warning("No finite %s values; plots skipped", metric)
            continue
        lower, upper = float(observed.min()), float(observed.max())
        if upper - lower < 1e-10:
            lower, upper = max(-1.0, lower - 0.01), min(1.0, upper + 0.01)
        axis_lower, axis_upper = zlim if zlim is not None else (lower, upper)
        if clim is not None:
            lower, upper = clim
        elif zlim is not None:
            lower, upper = zlim
        norm = Normalize(lower, upper)
        for split in ("train", "test"):
            subset = [r for r in rows if r["split"] == split]
            if not subset:
                continue
            ks = sorted({int(r["k"]) for r in subset})
            resolutions = sorted({float(r["resolution"]) for r in subset})
            epochs = sorted({int(r["epoch"]) for r in subset})
            X, Y = np.meshgrid(ks, resolutions)
            frames = []
            grids = []
            for epoch in epochs:
                lookup = {
                    (int(r["k"]), float(r["resolution"])): float(r[metric])
                    for r in subset
                    if int(r["epoch"]) == epoch
                }
                C = np.array(
                    [[lookup.get((k, res), np.nan) for k in ks] for res in resolutions]
                )
                if not np.isfinite(C).any():
                    continue
                grids.append((epoch, C))
                if len(ks) > 1 and len(resolutions) > 1:
                    trace = go.Surface(
                        x=X,
                        y=Y,
                        z=C,
                        surfacecolor=C,
                        cmin=lower,
                        cmax=upper,
                        colorscale="Viridis",
                        colorbar=dict(title="ARI"),
                        contours=dict(
                            z=dict(show=True, usecolormap=True, project_z=True)
                        ),
                        hovertemplate="k=%{x}<br>resolution=%{y}<br>ARI=%{z:.5f}<extra></extra>",
                    )
                else:
                    trace = go.Scatter3d(
                        x=X.ravel(),
                        y=Y.ravel(),
                        z=C.ravel(),
                        mode="lines+markers",
                        marker=dict(
                            color=C.ravel(),
                            colorscale="Viridis",
                            cmin=lower,
                            cmax=upper,
                            showscale=True,
                            colorbar=dict(title="ARI"),
                        ),
                    )
                frames.append(
                    go.Frame(
                        name=str(epoch),
                        data=[trace],
                        layout=dict(title=f"{split}: {metric} — epoch {epoch}"),
                    )
                )
            if not grids:
                continue
            selected_index = len(grids) - 1
            if chosen_epoch is not None:
                matches = [i for i, (e, _) in enumerate(grids) if e == chosen_epoch]
                if not matches:
                    raise ValueError(
                        f"Requested mesh epoch {chosen_epoch} has no finite {metric} for {split}"
                    )
                selected_index = matches[0]
            fig = go.Figure(data=frames[selected_index].data, frames=frames)
            fig.update_layout(
                title=f"{split}: {metric} — epoch {grids[selected_index][0]}",
                scene=dict(
                    xaxis_title="Leiden k",
                    yaxis_title="Resolution",
                    zaxis_title="ARI",
                    zaxis=dict(range=[axis_lower, axis_upper]),
                ),
                height=800,
                sliders=[
                    dict(
                        active=selected_index,
                        currentvalue=dict(prefix="Epoch: "),
                        steps=[
                            dict(
                                label=f.name,
                                method="animate",
                                args=[
                                    [f.name],
                                    dict(
                                        mode="immediate",
                                        frame=dict(duration=0, redraw=True),
                                        transition=dict(duration=0),
                                    ),
                                ],
                            )
                            for f in frames
                        ],
                    )
                ],
            )
            fig.write_html(out / f"{split}_{metric}_mesh.html", include_plotlyjs=True)
            epoch, C = grids[selected_index]
            static = plt.figure(figsize=(11, 8))
            ax = static.add_subplot(111, projection="3d")
            if min(X.shape) > 1:
                ax.plot_surface(
                    X,
                    Y,
                    C,
                    cmap="viridis",
                    norm=norm,
                    edgecolor="k",
                    linewidth=0.25,
                    antialiased=True,
                    rcount=C.shape[0],
                    ccount=C.shape[1],
                )
                if np.isfinite(C).all() and np.ptp(C) > 1e-10:
                    ax.contour(
                        X,
                        Y,
                        C,
                        zdir="z",
                        offset=axis_lower,
                        cmap="viridis",
                        norm=norm,
                        levels=8,
                    )
            else:
                ax.plot(X.ravel(), Y.ravel(), C.ravel(), color="gray")
                ax.scatter(
                    X.ravel(),
                    Y.ravel(),
                    C.ravel(),
                    c=C.ravel(),
                    cmap="viridis",
                    norm=norm,
                )
            ax.set(
                xlabel="Leiden k",
                ylabel="Resolution",
                zlabel="ARI",
                zlim=(axis_lower, axis_upper),
                title=f"{split}: {metric} — epoch {epoch}",
            )
            static.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
                ax=ax,
                shrink=0.65,
                label="ARI",
            )
            static.savefig(
                out / f"{split}_{metric}_mesh.png", dpi=180, bbox_inches="tight"
            )
            plt.close(static)
            # A scalar field over k × resolution × epoch: genuine 3D contours.
            if len(ks) > 1 and len(resolutions) > 1 and len(grids) > 1:
                K, R, E = np.meshgrid(
                    ks, resolutions, [e for e, _ in grids], indexing="ij"
                )
                V = np.stack([c.T for _, c in grids], axis=2)
                finite = V[np.isfinite(V)]
                if len(finite) and np.ptp(finite) > 1e-10:
                    vmin, vmax = float(finite.min()), float(finite.max())
                    volume = go.Figure(
                        go.Isosurface(
                            x=K.ravel(),
                            y=R.ravel(),
                            z=E.ravel(),
                            value=V.ravel(),
                            isomin=vmin + 0.1 * (vmax - vmin),
                            isomax=vmax - 0.1 * (vmax - vmin),
                            surface_count=7,
                            opacity=0.35,
                            cmin=lower,
                            cmax=upper,
                            colorscale="Viridis",
                            colorbar=dict(title="ARI"),
                            caps=dict(x_show=False, y_show=False, z_show=False),
                        )
                    )
                    volume.update_layout(
                        title=f"{split}: {metric} — 3D ARI contours (interpolated display)",
                        scene=dict(
                            xaxis_title="Leiden k",
                            yaxis_title="Resolution",
                            zaxis_title="Epoch",
                        ),
                        height=850,
                    )
                    volume.write_html(
                        out / f"{split}_{metric}_contours.html", include_plotlyjs=True
                    )
                else:
                    logging.info(
                        "Constant field: no nontrivial volume contours for %s %s",
                        split,
                        metric,
                    )
            plot_epoch_meshes(
                subset, out, split, metric, lower, upper, (axis_lower, axis_upper)
            )
            logging.info(
                "Saved %s %s mesh; ARI color limits %.6f to %.6f",
                split,
                metric,
                lower,
                upper,
            )


def read_umap_data(folder):
    """Load and validate both raw modality arrays for one saved checkpoint.

    Returns a split-indexed dictionary. No encoder or MRI image is loaded.
    Input row order is retained; all later joins use explicit subject IDs.
    """
    import numpy as np

    result = {}
    for split in ("train", "test"):
        path = folder / f"{split}_embeddings.npz"
        load_snapshot(path)
        with np.load(path, allow_pickle=False) as saved:
            result[split] = {
                key: saved[key].copy() for key in ("rs", "thick", "subject_id")
            }
            result[split]["subject_id"] = result[split]["subject_id"].astype(str)
            result[split]["site"] = (
                saved["site"].astype(str)
                if "site" in saved
                else np.full(len(saved["rs"]), "unknown")
            )
    if set(result["train"]["subject_id"]) & set(result["test"]["subject_id"]):
        raise ValueError(f"{folder}: overlapping train/test subject IDs")
    return result


def umap_vectors(data, space):
    """Return original stacked modality vectors or the Leiden community space.

    ``modalities`` exactly follows the training helper's interleaved order:
    thickness subject 0, BOLD subject 0, thickness subject 1, BOLD subject 1.
    ``community`` instead plots one normalized rs+thick vector per subject.
    """
    import numpy as np

    if space == "modalities":
        return np.stack([data["thick"], data["rs"]], axis=1).reshape(
            -1, data["rs"].shape[1]
        )
    z = data["rs"] + data["thick"]
    return (z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)).astype(
        np.float32
    )


def read_coordinates(path, data, space):
    """Validate a coordinate TSV and align its rows to the saved subject arrays.

    Duplicate, missing, extra, or nonfinite coordinate records are rejected.
    Existing TSVs must contain split, subject_id, modality, x, and y columns.
    Original training TSVs lack input fingerprints: reusing those confirms ID
    compatibility but cannot establish the original input or software provenance.
    """
    import numpy as np

    modalities = ("thick", "rs") if space == "modalities" else ("community",)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not {"split", "subject_id", "modality", "x", "y"}.issubset(
            reader.fieldnames or []
        ):
            raise ValueError(f"{path}: incompatible UMAP coordinate columns")
        lookup = {}
        for row in reader:
            key = (row["split"], row["subject_id"], row["modality"])
            if key in lookup:
                raise ValueError(f"{path}: duplicate coordinate key {key}")
            lookup[key] = [float(row["x"]), float(row["y"])]
    expected = {
        (s, subject, m)
        for s, d in data.items()
        for subject in d["subject_id"]
        for m in modalities
    }
    if set(lookup) != expected:
        raise ValueError(
            f"{path}: coordinates do not match current subject/split/modality keys"
        )
    xy = {
        s: np.asarray(
            [lookup[(s, subject, m)] for subject in d["subject_id"] for m in modalities]
        )
        for s, d in data.items()
    }
    if any(not np.isfinite(v).all() for v in xy.values()):
        raise ValueError(f"{path}: nonfinite coordinates")
    return xy


def write_coordinates(path, data, xy, space):
    """Save one reusable coordinate table per epoch, independent of Leiden settings."""
    modalities = ("thick", "rs") if space == "modalities" else ("community",)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["subject_id", "site", "split", "modality", "x", "y"])
        for split, d in data.items():
            for i, subject in enumerate(d["subject_id"]):
                for j, modality in enumerate(modalities):
                    x, y = xy[split][len(modalities) * i + j]
                    writer.writerow([subject, d["site"][i], split, modality, x, y])


def obtain_umap_coordinates(folder, output, data, args):
    """Reuse compatible coordinates or fit once on train and transform test.

    Own caches are validated against both input-file hashes and UMAP settings.
    Original training coordinates are reused only for the original modality
    space and default parameters. Use --umap-refit to bypass either cache.
    Reused legacy coordinates have ID validation, not verified fit provenance.
    Newly fitted models use one CPU job for deterministic seeded UMAP. The
    reducer is saved for provenance; this script never loads pickle files.
    """
    import hashlib
    import importlib.metadata
    import pickle

    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    seed = args.seed + args.fold
    signature = dict(
        space=args.umap_space,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        spread=args.umap_spread,
        metric=args.umap_metric,
        seed=seed,
        input_hashes={s: digest(folder / f"{s}_embeddings.npz") for s in data},
    )
    coordinate_path = output / "umap_coordinates.tsv"
    metadata_path = output / "umap_metadata.json"
    if not args.umap_refit and coordinate_path.is_file() and metadata_path.is_file():
        meta = json.loads(metadata_path.read_text())
        if meta.get("signature") == signature:
            logging.info("UMAP cache hit: %s", coordinate_path)
            return read_coordinates(coordinate_path, data, args.umap_space)
    original = folder / "umap_coordinates.tsv"
    defaults = (
        args.umap_space == "modalities"
        and args.umap_neighbors == 5
        and args.umap_min_dist == 0.1
        and args.umap_spread == 3.0
        and args.umap_metric == "euclidean"
    )
    if not args.umap_refit and defaults and original.is_file():
        try:
            xy = read_coordinates(original, data, args.umap_space)
        except (ValueError, KeyError) as exc:
            logging.warning(
                "Original coordinates incompatible; fitting train UMAP: %s", exc
            )
        else:
            write_coordinates(coordinate_path, data, xy, args.umap_space)
            metadata_path.write_text(
                json.dumps(
                    dict(
                        signature=signature,
                        source=str(original),
                        fit_provenance="Legacy training TSV reused; subject IDs validated; original fit settings/seed not verified",
                        original_coordinate_sha256=digest(original),
                    ),
                    indent=2,
                )
            )
            logging.info(
                "Reused original training coordinates (ID-validated): %s", original
            )
            return xy
    import umap

    train = umap_vectors(data["train"], args.umap_space)
    test = umap_vectors(data["test"], args.umap_space)
    neighbors = min(args.umap_neighbors, len(train) - 1)
    logging.info(
        "UMAP FIT TRAIN ONLY: shape=%s neighbors=%d space=%s seed=%d; n_jobs=1",
        train.shape,
        neighbors,
        args.umap_space,
        seed,
    )
    reducer = umap.UMAP(
        n_neighbors=neighbors,
        min_dist=args.umap_min_dist,
        spread=args.umap_spread,
        n_components=2,
        metric=args.umap_metric,
        init="random",
        random_state=seed,
        transform_seed=seed,
        n_jobs=1,
    )
    xy = {"train": reducer.fit_transform(train)}
    logging.info("UMAP TRANSFORM TEST ONLY (no fit): shape=%s", test.shape)
    xy["test"] = reducer.transform(test)
    write_coordinates(coordinate_path, data, xy, args.umap_space)
    with (output / "umap_train_fitted.pkl").open("wb") as handle:
        pickle.dump(reducer, handle)
    metadata_path.write_text(
        json.dumps(
            dict(
                signature=signature,
                source="new train-only UMAP fit",
                effective_neighbors=neighbors,
                versions={
                    m: importlib.metadata.version(m)
                    for m in ("numpy", "scikit-learn", "umap-learn")
                },
            ),
            indent=2,
        )
    )
    return xy


def load_epoch_partitions(labels_dir, epoch, data, args):
    """Read primary Leiden labels, validating and joining them by subject ID.

    Each train/test partition is independent. k, resolution, and seed are read
    from each label archive, not guessed from its filename or command defaults.
    --umap-only never runs Leiden or computes replacement partitions.
    """
    import numpy as np

    result = {}
    for split in ("train", "test"):
        folder = labels_dir / split / f"simclr_epoch_{epoch:04d}"
        files = sorted(folder.glob("*.npz"))
        if not files:
            raise FileNotFoundError(
                f"No saved partitions in {folder}; supply the existing sweep directory"
            )
        result[split] = {}
        for path in files:
            with np.load(path, allow_pickle=False) as d:
                k, res = int(d["k"]), float(d["resolution"])
                ids, labels = d["subject_id"].astype(str), d["labels"]
                if (
                    ids.ndim != 1
                    or labels.shape != ids.shape
                    or len(set(ids)) != len(ids)
                ):
                    raise ValueError(f"{path}: malformed labels or IDs")
                if set(ids) != set(data[split]["subject_id"]):
                    raise ValueError(
                        f"{path}: saved labels belong to different subjects"
                    )
                if int(d["epoch"]) != epoch:
                    raise ValueError(f"{path}: wrong epoch metadata")
                if (k, res) in result[split]:
                    raise ValueError(
                        f"Duplicate saved k/resolution combination in {folder}"
                    )
                lookup = dict(zip(ids, labels))
                aligned = np.array(
                    [lookup[s] for s in data[split]["subject_id"]], dtype=int
                )
                result[split][(k, res)] = dict(
                    labels=aligned, seed=int(d["primary_seed"]), source=str(path)
                )
        logging.info(
            "Loaded %d %s partitions at epoch %d", len(result[split]), split, epoch
        )
    if result["train"].keys() != result["test"].keys():
        raise ValueError(f"Epoch {epoch}: train/test parameter combinations differ")
    return result


def community_palette(labels, split):
    """Return categorical colors; IDs and colors do not match communities across splits."""
    import colorsys

    values = sorted(set(map(int, labels)))
    offset = 0.08 if split == "train" else 0.48
    return {
        v: "#%02x%02x%02x"
        % tuple(
            int(c * 255)
            for c in colorsys.hsv_to_rgb((offset + i * 0.61803398875) % 1.0, 0.68, 0.82)
        )
        for i, v in enumerate(values)
    }


def render_community_umaps(output, epoch, data, xy, partitions, args):
    """Write a compact offline HTML selector and optional PNG for every combination.

    Coordinates are stored once, not duplicated for every configuration. Each
    subject's two modality markers share its community color. Train/test panels
    have shared axis limits and independent community IDs. Optional long TSVs
    join coordinates and primary labels explicitly for each combination.
    """
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from plotly.offline import get_plotlyjs

    modalities = ("thick", "rs") if args.umap_space == "modalities" else ("community",)
    combos = sorted(partitions["train"])
    payload = dict(
        epoch=epoch,
        space=args.umap_space,
        coords={},
        combinations=[],
        note="Independent train/test communities. Community numbers/colors do not imply correspondence across splits, checkpoints, or parameter combinations.",
    )
    all_xy = np.concatenate(list(xy.values()))
    bounds = []
    for dim in (0, 1):
        lo, hi = float(all_xy[:, dim].min()), float(all_xy[:, dim].max())
        padding = max((hi - lo) * 0.05, 0.01)
        bounds.append([lo - padding, hi + padding])
    payload["bounds"] = bounds
    for split, d in data.items():
        payload["coords"][split] = dict(
            x=xy[split][:, 0].tolist(),
            y=xy[split][:, 1].tolist(),
            subject_id=np.repeat(d["subject_id"], len(modalities)).tolist(),
            site=np.repeat(d["site"], len(modalities)).tolist(),
            modalities=list(modalities),
        )
    for index, (k, res) in enumerate(combos):
        logging.info(
            "UMAP COLORS %d/%d epoch=%d k=%d resolution=%g (no new UMAP fit)",
            index + 1,
            len(combos),
            epoch,
            k,
            res,
        )
        entry = dict(k=k, resolution=res, splits={})
        for split in ("train", "test"):
            part = partitions[split][(k, res)]
            palette = community_palette(part["labels"], split)
            entry["splits"][split] = dict(
                labels=part["labels"].tolist(),
                seed=part["seed"],
                palette=palette,
                n_communities=len(palette),
            )
        payload["combinations"].append(entry)
        stem = f"communities_epoch_{epoch:04d}_k_{k}_resolution_{res!r}"
        if args.umap_write_tables:
            with (output / f"{stem}.tsv").open("w", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(
                    [
                        "epoch",
                        "split",
                        "subject_id",
                        "site",
                        "modality",
                        "x",
                        "y",
                        "k",
                        "resolution",
                        "seed",
                        "community",
                    ]
                )
                for split, d in data.items():
                    part = partitions[split][(k, res)]
                    for i, subject in enumerate(d["subject_id"]):
                        for j, modality in enumerate(modalities):
                            x, y = xy[split][len(modalities) * i + j]
                            writer.writerow(
                                [
                                    epoch,
                                    split,
                                    subject,
                                    d["site"][i],
                                    modality,
                                    x,
                                    y,
                                    k,
                                    res,
                                    part["seed"],
                                    part["labels"][i],
                                ]
                            )
        if not args.umap_no_png:
            fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
            for ax, split in zip(axes, ("train", "test")):
                part = partitions[split][(k, res)]
                palette = community_palette(part["labels"], split)
                colors = [palette[int(v)] for v in part["labels"]]
                for j, modality in enumerate(modalities):
                    coords = xy[split][j :: len(modalities)]
                    ax.scatter(
                        coords[:, 0],
                        coords[:, 1],
                        c=colors,
                        s=30,
                        alpha=0.65,
                        marker="^" if modality == "thick" else "o",
                        linewidths=0,
                    )
                counts = {v: int((part["labels"] == v).sum()) for v in palette}
                handles = [
                    Line2D(
                        [],
                        [],
                        marker="o",
                        linestyle="",
                        color=palette[v],
                        label=f"{split.upper()} C{v}: n={counts[v]}",
                    )
                    for v in palette
                ]
                #ax.legend(
                #    handles=handles,
                #    fontsize=10,
                #    ncol=max(1, (len(handles) + 19) // 20),
                #    loc="best",
                #)
                ax.set(
                    title=f'{split}: {len(palette)} communities; Leiden seed={part["seed"]}',
                    xlabel="UMAP 1",
                    ylabel="UMAP 2",
                    xlim=bounds[0],
                    ylim=bounds[1],
                )
                ax.title.set_fontsize(16)
                ax.xaxis.label.set_fontsize(14)
                ax.yaxis.label.set_fontsize(14)
                ax.tick_params(axis="both", labelsize=12)

            fig.suptitle(
                f"Epoch {epoch} | Leiden k={k}, resolution={res:g} | {args.umap_space}"
            )
            fig.text(
                0.5,
                0.015,
                (
                    "UMAP fit on train; test transformed. Community IDs are independent. Triangle: anatomy; circle: BOLD."
                    if args.umap_space == "modalities"
                    else "UMAP fit on train; test transformed. Independent communities; one point per subject."
                ),
                ha="center",
                fontsize=18,
            )
            fig.tight_layout(rect=[0, 0.04, 1, 0.95])
            fig.savefig(output / f"{stem}.png", dpi=args.umap_dpi)
            plt.close(fig)

    # generates the json object with from the leiden communities
    # Static JSON payload: safe embedding; selectors update coordinates/colors locally.
    encoded = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    html = """<!doctype html><html><head><meta charset="utf-8"><title>Leiden communities on UMAP</title>
<style>body{font:15px sans-serif;margin:24px;color:#162536}select{padding:7px;margin:8px}#plot{height:78vh;min-height:600px}</style>
<script>__PLOTLY__</script></head><body><h2>Leiden communities on fixed UMAP coordinates</h2>
<label>Combination <select id="combo"></select></label><div id="note"></div><div id="plot"></div>
<script>const D=__DATA__;const selector=document.getElementById('combo');
D.combinations.forEach((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`k=${c.k}, resolution=${c.resolution}`;selector.appendChild(o)});
document.getElementById('note').textContent=D.note;
function draw(){const c=D.combinations[Number(selector.value)];let traces=[];
['train','test'].forEach((split,panel)=>{let q=D.coords[split],part=c.splits[split],M=q.modalities.length;
Object.keys(part.palette).forEach(label=>{q.modalities.forEach((mod,j)=>{let x=[],y=[],text=[];
part.labels.forEach((v,i)=>{if(v===Number(label)){let n=i*M+j;x.push(q.x[n]);y.push(q.y[n]);text.push(`${split}: ${q.subject_id[n]}<br>site=${q.site[n]}<br>${mod}<br>community=${label}`)}});
traces.push({type:'scattergl',mode:'markers',x,y,text,hoverinfo:'text',name:`${split} C${label} ${mod}`,xaxis:panel?'x2':'x',yaxis:panel?'y2':'y',marker:{size:5,opacity:.7,color:part.palette[label],symbol:mod==='thick'?'triangle-up':'circle'}})})})});
Plotly.react('plot',traces,{title:`Epoch ${D.epoch} | k=${c.k}, resolution=${c.resolution} | Leiden seeds train=${c.splits.train.seed}, test=${c.splits.test.seed}`,grid:{rows:1,columns:2,pattern:'independent'},xaxis:{title:'Train UMAP 1',range:D.bounds[0]},yaxis:{title:'UMAP 2',range:D.bounds[1]},xaxis2:{title:'Test UMAP 1',range:D.bounds[0]},yaxis2:{title:'UMAP 2',range:D.bounds[1]},legend:{font:{size:10}},uirevision:`epoch-${D.epoch}`},{responsive:true});}
selector.onchange=draw;draw();</script></body></html>"""
    html = html.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", encoded)
    (output / "umap_communities.html").write_text(html, encoding="utf-8")
    (output / "community_plot_manifest.json").write_text(
        json.dumps(
            dict(
                epoch=epoch,
                space=args.umap_space,
                note=payload["note"],
                combinations=[
                    dict(
                        k=c["k"],
                        resolution=c["resolution"],
                        train_seed=c["splits"]["train"]["seed"],
                        test_seed=c["splits"]["test"]["seed"],
                    )
                    for c in payload["combinations"]
                ],
            ),
            indent=2,
        )
    )
    logging.info(
        "Saved epoch %d UMAP selector and %d PNGs",
        epoch,
        0 if args.umap_no_png else len(combos),
    )


def export_umap_communities(args):
    """Export checkpoint-wise UMAP plots using existing primary Leiden labels.

    The sweep directory contains ``labels/{split}/simclr_epoch_XXXX/*.npz``.
    Process every matching snapshot within epoch bounds, one at a time on CPU.
    --umap-only leaves stability.tsv, saved labels, and mesh plots untouched.
    """
    from threadpoolctl import threadpool_limits

    projections = args.input_dir.expanduser().resolve()
    sweep = (args.sweep_dir or args.output_dir).expanduser().resolve()
    output = args.output_dir.expanduser().resolve() / "umap_communities"
    output.mkdir(parents=True, exist_ok=True)
    snapshots = []
    for folder in projections.glob("simclr_epoch_*"):
        match = re.fullmatch(r"simclr_epoch_(\d+)", folder.name)
        if match and folder.is_dir():
            epoch = int(match.group(1))
            if epoch >= args.epoch_min and (
                args.epoch_max is None or epoch <= args.epoch_max
            ):
                snapshots.append((epoch, folder))
    snapshots.sort()
    if not snapshots or len({e for e, _ in snapshots}) != len(snapshots):
        raise ValueError("No matching snapshots or duplicate numeric epochs")
    if args.umap_only and (args.k_values or args.resolution_values):
        logging.info(
            "UMAP-only uses all combinations saved in labels; CLI k/resolution grids do not filter these"
        )
    with threadpool_limits(limits=args.cpu_threads):
        for number, (epoch, folder) in enumerate(snapshots, 1):
            logging.info(
                "UMAP checkpoint %d/%d epoch=%d", number, len(snapshots), epoch
            )
            destination = output / f"simclr_epoch_{epoch:04d}"
            destination.mkdir(parents=True, exist_ok=True)
            data = read_umap_data(folder)
            partitions = load_epoch_partitions(sweep / "labels", epoch, data, args)
            xy = obtain_umap_coordinates(folder, destination, data, args)
            render_community_umaps(destination, epoch, data, xy, partitions, args)
    logging.info("UMAP export complete: %s", output)


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
    if a.umap_only:
        a.output_dir = a.output_dir.expanduser().resolve()
        a.output_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(a.output_dir / "umap.log"),
            ],
        )
        export_umap_communities(a)
        return
    if a.plot_only is not None:
        a.output_dir = a.output_dir.expanduser().resolve()
        a.output_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(a.output_dir / "plot.log"),
            ],
        )
        with a.plot_only.expanduser().open(newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        plot_results(
            rows, a.output_dir, zlim=a.zlim, clim=a.clim, mesh_epoch=a.mesh_epoch
        )
        return
    a.input_dir = a.input_dir.expanduser().resolve()
    a.output_dir = a.output_dir.expanduser().resolve()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    ## comment to allow directory overwriting here**
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
    plot_results(rows, a.output_dir, zlim=a.zlim, clim=a.clim, mesh_epoch=a.mesh_epoch)
    if not a.skip_umap:
        export_umap_communities(a)
    logging.info("Complete. Results: %s", a.output_dir / "stability.tsv")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception(
            "Sweep failed; completed TSV rows and labels remain available"
        )
        sys.exit(1)
