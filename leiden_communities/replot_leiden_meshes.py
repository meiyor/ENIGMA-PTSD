#!/usr/bin/env python3
"""Replot saved Leiden stability results, with adjustable ARI height/color limits.

Usage
-----
python replot_leiden_meshes.py --tsv /path/stability.tsv --zlim 0.5 1.0

Outputs default to the TSV's own folder, replacing the matching plot files only.
No embeddings are loaded and no clustering is repeated. Both train/test and
both ARI metrics are plotted. PNGs show the last finite checkpoint; interactive
HTML meshes include an epoch slider. HTML volume contours keep epoch as their
third axis: --zlim applies to ARI surface height, not the contour epoch axis.

--zlim also sets color limits unless --clim is supplied separately. Values
outside the selected color range saturate; the saved measurements are unchanged.
Volume contours interpolate the measured grid only for display.

Dependencies: numpy matplotlib plotly
"""

import argparse
import csv
import logging
from pathlib import Path


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


def main():
    """Read a stability TSV and regenerate plots in its folder by default."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tsv", type=Path, required=True, help="Existing stability.tsv"
    )
    parser.add_argument("--output-dir", type=Path, help="Default: same folder as TSV")
    parser.add_argument(
        "--zlim",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="ARI surface height limits; also color limits by default",
    )
    parser.add_argument(
        "--clim",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Optional color limits independent of ARI height",
    )
    parser.add_argument(
        "--mesh-epoch",
        default="last",
        help="Parameter mesh checkpoint: last, best-train (highest mean training time-ARI over grid), or epoch number",
    )
    args = parser.parse_args()
    import math

    for name in ("zlim", "clim"):
        limits = getattr(args, name)
        if limits is not None and (
            not all(math.isfinite(v) for v in limits) or limits[0] >= limits[1]
        ):
            parser.error(f"--{name} needs two finite values with MIN < MAX")
    path = args.tsv.expanduser().resolve()
    out = args.output_dir.expanduser().resolve() if args.output_dir else path.parent
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out / "replot.log")],
    )
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"split", "epoch", "k", "resolution", "temporal_ari", "seed_ari"}
        if not required.issubset(reader.fieldnames or []):
            parser.error(f"TSV must contain columns: {sorted(required)}")
        rows = list(reader)
    if not rows:
        parser.error("TSV contains no result rows")
    logging.info("Reading %d results from %s; plotting only", len(rows), path)
    plot_results(rows, out, zlim=args.zlim, clim=args.clim, mesh_epoch=args.mesh_epoch)
    logging.info("Plots saved in %s", out)


if __name__ == "__main__":
    main()
