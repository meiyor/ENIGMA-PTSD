#!/usr/bin/env python3
"""Recreate training metric PNG/PDF plots from a fold's saved TSV/CSV files.

Requires pandas and matplotlib. No training, embeddings, or GPU are needed.
Existing figures with the same names in the output directory are overwritten.
Missing tables are reported and skipped; malformed tables raise an error.
"""

import argparse
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like
import pandas as pd


# Input stem, output stem, title, left column/label, right column/label,
# left axis label, right axis label, left limits, right limits.
PLOTS = (
    (
        "simclr_nmi_silhouette_by_epoch",
        "simclr_nmi_silhouette_twinx",
        "Training-community consistency and separation",
        "nmi_vs_previous", "NMI vs. previous Leiden check",
        "silhouette_cosine", "Cosine silhouette (full embedding)",
        "NMI (consecutive partitions)", "Cosine silhouette",
        (-0.02, 1.02), (-1.02, 1.02),
    ),
    (
        "simclr_subject_pair_nmi_silhouette_by_epoch",
        "simclr_subject_pair_nmi_silhouette_twinx",
        "BOLD–cortical thickness subject-pair alignment (minibatch mean)",
        "subject_pair_nmi", "Minibatch mean NMI",
        "subject_pair_silhouette_cosine", "Minibatch mean cosine silhouette",
        "NMI (subject pairing)", "Cosine silhouette",
        (-0.02, 1.02), (-1.02, 1.02),
    ),
    (
        "simclr_paired_cosine_mrr_by_epoch",
        "simclr_paired_cosine_mrr_twinx",
        "Full-cohort cosine margin and retrieval",
        "cosine_margin", "Positive-negative cosine margin",
        "mrr_bidirectional", "Bidirectional MRR",
        "Positive-negative cosine margin", "Mean reciprocal rank (MRR)",
        None, (-0.02, 1.02),
    ),
)


def replot_training_metrics(
    fold_dir,
    output_dir=None,
    *,
    left_color="tab:blue",
    right_color="tab:red",
    xtick_fontsize=10,
    ytick_fontsize=12,
    label_fontsize=14,
    title_fontsize=16,
    legend_fontsize=10,
    xtick_rotation=90,
    tick_every=1,
    marker_size=5,
    line_width=2.2,
    figsize=(12, 6.5),
    dpi=220,
):
    """Replot the three saved SimCLR paired-axis training metric figures.

    Parameters
    ----------
    fold_dir : str or pathlib.Path
        Main fold folder containing the three ``*_by_epoch.tsv`` tables
        named in PLOTS. CSV versions are accepted when the TSV is absent.
    output_dir : str or pathlib.Path, optional
        Destination; defaults to fold_dir. Original PNG/PDF names are reused.
    left_color, right_color : str
        Matplotlib color names or hex strings for curves and y-axis ticks.
    xtick_fontsize, ytick_fontsize : float
        Font sizes in points for epoch ticks and both y-axis ticks.
    label_fontsize, title_fontsize, legend_fontsize : float
        Font sizes in points for axis labels, figure titles, and legends.
    xtick_rotation : float
        Epoch-label rotation in degrees.
    tick_every : int
        Label every Nth saved epoch (not an epoch interval). All observations
        remain plotted, and the final epoch label is always included.
    marker_size, line_width : float
        Marker diameter in points and curve width in points.
    figsize : tuple of float
        Figure width and height in inches.
    dpi : int
        Output raster resolution.

    Returns
    -------
    list of pathlib.Path
        Paths of the PNG/PDF files successfully written.

    Notes
    -----
    Uses exactly the original metric columns, titles, legends, and y limits.
    Cosine margin retains automatic y limits. NaNs remain gaps; no smoothing,
    interpolation, or metric recomputation is performed. TSV takes precedence
    if both a TSV and CSV exist. Missing input tables produce warnings.
    """
    fold_dir = Path(fold_dir).expanduser()
    if not fold_dir.is_dir():
        raise NotADirectoryError(fold_dir)
    if not isinstance(tick_every, int) or tick_every < 1:
        raise ValueError("tick_every must be a positive integer")
    if not all(is_color_like(c) for c in (left_color, right_color)):
        raise ValueError("Colors must be valid Matplotlib names or hex strings")
    sizes = (xtick_fontsize, ytick_fontsize, label_fontsize, title_fontsize,
             legend_fontsize, marker_size, line_width, dpi, *figsize)
    if any(value <= 0 for value in sizes):
        raise ValueError("Font sizes, figure sizes, marker/line sizes, and dpi must be positive")
    destination = Path(output_dir).expanduser() if output_dir else fold_dir
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    for spec in PLOTS:
        (input_stem, output_stem, title, left_column, left_legend,
         right_column, right_legend, left_label, right_label,
         left_limits, right_limits) = spec
        source = next((fold_dir / f"{input_stem}.{ext}" for ext in ("tsv", "csv")
                       if (fold_dir / f"{input_stem}.{ext}").is_file()), None)
        if source is None:
            warnings.warn(f"Skipping missing table: {input_stem}.tsv/.csv", stacklevel=2)
            continue
        table = pd.read_csv(source, sep="\t" if source.suffix == ".tsv" else ",")
        required = ["epoch", left_column, right_column]
        missing = set(required) - set(table.columns)
        if missing:
            raise ValueError(f"{source}: missing columns {sorted(missing)}")
        table = table[required].apply(pd.to_numeric, errors="raise").sort_values("epoch")
        if table.empty or table["epoch"].isna().any() or table["epoch"].duplicated().any():
            raise ValueError(f"{source}: require nonempty data and unique nonmissing epochs")
        epochs = table["epoch"].to_numpy()
        ticks = list(epochs[::tick_every])
        if ticks[-1] != epochs[-1]:
            ticks.append(epochs[-1])
        fig, left = plt.subplots(figsize=figsize, dpi=dpi)
        try:
            right = left.twinx()
            lines = []
            for ax, column, label, color, marker in (
                (left, left_column, left_legend, left_color, "o"),
                (right, right_column, right_legend, right_color, "s"),
            ):
                line, = ax.plot(epochs, table[column], marker=marker, color=color,
                                linewidth=line_width, markersize=marker_size, label=label)
                lines.append(line)
                ax.tick_params(axis="y", colors=color, labelsize=ytick_fontsize)
                ax.spines["top"].set_visible(False)
            left.set_xlabel("SimCLR epoch", fontsize=label_fontsize)
            left.set_ylabel(left_label, fontsize=label_fontsize)
            right.set_ylabel(right_label, fontsize=label_fontsize)
            if left_limits is not None:
                left.set_ylim(*left_limits)
            else:
                left.axhline(0, color=left_color, alpha=0.25, linewidth=1)
                left.ticklabel_format(axis="y", style="plain", useOffset=False)
            right.set_ylim(*right_limits)
            left.set_xticks(ticks)
            left.tick_params(axis="x", labelrotation=xtick_rotation, labelsize=xtick_fontsize)
            left.grid(axis="y", color="#D9DFE6", linewidth=0.8, alpha=0.75)
            fig.suptitle(title, fontsize=title_fontsize)
            left.legend(handles=lines, loc="upper center", bbox_to_anchor=(0.5, 1.02),
                        frameon=False, ncol=2, fontsize=legend_fontsize)
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            for extension in ("png", "pdf"):
                target = destination / f"{output_stem}.{extension}"
                fig.savefig(target, dpi=dpi, bbox_inches="tight")
                written.append(target)
                print(f"Saved {target}")
        finally:
            plt.close(fig)
    if not written:
        raise FileNotFoundError(f"No matching metric tables in {fold_dir}")
    return written


def main():
    """Parse command-line style options and recreate available metric plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", required=True, help="Folder containing saved metric tables")
    parser.add_argument("--output-dir", help="Default: fold directory (overwrites matching figures)")
    parser.add_argument("--left-color", default="tab:blue")
    parser.add_argument("--right-color", default="tab:red")
    for name, default in (("xtick-fontsize", 10), ("ytick-fontsize", 12),
                          ("label-fontsize", 14), ("title-fontsize", 16),
                          ("legend-fontsize", 10), ("xtick-rotation", 90),
                          ("marker-size", 5), ("line-width", 2.2)):
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--tick-every", type=int, default=1, help="Label every Nth saved epoch")
    parser.add_argument("--figsize", type=float, nargs=2, default=(12, 6.5), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--dpi", type=int, default=220)
    replot_training_metrics(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
