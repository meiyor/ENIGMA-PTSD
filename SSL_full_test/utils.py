"""
   Module for adding utils function
   for SSL training and dataloader modules
   functionalities

   Utilities for ENIGMA-PTSD multimodal representation learning.

   This module provides:

   1. Subject-disjoint, site-stratified cross-validation folds.
   2. BOLD and cortical-thickness normalization.
   3. Training-fitted UMAP visualization.
   4. Age and sex metadata validation.
   5. Covariate construction for ComBat.
   6. Training-only ComBat harmonization and age/sex residualization.
   7. Cached harmonization transforms for multimodal training.

   Important
   ---------
   ComBat models and nuisance regressors are fitted exclusively on training
   subjects and subsequently applied to held-out subjects.

   The harmonization functions require overlapping sites between training
   and testing. They are not designed for leave-one-site-out evaluation.

   CTQ, diagnostic categories, and cluster assignments are not used by
   these preprocessing utilities.

"""
import sys
import os
import warnings
import json
import pickle
from importlib.metadata import version

import pandas as pd

# To ignore all warnings:
warnings.filterwarnings("ignore")
# To ignore specific categories of warnings (e.g., DeprecationWarning):
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # or "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.nn as nn

from pathlib import Path
import numpy as np

# import plotting tools
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

from loguru import logger

# set the seeds initialization here
seed = 42

# always define this seed before running your implementation!!
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# function for loading the latest checkpoint generated
def load_latest_ckpt_cort(
    ckpt_dir: str,
    device: torch.device,
    enc_4D_rsdata,
    enc_thick,
    optimizer_SSL=None,
    scheduler_SSL=None,
    ckpt_iter=None
):
    ckpt_dir = Path(ckpt_dir)

    if ckpt_iter is None:
       ckpt_files = sorted(ckpt_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    else:
       ckpt_files = sorted([p for p in ckpt_dir.glob("*.pth") if str(ckpt_iter) in p.name], key=lambda p: p.stat().st_mtime)

    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"No checkpoint files found in: {ckpt_dir}")

    latest = ckpt_files[-1]
    if ckpt_iter is None:
       logger.info(f"Loading latest checkpoint as {latest}")
    else:
       logger.info(f"Loading selected checkpoint as {latest}")

    ckpt = torch.load(latest, map_location=device)

    # ---- models loading process here
    enc_4D_rsdata.load_state_dict(ckpt["models"]["enc_4D_rsdata"], strict=True)
    enc_thick.load_state_dict(ckpt["models"]["enc_thick"], strict=True)

    # ---- optimizer / scheduler loading - necessary for replication
    if optimizer_SSL is not None and "optimizer_SSL" in ckpt:
        optimizer_SSL.load_state_dict(ckpt["optimizer_SSL"])

    if scheduler_SSL is not None and "scheduler_SSL" in ckpt:
        scheduler_SSL.load_state_dict(ckpt["scheduler_SSL"])

    start_iter = int(ckpt.get("iter", 0)) + 1

    return start_iter, str(latest)

# function for loading the latest checkpoint generated
def load_latest_ckpt_struct(
    ckpt_dir: str,
    device: torch.device,
    enc_surf,
    enc_thick,
    enc_vol,
    optimizer_SSL=None,
    scheduler_SSL=None,
    ckpt_iter=None
):
    ckpt_dir = Path(ckpt_dir)

    if ckpt_iter is None:
       ckpt_files = sorted(ckpt_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    else:
       ckpt_files = sorted([p for p in ckpt_dir.glob("*.pth") if str(ckpt_iter) in p.name], key=lambda p: p.stat().st_mtime)

    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"No checkpoint files found in: {ckpt_dir}")

    latest = ckpt_files[-1]
    if ckpt_iter is None:
       logger.info(f"Loading latest checkpoint as {latest}")
    else:
       logger.info(f"Loading selected checkpoint as {latest}")

    ckpt = torch.load(latest, map_location=device)

    # ---- models loading process here
    enc_vol.load_state_dict(ckpt["models"]["enc_vol"], strict=True)
    enc_thick.load_state_dict(ckpt["models"]["enc_thick"], strict=True)
    enc_surf.load_state_dict(ckpt["models"]["enc_surf"], strict=True)

    # ---- optimizer / scheduler loading - necessary for replication
    if optimizer_SSL is not None and "optimizer_SSL" in ckpt:
        optimizer_SSL.load_state_dict(ckpt["optimizer_SSL"])

    if scheduler_SSL is not None and "scheduler_SSL" in ckpt:
        scheduler_SSL.load_state_dict(ckpt["scheduler_SSL"])

    start_iter = int(ckpt.get("iter", 0)) + 1

    return start_iter, str(latest)

def load_latest_ckpt_4_mod(
    ckpt_dir: str,
    device: torch.device,
    enc_4D_rsdata,
    enc_alff,
    enc_falff,
    enc_reho,
    optimizer_SSL=None,
    scheduler_SSL=None,
    ckpt_iter=None
):
    ckpt_dir = Path(ckpt_dir)

    if ckpt_iter is None:
       ckpt_files = sorted(ckpt_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    else:
       ckpt_files = sorted([p for p in ckpt_dir.glob("*.pth") if str(ckpt_iter) in p.name], key=lambda p: p.stat().st_mtime)

    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"No checkpoint files found in: {ckpt_dir}")

    latest = ckpt_files[-1]
    logger.info(f"Loading latest checkpoint as {latest}")

    ckpt = torch.load(latest, map_location=device)

    # ---- models loading process here
    enc_4D_rsdata.load_state_dict(ckpt["models"]["enc_4D_rsdata"], strict=True)
    enc_alff.load_state_dict(ckpt["models"]["enc_alff"], strict=True)
    enc_falff.load_state_dict(ckpt["models"]["enc_falff"], strict=True)
    enc_reho.load_state_dict(ckpt["models"]["enc_reho"], strict=True)

    # ---- optimizer / scheduler loading - necessary for replication
    if optimizer_SSL is not None and "optimizer_SSL" in ckpt:
        optimizer_SSL.load_state_dict(ckpt["optimizer_SSL"])

    if scheduler_SSL is not None and "scheduler_SSL" in ckpt:
        scheduler_SSL.load_state_dict(ckpt["scheduler_SSL"])

    start_iter = int(ckpt.get("iter", 0)) + 1

    return start_iter, str(latest)

# function for reading the interim text files
def read_metric_txt(path_str: str):
    """
      reading the txt \n separated values
      for continuing with the training process
    """
    vals: list[float] = []
    for line in Path(path_str).read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        vals.append(float(s))
    return vals

# function for plotting the interim metrics
def plotting_twinx_variables(time_vector, data1, data2, title: str, x_label: str, y_label1: str, y_label2: str, folder_images: str, iter: int):
    """
    plot her twinx the variables you want to compare
    across number of epochs in this case.

    Plot two time series on twin y-axes and save.

    Parameters
    ----------
    time_vector : array-like
    data1, data2 : array-like
        Series aligned to `time_vector`.
    title, x_label, y_label1, y_label2 : str
    folder_images : str
    subj : str

    Returns
    -------
    None
    """

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(time_vector, data1, "b-", label=y_label1, linewidth=3)
    ax1.set_xlabel(x_label, fontsize=16)
    ax1.set_ylabel(y_label1, color="blue", fontsize=16)
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()

    ax2.plot(time_vector, data2, "r-", label=y_label2, linewidth=3)
    ax2.set_ylabel(y_label2, color="red", fontsize=16)
    ax2.tick_params(axis="y", labelcolor="red")

    ax1.grid(True)

    for tick in ax1.get_xticklabels():
        tick.set_fontsize(14)

    plt.title(title)
    fig.legend()
    fig.savefig(f"{folder_images}/{y_label1}_{y_label2}_{iter}.jpg")
    plt.close("all")

# function for define the metrics here..
def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def wasserstein_avg_over_dims(A: np.ndarray, B: np.ndarray) -> float:
    """
    A, B: (B, D) arrays.
    Computes 1D Wasserstein distance per dimension across the batch, then averages.
    """
    assert A.shape == B.shape, (A.shape, B.shape)
    D = A.shape[1]
    return float(np.mean([wasserstein_distance(A[:, d], B[:, d]) for d in range(D)]))


def pairwise_wasserstein(embeds_np: dict) -> dict:
    """
    embeds_np: dict name -> (B,D)
    returns dict (name_i, name_j) -> wasserstein distance
    """
    keys = list(embeds_np.keys())
    out = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ki, kj = keys[i], keys[j]
            out[(ki, kj)] = wasserstein_avg_over_dims(embeds_np[ki], embeds_np[kj])
    return out

def stack_modalities(embeds_np: dict):
    """
    Returns:
      X: (M*B, D)
      modality_labels: (M*B,) integers 0..M-1
    """
    keys = list(embeds_np.keys())
    X = np.concatenate([embeds_np[k] for k in keys], axis=0)
    B = embeds_np[keys[0]].shape[0]
    modality_labels = np.concatenate([np.full(B, i, dtype=int) for i in range(len(keys))], axis=0)
    return X, modality_labels, keys


def compute_nmi_via_kmeans(X: np.ndarray, true_labels: np.ndarray, n_clusters: int, seed: int = 42) -> float:
    """
    Cluster X, then compute NMI between cluster assignments and provided labels.
    """
    true_labels = np.asarray(true_labels)
    uniq = np.unique(true_labels)
    if uniq.size < 2:
        return float("nan")  # or 0.0 if you prefer
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed)
    pred = km.fit_predict(X)
    return float(normalized_mutual_info_score(true_labels, pred))

def compute_silhouette(X: np.ndarray, labels: np.ndarray, metric: str = "cosine") -> float:
    """
    Silhouette score for given labels.
    With normalized embeddings, cosine is usually a good choice.
    """
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if uniq.size < 2:
        return float("nan")  # or 0.0 if you prefer
    return float(silhouette_score(X, labels, metric=metric))

def make_subject_palette(subject_ids, palette_name="tab20"):
    """
    Create a deterministic subject->color mapping.
    subject_ids: iterable of subject labels (strings or ints)
    """
    uniq = list(dict.fromkeys(subject_ids))  # preserves first-seen order
    colors = sns.color_palette(palette_name, n_colors=len(uniq))
    return {sid: col for sid, col in zip(uniq, colors)}


def make_site_palette(site_names, palette_name="tab20"):
    """
    Create a deterministic site->color mapping (sorted by site name).
    """
    uniq = sorted(set(site_names))  # alphabetical order
    colors = sns.color_palette(palette_name, n_colors=len(uniq))
    return dict(zip(uniq, colors))


# initialization function here
def init_xavier(m: nn.Module, uniform: bool = True):
    """
    Initialize learnable weights of a module using Xavier/Glorot initialization.

    This helper is designed to be used with 'nn.Module.apply(...)' so it will be
    called recursively on every submodule of a model. It targets the common layers
    used in this project:

    - nn.Linear
    - nn.Conv3d
    - nn.MultiheadAttention (handles packed in_proj_* and out_proj)
    - Normalization layers (LayerNorm, BatchNorm3d) are set to identity init

    Parameters
    ----------
    m : torch.nn.Module
      The submodule currently visited by 'Module.apply'.
    uniform : bool, default=True
      If True, uses 'nn.init.xavier_uniform_'.
      If False, uses 'nn.init.xavier_normal_'.

    --------
    - For Linear/Conv3d:
      - Weight initialized with Xavier (uniform or normal).
      - Bias (if present) is zeroed.
    - For MultiheadAttention:
      - Initializes 'in_proj_weight' (or equivalent packed QKV weights) with Xavier.
      - Zeroes 'in_proj_bias' if present.
      - Initializes 'out_proj.weight' with Xavier and zeroes 'out_proj.bias'.
    - For LayerNorm/BatchNorm3d:
      - weight <- ones, bias <- zeros (safe identity init)
    """

    # Linear
    if isinstance(m, nn.Linear):
        if uniform:
            nn.init.xavier_uniform_(m.weight)
        else:
            nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    # Conv3d
    elif isinstance(m, nn.Conv3d):
        if uniform:
            nn.init.xavier_uniform_(m.weight)
        else:
            nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    # MultiheadAttention (handle its internal linear weights)
    elif isinstance(m, nn.MultiheadAttention):
        # in_proj (qkv packed) OR separate q/k/v weights
        if hasattr(m, "in_proj_weight") and m.in_proj_weight is not None:
            if uniform:
                nn.init.xavier_uniform_(m.in_proj_weight)
            else:
                nn.init.xavier_normal_(m.in_proj_weight)
            if m.in_proj_bias is not None:
                nn.init.zeros_(m.in_proj_bias)

        # out projection
        if uniform:
            nn.init.xavier_uniform_(m.out_proj.weight)
        else:
            nn.init.xavier_normal_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)

    # Norm layers (usually keep default, but safe to set)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.ones_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)

def auto_padding(kernel, dilation=(1, 1, 1)):
    """
    Compute "same-ish" symmetric padding for 3D kernels under dilation.

    For odd-valued kernel sizes, this returns the padding that preserves spatial
    dimensions for stride=1 (approximately "same" convolution). The formula used:

      padding_i = dilation_i * (kernel_i // 2)

    Parameters
    ----------
    kernel : tuple[int, int, int]
      3D kernel size (kD, kH, kW). Expected to be odd in most cases.
    dilation : tuple[int, int, int], default=(1,1,1)
      Dilation factors along each dimension.

    Returns
    -------
    tuple[int, int, int]
      Padding values (pD, pH, pW) suitable for nn.Conv3d(..., padding=...).
    """
    return tuple(d * (k // 2) for k, d in zip(kernel, dilation, strict=False))


# ================================================================
# 1. SUBJECT-DISJOINT CROSS-VALIDATION
# ================================================================

def make_subject_folds_by_site(
    frame,
    n_splits=5,
    seed=42,
):
    """
    Construct subject-disjoint, site-stratified cross-validation folds.

    Every selected site must occur in both training and testing
    partitions for every fold.

    Each subject is included in the testing partition exactly once.

    Stratification uses site labels, not diagnostic categories.

    Parameters
    ----------
    frame : pandas.DataFrame
        Subject manifest containing:
            participant_id : unique subject identifier
            site           : acquisition site

    n_splits : int, default=5
        Number of cross-validation folds.

    seed : int, default=42
        Random seed for reproducible stratification.

    Returns
    -------
    folds : list[dict]
        List of fold definitions containing training/testing indices
        and their corresponding site labels.

    Raises
    ------
    ValueError
        If subject IDs are duplicated, site labels are missing,
        or the selected sites contain too few subjects.

    Notes
    -----
    At least n_splits subjects per site are required.

    ComBat additionally requires at least two training subjects
    per site. No site is silently removed.
    """

    from sklearn.model_selection import StratifiedKFold

    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")

    if frame.participant_id.duplicated().any():
        raise ValueError(
            "Select one BOLD variant and one row per participant."
        )

    if (
        frame.site.isna().any()
        or (frame.site.astype(str).str.strip() == "").any()
    ):
        raise ValueError("Missing site labels.")

    counts = frame.groupby("site").size()

    if len(counts) < 2:
        raise ValueError(
            "At least two sites are required for ComBat."
        )

    too_small = counts[counts < n_splits]

    if len(too_small):
        raise ValueError(
            f"Every site needs >= {n_splits} eligible subjects: "
            f"{too_small.to_dict()}. "
            "Lower --n-splits or explicitly choose sites; "
            "no subjects/sites were silently dropped."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    folds = []
    tested = []
    expected = set(frame.site.astype(str))

    for k, (train, test) in enumerate(
        splitter.split(
            np.zeros(len(frame)),
            frame.site,
        ),
        start=1,
    ):

        a = frame.iloc[train]
        b = frame.iloc[test]

        if set(a.participant_id) & set(b.participant_id):
            raise AssertionError("Subject leakage.")

        if (
            set(a.site.astype(str)) != expected
            or set(b.site.astype(str)) != expected
        ):
            raise AssertionError(
                "Every site must occur in both train and test."
            )

        if (a.groupby("site").size() < 2).any():
            raise ValueError(
                "ComBat needs >= 2 training subjects/site; "
                "increase cohort or change folds."
            )

        folds.append(
            {
                "fold": k,
                "indices": {
                    "train": train.tolist(),
                    "test": test.tolist(),
                },
                "sites": {
                    "train": sorted(expected),
                    "test": sorted(expected),
                },
            }
        )

        tested.extend(test.tolist())

    if sorted(tested) != list(range(len(frame))):
        raise AssertionError(
            "Test coverage is not exactly once."
        )

    return folds


# ================================================================
# 2. SAMPLE NORMALIZATION
# ================================================================

def normalize_model_sample(item):
    """
    Normalize an individual BOLD and cortical-thickness sample.

    Applies the same normalization conventions as the original
    ENIGMA dataloader:

        BOLD:
            Spatial z-score normalization independently
            for every temporal volume.

        Thickness:
            Global z-score normalization across the 3D volume.

    Parameters
    ----------
    item : dict
        Individual sample from CleanENIGMADataset.

        Required fields:
            rs_data   : (D, H, W, T)
            thickness : (D, H, W)

    Returns
    -------
    bold : numpy.ndarray
        Spatially normalized BOLD data, float32.

    thick : numpy.ndarray
        Normalized thickness volume, float32.

    support_b : numpy.ndarray
        Boolean spatial support of the BOLD acquisition.

    support_t : numpy.ndarray
        Boolean spatial support of the cortical-thickness volume.

    Notes
    -----
    CleanENIGMADataset must be initialized with:

        normalize_bold=False
        normalize_thickness=False

    Otherwise, normalization would be applied twice.

    Temporal padding must not be present in an individual sample
    passed to this function.
    """

    def array(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()

        return np.asarray(value)

    raw_bold = array(
        item["rs_data"]
    ).astype(
        np.float32,
        copy=False,
    )

    raw_thick = array(
        item["thickness"]
    ).astype(
        np.float32,
        copy=False,
    )

    if raw_bold.ndim != 4 or raw_thick.ndim != 3:
        raise ValueError(
            "Expected 4D BOLD and 3D thickness."
        )

    if (
        not np.isfinite(raw_bold).all()
        or not np.isfinite(raw_thick).all()
    ):
        raise ValueError("Nonfinite image.")

    # Spatial support before normalization.
    support_b = np.any(
        raw_bold != 0,
        axis=-1,
    )

    support_t = raw_thick != 0

    # Per-frame spatial normalization.
    bold_mean = raw_bold.mean(
        axis=(0, 1, 2),
        keepdims=True,
    )

    bold_std = raw_bold.std(
        axis=(0, 1, 2),
        keepdims=True,
    )

    bold = (
        (raw_bold - bold_mean)
        / (bold_std + 1e-6)
    )

    # Global cortical-thickness normalization.
    thick = (
        (raw_thick - raw_thick.mean())
        / (raw_thick.std() + 1e-6)
    )

    return bold, thick, support_b, support_t


# ================================================================
# 3. TRAINING-FITTED UMAP PROJECTION
# ================================================================

def project_training_umap(
    embeddings,
    output,
    seed=42,
):
    """
    Fit UMAP on training embeddings and transform held-out subjects.

    The UMAP model is fitted exactly once on the training split.

    Validation and testing embeddings are transformed using the
    already fitted reducer, without fitting a new projection.

    Parameters
    ----------
    embeddings : dict
        Dictionary indexed by split names, for example:

            embeddings["train"]
            embeddings["test"]

        Each split must contain:

            thick      : (N, embedding_dim)
            rs         : (N, embedding_dim)
            subject_id : sequence of N identifiers
            site       : sequence of N site labels

        Structural and functional embeddings must be aligned
        by subject.

    output : str or pathlib.Path
        Directory where UMAP results are saved.

    seed : int, default=42
        Random seed for UMAP fitting and transformation.

    Returns
    -------
    frame : pandas.DataFrame
        Two-dimensional embedding coordinates with columns:

            subject_id
            site
            split
            modality
            x
            y

    Saved files
    -----------
    umap_train_fitted.pkl
        Fitted training UMAP object.

    umap_coordinates.tsv
        Coordinates for both modalities and every split.

    umap_train_test.png
        Scatterplot grouped by acquisition site and modality.

    Notes
    -----
    UMAP provides visualization, not direct evidence of stable
    clusters. Community discovery should use the original
    high-dimensional embeddings.
    """

    import umap
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    out = Path(output)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    def stacked(data):
        return np.stack(
            [
                data["thick"],
                data["rs"],
            ],
            axis=1,
        ).reshape(
            -1,
            data["rs"].shape[1],
        )

    # ------------------------------------------------------------
    # Fit the projection exclusively on training embeddings.
    # ------------------------------------------------------------

    train = stacked(
        embeddings["train"]
    )

    reducer = umap.UMAP(
        n_neighbors=min(5, len(train) - 1),
        min_dist=0.1,
        spread=3.0,
        n_components=2,
        metric="euclidean",
        init="random",
        random_state=seed,
        transform_seed=seed,
        n_jobs=1,
    )

    xy_train = reducer.fit_transform(train)

    # ------------------------------------------------------------
    # Save the trained UMAP reducer.
    # ------------------------------------------------------------

    with open(
        out / "umap_train_fitted.pkl",
        "wb",
    ) as file:
        pickle.dump(
            reducer,
            file,
        )

    # ------------------------------------------------------------
    # Transform the remaining splits without refitting.
    # ------------------------------------------------------------

    records = []

    for split, data in embeddings.items():

        if split == "train":
            xy = xy_train
        else:
            xy = reducer.transform(
                stacked(data)
            )

        for i, subject in enumerate(data["subject_id"]):

            for j, modality in enumerate(
                ("thick", "rs")
            ):

                records.append(
                    {
                        "subject_id": subject,
                        "site": data["site"][i],
                        "split": split,
                        "modality": modality,
                        "x": float(xy[2 * i + j, 0]),
                        "y": float(xy[2 * i + j, 1]),
                    }
                )

    frame = pd.DataFrame(records)

    frame.to_csv(
        out / "umap_coordinates.tsv",
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------
    # Prepare site-specific colors.
    # ------------------------------------------------------------

    levels = sorted(
        frame.site.unique()
    )

    cmap = plt.get_cmap("turbo")

    colors = {
        site: cmap(
            i / max(len(levels) - 1, 1)
        )
        for i, site in enumerate(levels)
    }

    # ------------------------------------------------------------
    # Plot each split using the same fitted coordinate system.
    # ------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        len(embeddings),
        figsize=(6 * len(embeddings), 6),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for ax, split in zip(
        axes[0],
        embeddings,
    ):

        subset = frame[
            frame.split == split
        ]

        for (site, modality), group in subset.groupby(
            ["site", "modality"]
        ):

            ax.scatter(
                group.x,
                group.y,
                color=colors[site],
                marker=(
                    "^"
                    if modality == "thick"
                    else "D"
                ),
                s=18,
                alpha=0.7,
                label=f"{site} / {modality}",
            )

        ax.set(
            title=split,
            xlabel="UMAP 1",
            ylabel="UMAP 2",
        )

        ##ax.legend(
        ##    fontsize=6,
        ##    loc="best",
        ##)

    fig.suptitle(
        "UMAP fitted on training subjects; "
        "test subjects transformed (sites overlap)"
    )

    fig.tight_layout()

    fig.savefig(
        out / "umap_train_test.png",
        dpi=180,
    )

    plt.close(fig)

    return frame


# ================================================================
# 4. AGE AND SEX METADATA
# ================================================================

def attach_age_sex_metadata(
    frame,
    metadata_path,
    age_column="Age",
    sex_column="Sex",
    missing="error",
):
    """
    Attach validated age and sex metadata to a subject manifest.

    Uses a one-to-one join by participant_id.

    No guessed ages, automatic Excel subject matching, or
    unverified numerical sex-code interpretation is permitted.

    Parameters
    ----------
    frame : pandas.DataFrame
        Input manifest containing participant_id and site.

    metadata_path : str or pathlib.Path
        Path to a tab-separated metadata table.

    age_column : str, default="Age"
        Column containing subject age.

    sex_column : str, default="Sex"
        Column containing sex labels.

    missing : {"error", "drop"}, default="error"
        Handling of incomplete or invalid metadata.

        "error":
            Raise an exception.

        "drop":
            Exclude invalid subjects and return an exclusion report.

    Returns
    -------
    result : pandas.DataFrame
        Manifest with two additional numerical columns:

            AGE
            SEX_M

    exclusions : pandas.DataFrame
        Excluded subject IDs, site labels, and reasons.

    Notes
    -----
    Accepted sex labels:

        M, MALE   -> 1
        F, FEMALE -> 0

    Other encodings require verified recoding in the source
    metadata before calling this function.
    """

    if missing not in ("error", "drop"):
        raise ValueError(
            "missing must be 'error' or 'drop'."
        )

    meta = pd.read_csv(
        metadata_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required = {
        "participant_id",
        age_column,
        sex_column,
    }

    if not required <= set(meta):
        raise ValueError(
            f"Metadata needs columns {sorted(required)}."
        )

    if meta.participant_id.duplicated().any():
        raise ValueError(
            "Duplicate metadata participant_id."
        )

    if frame.participant_id.duplicated().any():
        raise ValueError(
            "Duplicate manifest participant_id."
        )

    lookup = (
        meta.set_index("participant_id")
        .reindex(frame.participant_id.astype(str))
    )

    # ------------------------------------------------------------
    # Validate age.
    # ------------------------------------------------------------

    age = pd.to_numeric(
        lookup[age_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    # ------------------------------------------------------------
    # Validate and encode sex labels.
    # ------------------------------------------------------------

    labels = (
        lookup[sex_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    sex = labels.map(
        {
            "M": 1.0,
            "F": 0.0,
            "MALE": 1.0,
            "FEMALE": 0.0,
        }
    ).to_numpy(dtype=float)

    valid_age = (
        np.isfinite(age)
        & (age > 0)
        & (age <= 120)
    )

    valid_sex = np.isfinite(sex)

    good = valid_age & valid_sex

    # ------------------------------------------------------------
    # Construct explicit exclusion reasons.
    # ------------------------------------------------------------

    reasons = []

    for i in np.flatnonzero(~good):

        problems = []

        if not valid_age[i]:
            problems.append(
                "missing/non-numeric/out-of-range age"
            )

        if not valid_sex[i]:
            problems.append(
                "unrecognized/missing sex code: "
                f"{labels.iloc[i]!r}"
            )

        reasons.append(
            {
                "participant_id": frame.iloc[i].participant_id,
                "site": frame.iloc[i].site,
                "reason": "; ".join(problems),
            }
        )

    exclusions = pd.DataFrame(
        reasons,
        columns=[
            "participant_id",
            "site",
            "reason",
        ],
    )

    if len(exclusions) and missing == "error":
        raise ValueError(
            f"{len(exclusions)} subjects lack usable Age/Sex "
            "metadata. "
            f"Examples: {exclusions.head(5).to_dict('records')}. "
            "Correct metadata, or explicitly use "
            "--missing-covariates drop. "
            "No values were imputed or guessed."
        )

    result = frame.copy()

    result["AGE"] = age
    result["SEX_M"] = sex

    return (
        result.loc[good].reset_index(drop=True),
        exclusions,
    )


# ================================================================
# 5. CONSTRUCT COMBAT COVARIATES
# ================================================================

def covariates_for_frame(frame):
    """
    Construct the covariate table required by neuroHarmonize.

    Parameters
    ----------
    frame : pandas.DataFrame
        Manifest containing:
            site
            AGE
            SEX_M

    Returns
    -------
    pandas.DataFrame
        Covariates with columns, in this exact order:

            SITE
            AGE
            SEX_M

    Notes
    -----
    Rows must correspond exactly to the subject ordering
    of the feature matrix passed to ComBat.
    """

    return pd.DataFrame(
        {
            "SITE": frame.site.astype(str).to_numpy(),
            "AGE": frame.AGE.to_numpy(dtype=float),
            "SEX_M": frame.SEX_M.to_numpy(dtype=float),
        }
    )


# ================================================================
# 6. TRAINING-ONLY COMBAT HARMONIZATION
# ================================================================

def fit_transform_combat_age_sex(
    train_x,
    test_x,
    train_covars,
    test_covars,
    support=None,
    epsilon=1e-10,
    seed=42,
):
    """
    Fit ComBat on training subjects and apply it to held-out subjects.

    Uses the official neuroHarmonize APIs.

    After ComBat, linear age and sex effects are estimated
    exclusively from harmonized training features and removed
    from both training and testing features.

    Parameters
    ----------
    train_x : numpy.ndarray
        Training feature matrix, shape (N_train, F).

    test_x : numpy.ndarray
        Testing feature matrix, shape (N_test, F).

    train_covars : pandas.DataFrame
        Training covariates:
            SITE, AGE, SEX_M

    test_covars : pandas.DataFrame
        Testing covariates with identical columns.

    support : numpy.ndarray, optional
        Boolean mask of supported features.

    epsilon : float, default=1e-10
        Minimum acceptable feature variance.

    seed : int, default=42
        Random seed used by harmonizationLearn.

    Returns
    -------
    train_out : numpy.ndarray
        Harmonized and residualized training features.

    test_out : numpy.ndarray
        Harmonized and residualized testing features.

    state : dict
        Fitted ComBat parameters, nuisance regression,
        feature mask, and package metadata.

    Notes
    -----
    - ComBat is fitted only on training subjects.
    - Testing subjects never contribute to parameter fitting.
    - The current design requires overlapping acquisition sites.
    - Age/sex effects are protected during ComBat fitting.
    - Linear age/sex effects are then explicitly removed.
    - Features excluded by the mask remain unchanged.
    """

    from neuroHarmonize import (
        harmonizationLearn,
        harmonizationApply,
    )

    from sklearn.linear_model import LinearRegression

    # ------------------------------------------------------------
    # Standardize feature matrices and covariate tables.
    # ------------------------------------------------------------

    a = np.asarray(
        train_x,
        dtype=np.float64,
    )

    b = np.asarray(
        test_x,
        dtype=np.float64,
    )

    tc = train_covars.reset_index(
        drop=True
    ).copy()

    vc = test_covars.reset_index(
        drop=True
    ).copy()

    required = [
        "SITE",
        "AGE",
        "SEX_M",
    ]

    if (
        list(tc.columns) != required
        or list(vc.columns) != required
    ):
        raise ValueError(
            f"Covariate columns must be in order {required}."
        )

    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[1] != b.shape[1]
        or len(a) != len(tc)
        or len(b) != len(vc)
    ):
        raise ValueError(
            "Feature/covariate shape mismatch."
        )

    if (
        not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Nonfinite features.")

    # ------------------------------------------------------------
    # Validate numerical covariates.
    # ------------------------------------------------------------

    for cov in (tc, vc):

        if not np.isfinite(
            cov[["AGE", "SEX_M"]].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "Missing/nonfinite Age/Sex."
            )

        if not set(cov.SEX_M) <= {0, 1}:
            raise ValueError(
                "SEX_M must be coded 0 or 1."
            )

    levels, counts = np.unique(
        tc.SITE.astype(str),
        return_counts=True,
    )

    # ------------------------------------------------------------
    # Require overlapping acquisition sites.
    # ------------------------------------------------------------

    if set(vc.SITE.astype(str)) != set(levels):
        raise ValueError(
            "This train/test helper requires every training "
            "site in the test cohort."
        )

    if len(levels) < 2 or np.any(counts < 2):
        raise ValueError(
            "Need >= 2 training subjects/site and >= 2 sites."
        )

    # ------------------------------------------------------------
    # Validate the covariate design matrix.
    # ------------------------------------------------------------

    design = np.column_stack(
        [
            pd.get_dummies(
                tc.SITE,
                dtype=float,
            ).to_numpy(),
            tc[["AGE", "SEX_M"]].to_numpy(),
        ]
    )

    if (
        np.linalg.matrix_rank(design) < design.shape[1]
        or len(a) <= design.shape[1]
    ):
        raise ValueError(
            "Site/Age/Sex design is rank deficient or has "
            "no residual degrees of freedom. "
            "Age/sex effects cannot be separated reliably; "
            "inspect this training fold."
        )

    # ------------------------------------------------------------
    # Select supported features with sufficient global variability.
    # ------------------------------------------------------------

    mask = np.var(
        a,
        axis=0,
    ) > epsilon

    # Do not require a feature to vary within every acquisition site.
    # The within-site variance intersection removed all supported BOLD
    # log-SD features in this cohort.

    if support is not None:

        support = np.asarray(
            support,
            dtype=bool,
        )

        if support.shape != mask.shape:
            raise ValueError(
                "Training support shape mismatch."
            )

        mask &= support

    # Do not additionally filter by residual variance after site/age/sex
    # regression. ComBat fitting and finite-output checks remain unchanged.

    logger.info(
        f"ComBat features selected: {int(mask.sum())}/{mask.size} "
        "(global variance + common spatial support)."
    )

    if mask.sum() < 2:
        raise ValueError(
            "Too few supported, variable features "
            "for empirical-Bayes ComBat."
        )

    # ------------------------------------------------------------
    # Fit ComBat on TRAINING subjects only.
    # ------------------------------------------------------------

    combat_model, train_combat = harmonizationLearn(
        a[:, mask],
        tc,
        eb=True,
        seed=seed,
    )

    # ------------------------------------------------------------
    # Apply the SAME fitted model to testing subjects.
    # ------------------------------------------------------------

    test_combat = harmonizationApply(
        b[:, mask],
        vc,
        combat_model,
    )

    if (
        not np.isfinite(train_combat).all()
        or not np.isfinite(test_combat).all()
    ):
        raise ValueError(
            "Official ComBat returned nonfinite values; "
            "inspect training design/features."
        )

    # ------------------------------------------------------------
    # Estimate and remove linear age/sex effects.
    # ------------------------------------------------------------
    # ComBat preserves these covariate effects during fitting.
    #
    # The nuisance regression is fitted only on training data.
    # The training mean is retained as the reference level.
    # ------------------------------------------------------------

    nuisance_mean = tc[
        ["AGE", "SEX_M"]
    ].to_numpy(dtype=float).mean(axis=0)

    ctrain = (
        tc[["AGE", "SEX_M"]].to_numpy(dtype=float)
        - nuisance_mean
    )

    ctest = (
        vc[["AGE", "SEX_M"]].to_numpy(dtype=float)
        - nuisance_mean
    )

    regressor = LinearRegression().fit(
        ctrain,
        train_combat,
    )

    train_clean = (
        train_combat
        - (
            regressor.predict(ctrain)
            - regressor.intercept_
        )
    )

    test_clean = (
        test_combat
        - (
            regressor.predict(ctest)
            - regressor.intercept_
        )
    )

    # ------------------------------------------------------------
    # Restore complete feature matrices.
    # ------------------------------------------------------------

    train_out = a.copy()
    test_out = b.copy()

    train_out[:, mask] = train_clean
    test_out[:, mask] = test_clean

    # ------------------------------------------------------------
    # Save the fitted model and preprocessing metadata.
    # ------------------------------------------------------------

    state = {
        "format": (
            "official_neuroHarmonize_plus_"
            "sklearn_age_sex_v1"
        ),
        "combat_model": combat_model,
        "age_sex_regressor": regressor,
        "nuisance_mean": nuisance_mean,
        "covariate_columns": required,
        "feature_mask": mask,
        "train_sites": levels.tolist(),
        "feature_count": a.shape[1],
        "train_n": len(a),
        "age_training_range": [
            float(tc.AGE.min()),
            float(tc.AGE.max()),
        ],
        "packages": {
            package: version(package)
            for package in (
                "neuroHarmonize",
                "neuroCombat",
                "scikit-learn",
            )
        },
    }

    if (
        not np.isfinite(train_out).all()
        or not np.isfinite(test_out).all()
    ):
        raise ValueError(
            "Nonfinite residualized output."
        )

    return (
        train_out.astype(np.float32),
        test_out.astype(np.float32),
        state,
    )


# ================================================================
# 7. APPLY AN ALREADY FITTED COMBAT MODEL
# ================================================================

def apply_saved_combat_age_sex(
    x,
    covars,
    state,
):
    """
    Apply a previously fitted ComBat model and nuisance regressor.

    No ComBat fitting or nuisance regression fitting occurs here.

    Parameters
    ----------
    x : numpy.ndarray
        Feature matrix, shape (N, F).

    covars : pandas.DataFrame
        Covariates in the required order:
            SITE, AGE, SEX_M

    state : dict
        Fitted state returned by fit_transform_combat_age_sex.

    Returns
    -------
    output : numpy.ndarray
        Harmonized and residualized feature matrix, float32.

    Notes
    -----
    The input cohort must contain all acquisition sites represented
    in the saved training model.
    """

    from neuroHarmonize import harmonizationApply

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    if (
        x.ndim != 2
        or x.shape[1] != state["feature_count"]
        or len(x) != len(covars)
    ):
        raise ValueError("Shape mismatch.")

    if list(covars.columns) != state["covariate_columns"]:
        raise ValueError(
            "Covariate columns/order mismatch."
        )

    if set(covars.SITE.astype(str)) != set(
        state["train_sites"]
    ):
        raise ValueError(
            "Supply all known sites in this cohort."
        )

    if (
        not np.isfinite(x).all()
        or not np.isfinite(
            covars[["AGE", "SEX_M"]].to_numpy(dtype=float)
        ).all()
    ):
        raise ValueError("Nonfinite input.")

    if not set(covars.SEX_M) <= {0, 1}:
        raise ValueError(
            "SEX_M must be 0/1."
        )

    # ------------------------------------------------------------
    # Apply saved ComBat parameters.
    # ------------------------------------------------------------

    mask = state["feature_mask"]

    adjusted = harmonizationApply(
        x[:, mask],
        covars,
        state["combat_model"],
    )

    # ------------------------------------------------------------
    # Apply saved linear nuisance regression.
    # ------------------------------------------------------------

    regressor = state["age_sex_regressor"]

    c = (
        covars[["AGE", "SEX_M"]].to_numpy(dtype=float)
        - state["nuisance_mean"]
    )

    output = x.copy()

    output[:, mask] = (
        adjusted
        - (
            regressor.predict(c)
            - regressor.intercept_
        )
    )

    if not np.isfinite(output).all():
        raise ValueError("Nonfinite transform.")

    return output.astype(np.float32)


# ================================================================
# 8. PREPARE AND CACHE FOLD-SPECIFIC COMBAT TRANSFORMS
# ================================================================

def prepare_fold_combat_age_sex(
    raw_train,
    raw_test,
    output,
    modalities="both-logsd",
    seed=42,
):
    """
    Prepare fold-specific ComBat harmonization and cached maps.

    Training data are used for fitting both ComBat and the linear
    age/sex nuisance model.

    Testing data are transformed using the fitted parameters.

    Parameters
    ----------
    raw_train : CleanENIGMADataset
        Training dataset with unnormalized individual samples.

    raw_test : CleanENIGMADataset
        Testing dataset with unnormalized individual samples.

    output : str or pathlib.Path
        Output directory for fitted models, cached maps,
        and audit metadata.

    modalities : str, default="both-logsd"
        Harmonization mode:

            "none":
                No ComBat harmonization.

            "thickness":
                Harmonize cortical-thickness features only.

            "both-logsd":
                Harmonize cortical thickness and BOLD log-SD
                amplitude features.

    seed : int, default=42
        Random seed for ComBat fitting.

    Returns
    -------
    paths : dict or None
        Mapping of participant IDs to cached transformation paths.

    Notes
    -----
    BOLD harmonization operates on voxelwise log-standard-deviation
    amplitude features, not directly on the complete temporal signal.

    The fitted transforms are cached to avoid repeating ComBat
    optimization for individual minibatches during training.
    """

    from loguru import logger

    if modalities not in (
        "none",
        "thickness",
        "both-logsd",
    ):
        raise ValueError(
            "Invalid modalities."
        )

    if modalities == "none":
        return None

    out = Path(output)

    out.mkdir(
        parents=True,
        exist_ok=False,
    )

    # ------------------------------------------------------------
    # Verify that training and testing subjects are disjoint.
    # ------------------------------------------------------------

    train_ids = set(
        raw_train.frame.participant_id
    )

    test_ids = set(
        raw_test.frame.participant_id
    )

    if train_ids & test_ids:
        raise ValueError(
            "Train/test subjects overlap."
        )

    # ------------------------------------------------------------
    # Build subject-aligned covariate matrices.
    # ------------------------------------------------------------

    tc = covariates_for_frame(
        raw_train.frame
    )

    vc = covariates_for_frame(
        raw_test.frame
    )

    if (
        vc.AGE.min() < tc.AGE.min()
        or vc.AGE.max() > tc.AGE.max()
    ):
        logger.warning(
            "Some test ages lie outside the training age range; "
            "applying the frozen linear model by extrapolation."
        )

    matrices = {}
    supports = {}
    shapes = {}

    # ------------------------------------------------------------
    # Extract harmonization features from each subject.
    # ------------------------------------------------------------

    for split, ds in (
        ("train", raw_train),
        ("test", raw_test),
    ):

        thickness = []
        logsd = []

        support_t = None
        support_b = None

        for i in range(len(ds)):

            bold, thick, sb, st = normalize_model_sample(
                ds[i]
            )

            if not shapes:
                shapes = {
                    "thickness": thick.shape,
                    "bold": bold.shape[:-1],
                }

            if (
                thick.shape != shapes["thickness"]
                or bold.shape[:-1] != shapes["bold"]
            ):
                raise ValueError(
                    "Image shape/feature layout mismatch."
                )

            thickness.append(
                thick.ravel()
            )

            # ----------------------------------------------------
            # Training spatial support for thickness.
            # ----------------------------------------------------

            if split == "train":

                if support_t is None:
                    support_t = st.ravel().copy()
                else:
                    support_t &= st.ravel()

            # ----------------------------------------------------
            # BOLD log-standard-deviation features.
            # ----------------------------------------------------

            if modalities == "both-logsd":

                sd = bold.std(
                    axis=-1
                )

                sb &= sd > 1e-6

                logsd.append(
                    np.log(
                        np.maximum(
                            sd,
                            1e-6,
                        )
                    ).ravel()
                )

                if split == "train":

                    if support_b is None:
                        support_b = sb.ravel().copy()
                    else:
                        support_b &= sb.ravel()

            if (
                i % 25 == 0
                or i + 1 == len(ds)
            ):
                logger.info(
                    f"Extracting {split} features: "
                    f"{i + 1}/{len(ds)}; "
                    "fit uses TRAIN only."
                )

        matrices[split] = {
            "thickness": np.stack(thickness)
        }

        if modalities == "both-logsd":
            matrices[split]["bold_logsd"] = np.stack(
                logsd
            )

        if split == "train":
            supports = {
                "thickness": support_t,
                "bold_logsd": support_b,
            }

    # ------------------------------------------------------------
    # Prepare audit metadata.
    # ------------------------------------------------------------

    paths = {}

    audit = {
        "implementation": (
            "neuroHarmonize.harmonizationLearn / "
            "harmonizationApply"
        ),
        "age_sex_removal": (
            "sklearn LinearRegression; fitted on harmonized "
            "TRAIN, applied unchanged to TEST"
        ),
        "train_subjects": (
            raw_train.frame.participant_id.tolist()
        ),
        "test_subjects": (
            raw_test.frame.participant_id.tolist()
        ),
        "modalities": modalities,
        "fit_count_per_modality": 1,
        "reference": (
            "training mean age and sex proportions"
        ),
        "sex_coding": {
            "F": 0,
            "M": 1,
        },
        "features": {},
    }

    maps = out / "maps"

    maps.mkdir()

    # ------------------------------------------------------------
    # Fit each modality-specific ComBat model.
    # ------------------------------------------------------------

    for label in matrices["train"]:

        train_clean, test_clean, state = (
            fit_transform_combat_age_sex(
                matrices["train"][label],
                matrices["test"][label],
                tc,
                vc,
                support=supports[label],
                seed=seed,
            )
        )

        # --------------------------------------------------------
        # Save fitted harmonization state.
        # --------------------------------------------------------

        with open(
            out / f"{label}_model.pkl",
            "wb",
        ) as file:
            pickle.dump(
                state,
                file,
                pickle.HIGHEST_PROTOCOL,
            )

        audit["features"][label] = {
            "total": state["feature_count"],
            "adjusted": int(
                state["feature_mask"].sum()
            ),
            "packages": state["packages"],
        }

        # --------------------------------------------------------
        # Cache harmonized maps for each split.
        # --------------------------------------------------------

        for split, ds, clean in (
            ("train", raw_train, train_clean),
            ("test", raw_test, test_clean),
        ):

            if label == "bold_logsd":

                clean = np.exp(
                    clean.astype(np.float64)
                    - matrices[split][label]
                )

                if (
                    not np.isfinite(clean).all()
                    or clean.max() > 1e6
                    or clean.min() < 1e-6
                ):
                    raise ValueError(
                        "Extreme BOLD scaling."
                    )

            if label == "thickness":
                shape = shapes["thickness"]
            else:
                shape = shapes["bold"]

            for i, pid in enumerate(
                ds.frame.participant_id
            ):

                path = (
                    maps
                    / f"{split}_{i:06d}_{label}.npy"
                )

                np.save(
                    path,
                    clean[i].reshape(shape).astype(
                        np.float32
                    ),
                    allow_pickle=False,
                )

                paths.setdefault(
                    str(pid),
                    {},
                )[label] = str(
                    path.resolve()
                )

    # ------------------------------------------------------------
    # Save audit and cached transformation paths.
    # ------------------------------------------------------------

    (out / "audit.json").write_text(
        json.dumps(
            audit,
            indent=2,
        )
    )

    (out / "transforms.json").write_text(
        json.dumps(
            paths,
            indent=2,
        )
    )

    return paths


# ================================================================
# 9. APPLY CACHED HARMONIZATION MAPS
# ================================================================

def apply_cached_combat_age_sex(
    bold,
    thick,
    participant_id,
    transforms,
):
    """
    Apply previously cached ComBat and nuisance-correction maps.

    No model fitting or parameter estimation occurs here.

    Parameters
    ----------
    bold : numpy.ndarray
        Normalized BOLD data, shape (D, H, W, T).

    thick : numpy.ndarray
        Cortical-thickness volume, shape (D, H, W).

    participant_id : str
        Subject identifier.

    transforms : dict or None
        Cached transformation paths returned by
        prepare_fold_combat_age_sex.

    Returns
    -------
    bold : numpy.ndarray
        BOLD data with cached amplitude correction applied,
        if available.

    corrected : numpy.ndarray
        Harmonized cortical-thickness volume.

    Notes
    -----
    BOLD amplitude correction is applied across the complete
    temporal recording using a spatial scaling map.

    No temporal frames are cropped or discarded.
    """

    if transforms is None:
        return bold, thick

    entry = transforms[
        str(participant_id)
    ]

    # ------------------------------------------------------------
    # Load corrected cortical-thickness volume.
    # ------------------------------------------------------------

    corrected = np.load(
        entry["thickness"],
        allow_pickle=False,
    )

    if corrected.shape != thick.shape:
        raise ValueError(
            "Cached thickness shape mismatch."
        )

    # ------------------------------------------------------------
    # Apply cached BOLD log-SD amplitude correction.
    # ------------------------------------------------------------

    if "bold_logsd" in entry:

        scale = np.load(
            entry["bold_logsd"],
            allow_pickle=False,
        )

        if scale.shape != bold.shape[:-1]:
            raise ValueError(
                "Cached BOLD scale shape mismatch."
            )

        mean = bold.mean(
            axis=-1,
            keepdims=True,
        )

        bold = (
            mean
            + (bold - mean) * scale[..., None]
        ).astype(np.float32)

    if (
        not np.isfinite(bold).all()
        or not np.isfinite(corrected).all()
    ):
        raise ValueError(
            "Nonfinite cached inputs."
        )

    return bold, corrected


# ================================================================
# Sequential SimCLR / Leiden / GW utilities (additive only)
# ================================================================

def make_nuisance_reference(frame):
    """Fit numeric nuisance centering/scaling on TRAINING subjects only.

    Site vocabulary and age/sex means and scales are frozen for this fold.
    """
    import numpy as np

    sites = sorted(frame["site"].astype(str).unique().tolist())
    values = frame[["AGE", "SEX_M"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite age/sex nuisance metadata")
    scales = values.std(axis=0)
    scales[scales < 1e-6] = 1.0
    return {
        "sites": sites,
        "mean": values.mean(axis=0).tolist(),
        "scale": scales.tolist(),
    }


def nuisance_matrix(sites, ages, sexes, reference, device):
    """Build a frozen-reference nuisance design for a paired minibatch.

    Columns are one-hot acquisition sites and standardized AGE and SEX_M.
    This matrix is input to a differentiable linear cross-covariance penalty.
    """
    import torch

    site_names = list(reference["sites"])
    unknown = set(map(str, sites)) - set(site_names)
    if unknown:
        raise ValueError(f"Sites absent from training nuisance vocabulary: {unknown}")
    onehot = torch.zeros((len(sites), len(site_names)), device=device)
    for i, site in enumerate(sites):
        onehot[i, site_names.index(str(site))] = 1.0
    numeric = torch.stack(
        [ages.to(device).float(), sexes.to(device).float()], dim=1
    )
    mean = torch.tensor(reference["mean"], device=device, dtype=numeric.dtype)
    scale = torch.tensor(reference["scale"], device=device, dtype=numeric.dtype)
    return torch.cat([onehot, (numeric - mean) / scale], dim=1)


def linear_nuisance_penalty(embeddings, covariates, eps=1e-6):
    """Squared standardized linear cross-covariance with site/age/sex.

    Zero is attained when sampled embedding dimensions have zero *linear*
    correlation with the sampled covariate columns. Does not guarantee
    nonlinear or out-of-sample independence. Requires multiple subjects.
    """
    import torch

    if embeddings.ndim != 2 or covariates.ndim != 2:
        raise ValueError("Expected embeddings and covariates of rank 2")
    if len(embeddings) != len(covariates):
        raise ValueError("Different numbers of embedding and nuisance rows")
    if len(embeddings) < 2:
        return embeddings.sum() * 0.0
    z = embeddings - embeddings.mean(dim=0, keepdim=True)
    c = covariates - covariates.mean(dim=0, keepdim=True)
    z = z / torch.sqrt(z.square().mean(dim=0, keepdim=True) + eps)
    c = c / torch.sqrt(c.square().mean(dim=0, keepdim=True) + eps)
    return ((z.T @ c) / len(z)).square().mean()


def paired_simclr_loss(z_rs, z_thick, temperature=0.1):
    """Symmetric paired cross-modal InfoNCE; positives share subject index.

    This is CLIP-style cross-modal contrastive loss, commonly used as the
    cross-modal counterpart to SimCLR. No diagnosis or CTQ labels are used.
    """
    import torch
    import torch.nn.functional as F

    if z_rs.shape != z_thick.shape or z_rs.ndim != 2:
        raise ValueError("Paired embeddings must have identical (B, D) shapes")
    if len(z_rs) < 2 or temperature <= 0:
        raise ValueError("At least 2 subjects and positive temperature required")
    logits = F.normalize(z_rs, dim=1) @ F.normalize(z_thick, dim=1).T
    logits = logits / temperature
    indices = torch.arange(len(z_rs), device=z_rs.device)
    return 0.5 * (
        F.cross_entropy(logits, indices) + F.cross_entropy(logits.T, indices)
    )



def multimodal_multipositive_infonce(embeds: dict, temperature=0.07):
    """Original losses_SSL.py whole InfoNCE, including anchor self-similarity.

    Same-subject other-modality embeddings are positives; all candidates,
    including the anchor itself, enter the denominator.
    Historical objective preserved intentionally for controlled comparison.
    """
    import torch
    import torch.nn.functional as F

    names = list(embeds.keys())
    Z = [F.normalize(embeds[n], dim=1) for n in names]  # each (B,D)
    M = len(Z)
    B, D = Z[0].shape

    # stack to (M,B,D)
    Z = torch.stack(Z, dim=0)

    # We'll compute for each anchor modality i:
    # logits over all (modality j, sample k) pairs: shape (B, M*B)
    # positives are positions where j != i and k == b (same subject)
    total_loss = 0.0
    count = 0

    for i in range(M):
        anchor = Z[i]  # (B,D)

        # (B,D) @ (M*B,D)^T -> (B, M*B)
        others = Z.reshape(M * B, D)                        # (M*B,D)
        logits = (anchor @ others.T) / temperature          # (B,M*B)

        # build positive mask: (B, M*B)
        # index mapping: flat index = j*B + k
        pos_mask = torch.zeros((B, M * B), device=anchor.device, dtype=torch.bool)
        for j in range(M):
            if j == i:
                continue
            idxs = j * B + torch.arange(B, device=anchor.device)
            pos_mask[torch.arange(B, device=anchor.device), idxs] = True

        # logsumexp over all candidates - as a sofmax normalization and regularize metrics across batch
        log_den = torch.logsumexp(logits, dim=1)  # (B,)

        # logsumexp over positives (multi-positive)
        # set non-positives to -inf
        logits_pos = logits.masked_fill(~pos_mask, float("-inf"))
        log_num = torch.logsumexp(logits_pos, dim=1)  # (B,)

        # define the whole loss here den - num in the logarithmic domain
        loss_i = (log_den - log_num).mean()
        total_loss += loss_i
        count += 1

    return total_loss / count


def leiden_subject_communities(
    subject_embeddings, k=15, resolution=1.0, seed=42
):
    """Leiden communities on a symmetrized weighted k-NN cosine graph.

    Uses ORIGINAL high-dimensional subject vectors (never UMAP). Returns
    integer labels in subject row order; sizes can vary naturally.
    """
    import numpy as np
    import igraph as ig
    import leidenalg
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize

    z = np.asarray(subject_embeddings, dtype=np.float64)
    if z.ndim != 2 or len(z) < 4 or not np.isfinite(z).all():
        raise ValueError("Need >=4 finite subject embeddings")
    if k < 2 or resolution <= 0:
        raise ValueError("k >= 2 and resolution > 0 required")
    z = normalize(z)
    neighbors = NearestNeighbors(
        n_neighbors=min(k + 1, len(z)), metric="cosine"
    ).fit(z)
    distances, indices = neighbors.kneighbors(z)
    edges = {}
    for i in range(len(z)):
        for j, distance in zip(indices[i, 1:], distances[i, 1:]):
            if int(i) == int(j):
                continue
            edge = (min(i, int(j)), max(i, int(j)))
            similarity = max(1e-6, 1.0 - float(distance))
            edges[edge] = max(edges.get(edge, 0.0), similarity)
    if not edges:
        raise ValueError("Empty nearest-neighbor graph")
    graph = ig.Graph(n=len(z), edges=list(edges), directed=False)
    graph.es["weight"] = list(edges.values())
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
    )
    return np.asarray(partition.membership, dtype=np.int64)


def community_stability(current, previous, seed_agreement):
    """Compute ARI of successive train-only snapshots and Leiden seeds.

    ARI is invariant to cluster ID permutations. The returned values
    characterize reproducibility; they do not establish clinical validity.
    """
    from sklearn.metrics import adjusted_rand_score

    temporal_ari = (
        float(adjusted_rand_score(previous, current))
        if previous is not None else float("nan")
    )
    seed_ari = float(adjusted_rand_score(current, seed_agreement))
    return temporal_ari, seed_ari


def differentiable_entropic_gw(z_rs, z_thick, epsilon=0.1, max_iter=50):
    """Entropy-regularized GW within ONE community minibatch.

    POT solves the transport plan using detached distances (alternating
    optimization); the squared GW distortion is then differentiated through
    the live PyTorch distance matrices with the coupling held fixed.

    Coupling is unconstrained by subject identity. Do not confuse this with
    paired correspondence preservation. A separate paired contrastive anchor
    is recommended during refinement.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    import ot

    if z_rs.ndim != 2 or z_rs.shape != z_thick.shape or len(z_rs) < 2:
        raise ValueError("GW needs paired (N,D) embeddings and N >= 2")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    x = F.normalize(z_rs, dim=1)
    y = F.normalize(z_thick, dim=1)
    d1 = torch.cdist(x.float(), x.float()).square()
    d2 = torch.cdist(y.float(), y.float()).square()
    # Normalize costs to comparable scales, without detaching gradients.
    d1 = d1 / d1.detach().mean().clamp_min(1e-6)
    d2 = d2 / d2.detach().mean().clamp_min(1e-6)
    n = len(x)
    weights = np.ones(n, dtype=np.float64) / n
    with torch.no_grad():
        coupling_np = ot.gromov.entropic_gromov_wasserstein(
            d1.detach().cpu().numpy().astype(np.float64),
            d2.detach().cpu().numpy().astype(np.float64),
            weights,
            weights,
            loss_fun="square_loss",
            epsilon=epsilon,
            max_iter=max_iter,
            tol=1e-7,
            verbose=False,
        )
    if not np.isfinite(coupling_np).all() or coupling_np.sum() <= 0:
        raise FloatingPointError("Nonfinite/degenerate GW coupling")
    pi = torch.as_tensor(coupling_np, device=d1.device, dtype=d1.dtype)
    pi = pi / pi.sum().clamp_min(1e-12)
    p = pi.sum(dim=1)
    q = pi.sum(dim=0)
    # Efficient factorization of sum_ijkl (d1_ij - d2_kl)^2 pi_ik pi_jl.
    term_1 = (d1.square() * (p[:, None] * p[None, :])).sum()
    term_2 = (d2.square() * (q[:, None] * q[None, :])).sum()
    cross = (d1 * (pi @ d2 @ pi.T)).sum()
    return term_1 + term_2 - 2.0 * cross


# -----------------------------------------------------------------------------
# Checkpoint-only validation plotting (additive; training helpers unchanged)
# -----------------------------------------------------------------------------

def plot_checkpoint_nmi_silhouette(metrics, output_dir):
    """Plot epoch-wise Leiden NMI and 128D cosine silhouette on twin axes.

    Parameters
    ----------
    metrics : pandas.DataFrame
        One row per available SimCLR checkpoint, containing ``epoch``,
        ``nmi_previous``, and ``silhouette_cosine``. The first NMI is undefined
        because there is no preceding partition.
    output_dir : str or pathlib.Path
        Existing fold checkpoint directory. The PNG and PDF are written here.

    Notes
    -----
    NMI compares Leiden assignments to the preceding evaluated checkpoint;
    silhouette measures the partition in normalized joint embedding space.
    Missing values are left missing, not replaced with zeros.
    """
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import numpy as np

    output_dir = Path(output_dir)
    epochs = metrics["epoch"].to_numpy(dtype=int)
    nmi = metrics["nmi_previous"].to_numpy(dtype=float)
    silhouette = metrics["silhouette_cosine"].to_numpy(dtype=float)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(10.8, 5.8), constrained_layout=True)
    twin = ax.twinx()
    navy, vermilion = "#22577A", "#C65D35"

    first, = ax.plot(
        epochs, nmi, color=navy, linewidth=2.5, marker="o",
        markersize=4.5, label="NMI vs previous checkpoint", zorder=3,
    )
    second, = twin.plot(
        epochs, silhouette, color=vermilion, linewidth=2.5,
        marker="s", markersize=4.2, label="Cosine silhouette (128D)", zorder=3,
    )

    ax.set_xlabel("SimCLR training epoch", fontweight="medium", labelpad=10)
    ax.set_ylabel("Normalized mutual information", color=navy, labelpad=12)
    twin.set_ylabel("Cosine silhouette score", color=vermilion, labelpad=12)
    ax.tick_params(axis="y", colors=navy)
    twin.tick_params(axis="y", colors=vermilion)
    ax.set_ylim(-0.02, 1.02)
    twin.set_ylim(-1.02, 1.02)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_xlim(epochs.min(), epochs.max() if len(epochs) > 1 else epochs[0] + 1)
    ax.grid(axis="y", alpha=0.15, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_title("Leiden community consistency and separation", loc="left",
                 fontsize=15, fontweight="semibold", pad=19)
    ax.text(
        0, 1.012,
        "Train subjects · NMI between consecutive saved partitions · "
        "silhouette on full-dimensional joint representations",
        transform=ax.transAxes, fontsize=9, color="#64748B", va="bottom",
    )
    ax.legend(
        [first, second], [first.get_label(), second.get_label()],
        loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2,
        frameon=False, fontsize=10,
    )
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"simclr_nmi_silhouette_twinx.{ext}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_subject_pair_metrics(data, seed=42):
    """Compare paired BOLD/thickness embeddings using subject identity labels.

    Stack both modalities in the original D-dimensional space; each subject
    contributes exactly two vectors. KMeans uses N clusters for N subjects,
    as in the original subject-pair evaluation. No clinical labels are used.
    """
    rs = np.asarray(data["rs"])
    thick = np.asarray(data["thick"])
    if rs.ndim != 2 or rs.shape != thick.shape:
        raise ValueError("Expected aligned (N,D) BOLD and thickness embeddings")
    n = len(rs)
    if len(data["subject_id"]) != n or len(set(data["subject_id"])) != n:
        raise ValueError("Expected one unique subject ID per aligned pair")
    if n < 2:
        return float("nan"), float("nan")
    x, _, _ = stack_modalities({"rs": rs, "thick": thick})
    subject_labels = np.tile(np.arange(n), 2)
    return (
        compute_nmi_via_kmeans(x, subject_labels, n_clusters=n, seed=seed),
        compute_silhouette(x, subject_labels, metric="cosine"),
    )


def compute_minibatch_subject_pair_metrics(data, batch_size, seed=42):
    """Legacy unweighted mean of per-batch NMI and cosine silhouette.

    Use fixed contiguous groups in the ordered evaluation snapshot, including
    a short final batch. Singleton batches are undefined and excluded.
    """
    if batch_size < 2:
        return float("nan"), float("nan")
    n = len(data["subject_id"])
    rows = []
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        if stop - start < 2:
            continue
        batch = {key: data[key][start:stop] for key in ("rs", "thick", "subject_id")}
        # Match the original thickness-first KMeans input order and fixed seed.
        x, _, _ = stack_modalities({"thick": batch["thick"], "rs": batch["rs"]})
        labels = np.tile(np.arange(stop - start), 2)
        rows.append((compute_nmi_via_kmeans(x, labels, stop - start, seed=seed),
                     compute_silhouette(x, labels, metric="cosine")))
    if not rows:
        return float("nan"), float("nan")
    return tuple(float(v) for v in np.nanmean(np.asarray(rows), axis=0))


def compute_paired_cosine_mrr(data, block_size=256):
    """Full-cohort positive/negative cosine, margin and bidirectional MRR.

    Tied scores use expected reciprocal rank under uniform tie ordering.
    Blocked similarity calculations avoid allocating a full N-by-N matrix.
    """
    x = np.asarray(data["rs"], dtype=np.float64)
    y = np.asarray(data["thick"], dtype=np.float64)
    n = len(x)
    if x.ndim != 2 or x.shape != y.shape or n == 0:
        raise ValueError("Expected nonempty aligned (N,D) embeddings")
    if len(data["subject_id"]) != n or len(set(data["subject_id"])) != n:
        raise ValueError("Expected one unique subject ID per aligned pair")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite embeddings")
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    harmonic = np.r_[0., np.cumsum(1. / np.arange(1, n + 1))]

    def directional_mrr(query, gallery):
        total = 0.
        for start in range(0, n, block_size):
            stop = min(start + block_size, n)
            sim = query[start:stop] @ gallery.T
            target = sim[np.arange(stop - start), np.arange(start, stop)][:, None]
            tied = np.isclose(sim, target, rtol=0., atol=1e-12)
            greater = np.sum((sim > target) & ~tied, axis=1)
            ties = np.sum(tied, axis=1)
            total += np.sum((harmonic[greater + ties] - harmonic[greater]) / ties)
        return float(total / n)

    # Diagonal: same-subject positive pairs. Off-diagonal: all ordered
    # cross-modal pairs of distinct subjects, not within-modality pairs.
    diagonal = np.sum(x * y, axis=1)
    positive = float(np.clip(diagonal, -1., 1.).mean())
    # Sum of all entries of x @ y.T without allocating the N-by-N matrix.
    # Float64 accumulation preserves small margins in concentrated embeddings.
    if n > 1:
        total_similarity = np.dot(x.sum(axis=0), y.sum(axis=0))
        negative = float(np.clip(
            (total_similarity - diagonal.sum()) / (n * (n - 1)), -1., 1.
        ))
    else:
        negative = float("nan")

    forward = directional_mrr(x, y)
    backward = directional_mrr(y, x)
    return {
        "paired_cosine": positive,
        "negative_cosine": negative,
        "cosine_margin": positive - negative,
        "mrr_rs_to_thick": forward,
        "mrr_thick_to_rs": backward,
        "mrr_bidirectional": (forward + backward) / 2.,
    }
