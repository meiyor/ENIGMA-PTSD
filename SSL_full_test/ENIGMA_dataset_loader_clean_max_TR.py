"""Load exported ENIGMA-PTSD BOLD and cortical-thickness data.

Each complete 4D BOLD recording is resampled to the largest source TR
among the selected sites and participants, ensuring
downsampling only, without an arbitrary time window or frame crop. Subjects may
have different temporal lengths. A custom collate function pads each batch
and supplies ``rs_lengths`` for masked processing by the shared ResNet3D-TCN.

No source paths, spreadsheets, fALFF maps, or NPZ caches are required.
"""

from fractions import Fraction
from pathlib import Path
import warnings

from loguru import logger
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


def temporal_grid(n_frames: int, source_tr: float, target_tr: float) -> np.ndarray:
    """Return all target sample times falling within the observed BOLD span.

    The first source volume is sampled at 0 seconds and the final source
    volume at ``(n_frames - 1) * source_tr``. Neither the input acquisition
    nor its target grid is restricted to a fixed 200-second interval.

    Returns
    -------
    np.ndarray
        One-dimensional target timestamps, in seconds (float64).
    """
    if n_frames < 2:
        raise ValueError("At least two source BOLD frames are required.")
    if not np.isfinite([source_tr, target_tr]).all():
        raise ValueError("Source TR and target TR must be finite.")
    if source_tr <= 0 or target_tr <= 0:
        raise ValueError("Source TR and target TR must be positive.")

    span = (n_frames - 1) * source_tr
    n_target = int(np.floor(span / target_tr + 1e-9)) + 1
    if n_target < 2:
        raise ValueError(
            "Recording is too short for two samples at the target TR."
        )
    return np.arange(n_target, dtype=np.float64) * target_tr


def resample_complete_bold(
    bold: np.ndarray,
    source_tr: float,
    target_tr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a complete ``(D, H, W, T)`` BOLD array along time.

    Polyphase filtering provides low-pass antialiasing when downsampling.
    A rational approximation to source_TR/target_TR defines up/down factors.
    Output is trimmed to the target timestamp grid so no target sample lies
    beyond the final observed source time. All available source frames
    contribute; there is no fixed-length crop.

    Notes
    -----
    A target TR smaller than the source TR is rejected (no upsampling).
    Target samples are uniformly spaced at ``target_tr``. Approximation
    drift is checked before filtering. ``resample_poly`` handles boundaries
    using linear extrapolation rather than zero-padding.
    """
    if bold.ndim != 4:
        raise ValueError(f"Expected 4D BOLD, received shape {bold.shape}.")

    if target_tr < source_tr - 1e-9:
        raise ValueError(
            f"Upsampling is disabled: target TR {target_tr:g}s "
            f"is smaller than source TR {source_tr:g}s."
        )

    times = temporal_grid(bold.shape[-1], source_tr, target_tr)
    if np.isclose(source_tr, target_tr, rtol=0, atol=1e-10):
        return np.ascontiguousarray(bold, dtype=np.float32), times

    ratio = Fraction(float(source_tr / target_tr)).limit_denominator(10000)
    up, down = ratio.numerator, ratio.denominator
    approximate_tr = source_tr * down / up
    max_drift = (len(times) - 1) * abs(approximate_tr - target_tr)
    if max_drift > 1e-3 * target_tr:
        raise ValueError(
            "TR ratio cannot be approximated accurately enough; "
            "increase the rational denominator limit."
        )

    result = resample_poly(
        bold, up=up, down=down, axis=-1, padtype="line"
    )
    if result.shape[-1] < len(times):
        raise RuntimeError("Resampling yielded fewer samples than target grid.")
    result = np.ascontiguousarray(result[..., :len(times)], dtype=np.float32)
    return result, times


class CleanENIGMADataset(Dataset):
    """Paired rs-fMRI and structural MRI dataset with full temporal recordings.

    Parameters
    ----------
    root : str or Path
        Export directory containing manifest.tsv and exported modalities.
        Common target TR is the largest source TR among records retained
        AFTER selecting the BOLD variant, sites and participant IDs.
        The configured target_tr, if present, is not used.
    rs_stride, st_stride : int, default=4
        Spatial subsampling strides for BOLD and thickness respectively.
    normalize_bold, normalize_thickness : bool, default=True
        Preserve the existing per-volume BOLD z-score across spatial voxels
        and the per-map thickness z-score. Normalization is performed after
        temporal resampling.
    sites, participant_ids : iterable[str], optional
        Optional manifest filters.
    bold_variant : str, default="small"
        Required BOLD variant in manifest.tsv.
    validate_headers : bool, default=False
        If True, compare manifest frame counts with NIfTI headers. Headers
        are always read when the frame count is missing from the manifest.
    rs_time_window, rs_window_crop, temporal_policy : optional
        Deprecated compatibility arguments. They are ignored deliberately:
        the loader always retains the entire recording.

    Returns
    -------
    dict
        ``rs_data`` has shape (T_i, 1, D, H, W), ``rs_length`` is T_i,
        ``thickness`` has shape (D_st, H_st, W_st), and metadata includes
        subject_id, site, original/target TR, full duration and time grid.
    """

    def __init__(
        self,
        root,
        rs_time_window=None,
        rs_window_crop=None,
        rs_stride=4,
        st_stride=4,
        normalize_bold=True,
        normalize_thickness=True,
        temporal_policy=None,
        sites=None,
        participant_ids=None,
        bold_variant="small",
        validate_headers=False,
    ):
        self.root = Path(root).resolve()
        self.frame = pd.read_csv(
            self.root / "manifest.tsv",
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )
        available = sorted(self.frame.bold_variant.unique())
        self.frame = self.frame[self.frame.bold_variant == bold_variant].copy()
        if self.frame.empty:
            raise ValueError(
                f"Variant {bold_variant!r} unavailable; choose from {available}."
            )
        if self.frame.participant_id.duplicated().any():
            raise ValueError("Duplicate subjects in selected BOLD variant.")

        if not isinstance(rs_stride, int) or not isinstance(st_stride, int):
            raise ValueError("Spatial strides must be integers.")
        if min(rs_stride, st_stride) < 1:
            raise ValueError("Spatial strides must be positive.")

        # Apply selection first: the common temporal grid must reflect
        # precisely the selected sites/participants, never excluded sites.
        if sites is not None:
            self.frame = self.frame[self.frame.site.isin(sites)].copy()
        if participant_ids is not None:
            missing = set(participant_ids) - set(self.frame.participant_id)
            if missing:
                raise ValueError(
                    f"Requested IDs absent after site filter: {sorted(missing)}"
                )
            self.frame = self.frame[
                self.frame.participant_id.isin(participant_ids)
            ].copy()

        if self.frame.empty:
            raise ValueError("No subjects remain after site/participant filtering.")

        # The largest TR (in seconds) is the lowest sampling frequency.
        # Ignore dataset_config.json target_tr: use the selected data only.
        selected_trs = pd.to_numeric(
            self.frame["tr_seconds"], errors="coerce"
        )
        if (not np.isfinite(selected_trs.to_numpy()).all()
                or (selected_trs <= 0).any()):
            raise ValueError("Invalid TR values among selected subjects.")

        self.target_tr = float(selected_trs.max())
        logger.info(
            f"Selected sites={sorted(self.frame['site'].unique())}; "
            f"largest source TR={self.target_tr:g}s; "
            "common target TR uses downsampling only."
        )

        self.rs_stride = rs_stride
        self.st_stride = st_stride
        self.normalize_bold = normalize_bold
        self.normalize_thickness = normalize_thickness

        if any(
            argument is not None
            for argument in (rs_time_window, rs_window_crop, temporal_policy)
        ):
            warnings.warn(
                "rs_time_window, rs_window_crop and temporal_policy are "
                "deprecated and ignored; all BOLD frames are retained.",
                FutureWarning,
                stacklevel=2,
            )

        good, excluded = [], []
        for row in self.frame.itertuples():
            try:
                bold_path = self.root / row.bold_path
                thickness_path = self.root / row.thickness_path
                if not bold_path.is_file() or not thickness_path.is_file():
                    raise ValueError("Exported modality missing.")

                tr = float(row.tr_seconds)
                if not np.isfinite(tr) or tr <= 0:
                    raise ValueError("Invalid source TR.")

                raw_n_frames = getattr(row, "n_frames", "")
                n_frames = int(raw_n_frames) if raw_n_frames else None

                # Missing lengths must be discovered; optional full validation
                # also catches manifest/header mismatches.
                if n_frames is None or validate_headers:
                    logger.info(
                        f"Checking temporal eligibility: {row.participant_id}"
                    )
                    image = nib.load(str(bold_path), mmap=True)
                    if len(image.shape) != 4:
                        raise ValueError("BOLD is not 4D.")
                    header_n_frames = int(image.shape[-1])
                    if n_frames is not None and n_frames != header_n_frames:
                        raise ValueError("Manifest BOLD frame count differs from header.")
                    n_frames = header_n_frames
                    self.frame.loc[row.Index, "n_frames"] = str(n_frames)

                if n_frames is None:
                    raise ValueError("Unknown BOLD frame count.")
                target_times = temporal_grid(
                    n_frames, tr, self.target_tr
                )
                self.frame.loc[row.Index, "n_frames_resampled"] = str(
                    len(target_times)
                )
                good.append(row.Index)
            except (ValueError, OSError, KeyError) as exc:
                excluded.append(
                    {"participant_id": row.participant_id, "reason": str(exc)}
                )

        self.exclusions = pd.DataFrame(
            excluded, columns=["participant_id", "reason"]
        )
        self.frame = self.frame.loc[good].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError("No eligible paired recordings after validation.")

        self.sites_all = self.frame.site.to_numpy()
        logger.info(
            f"Full-recording loader: {len(self.frame)} subjects, "
            f"target TR={self.target_tr:g} s; no temporal cropping."
        )

    def __len__(self):
        """Return the number of eligible subjects."""
        return len(self.frame)

    def __getitem__(self, index):
        """Load, resample, and return one entire paired subject recording."""
        row = self.frame.iloc[index]
        tr = float(row.tr_seconds)
        img = nib.load(str(self.root / row.bold_path), mmap=True)
        if len(img.shape) != 4:
            raise ValueError(f"BOLD is not 4D for {row.participant_id}.")

        # A single proxy slice loads all temporal frames after spatial
        # subsampling; no frame-level repeated gzip decompression.
        bold = np.asarray(
            img.dataobj[
                ::self.rs_stride,
                ::self.rs_stride,
                ::self.rs_stride,
                :,
            ],
            dtype=np.float32,
        )
        if not np.isfinite(bold).all():
            raise ValueError(f"Nonfinite BOLD data for {row.participant_id}.")
        bold, times = resample_complete_bold(bold, tr, self.target_tr)

        thickness = np.array(
            np.load(
                self.root / row.thickness_path,
                mmap_mode="r",
                allow_pickle=False,
            )[
                ::self.st_stride,
                ::self.st_stride,
                ::self.st_stride,
            ],
            dtype=np.float32,
            copy=True,
        )
        if not np.isfinite(thickness).all():
            raise ValueError(f"Nonfinite thickness data for {row.participant_id}.")

        if self.normalize_bold:
            bold = (
                bold - bold.mean(axis=(0, 1, 2), keepdims=True)
            ) / (bold.std(axis=(0, 1, 2), keepdims=True) + 1e-6)
        if self.normalize_thickness:
            thickness = (
                thickness - thickness.mean()
            ) / (thickness.std() + 1e-6)

        # for adding it in the new encoder**
        # Convert (D, H, W, T) -> (T, 1, D, H, W).
        # Each item can have its own T; padding happens in collate.
        bold = np.ascontiguousarray(
            np.moveaxis(bold, -1, 0)[:, None, ...],
            dtype=np.float32,
        )
        return {
            "index": int(index),
            "rs_data": torch.from_numpy(bold),
            "rs_length": int(len(times)),
            "thickness": torch.from_numpy(
                np.ascontiguousarray(thickness, dtype=np.float32)
            ),
            "subject_id": row.participant_id,
            "site": row.site,
            # These are *resampled* frame indices, not source frame indices.
            "time_indices": torch.arange(len(times), dtype=torch.long),
            "time_seconds": torch.from_numpy(times.copy()),
            "tr_seconds": tr,
            "target_tr_seconds": self.target_tr,
            "n_frames_original": int(img.shape[-1]),
            "duration_seconds": float(img.shape[-1] * tr),
        }


def collate_variable_length_enigma(batch: list[dict]) -> dict:
    """Pad variable-length BOLD recordings and batch paired metadata.

    Returns
    -------
    dict
        rs_data : torch.FloatTensor, (B, T_max, 1, D, H, W)
            Zero-padded rs-fMRI sequences, directly accepted by the encoder.
        rs_lengths : torch.LongTensor, (B,)
            Actual valid resampled temporal lengths, before padding.
        rs_mask : torch.BoolTensor, (B, T_max)
            True for real frames, False for padding.
        thickness : torch.FloatTensor, (B, D_st, H_st, W_st)
            Stacked thickness maps.
        time_seconds : torch.FloatTensor, (B, T_max)
            Padded times in seconds; ignore entries with rs_mask=False.
        time_indices : torch.LongTensor, (B, T_max)
            Resampled frame indices; padded positions set to -1.
        ...
            Original subject metadata (index, site, subject_id, TR, duration).

    Raises
    ------
    ValueError
        If subjects have incompatible spatial shapes.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    rs_shapes = {tuple(item["rs_data"].shape[1:]) for item in batch}
    st_shapes = {tuple(item["thickness"].shape) for item in batch}
    if len(rs_shapes) != 1 or len(st_shapes) != 1:
        raise ValueError(
            "All BOLD and thickness maps in a batch must share their "
            "respective spatial dimensions; harmonize spatial grids first."
        )

    lengths = torch.tensor(
        [item["rs_length"] for item in batch], dtype=torch.long
    )
    rs_data = pad_sequence(
        [item["rs_data"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )
    max_t = rs_data.shape[1]
    mask = torch.arange(max_t).unsqueeze(0) < lengths.unsqueeze(1)

    return {
        "index": torch.tensor([item["index"] for item in batch], dtype=torch.long),
        "rs_data": rs_data,
        "rs_lengths": lengths,
        "rs_mask": mask,
        "thickness": torch.stack([item["thickness"] for item in batch]),
        "subject_id": [item["subject_id"] for item in batch],
        "site": [item["site"] for item in batch],
        "time_indices": pad_sequence(
            [item["time_indices"] for item in batch],
            batch_first=True,
            padding_value=-1,
        ),
        "time_seconds": pad_sequence(
            [item["time_seconds"] for item in batch],
            batch_first=True,
            padding_value=0.0,
        ),
        "tr_seconds": torch.tensor(
            [item["tr_seconds"] for item in batch], dtype=torch.float64
        ),
        "target_tr_seconds": torch.tensor(
            [item["target_tr_seconds"] for item in batch], dtype=torch.float64
        ),
        "n_frames_original": torch.tensor(
            [item["n_frames_original"] for item in batch], dtype=torch.long
        ),
        "duration_seconds": torch.tensor(
            [item["duration_seconds"] for item in batch], dtype=torch.float64
        ),
    }


def define_dataset_dataloader_ENIGMA(
    root,
    batch_size=10,
    shuffling=True,
    num_workers=0,
    **kwargs,
):
    """Return a DataLoader with temporal padding and a subject-level dataset.

    Typical encoder call::

        for batch in loader:
            z_rs = encoder(
                batch["rs_data"].to(device),
                lengths=batch["rs_lengths"].to(device),
            )

    Each BOLD sample contains its entire resampled recording. No window
    selection or fixed-frame truncation is performed by this loader.
    """
    dataset = CleanENIGMADataset(root, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffling,
        num_workers=num_workers,
        collate_fn=collate_variable_length_enigma,
    )
