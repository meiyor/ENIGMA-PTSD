"""Sequential ENIGMA-PTSD cross-modal SSL, Leiden communities, and GW.

Order (independently within each subject-disjoint, site-stratified fold):
    1. Train-only ComBat / age-sex residualization (optional ablations).
    2. Cross-modal paired SimCLR-style InfoNCE + linear nuisance penalty.
    3. Train-only Leiden stability checks (high-dimensional embeddings).
    4. GW refinement *inside* discovered communities, with paired anchors and
       the same linear site/age/sex penalty throughout refinement.
    5. Final held-out evaluation; test data never select stages or checkpoints.

Complete rs-fMRI recordings, resampled to the largest TR among selected sites,
pass through SharedResNet4D_TemporalConv with masked variable-length pooling.
No CTQ labels, diagnoses, or clinical categories enter optimization.

Needs the companion ``ENIGMA_sequential_utils_additions.py`` functions appended
at the end of your EXISTING ``utils.py``. Install python-igraph, leidenalg, POT,
neuroHarmonize, sklearn, and nibabel in the training environment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from ENIGMA_dataset_loader_clean_max_TR import (
    CleanENIGMADataset,
    collate_variable_length_enigma,
    temporal_grid,
)
from ResNet_Encoders_definition import (
    ResNet3DEncoder,
    LateFusion4DResNet_TemporalConv,
    SharedResNet4D_TemporalConv,
    auto_padding,
)
from utils import (
    apply_cached_combat_age_sex,
    attach_age_sex_metadata,
    community_stability,
    differentiable_entropic_gw,
    init_xavier,
    leiden_subject_communities,
    linear_nuisance_penalty,
    make_nuisance_reference,
    make_subject_folds_by_site,
    normalize_model_sample,
    nuisance_matrix,
    paired_simclr_loss,
    multimodal_multipositive_infonce,
    prepare_fold_combat_age_sex,
    project_training_umap,
)


def seed_everything(seed: int) -> None:
    """Initialize independent Python, NumPy, and PyTorch random states."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def subset_clean_dataset(base: CleanENIGMADataset, indices):
    """Shallow-copy a lazy dataset but restrict its subject manifest."""
    subset = copy.copy(base)
    subset.frame = base.frame.iloc[list(indices)].reset_index(drop=True).copy()
    subset.sites_all = subset.frame["site"].to_numpy()
    return subset


class CombatSpatialView(Dataset):
    """Translate new (T,1,D,H,W) samples into legacy (D,H,W,T).

    This adapter is used ONLY at the boundary to the existing ComBat helpers.
    It prevents inadvertently treating the temporal axis as a spatial axis.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.frame = dataset.frame

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        bold = item["rs_data"].numpy()  # (T,1,D,H,W)
        item["rs_data"] = np.moveaxis(bold[:, 0], 0, -1).copy()
        return item


class HarmonizedModelDataset(Dataset):
    """Normalize recording, apply cached ComBat, optionally center-crop, add covariates.

    ComBat harmonization is prepared from train-only fits before optimization.
    Returned sequences use (T,1,D,H,W), as required by the shared TCN.
    """

    def __init__(self, raw, transforms, crop_frames=None):
        self.raw = raw
        self.transforms = transforms
        self.frame = raw.frame
        self.crop_frames = crop_frames

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index):
        item = dict(self.raw[index])
        spatial = np.moveaxis(item["rs_data"].numpy()[:, 0], 0, -1)
        compatibility_item = {"rs_data": spatial, "thickness": item["thickness"]}
        bold, thick, _, _ = normalize_model_sample(compatibility_item)
        bold, thick = apply_cached_combat_age_sex(
            bold, thick, item["subject_id"], self.transforms
        )
        # Apply a deterministic center crop AFTER complete-recording resampling
        # and fold-fitted harmonization. This keeps the ComBat fitting procedure
        # unchanged, and produces aligned time indices and timestamps.
        if self.crop_frames is not None:
            n_available = bold.shape[-1]
            if n_available < self.crop_frames:
                raise ValueError(
                    f"Subject {item['subject_id']} has {n_available} resampled "
                    f"frames, fewer than the requested {self.crop_frames}. "
                    "Reduce --rs-crop-seconds."
                )
            first = (n_available - self.crop_frames) // 2
            last = first + self.crop_frames
            bold = bold[..., first:last]
            item["time_indices"] = item["time_indices"][first:last].clone()
            item["time_seconds"] = item["time_seconds"][first:last].clone()
            item["rs_length"] = self.crop_frames
        bold = np.ascontiguousarray(np.moveaxis(bold, -1, 0)[:, None], dtype=np.float32)
        item["rs_data"] = torch.from_numpy(bold)
        item["thickness"] = torch.from_numpy(
            np.ascontiguousarray(thick, dtype=np.float32)
        )
        item["AGE"] = float(self.frame.iloc[index]["AGE"])
        item["SEX_M"] = float(self.frame.iloc[index]["SEX_M"])
        return item


def collate_training(batch):
    """Use the supplied variable-length collate and append age/sex vectors."""
    result = collate_variable_length_enigma(batch)
    result["AGE"] = torch.tensor([b["AGE"] for b in batch], dtype=torch.float32)
    result["SEX_M"] = torch.tensor([b["SEX_M"] for b in batch], dtype=torch.float32)
    return result


def make_loader(dataset, batch_size, shuffle, seed, workers, device):
    """Return an independent, masked variable-length subject loader."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_training,
    )


def build_models(args) -> nn.ModuleDict:
    """Build structural ResNet and selected shared/independent RS TCN."""
    thick = ResNet3DEncoder(
        in_channels=1,
        use_stages=(True, False, False, False),
        out_dim=args.out_dim,
        emb_dim=args.out_dim * 2,
        pretrained=False,
        attn_heads=1,
        activation_function="gelu",
        attention_projection=True,
        conv_overrides=[
            {
                "pattern": r"^layer1\.\d+\.conv1$",
                "kernel_size": (1, 3, 3),
                "padding": auto_padding((1, 3, 3)),
            },
            {
                "pattern": r"^layer1\.\d+\.conv2$",
                "kernel_size": (3, 3, 3),
                "padding": auto_padding((3, 3, 3)),
            },
        ],
    )
    if args.rs_model == "independent":
        if args.rs_crop_frames is None:
            raise ValueError("Independent-stream ResNet needs --rs-crop-seconds")
        # This is the historical LateFusion model: one distinct ResNet3D per
        # resampled time point, Conv1D+BatchNorm, mean/last pooling and the
        # historical single-token attention projection.
        rs = LateFusion4DResNet_TemporalConv(
            n_streams=args.rs_crop_frames,
            emb_dim=args.out_dim * 2,
            out_dim=args.out_dim,
            pretrained=False,
            kernel_size=args.tcn_kernel,
            n_blocks=args.tcn_blocks,
            hidden_channels=args.tcn_channels,
            dilation_growth=args.tcn_dilation,
            attention_projection=True,
            attn_heads=1,
            pool="mean" if args.pool != "last" else "last",
            out_dim_final=args.out_dim,
        )
    else:
        rs = SharedResNet4D_TemporalConv(
            emb_dim=args.out_dim * 2,
            out_dim=args.out_dim,
            pretrained=False,
            kernel_size=args.tcn_kernel,
            n_blocks=args.tcn_blocks,
            hidden_channels=args.tcn_channels,
            dilation_growth=args.tcn_dilation,
            pool=args.pool,
            attention_heads=args.attention_heads,
            attention_dropout=args.attention_dropout,
            out_dim_final=args.out_dim,
            frame_chunk=args.frame_chunk,
        )
    models = nn.ModuleDict({"thick": thick, "rs": rs})
    models.apply(lambda module: init_xavier(module, uniform=True))
    return models


def encode_batch(models, batch, device):
    """Obtain two L2-normalized (B,D) vectors for paired subject IDs."""
    rs = batch["rs_data"].to(device, non_blocking=True)
    lengths = batch["rs_lengths"].to(device, non_blocking=True)
    thick = batch["thickness"].to(device, non_blocking=True)
    if isinstance(models["rs"], LateFusion4DResNet_TemporalConv):
        if not torch.all(lengths == rs.shape[1]):
            raise ValueError("Independent streams require equal unpadded lengths")
        if rs.shape[1] != len(models["rs"].encoders):
            raise ValueError("Independent ResNet count does not match crop frames")
        z_rs = models["rs"]([rs[:, t] for t in range(rs.shape[1])])
    else:
        z_rs = models["rs"](rs, lengths=lengths)
    z_st = models["thick"](thick.unsqueeze(1))
    return {"rs": F.normalize(z_rs, dim=1), "thick": F.normalize(z_st, dim=1)}


def batch_nuisance_loss(emb, batch, reference, device):
    """Penalize linear site/age/sex dependence in BOTH embedding modalities."""
    design = nuisance_matrix(
        batch["site"], batch["AGE"], batch["SEX_M"], reference, device
    )
    return 0.5 * (
        linear_nuisance_penalty(emb["rs"], design)
        + linear_nuisance_penalty(emb["thick"], design)
    )


@torch.no_grad()
def collect_embeddings(models, loader, device):
    """Collect all subject-aligned final embeddings, without selecting on test."""
    models.eval()
    collected = {"rs": [], "thick": [], "subject_id": [], "site": []}
    for batch in loader:
        z = encode_batch(models, batch, device)
        collected["rs"].append(z["rs"].cpu().numpy())
        collected["thick"].append(z["thick"].cpu().numpy())
        collected["subject_id"].extend(batch["subject_id"])
        collected["site"].extend(batch["site"])
    result = {
        "rs": np.concatenate(collected["rs"]),
        "thick": np.concatenate(collected["thick"]),
        "subject_id": np.asarray(collected["subject_id"], dtype=str),
        "site": np.asarray(collected["site"], dtype=str),
    }
    return result


def community_space(data):
    """Symmetric subject-level space; modality averages are not CTQ targets."""
    z = data["rs"] + data["thick"]
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    return (z / np.maximum(norms, 1e-8)).astype(np.float32)


def cluster_snapshot(data, args, seed):
    """Discover Leiden communities and quantify random-seed reproducibility."""
    z = community_space(data)
    labels = leiden_subject_communities(
        z, k=args.leiden_k, resolution=args.leiden_resolution, seed=seed
    )
    rerun = leiden_subject_communities(
        z, k=args.leiden_k, resolution=args.leiden_resolution, seed=seed + 103
    )
    sizes = pd.Series(labels).value_counts()
    supported = int(sizes[sizes >= args.min_community_size].sum())
    return labels, rerun, {
        "n_communities": int(len(sizes)),
        "sizes": {str(k): int(v) for k, v in sizes.items()},
        "supported_fraction": supported / len(labels),
        "n_supported": int(sum(sizes >= args.min_community_size)),
    }


def write_checkpoint(path, models, optimizer, epoch, stage, fold, args, **extra):
    """Save model state for reproducible train-only phase transition."""
    torch.save(
        {
            "epoch": epoch,
            "stage": stage,
            "fold": fold,
            "models": models.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            **extra,
        },
        path,
    )


def find_fold_restart(fold_out, fold, device, args):
    """Find the latest valid checkpoint in an existing fold directory.

    A completed fold is handled by run(); unfinished GW takes precedence over
    the final SimCLR checkpoint. Checkpoints are local trusted training files.
    """
    if not fold_out.is_dir():
        return None

    candidates = []
    for stage in ("simclr", "gw"):
        for path in fold_out.glob(f"{stage}_epoch_*.pt"):
            match = re.fullmatch(rf"{stage}_epoch_(\d+)\.pt", path.name)
            if match:
                candidates.append((stage, int(match.group(1)), path))

    final_simclr = fold_out / "simclr_final.pt"
    if final_simclr.is_file():
        candidates.append(("simclr", -1, final_simclr))

    # Latest GW checkpoint wins, then final SimCLR, then latest SSL epoch.
    candidates.sort(
        key=lambda x: (x[0] == "gw", x[2].name == "simclr_final.pt", x[1]),
        reverse=True,
    )
    for stage, _, path in candidates:
        try:
            state = torch.load(path, map_location=device, weights_only=False)
            if state.get("stage") != stage or int(state.get("fold", -1)) != fold:
                raise ValueError("Checkpoint stage/fold does not match file location")
            saved = state.get("args", {})
            if saved.get("ssl_loss", "pairwise") != args.ssl_loss:
                raise ValueError(
                    f"Checkpoint {path} was trained with "
                    f"ssl_loss={saved.get('ssl_loss', 'pairwise')!r}, "
                    f"but requested {args.ssl_loss!r}. Use a different "
                    "--output-root when changing the contrastive objective."
                )
            critical = (
                "bold_variant", "rs_stride", "st_stride", "out_dim", "pool",
                "rs_model", "rs_crop_seconds", "rs_crop_frames",
                "attention_heads", "tcn_kernel", "tcn_blocks", "tcn_channels",
                "tcn_dilation", "frame_chunk", "combat", "seed", "n_splits",
                "temperature", "ssl_loss", "batch_size", "learning_rate", "max_ssl_epochs",
                "cluster_check_every", "leiden_k", "leiden_resolution",
                "cluster_stability_threshold", "cluster_seed_threshold",
                "stability_patience", "min_community_size", "min_communities",
                "min_community_coverage", "gw_learning_rate", "gw_weight",
                "gw_epsilon", "gw_batch_size", "gw_sinkhorn_iterations",
                "anchor_weight", "nuisance_weight", "weight_decay",
                "min_lr", "missing_covariates", "sites", "age_column", "sex_column",
            )
            mismatched = [k for k in critical if k in saved and saved[k] != getattr(args, k)]
            if mismatched:
                raise ValueError(
                    f"Resume checkpoint {path} has different settings: {mismatched}. "
                    "Use the original settings or a new output directory."
                )
            logger.info(f"Resume candidate: {path} (stage={stage}, epoch={state['epoch']})")
            return state
        except (OSError, EOFError, RuntimeError) as exc:
            logger.warning(f"Skipping unreadable checkpoint {path}: {exc}")
    return None


def restore_rng_state(checkpoint):
    """Restore RNG state, converting saved states to PyTorch's required types."""

    if "rng_python" not in checkpoint:
        logger.warning("Checkpoint lacks RNG state; skipping restoration.")
        return

    random.setstate(checkpoint["rng_python"])
    np.random.set_state(checkpoint["rng_numpy"])

    def as_byte_tensor(state):
        if isinstance(state, torch.Tensor):
            return state.detach().to(
                device="cpu", dtype=torch.uint8
            ).contiguous()

        return torch.as_tensor(
            state, dtype=torch.uint8, device="cpu"
        ).contiguous()

    if checkpoint.get("rng_torch") is not None:
        torch.set_rng_state(
            as_byte_tensor(checkpoint["rng_torch"])
        )

    if torch.cuda.is_available() and checkpoint.get("rng_cuda") is not None:
        cuda_states = [
            as_byte_tensor(state)
            for state in checkpoint["rng_cuda"]
        ]

        torch.cuda.set_rng_state_all(cuda_states)

def load_saved_fold_transforms(fold_out, args, train_frame, test_frame):
    """Reuse only a completed fold-specific ComBat cache, not a partial fit."""
    if args.combat == "none":
        return None, True
    directory = fold_out / "combat"
    audit_path = directory / "audit.json"
    paths_path = directory / "transforms.json"
    if not audit_path.is_file() or not paths_path.is_file():
        return None, False
    audit = json.loads(audit_path.read_text())
    paths = json.loads(paths_path.read_text())
    if (audit.get("train_subjects") != train_frame.participant_id.tolist()
            or audit.get("test_subjects") != test_frame.participant_id.tolist()
            or audit.get("modalities") != args.combat):
        raise ValueError("Existing ComBat cache is for a different split/configuration")
    needed = ("thickness", "bold_logsd") if args.combat == "both-logsd" else ("thickness",)
    for pid in [*train_frame.participant_id.tolist(), *test_frame.participant_id.tolist()]:
        if str(pid) not in paths or any(
            key not in paths[str(pid)] or not Path(paths[str(pid)][key]).is_file()
            for key in needed
        ):
            raise ValueError(f"Incomplete existing ComBat cache: {pid}")
    logger.info(f"Reusing harmonization cache from {directory}")
    return paths, True


def train_simclr_stage(
    models,
    train_dataset,
    train_eval,
    test_eval,
    reference,
    args,
    fold_out,
    fold,
    device,
    resume_state=None,
):
    """Train InfoNCE until stable train-only communities or epoch cap.

    Transition requires successive snapshots AND distinct Leiden seeds to
    exceed the ARI threshold for `stability_patience` checks. Otherwise the
    pipeline falls back to the last SimCLR epoch, without claiming stability.

    Save SimCLR checkpoints every `checkpoint_every` epochs and at the
    final epoch.

    UMAP plots are generated at each Leiden clustering-check epoch.
    """

    loader = make_loader(
        train_dataset,
        args.batch_size,
        True,
        args.seed + fold,
        args.num_workers,
        device,
    )

    optimizer = torch.optim.AdamW(
        models.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_ssl_epochs,
        eta_min=args.min_lr,
    )

    history = []
    checks = []
    previous = None
    stable_checks = 0
    selected_labels = None
    phase_status = "max_epochs_without_stability"
    start_epoch = 1
    if resume_state is not None:
        models.load_state_dict(resume_state["models"], strict=True)
        optimizer.load_state_dict(resume_state["optimizer"])
        start_epoch = int(resume_state["epoch"]) + 1
        if "scheduler" in resume_state:
            scheduler.load_state_dict(resume_state["scheduler"])
        else:
            # Legacy checkpoint: reconstruct epoch-aligned cosine phase.
            scheduler.step(int(resume_state["epoch"]))
            logger.warning("Legacy checkpoint: cosine scheduler reconstructed from epoch")
        history = resume_state.get("history", [])
        checks = resume_state.get("checks", [])
        previous = resume_state.get("previous")
        stable_checks = int(resume_state.get("stable_checks", 0))
        selected_labels = resume_state.get("selected_labels")
        if previous is None and start_epoch > args.cluster_check_every:
            logger.warning("Legacy checkpoint lacks Leiden history; stability streak restarts")
        if "loader_rng" in resume_state:
            #loader.generator.set_state(resume_state["loader_rng"])
            rng_state = resume_state["loader_rng"]

            if not isinstance(rng_state, torch.Tensor):
               rng_state = torch.as_tensor(rng_state, dtype=torch.uint8)
            else:
               rng_state = rng_state.detach().to(
                  device="cpu", dtype=torch.uint8
               )

            loader.generator.set_state(rng_state.contiguous())

        restore_rng_state(resume_state)
        logger.info(f"Resuming fold {fold} SimCLR at epoch {start_epoch}")

    # A checkpoint may have been saved at the last scheduled epoch just
    # before the final SimCLR metadata files were written.
    epoch = start_epoch - 1
    if epoch >= args.max_ssl_epochs:
        if selected_labels is None:
            logger.warning("Legacy final-epoch checkpoint has no Leiden state; "
                           "reconstructing its last training-only snapshot")
            restored_data = collect_embeddings(models, train_eval, device)
            selected_labels, _, _ = cluster_snapshot(
                restored_data, args, args.seed + epoch
            )
        if stable_checks >= args.stability_patience:
            phase_status = "train_leiden_stability_threshold_met"

    for epoch in range(start_epoch, args.max_ssl_epochs + 1):

        models.train()

        running = {
            "ssl": 0.0,
            "nuis": 0.0,
            "total": 0.0,
        }

        seen = 0

        for batch in loader:

            n = len(batch["subject_id"])

            if n < 2:
                continue

            optimizer.zero_grad(set_to_none=True)

            emb = encode_batch(
                models,
                batch,
                device,
            )

            if args.ssl_loss == "whole":
                loss_ssl = multimodal_multipositive_infonce(
                    emb, temperature=args.temperature,
                )
            else:
                loss_ssl = paired_simclr_loss(
                    emb["rs"],
                    emb["thick"],
                    args.temperature,
                )

            loss_nuis = batch_nuisance_loss(
                emb,
                batch,
                reference,
                device,
            )

            total = (
                loss_ssl
                + args.nuisance_weight * loss_nuis
            )

            if not torch.isfinite(total):
                raise FloatingPointError(
                    "Nonfinite SimCLR/nuisance loss"
                )

            total.backward()
            optimizer.step()

            for key, value in (
                ("ssl", loss_ssl),
                ("nuis", loss_nuis),
                ("total", total),
            ):
                running[key] += float(value.detach()) * n

            seen += n

        if seen == 0:
            raise ValueError(
                "Training needs minibatches with >=2 subjects"
            )

        scheduler.step()

        record = {
            "stage": "simclr",
            "epoch": epoch,
            "n": seen,
        }

        record.update(
            {
                key: val / seen
                for key, val in running.items()
            }
        )

        history.append(record)

        # Persist SSL loss metrics during training, not only at stage completion.
        if epoch % 5 == 0:
            pd.DataFrame(history).to_csv(
                fold_out / "simclr_history.tsv", sep="\t", index=False,
            )

        logger.info(
            f"Fold {fold} SSL epoch {epoch}: "
            f"{record['total']:.5f}"
        )

        # =========================================================
        # NEW: SAVE PERIODIC SIMCLR CHECKPOINTS
        # =========================================================

        if epoch % args.checkpoint_every == 0:

            checkpoint_path = (
                fold_out / f"simclr_epoch_{epoch:04d}.pt"
            )

            write_checkpoint(
                checkpoint_path,
                models,
                optimizer,
                epoch,
                "simclr",
                fold,
                args,
                scheduler=scheduler.state_dict(),
                loader_rng=loader.generator.get_state(),
                history=history,
                checks=checks,
                previous=previous,
                stable_checks=stable_checks,
                selected_labels=selected_labels,
            )

            logger.info(
                f"Fold {fold}: SimCLR checkpoint saved: "
                f"{checkpoint_path}"
            )

        # =========================================================
        # LEIDEN CLUSTERING AND UMAP CHECKS
        # =========================================================

        should_check = (
            epoch % args.cluster_check_every == 0
            or epoch == args.max_ssl_epochs
        )

        if not should_check:
            continue

        # Train embeddings determine Leiden communities.
        train_data = collect_embeddings(
            models,
            train_eval,
            device,
        )

        # Test embeddings are used only for visualization.
        if not args.skip_umap:

            test_data = collect_embeddings(
                models,
                test_eval,
                device,
            )

            project_training_umap(
                {
                    "train": train_data,
                    "test": test_data,
                },
                fold_out
                / "projections"
                / f"simclr_epoch_{epoch:04d}",
                seed=args.seed + fold,
            )

        labels, other_seed, info = cluster_snapshot(
            train_data,
            args,
            args.seed + epoch,
        )

        temporal_ari, seed_ari = community_stability(
            labels,
            previous,
            other_seed,
        )

        eligible = (
            epoch >= args.min_ssl_epochs
            and np.isfinite(temporal_ari)
            and temporal_ari >= args.cluster_stability_threshold
            and seed_ari >= args.cluster_seed_threshold
            and info["supported_fraction"]
            >= args.min_community_coverage
            and info["n_supported"] >= args.min_communities
        )

        stable_checks = (
            stable_checks + 1
            if eligible
            else 0
        )

        check = {
            "epoch": epoch,
            "temporal_ari": temporal_ari,
            "seed_ari": seed_ari,
            "stable_checks": stable_checks,
            **info,
        }

        checks.append(check)

        # Persist each Leiden check immediately (default: every 5 epochs).
        pd.DataFrame(checks).to_csv(
            fold_out / "leiden_stability.tsv", sep="\t", index=False,
        )

        logger.info(
            f"Fold {fold} Leiden train check: {check}"
        )

        previous = labels.copy()
        selected_labels = labels.copy()

        # Update the periodic checkpoint with Leiden state after this check.
        if epoch % args.checkpoint_every == 0:
            write_checkpoint(
                fold_out / f"simclr_epoch_{epoch:04d}.pt",
                models, optimizer, epoch, "simclr", fold, args,
                scheduler=scheduler.state_dict(),
                loader_rng=loader.generator.get_state(),
                history=history,
                checks=checks,
                previous=previous,
                stable_checks=stable_checks,
                selected_labels=selected_labels,
            )

        if stable_checks >= args.stability_patience:

            phase_status = (
                "train_leiden_stability_threshold_met"
            )

            break

    # =============================================================
    # SAVE TRAINING HISTORY
    # =============================================================

    pd.DataFrame(history).to_csv(
        fold_out / "simclr_history.tsv",
        sep="\t",
        index=False,
    )

    pd.DataFrame(checks).to_csv(
        fold_out / "leiden_stability.tsv",
        sep="\t",
        index=False,
    )

    if selected_labels is None:
        raise RuntimeError(
            "No training cluster snapshot was evaluated"
        )

    train_data = collect_embeddings(
        models,
        train_eval,
        device,
    )

    final_labels, other_seed, info = cluster_snapshot(
        train_data,
        args,
        args.seed + epoch,
    )

    np.save(
        fold_out / "simclr_train_community_labels.npy",
        final_labels,
    )

    np.savez_compressed(
        fold_out / "simclr_train_embeddings.npz",
        **train_data,
    )

    (fold_out / "phase_transition.json").write_text(
        json.dumps(
            {
                "status": phase_status,
                "simclr_epochs": epoch,
                "final_community_summary": info,
                "final_seed_ari": community_stability(
                    final_labels,
                    None,
                    other_seed,
                )[1],
            },
            indent=2,
        )
    )

    # =============================================================
    # SAVE FINAL SIMCLR CHECKPOINT
    # =============================================================

    write_checkpoint(
        fold_out / "simclr_final.pt",
        models,
        optimizer,
        epoch,
        "simclr",
        fold,
        args,
        scheduler=scheduler.state_dict(),
        loader_rng=loader.generator.get_state(),
        history=history,
        checks=checks,
        previous=previous,
        stable_checks=stable_checks,
        selected_labels=final_labels,
    )

    logger.info(
        f"Fold {fold}: SimCLR completed at epoch {epoch}. "
        "Final checkpoint saved."
    )

    return final_labels, phase_status, epoch



def community_batches(dataset, labels, args, fold, epoch, device):
    """Yield all available subjects per community in variable-size mini-batches.

    For communities larger than gw_batch_size, GW is a stochastic approximation
    to full within-community GW. There is NO equal-size cluster requirement.
    """
    values, counts = np.unique(labels, return_counts=True)
    rng = np.random.default_rng(args.seed + 10007 * fold + epoch)
    order = list(rng.permutation(len(values)))
    for index in order:
        community = int(values[index])
        size = int(counts[index])
        if size < args.min_community_size:
            continue
        indices = np.flatnonzero(labels == community).tolist()
        generator_seed = args.seed + 1009 * fold + 17 * epoch + community
        loader = make_loader(
            Subset(dataset, indices), min(args.gw_batch_size, size),
            True, generator_seed, args.num_workers, device,
        )
        for batch in loader:
            if len(batch["subject_id"]) >= 2:
                yield community, size, batch


def train_gw_stage(
    models, train_dataset, train_eval, test_eval, labels, reference,
    args, fold_out, fold, device, resume_state=None,
):
    """Refine only reproducibly discovered communities with entropic GW.

    Uses no test information for optimization. Paired InfoNCE anchors preserve
    subject correspondence, while the linear nuisance penalty remains active.
    Fit UMAP on train embeddings and transform test embeddings only for
    visualization at every checkpoint epoch and at the final GW epoch.
    """
    optimizer = torch.optim.AdamW(
        models.parameters(), lr=args.gw_learning_rate, weight_decay=args.weight_decay
    )
    history = []
    start_epoch = 1
    if resume_state is not None:
        models.load_state_dict(resume_state["models"], strict=True)
        optimizer.load_state_dict(resume_state["optimizer"])
        start_epoch = int(resume_state["epoch"]) + 1
        history = resume_state.get("history", [])
        restore_rng_state(resume_state)
        logger.info(f"Resuming fold {fold} GW at epoch {start_epoch}")
    for epoch in range(start_epoch, args.gw_epochs + 1):
        models.train()
        aggregate = {"gw": 0.0, "anchor": 0.0, "nuis": 0.0, "total": 0.0}
        seen = 0
        for community, community_size, batch in community_batches(
            train_dataset, labels, args, fold, epoch, device
        ):
            n = len(batch["subject_id"])
            optimizer.zero_grad(set_to_none=True)
            z = encode_batch(models, batch, device)
            loss_gw = differentiable_entropic_gw(
                z["rs"], z["thick"],
                epsilon=args.gw_epsilon, max_iter=args.gw_sinkhorn_iterations,
            )
            loss_anchor = paired_simclr_loss(
                z["rs"], z["thick"], args.temperature
            )
            loss_nuis = batch_nuisance_loss(z, batch, reference, device)
            total = (
                args.gw_weight * loss_gw
                + args.anchor_weight * loss_anchor
                + args.nuisance_weight * loss_nuis
            )
            if not torch.isfinite(total):
                raise FloatingPointError(
                    f"Nonfinite GW loss: fold {fold}, community {community}"
                )
            total.backward()
            optimizer.step()
            for name, value in (
                ("gw", loss_gw), ("anchor", loss_anchor),
                ("nuis", loss_nuis), ("total", total),
            ):
                aggregate[name] += float(value.detach()) * n
            seen += n
        if seen == 0:
            raise ValueError(
                "No eligible community minibatches; reduce --min-community-size "
                "or increase available training subjects."
            )
        entry = {"stage": "gw", "epoch": epoch, "n": seen}
        entry.update({key: value / seen for key, value in aggregate.items()})
        history.append(entry)
        logger.info(f"Fold {fold} GW epoch {epoch}: {entry['total']:.5f}")
        if epoch % args.checkpoint_every == 0 or epoch == args.gw_epochs:
            write_checkpoint(
                fold_out / f"gw_epoch_{epoch:04d}.pt",
                models, optimizer, epoch, "gw", fold, args,
                history=history,
            )
            if not args.skip_umap:
                train_data = collect_embeddings(models, train_eval, device)
                test_data = collect_embeddings(models, test_eval, device)
                project_training_umap(
                    {"train": train_data, "test": test_data},
                    fold_out / "projections" / f"gw_epoch_{epoch:04d}",
                    seed=args.seed + fold,
                )
    pd.DataFrame(history).to_csv(fold_out / "gw_history.tsv", sep="\t", index=False)
    return history


def paired_retrieval_metrics(data, temperature):
    """Subject-level paired retrieval and cosine alignment on a full split."""
    x = torch.from_numpy(data["rs"]).float()
    y = torch.from_numpy(data["thick"]).float()
    n = len(x)
    targets = torch.arange(n)
    similarity = x @ y.T
    return {
        "n_subjects": n,
        "rs_to_thick_retrieval": float((similarity.argmax(1) == targets).float().mean()),
        "thick_to_rs_retrieval": float((similarity.argmax(0) == targets).float().mean()),
        "paired_cosine": float((x * y).sum(1).mean()),
        "chance_retrieval": 1.0 / n,
    }


def evaluate_final(models, datasets, args, fold_out, fold, device, prefix=""):
    """Evaluate held-out subjects only after ALL training decisions are fixed."""
    loaders = {
        name: make_loader(
            ds, args.eval_batch_size, False, args.seed + fold,
            args.num_workers, device,
        )
        for name, ds in datasets.items()
    }
    results = {}
    embeds = {}
    for split, loader in loaders.items():
        data = collect_embeddings(models, loader, device)
        embeds[split] = data
        np.savez_compressed(fold_out / f"{prefix}{split}_embeddings.npz", **data)
        results[split] = paired_retrieval_metrics(data, args.temperature)
    if not args.skip_umap and not prefix:
        project_training_umap(embeds, fold_out / "projections", seed=args.seed + fold)
    return results


def run(args):
    """Run folds with fold-local harmonization and training-only transitions."""
    seed_everything(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Device={device}; sequential SSL -> Leiden -> within-community GW")

    raw = CleanENIGMADataset(
        root=args.data_root,
        bold_variant=args.bold_variant,
        rs_stride=args.rs_stride,
        st_stride=args.st_stride,
        normalize_bold=False,
        normalize_thickness=False,
        sites=args.sites,
        validate_headers=True,
    )
    metadata = (
        Path(args.metadata) if args.metadata
        else Path(args.data_root) / "participant_metadata.tsv"
    )
    raw.frame, metadata_exclusions = attach_age_sex_metadata(
        raw.frame, metadata, args.age_column,
        args.sex_column, args.missing_covariates,
    )
    if raw.frame.empty:
        raise ValueError("No eligible subjects after metadata validation")
    raw.sites_all = raw.frame["site"].to_numpy()
    # If complete-case exclusions changed the maximum source TR, fix the target
    # for the retained cohort BEFORE any ComBat extraction or training.
    raw.target_tr = float(raw.frame["tr_seconds"].astype(float).max())
    raw.frame["n_frames_resampled"] = [
        str(len(temporal_grid(int(row.n_frames), float(row.tr_seconds), raw.target_tr)))
        for row in raw.frame.itertuples()
    ]
    logger.info(f"Final selected-cohort target TR={raw.target_tr:g}s")
    # Window length is specified in seconds, never in a hard-coded frame count.
    # All selected subjects have been resampled to the SAME target TR.
    if args.rs_crop_seconds is not None:
        args.rs_crop_frames = max(1, int(np.floor(
            args.rs_crop_seconds / raw.target_tr + 1e-9
        )))
        if args.rs_crop_frames < 2:
            raise ValueError("--rs-crop-seconds must contain at least two target-TR frames")
        minimum = raw.frame["n_frames_resampled"].astype(int).min()
        if minimum < args.rs_crop_frames:
            raise ValueError(
                f"Requested {args.rs_crop_seconds:g}s = {args.rs_crop_frames} "
                f"frames at target TR {raw.target_tr:g}s, but the shortest "
                f"recording contains only {minimum} frames. Choose a shorter window."
            )
        logger.info(
            f"Fixed center crop: requested={args.rs_crop_seconds:g}s, "
            f"frames={args.rs_crop_frames}, "
            f"effective sampled duration={args.rs_crop_frames * raw.target_tr:g}s "
            f"(TR={raw.target_tr:g}s)"
        )
    else:
        args.rs_crop_frames = None
        if args.rs_model == "independent":
            raise ValueError("--rs-model independent needs --rs-crop-seconds")

    folds = make_subject_folds_by_site(raw.frame, args.n_splits, args.seed)
    if args.fold is not None and not 1 <= args.fold <= args.n_splits:
        raise ValueError("--fold out of range")
    out = Path(args.output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["selected_target_tr"] = raw.target_tr
    config["eligible_ids"] = raw.frame["participant_id"].tolist()
    config["manifest_sha256"] = hashlib.sha256(
        (Path(args.data_root) / "manifest.tsv").read_bytes()
    ).hexdigest()
    config["metadata_sha256"] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    config_path = out / "run_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text())
        # Existing runs without this field used the pairwise objective.
        old.setdefault("ssl_loss", "pairwise")
        ignored = {"output_root", "fold", "skip_umap", "checkpoint_every"}
        incompatible = [
            k for k in config.keys() & old.keys()
            if k not in ignored and config[k] != old[k]
        ]
        if incompatible:
            raise ValueError(
                f"Output directory has different experiment settings: {incompatible}. "
                "Choose a new --output-root."
            )
    else:
        config_path.write_text(json.dumps(config, indent=2))
    (out / "folds.json").write_text(json.dumps(folds, indent=2))
    raw.frame.to_csv(out / "eligible_manifest.tsv", sep="\t", index=False)
    metadata_exclusions.to_csv(out / "metadata_exclusions.tsv", sep="\t", index=False)
    raw.exclusions.to_csv(out / "temporal_exclusions.tsv", sep="\t", index=False)
    summary = []

    for fold_spec in folds:
        fold = int(fold_spec["fold"])
        if args.fold is not None and args.fold != fold:
            continue
        seed_everything(args.seed + fold)
        fold_out = out / f"fold_{fold:02d}"
        if (fold_out / "metrics.json").is_file():
            logger.info(f"Fold {fold} already complete; skipping")
            summary.append(json.loads((fold_out / "metrics.json").read_text()))
            continue
        fold_out.mkdir(exist_ok=True)
        split_path = fold_out / "split.json"
        if split_path.exists() and json.loads(split_path.read_text()) != fold_spec:
            raise ValueError(f"Fold {fold} split differs from saved run")
        split_path.write_text(json.dumps(fold_spec, indent=2))
        restart_state = find_fold_restart(fold_out, fold, device, args)
        subsets = {
            key: subset_clean_dataset(raw, ids)
            for key, ids in fold_spec["indices"].items()
        }
        for name, ds in subsets.items():
            ds.frame.to_csv(fold_out / f"{name}_manifest.tsv", sep="\t", index=False)
        reference = make_nuisance_reference(subsets["train"].frame)
        # HARMONIZATION FIRST: fit TRAIN only and apply to TEST, no model gradient yet.
        combat_views = {
            key: CombatSpatialView(ds) for key, ds in subsets.items()
        }

        # breakpoint()
        transforms, cache_complete = load_saved_fold_transforms(
            fold_out, args, subsets["train"].frame, subsets["test"].frame
        )
        if not cache_complete:
            if restart_state is None and (fold_out / "combat").exists():
                raise ValueError(
                    "Existing fold contains an incomplete ComBat directory but no "
                    "usable checkpoint. Move that partial directory aside or use "
                    "a fresh output directory."
                )
            if restart_state is not None:
                raise ValueError(
                    "Checkpoint exists but ComBat cache is incomplete; cannot safely resume."
                )
            transforms = prepare_fold_combat_age_sex(
                combat_views["train"], combat_views["test"],
                fold_out / "combat", modalities=args.combat, seed=args.seed + fold,
            )
        datasets = {
            key: HarmonizedModelDataset(ds, transforms, crop_frames=args.rs_crop_frames)
            for key, ds in subsets.items()
        }
        train_eval = make_loader(
            datasets["train"], args.eval_batch_size, False, args.seed + fold,
            args.num_workers, device,
        )
        test_eval = make_loader(
            datasets["test"], args.eval_batch_size, False, args.seed + fold,
            args.num_workers, device,
        )
        models = build_models(args).to(device)
        if restart_state is not None and restart_state["stage"] == "gw":
            transition_file = fold_out / "phase_transition.json"
            labels_file = fold_out / "simclr_train_community_labels.npy"
            if not transition_file.is_file() or not labels_file.is_file():
                raise ValueError("Cannot resume GW: saved SimCLR transition/labels missing")
            transition_info = json.loads(transition_file.read_text())
            labels = np.load(labels_file, allow_pickle=False)
            transition = transition_info["status"]
            ssl_epochs = int(transition_info["simclr_epochs"])
        elif restart_state is not None and (fold_out / "simclr_final.pt").is_file():
            transition_file = fold_out / "phase_transition.json"
            labels_file = fold_out / "simclr_train_community_labels.npy"
            if not transition_file.is_file() or not labels_file.is_file():
                raise ValueError("Final SimCLR checkpoint exists without transition/labels")
            models.load_state_dict(restart_state["models"], strict=True)
            transition_info = json.loads(transition_file.read_text())
            labels = np.load(labels_file, allow_pickle=False)
            transition = transition_info["status"]
            ssl_epochs = int(transition_info["simclr_epochs"])
            logger.info(f"Restored completed SimCLR for fold {fold}; starting GW if eligible")
        else:
            labels, transition, ssl_epochs = train_simclr_stage(
                models, datasets["train"], train_eval, test_eval, reference,
                args, fold_out, fold, device,
                resume_state=restart_state,
            )
        eligible = sum(
            count >= args.min_community_size
            for count in np.unique(labels, return_counts=True)[1]
        )
        gw_performed = False
        if (
            transition == "train_leiden_stability_threshold_met"
            and eligible >= args.min_communities
            and args.gw_epochs > 0
        ):
            train_gw_stage(
                models, datasets["train"], train_eval, test_eval, labels,
                reference, args, fold_out, fold, device,
                resume_state=(restart_state if restart_state is not None
                              and restart_state["stage"] == "gw" else None),
            )
            gw_performed = True
        else:
            logger.warning(
                f"Fold {fold}: GW NOT started (stability threshold unmet "
                "or too few eligible communities). No forced clusters."
            )
        # Test access happens only after the stage transition and refinement
        # are completely determined. Compare BOTH stages without test-driven
        # early stopping, parameter changes or checkpoint selection.
        final_results = evaluate_final(
            models, datasets, args, fold_out, fold, device
        )
        if gw_performed:
            checkpoint = torch.load(
                fold_out / "simclr_final.pt", map_location=device,
                weights_only=False,  # Only load checkpoints created by this run.
            )
            gw_final = fold_out / f"gw_epoch_{args.gw_epochs:04d}.pt"
            if not gw_final.is_file():
                raise FileNotFoundError(gw_final)
            models.load_state_dict(checkpoint["models"], strict=True)
            baseline_results = evaluate_final(
                models, datasets, args, fold_out, fold, device,
                prefix="simclr_baseline_",
            )
            checkpoint = torch.load(
                gw_final, map_location=device, weights_only=False
            )
            models.load_state_dict(checkpoint["models"], strict=True)
        else:
            baseline_results = final_results
        report = {
            "fold": fold,
            "target_tr": raw.target_tr,
            "combat": args.combat,
            "simclr_epochs": ssl_epochs,
            "transition": transition,
            "gw_performed": gw_performed,
            "train_communities": int(len(np.unique(labels))),
            "evaluation": final_results,
            "simclr_baseline_evaluation": baseline_results,
            "interpretation": (
                "Subject-disjoint known-site folds; no CTQ supervision, "
                "no test-driven training decisions."
            ),
        }
        (fold_out / "metrics.json").write_text(json.dumps(report, indent=2))
        summary.append(report)
        del models, datasets, subsets
        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.success(f"Completed fold {fold}: {fold_out}")

    suffix = f"_fold_{args.fold:02d}" if args.fold else ""
    (out / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2))


def parse_args():
    """Provide explicit phase, community-stability, and nuisance controls."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-root", default="./enigma_sequential_ssl_gw")
    p.add_argument("--metadata")
    p.add_argument("--age-column", default="Age")
    p.add_argument("--sex-column", default="Sex")
    p.add_argument("--missing-covariates", choices=("error", "drop"), default="error")
    p.add_argument("--sites", nargs="+")
    p.add_argument("--bold-variant", default="small8")
    p.add_argument("--rs-stride", type=int, default=4)
    p.add_argument("--rs-model", choices=("shared", "independent"), default="shared",
                   help="Shared full-recording ResNet or historical independent-stream ResNets")
    p.add_argument("--rs-crop-seconds", type=float, default=None,
                   help="Fixed center-crop duration in seconds after common-target-TR resampling; "
                        "omit for full-length shared model")
    p.add_argument("--st-stride", type=int, default=4)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--fold", type=int)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--out-dim", type=int, default=64)
    p.add_argument("--tcn-kernel", type=int, default=5)
    p.add_argument("--tcn-blocks", type=int, default=3)
    p.add_argument("--tcn-channels", type=int, default=100)
    p.add_argument("--tcn-dilation", type=int, default=2)
    p.add_argument("--pool", choices=("mean", "mean_std", "last", "attention"), default="attention")
    p.add_argument("--attention-heads", type=int, default=4)
    p.add_argument("--attention-dropout", type=float, default=0.0)
    p.add_argument("--frame-chunk", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--ssl-loss", choices=("whole", "pairwise"), default="whole")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--max-ssl-epochs", type=int, default=100)
    p.add_argument("--min-ssl-epochs", type=int, default=20)
    p.add_argument("--cluster-check-every", type=int, default=5)
    p.add_argument("--leiden-k", type=int, default=15)
    p.add_argument("--leiden-resolution", type=float, default=1.0)
    p.add_argument("--cluster-stability-threshold", type=float, default=0.90)
    p.add_argument("--cluster-seed-threshold", type=float, default=0.90)
    p.add_argument("--stability-patience", type=int, default=3)
    p.add_argument("--min-community-size", type=int, default=5)
    p.add_argument("--min-communities", type=int, default=2)
    p.add_argument("--min-community-coverage", type=float, default=0.70)
    p.add_argument("--gw-epochs", type=int, default=15)
    p.add_argument("--gw-batch-size", type=int, default=16)
    p.add_argument("--gw-learning-rate", type=float, default=2e-5)
    p.add_argument("--gw-weight", type=float, default=0.1)
    p.add_argument("--gw-epsilon", type=float, default=0.1)
    p.add_argument("--gw-sinkhorn-iterations", type=int, default=50)
    p.add_argument("--anchor-weight", type=float, default=1.0)
    p.add_argument("--nuisance-weight", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--combat", choices=("both-logsd", "thickness", "none"), default="both-logsd")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu-threads", type=int, default=3)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device")
    p.add_argument("--skip-umap", action="store_true")
    args = p.parse_args()
    if args.rs_crop_seconds is not None and (not np.isfinite(args.rs_crop_seconds) or args.rs_crop_seconds <= 0):
        p.error("--rs-crop-seconds must be a positive finite number")
    if args.rs_model == "independent" and args.rs_crop_seconds is None:
        p.error("--rs-model independent requires --rs-crop-seconds")
    if args.rs_model == "independent" and args.pool not in ("mean", "last", "attention"):
        p.error("Independent historical encoder supports mean/last pooling; attention maps to mean")
    if args.min_ssl_epochs > args.max_ssl_epochs:
        p.error("--min-ssl-epochs must not exceed --max-ssl-epochs")
    for name in (
        "n_splits", "batch_size", "eval_batch_size", "out_dim", "tcn_blocks",
        "tcn_channels", "frame_chunk", "cluster_check_every", "stability_patience",
        "gw_batch_size", "min_community_size", "min_communities", "checkpoint_every",
    ):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.batch_size < 2 or args.gw_batch_size < 2 or args.n_splits < 2:
        p.error("Training batches and n_splits require at least two")
    if args.tcn_channels % args.attention_heads and args.pool == "attention":
        p.error("--tcn-channels must be divisible by --attention-heads")
    if args.gw_epochs < 0 or args.num_workers < 0:
        p.error("GW epochs and workers cannot be negative")
    if not (0 < args.min_community_coverage <= 1):
        p.error("--min-community-coverage must be in (0,1]")
    if not (0 <= args.cluster_stability_threshold <= 1):
        p.error("--cluster-stability-threshold must be in [0,1]")
    if not (0 <= args.cluster_seed_threshold <= 1):
        p.error("--cluster-seed-threshold must be in [0,1]")
    if min(
        args.temperature, args.learning_rate, args.gw_learning_rate,
        args.gw_epsilon, args.leiden_resolution,
    ) <= 0:
        p.error("Temperatures, rates, epsilon and Leiden resolution must be > 0")
    if min(args.gw_weight, args.anchor_weight, args.nuisance_weight) < 0:
        p.error("Loss weights cannot be negative")
    return args


if __name__ == "__main__":
    run(parse_args())
