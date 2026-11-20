#!/usr/bin/env python3
"""Evaluate saved ENIGMA SimCLR checkpoints without retraining or UMAP.

One checkpoint is loaded at a time. Leiden NMI comes directly from stored
``selected_labels``; cosine silhouette requires one no-gradient forward pass
through the original train subjects. Uses the existing fold-specific ComBat
cache and does not refit harmonization. Writes results into --fold-dir.

Run alongside the training script, dataset loader, encoder definitions and
updated utils.py. The dataset stays lazy; use --eval-batch-size 1 for minimum
peak activation memory. --nmi-only does not read images or load the encoders.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import re
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import normalized_mutual_info_score, silhouette_score


def parse_args():
    """Parse checkpoint folder and memory-conscious evaluation settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--training-script", type=Path, default=Path(__file__).with_name(
        "train_test_cort_rest_regularized_cross_val_combat_age_sex_umap_epochs_model_select_duration_filter.py"
    ))
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Override original data-root if data were moved.")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--nmi-only", action="store_true",
                        help="Compute NMI only; requires no MRI data or GPU.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recalculate previously saved silhouette rows.")
    args = parser.parse_args()
    if args.eval_batch_size < 1:
        parser.error("--eval-batch-size must be positive")
    return args


def load_training_module(script_path):
    """Import existing model/dataset functions without starting training."""
    if not script_path.is_file():
        raise FileNotFoundError(f"Training script not found: {script_path}")
    spec = importlib.util.spec_from_file_location("enigma_training_for_eval", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_catalog(fold_dir):
    """Return numeric-epoch-ordered SimCLR checkpoints only."""
    files = []
    for path in fold_dir.glob("simclr_epoch_*.pt"):
        match = re.fullmatch(r"simclr_epoch_(\d+)\.pt", path.name)
        if match:
            files.append((int(match.group(1)), path))
    return sorted(files)


def load_checkpoint(path):
    """Memory-map a trusted local PyTorch checkpoint on CPU where supported."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def prepare_train_loader(train_module, fold_dir, configuration, batch_size, device, data_root):
    """Rebuild only the original eligible training split and reuse ComBat maps.

    The retained subject IDs in eligible_manifest.tsv determine the exact
    fold indexing; no metadata joins or harmonization fitting are repeated.
    """
    root = Path(data_root or configuration["data_root"])
    manifest = pd.read_csv(fold_dir.parent / "eligible_manifest.tsv", sep="\t",
                           dtype={"participant_id": str})
    fold_spec = json.loads((fold_dir / "split.json").read_text())
    train_indices = fold_spec["indices"]["train"]
    selected = manifest.iloc[train_indices].reset_index(drop=True).copy()

    raw = train_module.CleanENIGMADataset(
        root=root,
        bold_variant=configuration["bold_variant"],
        rs_stride=configuration["rs_stride"],
        st_stride=configuration["st_stride"],
        normalize_bold=False,
        normalize_thickness=False,
        sites=configuration.get("sites"),
        validate_headers=False,
    )
    raw.frame["participant_id"] = raw.frame["participant_id"].astype(str)
    if raw.frame["participant_id"].duplicated().any():
        raise ValueError("Duplicate IDs in current dataset; cannot recover fold safely")
    indexed = raw.frame.set_index("participant_id", drop=False)
    missing = set(selected["participant_id"]) - set(indexed.index)
    if missing:
        raise ValueError(f"Missing {len(missing)} training subjects in data-root")
    raw.frame = indexed.loc[selected["participant_id"].tolist()].reset_index(drop=True)
    for column in ("AGE", "SEX_M"):
        raw.frame[column] = selected[column].to_numpy()
    raw.sites_all = raw.frame["site"].to_numpy()
    raw.target_tr = float(configuration.get("selected_target_tr",
                                           selected["tr_seconds"].max()))

    mode = configuration.get("combat", "both-logsd")
    if mode == "none":
        transforms = None
    else:
        cache_file = fold_dir / "combat" / "transforms.json"
        if not cache_file.is_file():
            raise FileNotFoundError(f"Required saved ComBat cache: {cache_file}")
        transforms = json.loads(cache_file.read_text())

    crop = configuration.get("rs_crop_frames")
    if crop is None and configuration.get("rs_crop_seconds") is not None:
        crop = max(1, int(np.floor(float(configuration["rs_crop_seconds"])
                                   / raw.target_tr + 1e-9)))
    dataset = train_module.HarmonizedModelDataset(raw, transforms, crop_frames=crop)
    loader = train_module.make_loader(
        dataset, batch_size=batch_size, shuffle=False,
        seed=int(configuration.get("seed", 42)) + int(fold_spec["fold"]),
        workers=0, device=device,
    )
    return loader, selected["participant_id"].astype(str).to_numpy()


def extract_joint_embeddings(model, loader, device, expected_ids, encode_batch):
    """Stream deterministic train inference; keep only small 128D arrays."""
    model.eval()
    vectors = []
    identifiers = []
    with torch.inference_mode():
        for batch in loader:
            outputs = encode_batch(model, batch, device)
            joint = outputs["rs"] + outputs["thick"]
            joint = torch.nn.functional.normalize(joint, dim=1)
            vectors.append(joint.cpu().numpy())
            identifiers.extend(str(s) for s in batch["subject_id"])
            del outputs, joint, batch
    if not np.array_equal(np.asarray(identifiers), expected_ids):
        raise ValueError("Train subject order mismatch; refusing invalid silhouette")
    return np.concatenate(vectors, axis=0)


def main():
    """Read checkpoints sequentially, cache metric rows, and plot twin axes."""
    args = parse_args()
    fold_dir = args.fold_dir.resolve()
    catalog = checkpoint_catalog(fold_dir)
    if not catalog:
        raise FileNotFoundError(f"No simclr_epoch_*.pt checkpoints in {fold_dir}")

    config_path = fold_dir.parent / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run configuration not found: {config_path}")
    configuration = json.loads(config_path.read_text())
    table_path = fold_dir / "simclr_nmi_silhouette_by_epoch.tsv"
    previous_rows = {}
    if table_path.exists() and not args.overwrite:
        previous_rows = {
            int(record["epoch"]): record
            for record in pd.read_csv(table_path, sep="\t").to_dict("records")
        }

    device = torch.device(args.device)
    module = None
    loader = None
    ids = None
    rows = []
    previous_labels = None
    for epoch, path in catalog:
        state = load_checkpoint(path)
        labels = state.get("selected_labels")
        if labels is not None:
            labels = np.asarray(labels).reshape(-1)
        if labels is None:
            print(f"Epoch {epoch}: missing selected_labels (not a clustering epoch)")
            del state
            continue

        # NMI is always derived from the actual checkpoint labels.
        nmi = (float(normalized_mutual_info_score(previous_labels, labels))
               if previous_labels is not None else float("nan"))
        previous_labels = labels.copy()
        existing = previous_rows.get(epoch, {})
        silhouette = float(existing.get("silhouette_cosine", float("nan")))
        if not args.nmi_only and not np.isfinite(silhouette):
            if module is None:
                module = load_training_module(args.training_script.resolve())
                loader, ids = prepare_train_loader(
                    module, fold_dir, configuration, args.eval_batch_size,
                    device, args.data_root,
                )
            saved_args = dict(configuration)
            saved_args.update(state.get("args", {}))
            saved_args["rs_crop_frames"] = (saved_args.get("rs_crop_frames") or
                (max(1, int(np.floor(float(saved_args["rs_crop_seconds"])
                                      / float(configuration["selected_target_tr"]) + 1e-9)))
                 if saved_args.get("rs_crop_seconds") is not None else None))
            model = module.build_models(Namespace(**saved_args))
            model.load_state_dict(state["models"], strict=True)
            model.to(device)
            del state
            embeddings = extract_joint_embeddings(model, loader, device,
                                                  ids, module.encode_batch)
            if len(labels) != len(embeddings):
                raise ValueError(f"Checkpoint {epoch} labels/embedding count mismatch")
            sizes = np.unique(labels, return_counts=True)[1]
            if len(sizes) > 1 and len(sizes) < len(labels):
                silhouette = float(silhouette_score(embeddings, labels, metric="cosine"))
            del model, embeddings
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        else:
            del state

        rows.append({
            "epoch": epoch,
            "nmi_previous": nmi,
            "silhouette_cosine": silhouette,
            "n_communities": len(np.unique(labels)),
            "n_subjects": len(labels),
        })
        # Save progress after every epoch to survive preemption safely.
        pd.DataFrame(rows).to_csv(table_path, sep="\t", index=False)
        print(f"Epoch {epoch:4d}: NMI={nmi:.4f}, silhouette={silhouette:.4f}",
              flush=True)

    if not rows:
        raise RuntimeError("No checkpoints with saved Leiden labels")
    from utils import plot_checkpoint_nmi_silhouette
    plot_checkpoint_nmi_silhouette(pd.DataFrame(rows), fold_dir)
    print(f"Saved metrics and twinx plots in: {fold_dir}")


if __name__ == "__main__":
    main()
