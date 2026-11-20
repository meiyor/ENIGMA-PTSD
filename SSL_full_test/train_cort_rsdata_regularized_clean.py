"""RS BOLD + cortical-thickness SSL training on the clean manifest export.
Keep ENIGMA_dataset_loader_clean.py beside this file. Existing losses_SSL.py,
utils.py and corrected ResNet_Encoders_definition.py are required.
Preserves the original encoders, losses, normalization and diagnostic plots.
This is the original single-cohort training workflow, NOT site-held-out CV or ComBat.
Eight original positional arguments are retained; --data-root is required.
"""
import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # or "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# import plotting tools
import matplotlib.pyplot as plt
import seaborn as sns

import umap.umap_ as umap
from sklearn.manifold import TSNE

from loguru import logger
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

# get the dataloder and dataset definition here***
from ENIGMA_dataset_loader_clean import CleanENIGMADataset
from torch.utils.data import DataLoader

# get the model definitions here***
from ResNet_Encoders_definition import LateFusion4DResNet, LateFusion4DResNet_LSTM, ResNet3DEncoder, LateFusion4DResNet_TemporalConv

# import losses for SSL and alternative regularization
from losses_SSL import multimodal_pairwise_clip_loss, multimodal_multipositive_infonce_whole

# get the utils functions from the utils module **DO THAT FOR ALL THE CODE**

# define the device here
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# import the utils function from tehe utils module
from utils import ( load_latest_ckpt_cort,
                    read_metric_txt,
                    plotting_twinx_variables,
                    to_np,
                    make_subject_palette,
                    wasserstein_avg_over_dims,
                    pairwise_wasserstein,
                    stack_modalities,
                    init_xavier,
                    compute_silhouette,
                    compute_nmi_via_kmeans,
                    auto_padding
                  )

# set the seeds initialization here
seed = 42

# always define this seed before running your implementation!!
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# get the unique functions per modules here
def get_concat_embedding(rs_DATA, st_DATA,
                        enc_thick,
                        enc_4D_rsdata, batch_sizes):
    """
      get all the embeddings from each iteration here
      project all the data with the models trained after that
    """
    out_thick = enc_thick(st_DATA.unsqueeze(1))

    rs_ = rs_DATA.permute(0, 4, 1, 2, 3).unsqueeze(2)  # (B,T,1,D,H,W)
    RSDATA = [rs_[:, t] for t in range(rs_.shape[1])]
    out_rs = enc_4D_rsdata(RSDATA)

    embeds_all = {
        "thick": out_thick, "rs": out_rs
    }
    embeds_all = {k: F.normalize(v, dim=1) for k, v in embeds_all.items()}

    # stack this in to_np way
    embeds_np = {k: to_np(v) for k, v in embeds_all.items()}

    wd = pairwise_wasserstein(embeds_np)

    X_all, mod_labels, mod_keys = stack_modalities(embeds_np)

    val_clust = np.tile(np.arange(batch_sizes), 2)

    if batch_sizes > 1:
       sil_mod = compute_silhouette(X_all, val_clust, metric="cosine")
       nmi_mod = compute_nmi_via_kmeans(X_all, val_clust, n_clusters=batch_sizes)
    else:
       sil_mod = np.nan
       nmi_mod = np.nan

    z = torch.cat([embeds_all[k] for k in ["thick","rs"]], dim=1)
    return z, wd, sil_mod, nmi_mod, embeds_all

## get the UMAP and tSNE projections here..
def get_UMAP_tSNE_projections(Z, proj_components:int=2, ndim:int=128):
    """
      get the UMAP and TSNE projects
      given the eval embeddings
    """
    umap_map = umap.UMAP(n_neighbors=5, min_dist=0.1, spread=3.0, n_components=proj_components, random_state=42, verbose=True, metric="euclidean", init="random")
    # take into account these values are concat projections of ndim values
    # reshape here the input values here
    Z_stacked = np.vstack(Z)
    Z_reshaped = Z_stacked.reshape(Z_stacked.shape[0], 2, ndim)
    Z_final = Z_reshaped.reshape(-1, ndim)
    umap_proj = umap_map.fit_transform(Z_final)

    return umap_proj

# plot the projections here
def plot_proj_feat(proj_feat, sub_labels, subject_order, subject_palette, modalities, title: str, epoch: str, suffix: str, folder_name: str):
    """
    Scatter-plot 2D t-SNE embeddings and save to disk.

    Parameters
    ----------
    proj_feat : np.ndarray, shape (N, 2) -> this can be tsne_feat or umap_feat input representations
    labels : array-like
        Class/subject labels for coloring.
    title, epoch, suffix, folder_name, class_value, subj : str
        Strings for titling, legend label, and output filename.

    Returns
    -------
    None
    """

    markers=["^", "D"]
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=proj_feat[:, 0], y=proj_feat[:, 1], hue=sub_labels, hue_order=subject_order, style=modalities, markers=markers, palette=subject_palette, s=180, edgecolor="k", alpha=0.7, legend=False)
    plt.title(title)
    # plt.legend(title=class_value, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"{folder_name}/{suffix}_iter_{epoch}.jpg")
    plt.close("all")

if __name__ == "__main__":
    """
      **Main section of the code**
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('time_samples', type=int)
    parser.add_argument('iterations', type=int, help='Total epochs')
    parser.add_argument('batch_size', type=int)
    parser.add_argument('learning_rate', type=float)
    parser.add_argument('loss_selector', type=int, choices=(0, 1))
    parser.add_argument('out_dim', type=int)
    parser.add_argument('rs_data_model_sel', type=int, choices=(0, 1, 2))
    parser.add_argument('temperature', type=float)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--bold-variant', default='small')
    parser.add_argument('--rs-stride', type=int, default=4,
                        help='Additional spatial stride; use 1 for already downsampled BOLD')
    parser.add_argument('--st-stride', type=int, default=4)
    parser.add_argument('--rs-time-window', type=float, default=200)
    parser.add_argument('--temporal-policy', choices=('legacy', 'strict'), default='legacy')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--validate-headers', action='store_true')
    parser.add_argument('--sites', nargs='+', help='Optional exact site names from manifest.tsv')
    parser.add_argument('--shuffle', action='store_true', help='Original default is unshuffled')
    parser.add_argument('--output-root', default='./clean_training_runs')
    parser.add_argument('--resume', action='store_true', help='Resume this clean-data run only')
    args = parser.parse_args()
    time_samples, iterations, batch_size = args.time_samples, args.iterations, args.batch_size
    learning_rate, loss_selector, out_dim = args.learning_rate, args.loss_selector, args.out_dim
    rs_data_model_sel, temperature = args.rs_data_model_sel, args.temperature
    if min(time_samples, iterations, out_dim, args.rs_stride, args.st_stride) < 1:
        parser.error('Dimensions, epochs and strides must be positive')
    if batch_size < 2 or learning_rate <= 0 or temperature <= 0 or args.num_workers < 0:
        parser.error('Use batch_size >= 2, positive learning rate/temperature and nonnegative workers')
    if rs_data_model_sel == 1 and out_dim < 2:
        parser.error('LSTM model needs out_dim >= 2')
    run_name = (f'clean_{args.bold_variant}_T{time_samples}_rs{args.rs_stride}_st{args.st_stride}_'
                f'{iterations}_{batch_size}_{learning_rate}_{loss_selector}_{out_dim}_'
                f'{rs_data_model_sel}_{temperature}')
    model_path = str(Path(args.output_root) / run_name / 'models')
    vis_path = str(Path(args.output_root) / run_name / 'visualization')
    config_path = Path(model_path) / 'run_config.json'
    config = vars(args).copy()
    config.pop('resume')
    config['data_root'] = str(Path(args.data_root).resolve())
    if args.resume:
        if not config_path.exists() or json.loads(config_path.read_text()) != config:
            raise ValueError('Resume requires an identical clean-data run configuration')
    elif Path(model_path).exists() and any(Path(model_path).iterdir()):
        raise FileExistsError('Run already exists: use --resume or a different --output-root')

    # Define here the models and plot folders for interim results visualization
    if not os.path.exists(model_path):
       os.makedirs(model_path, exist_ok=True)
    if not os.path.exists(vis_path):
       os.makedirs(vis_path, exist_ok=True)

    config_path.write_text(json.dumps(config, indent=2))

    # Only cortical thickness and RS BOLD are instantiated.
    enc_thick = ResNet3DEncoder(
        in_channels=1,
        # interrupting the action of some layers if conside or not
        use_stages=(True, False, False, False),
        out_dim=out_dim,
        emb_dim=out_dim*2,
        attn_heads=1,
        activation_function="gelu",
        attention_projection=True,
        conv_overrides=[
            # alway define this kernel size and padding  as odd values NOT event. Just
            # work on the convolutional kernels
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

    # define the 4D model for the RSData
    if rs_data_model_sel == 0:
       enc_4D_rsdata = LateFusion4DResNet(
           n_streams=time_samples,
           emb_dim=out_dim*2,
           fusion="concat",
           out_dim=out_dim,
           pretrained=False,
       )
    elif rs_data_model_sel == 1:
       enc_4D_rsdata = LateFusion4DResNet_LSTM(
           n_streams=time_samples,
           emb_dim=out_dim*2,
           fusion="lstm",          # kept for compatibility; you can ignore
           out_dim=int(out_dim/2),             # each ResNet3DEncoder output dim
           pretrained=False,
           lstm_hidden=int(out_dim/2),
           lstm_layers=2,
           bidirectional=True,
           lstm_dropout= 0.75,
           pool= "mean",            # "last" | "mean"
           out_dim_final=out_dim,
       )
    else:
       enc_4D_rsdata = LateFusion4DResNet_TemporalConv(
           n_streams=time_samples,
           emb_dim=out_dim*2,
           out_dim=out_dim,            # per-branch output channels C
           pretrained=False,
           kernel_size=5,
           n_blocks=3,
           hidden_channels=100,
           dilation_growth=2,
           attention_projection=True,
           attn_heads=1,
           pool="mean",
           out_dim_final=out_dim,
       )

    # initialize weights models
    for model in [enc_4D_rsdata, enc_thick]:
        model.to(device)
        model.apply(lambda m: init_xavier(m, uniform=True))


    # define the optimizer here
    optimizer_SSL = torch.optim.AdamW(list(enc_4D_rsdata.parameters()) + list(enc_thick.parameters()), lr=learning_rate)
    scheduler_SSL = CosineAnnealingLR(optimizer_SSL, T_max=iterations, eta_min=1e-5)

    # validates this first!!
    if args.resume:
       start_iter, _ = load_latest_ckpt_cort(model_path, device, enc_4D_rsdata, enc_thick, optimizer_SSL, scheduler_SSL)
       MI_vals = read_metric_txt(f"{model_path}/mutual_information_interim.txt")
       SIL_vals = read_metric_txt(f"{model_path}/silhoutte_interim.txt")
       WD_vals = read_metric_txt(f"{model_path}/wasserstein_distance_interim.txt")
       LOSS_vals = read_metric_txt(f"{model_path}/losses_interim.txt")
       ITERS = list(range(0, start_iter, 10))
    else:
       start_iter = 0
       MI_vals = []
       SIL_vals = []
       WD_vals = []
       ITERS = []
       LOSS_vals = []

    logger.info("All encoders has been defined!!")

    # set the models in train mode
    enc_4D_rsdata.train()
    enc_thick.train()

    dataset = CleanENIGMADataset(
        root=args.data_root, bold_variant=args.bold_variant,
        rs_time_window=args.rs_time_window, rs_window_crop=time_samples,
        rs_stride=args.rs_stride, st_stride=args.st_stride,
        temporal_policy=args.temporal_policy, sites=args.sites,
        validate_headers=args.validate_headers,
    )
    if len(dataset) < 2:
        raise ValueError('At least two eligible subjects are required for contrastive training')
    dataset.frame.to_csv(Path(model_path) / 'selected_manifest.tsv', sep='\t', index=False)
    dataset.exclusions.to_csv(Path(model_path) / 'loader_exclusions.tsv', sep='\t', index=False)
    loader_options = dict(batch_size=batch_size, num_workers=args.num_workers,
                          pin_memory=device.type == 'cuda')
    data_loader_ENIGMA = DataLoader(dataset, shuffle=args.shuffle, **loader_options)
    data_loader_ENIGMA_eval = DataLoader(dataset, shuffle=False, **loader_options)
    logger.info(f'Clean dataset: {len(dataset)} subjects, {len(set(dataset.sites_all))} sites; '
                f'BOLD={args.bold_variant}, additional strides RS={args.rs_stride}, thickness={args.st_stride}')
    if args.bold_variant != 'small' and args.rs_stride > 1:
        logger.warning('A spatial stride is being applied on top of the selected smallN BOLD variant.')
    logger.info('Training-cohort diagnostics only: this script does not implement held-out CV or ComBat.')

    logger.info("Start SSL training!!..")

    if loss_selector == 0:
       logger.info("Using SSL loss per modality pairs")
    else:
       logger.info("Using SSL loss across ALL the modalities")

    for iter in range(start_iter, iterations):

        # check if the dataloader works
        idx_sample = []
        sites_sample = []
        tr_values = []
        time_sub_values = []
        subjects_ids = []

        loss_interim = []

        for batch_data in data_loader_ENIGMA:
            if batch_data is None:  # validate this when batch is None and skip
                continue

            idx = batch_data['index']
            if len(idx) < 2:
                logger.warning('Skipping singleton training batch: contrastive loss needs negatives')
                continue
            rs_DATA = batch_data['rs_data'].to(device, non_blocking=True)
            st_DATA = batch_data['thickness'].to(device, non_blocking=True)
            subject_index = batch_data['subject_id']
            sites_idx = batch_data['site']
            TRs = batch_data['tr_seconds']
            time_subject = batch_data['duration_seconds']

            # get the inputs on each modality and get the embeddings
            out_thick = enc_thick(st_DATA.unsqueeze(1))
            # create the list of tensor the 4D image timesamples. Do this permutation before transforming the tensor to a list of tensors!!
            rs_DATA = rs_DATA.permute(0, 4, 1, 2, 3).unsqueeze(2)
            RSDATA = [rs_DATA[:, t] for t in range(rs_DATA.shape[1])]
            out_rsdata = enc_4D_rsdata(RSDATA)

            embeds_all = {
              "thick":  out_thick,    # (B,D)
              "rs":    out_rsdata,  # (B,D)
            }

            # normalize the embeddings here to make it easier map in the l2-sphere later..
            embeds_all = {k_embeds: F.normalize(v_embeds, dim=1) for k_embeds, v_embeds in embeds_all.items()}

            # define and update the loss here. Use the current embeddings to calculate the loss but NOT the ones for the clustering representation
            if loss_selector == 0:
               loss_SSL = multimodal_pairwise_clip_loss(embeds=embeds_all, temperature=temperature)
            else:
               loss_SSL = multimodal_multipositive_infonce_whole(embeds=embeds_all, temperature=temperature)

            loss_interim.append(loss_SSL.detach())

            # do the loss backwards and the zero grad update
            optimizer_SSL.zero_grad()
            loss_SSL.backward()
            optimizer_SSL.step()


            # do the for across all the modalities
            tr_values.append(TRs)
            idx_sample.append(idx)
            time_sub_values.append(time_subject)
            sites_sample.append(sites_idx)
            subjects_ids.append(subject_index)
            # logger.info(f"Reading modalities for subject {subject_index}")
            # logger.success(f"batch size index {idx}")

        # get the embeddings projected in eval mode here
        # report the interim loss here
        if not loss_interim:
            raise RuntimeError('No training batches contained at least two subjects')
        scheduler_SSL.step()  # T_max is epochs, so step once per epoch.
        mean_loss = torch.stack(loss_interim).mean()
        mean_loss_value = mean_loss.item()
        logger.info(f"Training iteration {iter} with loss: {mean_loss_value}..")

        models = [enc_4D_rsdata, enc_thick]
        # processing the projected embeddings in eval mode
        # ---- END OF EPOCH: EMBEDDING SNAPSHOT IN EVAL MODE ----
        # (optional: only every few epochs)
        Z_vals = []
        sites_evals = []
        subs_evals = []
        mi_vals = []
        sil_vals = []
        wd_vals = []

        if iter == 0 or iter % 10 == 0 or iter == iterations - 1:
           prev_modes = [m.training for m in models]
           for model in models:
               model.eval()

           # collect a limited number of batches to keep it light
           with torch.no_grad():
               for j, batch_data_eval in enumerate(data_loader_ENIGMA_eval):

                   if batch_data_eval is None:  # validate this when batch is None and skip
                       continue

                   idx_eval = batch_data_eval['index']
                   subject_index_eval = batch_data_eval['subject_id']
                   sites_idx_eval = batch_data_eval['site']
                   rs_DATA_eval = batch_data_eval['rs_data'].to(device, non_blocking=True)
                   st_DATA_eval = batch_data_eval['thickness'].to(device, non_blocking=True)

                   z, wd, sil, nmi, _ = get_concat_embedding(
                      rs_DATA_eval, st_DATA_eval,
                      enc_thick, enc_4D_rsdata, len(idx_eval)
                   )

                   Z_vals.append(z.detach().cpu().numpy())
                   sites_evals.append(sites_idx_eval)
                   subs_evals.append(subject_index_eval)

                   mi_vals.append(nmi)
                   sil_vals.append(sil)
                   wd_vals.append(sum(wd.values()) / len(wd))

           # Save embeddings with exact subject/site row alignment, including final singletons.
           np.savez_compressed(
               Path(vis_path) / f'embeddings_iter_{iter:05d}.npz',
               embeddings=np.vstack(Z_vals),
               subject_id=np.asarray([s for batch in subs_evals for s in batch]),
               site=np.asarray([s for batch in sites_evals for s in batch]),
               modality_order=np.asarray(['thick', 'rs']),
           )
           # get UMAP and t-SNE projections
           umap_projections = get_UMAP_tSNE_projections(Z=Z_vals, proj_components=2, ndim=out_dim)
           modality_ids = np.tile(np.arange(2), np.vstack(Z_vals).shape[0])
           subs_evals = [str(s) for sublist in subs_evals for s in sublist]
           subs_evals = np.array(subs_evals)  # convert list → array
           subs_evals_long = np.repeat(subs_evals, 2)

           MI_vals.append(np.nanmean(np.array(mi_vals)))
           SIL_vals.append(np.nanmean(np.array(sil_vals)))
           WD_vals.append(np.nanmean(np.array(wd_vals)))
           LOSS_vals.append(mean_loss_value)
           ITERS.append(iter)

           ## append the values in the last part of the metrics file
           with open(f"{model_path}/mutual_information_interim.txt", "a") as f_mi:
              f_mi.write(str(np.nanmean(np.array(mi_vals))) + '\n')
           with open(f"{model_path}/wasserstein_distance_interim.txt", "a") as f_wd:
              f_wd.write(str(np.nanmean(np.array(wd_vals))) + '\n')
           with open(f"{model_path}/silhoutte_interim.txt", "a") as f_sil:
              f_sil.write(str(np.nanmean(np.array(sil_vals))) + '\n')
           with open(f"{model_path}/losses_interim.txt", "a") as f_loss:
              f_loss.write(str(mean_loss_value) + '\n')

           # plotting the metrics here
           logger.info(f"Plotting metrics for iteration {iter}!!")
           plotting_twinx_variables(np.array(ITERS), np.array(LOSS_vals), np.array(MI_vals), title=f"SSL loss and MI", x_label="epochs", y_label1="SSL Loss", y_label2="NMI", folder_images=vis_path, iter=iter)
           plotting_twinx_variables(np.array(ITERS), np.array(LOSS_vals), np.array(SIL_vals), title=f"SSL loss and Silhoutte", x_label="epochs", y_label1="SSL Loss", y_label2="Silhoutte", folder_images=vis_path, iter=iter)
           plotting_twinx_variables(np.array(ITERS), np.array(LOSS_vals), np.array(WD_vals), title=f"SSL loss and Wasserstein Distance", x_label="epochs", y_label1="SSL Loss", y_label2="Wasserstein Distance", folder_images=vis_path, iter=iter)
           plotting_twinx_variables(np.array(ITERS), np.array(MI_vals), np.array(WD_vals), title=f"MI and Wasserstein Distance", x_label="epochs", y_label1="NMI", y_label2="Wasserstein Distance", folder_images=vis_path, iter=iter)
           plotting_twinx_variables(np.array(ITERS), np.array(MI_vals), np.array(SIL_vals), title=f"MI and Silhoutte", x_label="epochs", y_label1="NMI", y_label2="Silhoutte", folder_images=vis_path, iter=iter)

           if True:  # Build palette on every snapshot, including resumed runs.
              subject_order = sorted(np.unique(subs_evals_long).tolist())
              subject_palette = make_subject_palette(subject_order, palette_name="tab20")

           logger.info(f"Plotting UMAP projections for iteration {iter}!!")
           # plotting the UMAP projections here
           plot_proj_feat(proj_feat=umap_projections, sub_labels=subs_evals_long, subject_order=subject_order, subject_palette=subject_palette, modalities=modality_ids, title=f"UMAP projections for epoch {iter} - SSL loss {mean_loss_value}", epoch=str(iter), suffix="umap_projections", folder_name=vis_path)

           # restore to train modes back
           for model, was_train in zip(models, prev_modes):
               model.train(was_train)

           # save the checkpoint here
           ckpt_path = os.path.join(model_path, f"ssl_ckpt_iter_{iter:05d}.pth")

           ckpt = {
                "iter": iter,
                "models": {
                   "enc_4D_rsdata": enc_4D_rsdata.state_dict(),
                   "enc_thick": enc_thick.state_dict(),
                },
                "optimizer_SSL": optimizer_SSL.state_dict(),
                "scheduler_SSL": scheduler_SSL.state_dict(),
           }

           torch.save(ckpt, ckpt_path)
           logger.success(f"Saving checkpoint for iteration {iter}!!")

