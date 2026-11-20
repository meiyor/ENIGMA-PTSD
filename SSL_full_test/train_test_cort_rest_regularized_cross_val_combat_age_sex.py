"""K-fold SSL on different subjects from the SAME sites (train/test only).
Official neuroHarmonize ComBat plus sklearn age/sex removal; TRAIN fit and TEST apply.
Fit UMAP on final training embeddings only; use transform for test embeddings.
No test-based checkpoint selection, early stopping, or hyperparameter tuning.
Original eight positional arguments and model/loss configurations are retained.
Requires corrected ResNet_Encoders_definition.py, original losses_SSL.py/utils.py,
ENIGMA_dataset_loader_clean.py and utils_combat_age_sex.py alongside/on PYTHONPATH.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from loguru import logger

from ENIGMA_dataset_loader_clean import CleanENIGMADataset
from ResNet_Encoders_definition import (ResNet3DEncoder, LateFusion4DResNet,
    LateFusion4DResNet_LSTM, LateFusion4DResNet_TemporalConv)
from losses_SSL import multimodal_pairwise_clip_loss, multimodal_multipositive_infonce_whole
from utils import auto_padding, init_xavier
# If appending the helper file to utils.py, change this module name to utils.
from utils_combat_age_sex import (
    make_subject_folds_by_site, attach_age_sex_metadata,
    prepare_fold_combat_age_sex, normalize_model_sample,
    apply_cached_combat_age_sex, project_training_umap,
)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def subset_clean_dataset(base, indices):
    subset = copy.copy(base)
    subset.frame = base.frame.iloc[indices].reset_index(drop=True).copy()
    subset.sites_all = subset.frame.site.to_numpy()
    return subset


class CombatModelDataset(Dataset):
    """Normalize as before, then apply frozen fold-specific ComBat on CPU.
    Images stay lazy; feature arrays are processed once per fold and adjusted maps cached.
    """
    def __init__(self, raw, combat):
        self.raw, self.combat = raw, combat
    def __len__(self): return len(self.raw)
    def __getitem__(self, index):
        item = self.raw[index]
        bold, thick, _, _ = normalize_model_sample(item)
        bold, thick = apply_cached_combat_age_sex(bold,thick,item['subject_id'],self.combat)
        item['rs_data'] = torch.from_numpy(np.ascontiguousarray(bold,dtype=np.float32))
        item['thickness'] = torch.from_numpy(np.ascontiguousarray(thick,dtype=np.float32))
        return item


def guard_batched_attention(module, inputs):
    """Fail early if the original cross-subject attention bug has not been fixed."""
    if inputs[0].ndim != 3:
        raise RuntimeError('MultiheadAttention received a 2D subjects x features tensor. '
                           'Apply the unsqueeze(1)/squeeze(1) fixes in ResNet3DEncoder '
                           'and LateFusion4DResNet_TemporalConv before training.')


def build_models(args):
    out_dim = args.out_dim; time_samples = args.time_samples
    rs_data_model_sel = args.rs_data_model_sel
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

    models = nn.ModuleDict({'thick':enc_thick,'rs':enc_4D_rsdata})
    for model in models.values():
        model.apply(lambda m: init_xavier(m, uniform=True))
    for module in models.modules():
        if isinstance(module, nn.MultiheadAttention):
            module.register_forward_pre_hook(guard_batched_attention)
    return models

def encode_batch(models, batch, device):
    bold = batch['rs_data'].to(device,non_blocking=True)
    thick = batch['thickness'].to(device,non_blocking=True)
    # Exactly the original B,X,Y,Z,T -> list of B,1,X,Y,Z conversion.
    seq = bold.permute(0,4,1,2,3).unsqueeze(2)
    embeddings = {'thick':models['thick'](thick.unsqueeze(1)),
                  'rs':models['rs']([seq[:,t] for t in range(seq.shape[1])])}
    return {k:F.normalize(v,dim=1) for k,v in embeddings.items()}


def ssl_loss(embeddings,args):
    if args.loss_selector == 0:
        return multimodal_pairwise_clip_loss(embeds=embeddings,temperature=args.temperature)
    return multimodal_multipositive_infonce_whole(embeds=embeddings,temperature=args.temperature)


@torch.no_grad()
def collect_embeddings(models,loader,device):
    models.eval(); thick=[]; bold=[]; subjects=[]; sites=[]
    for batch in loader:
        emb = encode_batch(models,batch,device)
        thick.append(emb['thick'].cpu().numpy()); bold.append(emb['rs'].cpu().numpy())
        subjects.extend(batch['subject_id']); sites.extend(batch['site'])
    result = {'thick':np.concatenate(thick),'rs':np.concatenate(bold),
              'subject_id':np.asarray(subjects,dtype=str),'site':np.asarray(sites,dtype=str)}
    if not np.isfinite(result['thick']).all() or not np.isfinite(result['rs']).all():
        raise ValueError('Nonfinite embeddings')
    return result


def chunked_alignment_metrics(data,args):
    """Full split evaluation without allocating a full N x N matrix.
    Mirrors the original two-modality objectives. Selector 1 intentionally retains
    self-similarity in its denominator, matching the supplied losses_SSL.py.
    Retrieval searches ALL subjects in this split, not a changing minibatch.
    """
    a=F.normalize(torch.from_numpy(data['thick']),dim=1)
    b=F.normalize(torch.from_numpy(data['rs']),dim=1)
    n=len(a); total=0.; correct_ab=0; correct_ba=0
    for start in range(0,n,args.metric_chunk):
        end=min(start+args.metric_chunk,n); targets=torch.arange(start,end)
        ab=a[start:end] @ b.T; ba=b[start:end] @ a.T
        correct_ab += int((ab.argmax(1)==targets).sum())
        correct_ba += int((ba.argmax(1)==targets).sum())
        logits_ab=ab/args.temperature; logits_ba=ba/args.temperature
        if args.loss_selector == 0:
            total += float((F.cross_entropy(logits_ab,targets,reduction='sum')+
                            F.cross_entropy(logits_ba,targets,reduction='sum'))*.5)
        else:
            aa=(a[start:end]@a.T)/args.temperature
            bb=(b[start:end]@b.T)/args.temperature
            rows=torch.arange(end-start)
            loss_a=torch.logsumexp(torch.cat([aa,logits_ab],dim=1),dim=1)-logits_ab[rows,targets]
            loss_b=torch.logsumexp(torch.cat([logits_ba,bb],dim=1),dim=1)-logits_ba[rows,targets]
            total += float((loss_a.sum()+loss_b.sum())*.5)
    return {'n_subjects':n,'full_split_ssl_loss':total/n,
            'retrieval_thick_to_rs':correct_ab/n,'retrieval_rs_to_thick':correct_ba/n,
            'paired_cosine':float((a*b).sum(1).mean()),'chance_retrieval':1/n}


def write_loss_plot(history,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame=pd.DataFrame(history)
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(frame.epoch,frame.train_loss)
    ax.set(xlabel='Epoch',ylabel='Training minibatch SSL loss',title='Training only; test evaluated after final epoch')
    fig.tight_layout(); fig.savefig(out/'training_loss.png',dpi=160); plt.close(fig)


def run(args):
    seed_everything(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device=torch.device(args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    logger.info(f'Device: {device}; {args.n_splits} folds; same sites, different train/test subjects')
    # Header-only eligibility checks BEFORE splitting; no pooled feature statistics.
    raw=CleanENIGMADataset(root=args.data_root,bold_variant=args.bold_variant,
        rs_time_window=args.rs_time_window,rs_window_crop=args.time_samples,
        rs_stride=args.rs_stride,st_stride=args.st_stride,temporal_policy=args.temporal_policy,
        normalize_bold=False,normalize_thickness=False,sites=args.sites,validate_headers=True)
    metadata=Path(args.metadata) if args.metadata else Path(args.data_root)/'participant_metadata.tsv'
    raw.frame,metadata_exclusions=attach_age_sex_metadata(
        raw.frame,metadata,args.age_column,args.sex_column,args.missing_covariates)
    raw.sites_all=raw.frame.site.to_numpy()
    folds=make_subject_folds_by_site(raw.frame,args.n_splits,args.seed)
    if args.fold is not None and not 1 <= args.fold <= args.n_splits:
        raise ValueError('--fold must be between 1 and --n-splits')
    if args.bold_variant!='small' and args.rs_stride>1:
        logger.warning('Additional spatial stride is applied on top of your selected smallN BOLD variant')
    out=Path(args.output_root).resolve()
    out.mkdir(parents=True,exist_ok=True)
    config=vars(args).copy(); config.pop('fold')
    config['data_root']=str(Path(args.data_root).resolve())
    config['manifest_sha256']=hashlib.sha256((Path(args.data_root)/'manifest.tsv').read_bytes()).hexdigest()
    config['dataset_config_sha256']=hashlib.sha256((Path(args.data_root)/'dataset_config.json').read_bytes()).hexdigest()
    config['eligible_ids']=raw.frame.participant_id.tolist()
    config['metadata_sha256']=hashlib.sha256(metadata.read_bytes()).hexdigest()
    config_path=out/'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise ValueError('Output contains a different run; choose a new --output-root')
    config_path.write_text(json.dumps(config,indent=2))
    (out/'folds.json').write_text(json.dumps(folds,indent=2))
    raw.frame.to_csv(out/'eligible_manifest.tsv',sep='\t',index=False)
    raw.exclusions.to_csv(out/'temporal_exclusions.tsv',sep='\t',index=False)
    metadata_exclusions.to_csv(out/'age_sex_exclusions.tsv',sep='\t',index=False)
    summary=[]
    for fold in folds:
        k=fold['fold']
        if args.fold is not None and args.fold!=k: continue
        fold_out=out/f'fold_{k:02d}'
        fold_out.mkdir(exist_ok=False)  # never overwrite or accidentally reuse a fold
        seed_everything(args.seed+k)
        (fold_out/'split.json').write_text(json.dumps(fold,indent=2))
        subsets={s:subset_clean_dataset(raw,idx) for s,idx in fold['indices'].items()}
        for split,ds in subsets.items():
            ds.frame.to_csv(fold_out/f'{split}_manifest.tsv',sep='\t',index=False)
        counts=pd.concat([d.frame.assign(split=s) for s,d in subsets.items()]).groupby(['site','split']).size()
        counts.rename('n_subjects').to_csv(fold_out/'site_counts.tsv',sep='\t')
        logger.info(f'Fold {k}: train={len(subsets["train"])} test={len(subsets["test"])}; '
                    f'{len(fold["sites"]["train"])} sites in both')
        # Exactly one TRAIN-only fit per enabled modality, before any gradient step.
        combat=prepare_fold_combat_age_sex(subsets['train'],subsets['test'],
            fold_out/'combat',args.combat,seed=args.seed+k)
        datasets={s:CombatModelDataset(ds,combat) for s,ds in subsets.items()}
        generator=torch.Generator().manual_seed(args.seed+k)
        options=dict(batch_size=args.batch_size,num_workers=args.num_workers,pin_memory=device.type=='cuda')
        train_loader=DataLoader(datasets['train'],shuffle=True,generator=generator,**options)
        eval_loaders={s:DataLoader(ds,shuffle=False,**options) for s,ds in datasets.items()}
        models=build_models(args).to(device)
        optimizer=torch.optim.AdamW(models.parameters(),lr=args.learning_rate,weight_decay=.01)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.iterations,eta_min=1e-5)
        history=[]
        for epoch in range(1,args.iterations+1):
            models.train(); loss_sum=0.; seen=0; skipped=0
            for batch in train_loader:
                size=len(batch['subject_id'])
                if size<2:
                    skipped+=size
                    continue
                optimizer.zero_grad(set_to_none=True)
                emb=encode_batch(models,batch,device); loss=ssl_loss(emb,args)
                if not torch.isfinite(loss): raise ValueError(f'Nonfinite loss in fold {k}, epoch {epoch}')
                loss.backward(); optimizer.step()
                loss_sum+=float(loss.detach())*size; seen+=size
            if seen==0: raise ValueError('No train batch contains >=2 subjects')
            history.append({'epoch':epoch,'train_loss':loss_sum/seen,'train_seen':seen,
                            'singleton_skipped':skipped,'learning_rate':optimizer.param_groups[0]['lr']})
            scheduler.step()
            pd.DataFrame(history).to_csv(fold_out/'training_history.tsv',sep='\t',index=False)
            logger.info(f'Fold {k}/{args.n_splits}, epoch {epoch}/{args.iterations}, loss={loss_sum/seen:.6f}')
            # Saving does not evaluate or select on test data.
            if epoch==args.iterations or epoch%args.checkpoint_every==0:
                torch.save({'epoch':epoch,'fold':k,'models':models.state_dict(),
                    'optimizer_SSL':optimizer.state_dict(),'scheduler_SSL':scheduler.state_dict(),
                    'arguments':vars(args),'train_ids':subsets['train'].frame.participant_id.tolist()},
                    fold_out/f'checkpoint_epoch_{epoch:04d}.pt')
        # Final epoch fixed in advance; no test-driven choice of model/checkpoint.
        embeddings={s:collect_embeddings(models,loader,device) for s,loader in eval_loaders.items()}
        results={'fold':k,'epoch':args.iterations,'combat':args.combat,
                 'design':'subject-disjoint train/test, overlapping sites',
                 'nuisances':'site plus linear age/sex effects (when ComBat enabled)',
                 'selection':'fixed final epoch; no validation and no test selection'}
        for split,data in embeddings.items():
            np.savez_compressed(fold_out/f'{split}_embeddings.npz',**data)
            results[split]=chunked_alignment_metrics(data,args)
        (fold_out/'metrics.json').write_text(json.dumps(results,indent=2))
        if not args.skip_umap:
            project_training_umap(embeddings,fold_out/'projections',seed=args.seed+k)
        write_loss_plot(history,fold_out)
        summary.append(results)
        del models,optimizer,scheduler,train_loader,eval_loaders,datasets,subsets,combat
        if device.type=='cuda': torch.cuda.empty_cache()
        logger.success(f'Completed fold {k}: {fold_out}')
    report={'completed_folds_this_invocation':summary,
            'interpretation':'New subjects from known sites, not generalization to unseen sites. '
                             'Retrieval measures paired-modality alignment, not PTSD prediction. '
                             'Full-split SSL loss and retrieval depend on candidate cohort size. '
                             'Different folds have independent encoders and UMAP coordinate systems.'}
    test=[r['test'] for r in summary]
    report['test_fold_macro_mean_sd']={key:{'mean':float(np.mean([r[key] for r in test])),
        'sd':float(np.std([r[key] for r in test],ddof=1)) if len(test)>1 else None}
        for key in test[0] if key!='n_subjects'}
    report['test_subjects_this_invocation']=sum(r['n_subjects'] for r in test)
    suffix=f'_fold_{args.fold:02d}' if args.fold is not None else ''
    (out/f'summary{suffix}.json').write_text(json.dumps(report,indent=2))


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('time_samples',type=int)
    p.add_argument('iterations',type=int)
    p.add_argument('batch_size',type=int)
    p.add_argument('learning_rate',type=float)
    p.add_argument('loss_selector',type=int,choices=(0,1))
    p.add_argument('out_dim',type=int)
    p.add_argument('rs_data_model_sel',type=int,choices=(0,1,2))
    p.add_argument('temperature',type=float)
    p.add_argument('--data-root',default='/scratch/jmm1987/ENIGMA_PTSD_Clean_OriginalLogic')
    p.add_argument('--output-root',default='./cort_rest_subject_cv_combat_age_sex')
    p.add_argument('--n-splits',type=int,default=5)
    p.add_argument('--fold',type=int,help='Run only this 1-based fold; omit to run all folds')
    p.add_argument('--bold-variant',default='small')
    p.add_argument('--rs-stride',type=int,default=4)
    p.add_argument('--st-stride',type=int,default=4)
    p.add_argument('--rs-time-window',type=float,default=200)
    p.add_argument('--temporal-policy',choices=('legacy','strict'),default='legacy')
    p.add_argument('--combat',choices=('both-logsd','thickness','none'),default='both-logsd',
                   help='both-logsd: normalized thickness + BOLD log temporal SD; not connectivity harmonization')
    p.add_argument('--metadata',help='Defaults to data-root/participant_metadata.tsv')
    p.add_argument('--age-column',default='Age')
    p.add_argument('--sex-column',default='Sex')
    p.add_argument('--missing-covariates',choices=('error','drop'),default='error',
                   help='Default rejects missing/unrecognized Age/Sex; drop is explicit complete-case analysis')
    p.add_argument('--sites',nargs='+')
    p.add_argument('--num-workers',type=int,default=0)
    p.add_argument('--cpu-threads',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device')
    p.add_argument('--metric-chunk',type=int,default=256)
    p.add_argument('--checkpoint-every',type=int,default=10)
    p.add_argument('--skip-umap',action='store_true',help='Only for smoke tests or embedding-only runs')
    a=p.parse_args()
    if min(a.time_samples,a.iterations,a.rs_stride,a.st_stride,a.cpu_threads,a.metric_chunk,a.checkpoint_every)<1:
        p.error('Sample/epoch counts, strides, threads, metric chunk and checkpoint interval must be positive')
    if a.batch_size<2 or a.out_dim<2 or a.n_splits<2 or a.num_workers<0:
        p.error('batch_size/out_dim/n_splits >=2, num_workers >=0 required')
    if min(a.learning_rate,a.temperature,a.rs_time_window)<=0:
        p.error('Learning rate, temperature and time window must be positive')
    return a


if __name__=='__main__':
    run(parse_args())
