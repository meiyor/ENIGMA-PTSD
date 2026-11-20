"""Subject splits within sites, train-only ComBat, and train-fitted UMAP.
These helpers can remain in this module or be appended to the training utils.py.
Only numpy/pandas are imported eagerly. neuroCombat and UMAP are imported on use.
No held-out subject is ever used to fit ComBat or UMAP. Sites overlap across splits.
"""
from pathlib import Path
import json
import pickle
import warnings
import numpy as np
import pandas as pd


def make_subject_folds_by_site(frame, n_splits=5, seed=42):
    """Subject-disjoint K folds, with EVERY site in train and test in EVERY fold.
    Each subject is tested exactly once. Requires >=K subjects per site, and at
    least two training subjects per site for ComBat. Never silently drops sites.
    Uses stratification by site, not diagnosis; all rows must be one subject each.
    """
    from sklearn.model_selection import StratifiedKFold
    if n_splits < 2: raise ValueError('n_splits must be >=2')
    if frame.participant_id.duplicated().any():
        raise ValueError('Select one BOLD variant and one row per participant')
    if frame.site.isna().any() or (frame.site.astype(str).str.strip() == '').any():
        raise ValueError('Missing site labels')
    counts = frame.groupby('site').size()
    if len(counts) < 2: raise ValueError('At least two sites are required for ComBat')
    too_small = counts[counts < n_splits]
    if len(too_small):
        raise ValueError(f'Every site needs >= {n_splits} eligible subjects: '
                         f'{too_small.to_dict()}. Lower --n-splits or explicitly choose sites; '
                         'no subjects/sites were silently dropped.')
    splitter = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    folds=[]; tested=[]; expected=set(frame.site.astype(str))
    for k,(train,test) in enumerate(splitter.split(np.zeros(len(frame)),frame.site),start=1):
        a,b = frame.iloc[train],frame.iloc[test]
        if set(a.participant_id) & set(b.participant_id): raise AssertionError('Subject leakage')
        if set(a.site.astype(str)) != expected or set(b.site.astype(str)) != expected:
            raise AssertionError('Every site must occur in both train and test')
        if (a.groupby('site').size() < 2).any():
            raise ValueError('ComBat needs >=2 training subjects/site; increase cohort or change folds')
        folds.append({'fold':k,'indices':{'train':train.tolist(),'test':test.tolist()},
                      'sites':{'train':sorted(expected),'test':sorted(expected)}})
        tested.extend(test.tolist())
    if sorted(tested) != list(range(len(frame))): raise AssertionError('Test coverage is not exactly once')
    return folds


def fit_combat_train(x, sites, support=None, epsilon=1e-10):
    """Fit ordinary parametric EB ComBat on subjects x features, training ONLY.
    No clinical covariates are included: this is not a claim of preserving diagnosis,
    trauma, age or sex effects when confounded with site. Constant/unsupported features
    remain unchanged. Exact repeated parcel profiles count once in the EB fit.
    """
    from neuroCombat import neuroCombat
    x = np.asarray(x, dtype=np.float64)
    sites = np.asarray(sites, dtype=str)
    if x.ndim != 2 or len(x) != len(sites) or not np.isfinite(x).all():
        raise ValueError('Expected finite subjects x features aligned with sites')
    levels, counts = np.unique(sites, return_counts=True)
    if len(levels) < 2 or np.any(counts < 2):
        raise ValueError('ComBat requires >=2 training sites, each with >=2 subjects')
    mask = np.var(x, axis=0) > epsilon
    for site in levels: mask &= np.var(x[sites == site], axis=0) > epsilon
    if support is not None:
        support = np.asarray(support, dtype=bool)
        if support.shape != mask.shape: raise ValueError('Support shape mismatch')
        mask &= support
    if mask.sum() < 2: raise ValueError('Too few variable supported features for ComBat')
    unique, inverse = np.unique(x[:, mask].T, axis=0, return_inverse=True)
    if len(unique) < 2: raise ValueError('Too few unique feature profiles for empirical Bayes')
    result = neuroCombat(dat=unique, covars=pd.DataFrame({'site':sites}), batch_col='site')
    est = result['estimates']
    # No biological covariates: mod.mean is zero. Retain its mean for parity.
    mean = np.asarray(est['stand.mean'])[:, 0] + np.asarray(est['mod.mean']).mean(axis=1)
    variance = np.asarray(est['var.pooled']).reshape(-1)
    gamma = np.asarray(est['gamma.star']); delta = np.asarray(est['delta.star'])
    if (not all(np.isfinite(a).all() for a in (mean,variance,gamma,delta))
            or np.any(variance <= 0) or np.any(delta <= 0)):
        raise ValueError('Invalid ComBat fit; inspect sample counts and feature variance')
    state = {'feature_count':x.shape[1], 'mask':mask, 'inverse':inverse,
             'sites':np.asarray(est['batches'],dtype=str), 'mean':mean,
             'variance':variance, 'gamma':gamma, 'delta':delta,
             'fit_n':len(x), 'covariates':[], 'unique_features':len(unique)}
    adjusted, _ = apply_combat_train(x, sites, state, unseen_policy='error')
    if not np.allclose(adjusted[:, mask], result['data'].T[:, inverse], rtol=2e-5, atol=2e-5):
        raise AssertionError('Frozen-parameter transform differs from the fitted ComBat output')
    return state


def apply_combat_train(x, sites, state, unseen_policy='error'):
    """Apply frozen training parameters; NEVER fit on these inputs.
    identity explicitly returns unseen sites unchanged (NOT ComBat-harmonized).
    For seen sites this is the no-covariate neuroCombatFromTraining formula.
    """
    if unseen_policy not in ('error','identity'): raise ValueError('Invalid unseen-site policy')
    x = np.asarray(x, dtype=np.float64); sites = np.asarray(sites, dtype=str)
    if x.ndim != 2 or x.shape[1] != state['feature_count'] or len(x) != len(sites):
        raise ValueError('ComBat input shape/site mismatch')
    if not np.isfinite(x).all(): raise ValueError('Nonfinite ComBat input')
    lookup = {str(s):i for i,s in enumerate(state['sites'])}
    unseen = sorted(set(sites)-set(lookup))
    if unseen and unseen_policy == 'error':
        raise ValueError(f'No training ComBat parameters for unseen sites {unseen}. '
                         'Use identity only if explicitly accepting NO correction on these sites.')
    y = x.copy(); mask = state['mask']; inv = state['inverse']
    mean = state['mean'][inv]; sd = np.sqrt(state['variance'][inv])
    for site in sorted(set(sites) & set(lookup)):
        rows = np.flatnonzero(sites == site); k = lookup[site]
        selected = x[np.ix_(rows, np.flatnonzero(mask))]
        standardized = (selected - mean) / sd
        corrected = ((standardized - state['gamma'][k,inv]) /
                     np.sqrt(state['delta'][k,inv])) * sd + mean
        y[np.ix_(rows, np.flatnonzero(mask))] = corrected
    if not np.isfinite(y).all(): raise ValueError('Nonfinite ComBat transform')
    return y.astype(np.float32), {'unseen_sites':unseen, 'unseen_policy':unseen_policy,
                                  'n_corrected':sum(s in lookup for s in sites), 'n_total':len(x)}


def normalize_model_sample(item):
    """Same per-frame BOLD and per-volume thickness normalization as original loader.
    Input must come from CleanENIGMADataset with both normalization flags False.
    """
    def array(v):
        return v.detach().cpu().numpy() if hasattr(v,'detach') else np.asarray(v)
    raw_bold = array(item['rs_data']).astype(np.float32,copy=False)
    raw_thick = array(item['thickness']).astype(np.float32,copy=False)
    if raw_bold.ndim != 4 or raw_thick.ndim != 3: raise ValueError('Expected 4D BOLD and 3D thickness')
    if not np.isfinite(raw_bold).all() or not np.isfinite(raw_thick).all():
        raise ValueError('Nonfinite image')
    support_b = np.any(raw_bold != 0, axis=-1)
    support_t = raw_thick != 0
    bold = (raw_bold-raw_bold.mean(axis=(0,1,2),keepdims=True))/(raw_bold.std(axis=(0,1,2),keepdims=True)+1e-6)
    thick = (raw_thick-raw_thick.mean())/(raw_thick.std()+1e-6)
    return bold, thick, support_b, support_t


def fit_image_combat(raw_train_dataset, output, modalities='both-logsd'):
    """Fit once before epoch 1, using training subjects only.
    Thickness: voxel features AFTER original per-volume normalization.
    BOLD: log temporal SD per voxel AFTER original per-frame normalization.
    BOLD adjustment rescales deviations around temporal mean; it does not harmonize
    correlations, temporal structure, spectral content or connectivity.
    """
    from loguru import logger
    if modalities not in ('none','thickness','both-logsd'): raise ValueError('Invalid modalities')
    out = Path(output); out.mkdir(parents=True,exist_ok=False)
    if modalities == 'none': return None
    state = {'modalities':modalities, 'thickness':None, 'bold_logsd':None}
    thickness, logsd = [], []
    support_t = support_b = None; shape_t = shape_b = None
    frame = raw_train_dataset.frame
    for i in range(len(raw_train_dataset)):
        item = raw_train_dataset[i]
        bold, thick, sb, st = normalize_model_sample(item)
        if shape_t is None: shape_t, shape_b = thick.shape, bold.shape[:-1]
        if thick.shape != shape_t or bold.shape[:-1] != shape_b:
            raise ValueError('Images have different model-input shapes; harmonization requires aligned features')
        thickness.append(thick.ravel())
        support_t = st.ravel().copy() if support_t is None else support_t & st.ravel()
        if modalities == 'both-logsd':
            sd = bold.std(axis=-1); sb &= sd > 1e-6
            support_b = sb.ravel().copy() if support_b is None else support_b & sb.ravel()
            logsd.append(np.log(np.maximum(sd,1e-6)).ravel())
        if i % 25 == 0 or i+1 == len(raw_train_dataset):
            logger.info(f'ComBat TRAIN features: {i+1}/{len(raw_train_dataset)}')
    state['thickness'] = fit_combat_train(np.stack(thickness), frame.site, support_t)
    del thickness
    if modalities == 'both-logsd':
        state['bold_logsd'] = fit_combat_train(np.stack(logsd), frame.site, support_b)
        del logsd
    state.update(thickness_shape=shape_t,bold_shape=shape_b,
                 train_subjects=frame.participant_id.tolist(), train_sites=sorted(frame.site.unique()))
    save_combat_state(state, out/'combat_train.pkl')
    audit = {'fit_before_epoch':1, 'fit_count_per_enabled_modality':1,
             'train_subjects':state['train_subjects'], 'train_sites':state['train_sites'],
             'modalities':modalities, 'covariates':[],
             'heldout':'Test subjects receive frozen training estimates for their known site; no test fitting',
             'bold_method':'log temporal SD of normalized BOLD; rescale deviations, not connectivity harmonization',
             'thickness_method':'ComBat on normalized thickness voxel values',
             'feature_support':'Training-only intersection of nonzero source-image voxels and variable features',
             'requires_aligned_voxels':'Equal shapes do not establish spatial alignment; verify source geometry',
             'features':{k:{'total':state[k]['feature_count'],'adjusted':int(state[k]['mask'].sum()),
                            'unique':state[k]['unique_features']}
                         for k in ('thickness','bold_logsd') if state[k] is not None}}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    return state


def transform_image_sample(bold, thick, site, state, unseen_policy='error'):
    """Apply frozen model, preserving original spatial and time dimensions."""
    if state is None: return bold, thick
    if tuple(thick.shape)!=tuple(state['thickness_shape']) or tuple(bold.shape[:-1])!=tuple(state['bold_shape']):
        raise ValueError('Held-out image shape differs from training feature layout')
    if str(site) not in state['train_sites']:
        if unseen_policy == 'identity': return bold, thick
        raise ValueError(f'Unseen site {site}: no fitted ComBat correction')
    t,_ = apply_combat_train(thick.reshape(1,-1),[site],state['thickness'],unseen_policy)
    thick = t.reshape(thick.shape)
    if state['bold_logsd'] is not None:
        sd = bold.std(axis=-1)
        original = np.log(np.maximum(sd,1e-6)).reshape(1,-1)
        adjusted,_ = apply_combat_train(original,[site],state['bold_logsd'],unseen_policy)
        ratio = np.exp(adjusted.astype(np.float64)-original).reshape(sd.shape)
        if not np.isfinite(ratio).all() or ratio.max()>1e6 or ratio.min()<1e-6:
            raise ValueError('Extreme BOLD scaling; inspect training ComBat fit')
        mean = bold.mean(axis=-1,keepdims=True)
        bold = (mean + (bold-mean)*ratio[...,None]).astype(np.float32)
    if not np.isfinite(bold).all() or not np.isfinite(thick).all(): raise ValueError('Nonfinite harmonized image')
    return bold, thick


def save_combat_state(state, path):
    with open(path,'wb') as f: pickle.dump(state,f,pickle.HIGHEST_PROTOCOL)


def load_combat_state(path):
    """Only load your own trusted pickle files."""
    with open(path,'rb') as f: return pickle.load(f)


def project_training_umap(embeddings, output, seed=42):
    """Fit exactly once on train, transform val/test without fitting them.
    embeddings[split] has thick, rs, subject_id, site, in aligned subject order.
    """
    import umap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(output); out.mkdir(parents=True,exist_ok=True)
    def stacked(data):
        return np.stack([data['thick'],data['rs']],axis=1).reshape(-1,data['rs'].shape[1])
    train = stacked(embeddings['train'])
    reducer = umap.UMAP(n_neighbors=min(5,len(train)-1), min_dist=.1, spread=3.,
                        n_components=2, metric='euclidean', init='random',
                        random_state=seed,transform_seed=seed,n_jobs=1)
    xy_train = reducer.fit_transform(train)
    with open(out/'umap_train_fitted.pkl','wb') as f: pickle.dump(reducer,f)
    records = []
    for split,data in embeddings.items():
        xy = xy_train if split=='train' else reducer.transform(stacked(data))
        for i,subject in enumerate(data['subject_id']):
            for j,modality in enumerate(('thick','rs')):
                records.append({'subject_id':subject,'site':data['site'][i], 'split':split,
                                'modality':modality,'x':float(xy[2*i+j,0]),'y':float(xy[2*i+j,1])})
    frame = pd.DataFrame(records)
    frame.to_csv(out/'umap_coordinates.tsv',sep='\t',index=False)
    levels = sorted(frame.site.unique()); cmap = plt.get_cmap('turbo')
    colors = {s:cmap(i/max(len(levels)-1,1)) for i,s in enumerate(levels)}
    fig,axes = plt.subplots(1,len(embeddings),figsize=(6*len(embeddings),6),squeeze=False,sharex=True,sharey=True)
    for ax,split in zip(axes[0],embeddings):
        subset = frame[frame.split==split]
        for (site,modality),group in subset.groupby(['site','modality']):
            ax.scatter(group.x,group.y,color=colors[site],marker='^' if modality=='thick' else 'D',
                       s=18,alpha=.7,label=f'{site} / {modality}')
        ax.set(title=split,xlabel='UMAP 1',ylabel='UMAP 2')
        ax.legend(fontsize=6,loc='best')
    fig.suptitle('UMAP fitted on training subjects; test subjects transformed (sites overlap)')
    fig.tight_layout(); fig.savefig(out/'umap_train_test.png',dpi=180); plt.close(fig)
    return frame
