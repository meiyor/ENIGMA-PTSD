"""Official neuroHarmonize ComBat + sklearn age/sex regression, TRAIN fitted.
No ComBat estimation or adjustment equations are implemented in this file.
The only image operations are feature extraction and applying saved outputs.
"""
from pathlib import Path
import json
import pickle
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


def attach_age_sex_metadata(frame, metadata_path, age_column='Age', sex_column='Sex',
                            missing='error'):
    """One-to-one ID join; no Excel rematching, guessed ages or numeric sex mapping.
    Accepted sex labels M/F, MALE/FEMALE. Other codes require verified recoding in
    your metadata. Default error; explicit drop returns an exclusion report.
    """
    if missing not in ('error','drop'): raise ValueError('missing must be error or drop')
    meta=pd.read_csv(metadata_path,sep='\t',dtype=str,keep_default_na=False)
    required={'participant_id',age_column,sex_column}
    if not required <= set(meta): raise ValueError(f'Metadata needs columns {sorted(required)}')
    if meta.participant_id.duplicated().any(): raise ValueError('Duplicate metadata participant_id')
    if frame.participant_id.duplicated().any(): raise ValueError('Duplicate manifest participant_id')
    lookup=meta.set_index('participant_id').reindex(frame.participant_id.astype(str))
    age=pd.to_numeric(lookup[age_column],errors='coerce').to_numpy(dtype=float)
    labels=lookup[sex_column].fillna('').astype(str).str.strip().str.upper()
    sex=labels.map({'M':1.,'F':0.,'MALE':1.,'FEMALE':0.}).to_numpy(dtype=float)
    valid_age=np.isfinite(age)&(age>0)&(age<=120)
    valid_sex=np.isfinite(sex)
    good=valid_age&valid_sex
    reasons=[]
    for i in np.flatnonzero(~good):
        problems=[]
        if not valid_age[i]: problems.append('missing/non-numeric/out-of-range age')
        if not valid_sex[i]: problems.append(f'unrecognized/missing sex code: {labels.iloc[i]!r}')
        reasons.append({'participant_id':frame.iloc[i].participant_id,'site':frame.iloc[i].site,
                        'reason':'; '.join(problems)})
    exclusions=pd.DataFrame(reasons,columns=['participant_id','site','reason'])
    if len(exclusions) and missing=='error':
        raise ValueError(f'{len(exclusions)} subjects lack usable Age/Sex metadata. '
                         f'Examples: {exclusions.head(5).to_dict("records")}. '
                         'Correct metadata, or explicitly use --missing-covariates drop. '
                         'No values were imputed or guessed.')
    result=frame.copy();result['AGE']=age;result['SEX_M']=sex
    return result.loc[good].reset_index(drop=True),exclusions


def covariates_for_frame(frame):
    """Official API expects SITE and numerical covariates in identical row order."""
    return pd.DataFrame({'SITE':frame.site.astype(str).to_numpy(),
                         'AGE':frame.AGE.to_numpy(dtype=float),
                         'SEX_M':frame.SEX_M.to_numpy(dtype=float)})


def fit_transform_combat_age_sex(train_x,test_x,train_covars,test_covars,
                                support=None,epsilon=1e-10,seed=42):
    """Call official package APIs, then remove linear age/sex effects via sklearn.
    Training and test feature matrices: subjects x corresponding aligned features.
    Sites must overlap, subjects must not. This function never fits on test rows.
    Covariate effects are protected during ComBat and then explicitly regressed out.
    """
    from neuroHarmonize import harmonizationLearn,harmonizationApply
    from sklearn.linear_model import LinearRegression
    from importlib.metadata import version
    a=np.asarray(train_x,dtype=np.float64);b=np.asarray(test_x,dtype=np.float64)
    tc=train_covars.reset_index(drop=True).copy();vc=test_covars.reset_index(drop=True).copy()
    required=['SITE','AGE','SEX_M']
    if list(tc.columns)!=required or list(vc.columns)!=required:
        raise ValueError(f'Covariate columns must be in order {required}')
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1] or len(a)!=len(tc) or len(b)!=len(vc):
        raise ValueError('Feature/covariate shape mismatch')
    if not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('Nonfinite features')
    for cov in (tc,vc):
        if not np.isfinite(cov[['AGE','SEX_M']].to_numpy(dtype=float)).all(): raise ValueError('Missing/nonfinite Age/Sex')
        if not set(cov.SEX_M)<= {0,1}: raise ValueError('SEX_M must be coded 0 or 1')
    levels,counts=np.unique(tc.SITE.astype(str),return_counts=True)
    # Full-cohort apply is intentional: the package must receive all trained sites,
    # rather than a one-site minibatch with an incomplete design matrix.
    if set(vc.SITE.astype(str)) != set(levels):
        raise ValueError('This train/test helper requires every training site in the test cohort')
    if len(levels)<2 or np.any(counts<2): raise ValueError('Need >=2 training subjects/site and >=2 sites')
    design=np.column_stack([pd.get_dummies(tc.SITE,dtype=float).to_numpy(),tc[['AGE','SEX_M']].to_numpy()])
    if np.linalg.matrix_rank(design)<design.shape[1] or len(a)<=design.shape[1]:
        raise ValueError('Site/Age/Sex design is rank deficient or has no residual degrees of freedom. '
                         'Age/sex effects cannot be separated reliably; inspect this training fold.')
    mask=np.var(a,axis=0)>epsilon
    for site in levels:mask &= np.var(a[tc.SITE.astype(str)==site],axis=0)>epsilon
    if support is not None:
        support=np.asarray(support,dtype=bool)
        if support.shape!=mask.shape:raise ValueError('Training support shape mismatch')
        mask &= support
    # Also reject features with zero residual variance after the full covariate design.
    residual=a[:,mask]-design@np.linalg.lstsq(design,a[:,mask],rcond=None)[0]
    selected=np.flatnonzero(mask)
    mask[selected[np.var(residual,axis=0)<=epsilon]]=False
    if mask.sum()<2:raise ValueError('Too few supported, variable features for empirical-Bayes ComBat')

    # Official repository implementation: fit ONCE on training subjects.
    combat_model,train_combat=harmonizationLearn(a[:,mask],tc,eb=True,seed=seed)
    # Apply the SAME package model to test subjects: no learning/refitting here.
    test_combat=harmonizationApply(b[:,mask],vc,combat_model)
    if not np.isfinite(train_combat).all() or not np.isfinite(test_combat).all():
        raise ValueError('Official ComBat returned nonfinite values; inspect training design/features')

    # ComBat intentionally preserves Age/Sex. Remove those linear effects explicitly
    # using sklearn; retain the training mean instead of mapping everyone to age zero.
    nuisance_mean=tc[['AGE','SEX_M']].to_numpy(dtype=float).mean(axis=0)
    ctrain=tc[['AGE','SEX_M']].to_numpy(dtype=float)-nuisance_mean
    ctest=vc[['AGE','SEX_M']].to_numpy(dtype=float)-nuisance_mean
    regressor=LinearRegression().fit(ctrain,train_combat)
    train_clean=train_combat-(regressor.predict(ctrain)-regressor.intercept_)
    test_clean=test_combat-(regressor.predict(ctest)-regressor.intercept_)
    train_out=a.copy();test_out=b.copy()
    train_out[:,mask]=train_clean;test_out[:,mask]=test_clean
    state={'format':'official_neuroHarmonize_plus_sklearn_age_sex_v1',
           'combat_model':combat_model,'age_sex_regressor':regressor,
           'nuisance_mean':nuisance_mean,'covariate_columns':required,'feature_mask':mask,
           'train_sites':levels.tolist(),'feature_count':a.shape[1],'train_n':len(a),
           'age_training_range':[float(tc.AGE.min()),float(tc.AGE.max())],
           'packages':{p:version(p) for p in ('neuroHarmonize','neuroCombat','scikit-learn')}}
    if not np.isfinite(train_out).all() or not np.isfinite(test_out).all():raise ValueError('Nonfinite residualized output')
    return train_out.astype(np.float32),test_out.astype(np.float32),state


def apply_saved_combat_age_sex(x,covars,state):
    """Reuse only official ComBat apply plus fitted sklearn predict; no refitting.
    Supply an entire cohort containing all training sites, in aligned row order.
    """
    from neuroHarmonize import harmonizationApply
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=state['feature_count'] or len(x)!=len(covars):raise ValueError('Shape mismatch')
    if list(covars.columns)!=state['covariate_columns']:raise ValueError('Covariate columns/order mismatch')
    if set(covars.SITE.astype(str))!=set(state['train_sites']):raise ValueError('Supply all known sites in this cohort')
    if not np.isfinite(x).all() or not np.isfinite(covars[['AGE','SEX_M']].to_numpy(dtype=float)).all():raise ValueError('Nonfinite input')
    if not set(covars.SEX_M)<= {0,1}:raise ValueError('SEX_M must be 0/1')
    mask=state['feature_mask']
    adjusted=harmonizationApply(x[:,mask],covars,state['combat_model'])
    regressor=state['age_sex_regressor']
    c=covars[['AGE','SEX_M']].to_numpy(dtype=float)-state['nuisance_mean']
    output=x.copy();output[:,mask]=adjusted-(regressor.predict(c)-regressor.intercept_)
    if not np.isfinite(output).all():raise ValueError('Nonfinite transform')
    return output.astype(np.float32)


def prepare_fold_combat_age_sex(raw_train,raw_test,output,modalities='both-logsd',seed=42):
    """Extract features once/split, fit TRAIN only, apply TEST once, cache small maps.
    BOLD is still log-SD amplitude harmonization, not full time-series harmonization.
    Caches avoid calling the package on single-site image minibatches every epoch.
    """
    from loguru import logger
    if modalities not in ('none','thickness','both-logsd'):raise ValueError('Invalid modalities')
    if modalities=='none':return None
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    if set(raw_train.frame.participant_id)&set(raw_test.frame.participant_id):raise ValueError('Train/test subjects overlap')
    tc=covariates_for_frame(raw_train.frame);vc=covariates_for_frame(raw_test.frame)
    if vc.AGE.min()<tc.AGE.min() or vc.AGE.max()>tc.AGE.max():
        logger.warning('Some test ages lie outside the training age range; applying the frozen linear model by extrapolation.')
    matrices={};supports={};shapes={}
    for split,ds in [('train',raw_train),('test',raw_test)]:
        thickness=[];logsd=[];support_t=support_b=None
        for i in range(len(ds)):
            bold,thick,sb,st=normalize_model_sample(ds[i])
            if not shapes:shapes={'thickness':thick.shape,'bold':bold.shape[:-1]}
            if thick.shape!=shapes['thickness'] or bold.shape[:-1]!=shapes['bold']:raise ValueError('Image shape/feature layout mismatch')
            thickness.append(thick.ravel())
            if split=='train':support_t=st.ravel().copy() if support_t is None else support_t&st.ravel()
            if modalities=='both-logsd':
                sd=bold.std(axis=-1);sb &= sd>1e-6
                logsd.append(np.log(np.maximum(sd,1e-6)).ravel())
                if split=='train':support_b=sb.ravel().copy() if support_b is None else support_b&sb.ravel()
            if i%25==0 or i+1==len(ds):logger.info(f'Extracting {split} features: {i+1}/{len(ds)}; fit uses TRAIN only')
        matrices[split]={'thickness':np.stack(thickness)}
        if modalities=='both-logsd':matrices[split]['bold_logsd']=np.stack(logsd)
        if split=='train':supports={'thickness':support_t,'bold_logsd':support_b}
    paths={};audit={'implementation':'neuroHarmonize.harmonizationLearn / harmonizationApply',
        'age_sex_removal':'sklearn LinearRegression; fitted on harmonized TRAIN, applied unchanged to TEST',
        'train_subjects':raw_train.frame.participant_id.tolist(),'test_subjects':raw_test.frame.participant_id.tolist(),
        'modalities':modalities,'fit_count_per_modality':1,'reference':'training mean age and sex proportions',
        'sex_coding':{'F':0,'M':1},'features':{}}
    maps=out/'maps';maps.mkdir()
    for label in matrices['train']:
        train_clean,test_clean,state=fit_transform_combat_age_sex(
            matrices['train'][label],matrices['test'][label],tc,vc,support=supports[label],seed=seed)
        with open(out/f'{label}_model.pkl','wb') as f:pickle.dump(state,f,pickle.HIGHEST_PROTOCOL)
        audit['features'][label]={'total':state['feature_count'],'adjusted':int(state['feature_mask'].sum()),'packages':state['packages']}
        for split,ds,clean in [('train',raw_train,train_clean),('test',raw_test,test_clean)]:
            if label=='bold_logsd':
                clean=np.exp(clean.astype(np.float64)-matrices[split][label])
                if not np.isfinite(clean).all() or clean.max()>1e6 or clean.min()<1e-6:raise ValueError('Extreme BOLD scaling')
            shape=shapes['thickness'] if label=='thickness' else shapes['bold']
            for i,pid in enumerate(ds.frame.participant_id):
                path=maps/f'{split}_{i:06d}_{label}.npy'
                np.save(path,clean[i].reshape(shape).astype(np.float32),allow_pickle=False)
                paths.setdefault(str(pid),{})[label]=str(path.resolve())
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    (out/'transforms.json').write_text(json.dumps(paths,indent=2))
    return paths


def apply_cached_combat_age_sex(bold,thick,participant_id,transforms):
    """Read the already transformed maps; no fitting or parameter estimation."""
    if transforms is None:return bold,thick
    entry=transforms[str(participant_id)]
    corrected=np.load(entry['thickness'],allow_pickle=False)
    if corrected.shape!=thick.shape:raise ValueError('Cached thickness shape mismatch')
    if 'bold_logsd' in entry:
        scale=np.load(entry['bold_logsd'],allow_pickle=False)
        if scale.shape!=bold.shape[:-1]:raise ValueError('Cached BOLD scale shape mismatch')
        mean=bold.mean(axis=-1,keepdims=True)
        bold=(mean+(bold-mean)*scale[...,None]).astype(np.float32)
    if not np.isfinite(bold).all() or not np.isfinite(corrected).all():raise ValueError('Nonfinite cached inputs')
    return bold,corrected
