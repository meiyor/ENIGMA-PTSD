"""Read exported BOLD/thickness only. No source paths, Excel, fALFF or NPZ caches."""
from pathlib import Path
import json
from loguru import logger
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader


def time_indices(nframes,tr,target_tr,window,crop,policy='legacy'):
    if not np.isfinite([tr,target_tr,window]).all() or min(tr,target_tr,window)<=0 or crop<1:
        raise ValueError('Invalid temporal parameters')
    if nframes*tr<=200:
        raise ValueError('Original loader excludes scans of duration <=200 seconds')
    fs_current=1/tr; fs_new=1/target_tr
    n=min(int(window*fs_current),nframes)
    if n<crop: raise ValueError('Too few source indices for requested crop')
    # Same nearest-index calculation and clipping as the attached code.
    idx=np.round(np.arange(crop)*fs_current/fs_new).astype(int)
    if policy=='strict' and (np.any(idx>=n) or len(np.unique(idx))!=len(idx)):
        raise ValueError('Target grid repeats or exceeds source frames; lower crop or adjust window')
    return np.clip(idx,0,n-1)


class CleanENIGMADataset(Dataset):
    def __init__(self,root,rs_time_window=200,rs_window_crop=56,rs_stride=4,st_stride=4,
                 normalize_bold=True,normalize_thickness=True,temporal_policy='legacy',
                 sites=None,participant_ids=None,bold_variant='small',validate_headers=False):
        self.root=Path(root).resolve()
        self.frame=pd.read_csv(self.root/'manifest.tsv',sep='\t',dtype=str,keep_default_na=False)
        available=sorted(self.frame.bold_variant.unique())
        self.frame=self.frame[self.frame.bold_variant==bold_variant]
        if self.frame.empty: raise ValueError(f'Variant {bold_variant} unavailable; choose from {available}')
        if self.frame.participant_id.duplicated().any(): raise ValueError('Duplicate subjects in selected variant')
        self.target_tr=float(json.loads((self.root/'dataset_config.json').read_text())['target_tr'])
        if temporal_policy not in ('legacy','strict'): raise ValueError('policy must be legacy or strict')
        if min(rs_stride,st_stride)<1: raise ValueError('Spatial strides must be positive')
        if sites is not None: self.frame=self.frame[self.frame.site.isin(sites)]
        if participant_ids is not None:
            missing=set(participant_ids)-set(self.frame.participant_id)
            if missing: raise ValueError(f'Requested IDs absent after site filter: {sorted(missing)}')
            self.frame=self.frame[self.frame.participant_id.isin(participant_ids)]
        self.rs_time_window=rs_time_window; self.rs_window_crop=rs_window_crop
        self.rs_stride=rs_stride; self.st_stride=st_stride
        self.normalize_bold=normalize_bold; self.normalize_thickness=normalize_thickness
        self.temporal_policy=temporal_policy
        good=[]; excluded=[]
        for row in self.frame.itertuples():
            try:
                n_frames=getattr(row,'n_frames','')
                if not n_frames and validate_headers:
                    logger.info(f'Checking temporal eligibility: {row.participant_id}')
                    image=nib.load(self.root/row.bold_path,mmap=True)
                    if len(image.shape)!=4: raise ValueError('BOLD is not 4D')
                    n_frames=image.shape[3]
                    self.frame.loc[row.Index,'n_frames']=str(n_frames)
                if n_frames:
                    time_indices(int(n_frames),float(row.tr_seconds),self.target_tr,rs_time_window,rs_window_crop,temporal_policy)
                if not (self.root/row.bold_path).is_file() or not (self.root/row.thickness_path).is_file():
                    raise ValueError('Exported modality missing')
                good.append(row.Index)
            except ValueError as e: excluded.append({'participant_id':row.participant_id,'reason':str(e)})
        self.exclusions=pd.DataFrame(excluded,columns=['participant_id','reason'])
        self.frame=self.frame.loc[good].reset_index(drop=True)
        if self.frame.empty: raise ValueError('No eligible samples for requested temporal configuration')
        self.sites_all=self.frame.site.to_numpy()
        if not validate_headers:
            logger.info('Lazy loader: temporal eligibility for unknown lengths is checked on first access, not during export.')

    def __len__(self): return len(self.frame)

    def __getitem__(self,index):
        row=self.frame.iloc[index]; tr=float(row.tr_seconds)
        img=nib.load(self.root/row.bold_path,mmap=True)
        idx=time_indices(img.shape[-1],tr,self.target_tr,self.rs_time_window,self.rs_window_crop,self.temporal_policy)
        # Single contiguous read avoids decompressing a .nii.gz once per frame.
        bold=np.asarray(img.dataobj[::self.rs_stride,::self.rs_stride,::self.rs_stride,:int(idx.max())+1],dtype=np.float32)[...,idx].copy()
        thickness=np.array(np.load(self.root/row.thickness_path,mmap_mode='r',allow_pickle=False)[::self.st_stride,::self.st_stride,::self.st_stride],dtype=np.float32,copy=True)
        if not np.isfinite(bold).all() or not np.isfinite(thickness).all():
            raise ValueError(f'Nonfinite data for {row.participant_id}')
        if self.normalize_bold:
            bold=(bold-bold.mean(axis=(0,1,2),keepdims=True))/(bold.std(axis=(0,1,2),keepdims=True)+1e-6)
        if self.normalize_thickness:
            thickness=(thickness-thickness.mean())/(thickness.std()+1e-6)
        # return in order the batch index., the torch tensor with the BOLD information from 4D resting state sample,
        # the numpy 3D tesnros with the thickness of the same subject, subject_id of the corresponding subjects, the corresponding site of this subject
        # the time_indeces of the 4D RSData sample, the TR in seconds of the 4D RSData image, and the duration of the RSData 4D data
        return {'index':index,'rs_data':torch.from_numpy(bold),'thickness':torch.from_numpy(thickness),
                'subject_id':row.participant_id,'site':row.site,'time_indices':torch.from_numpy(idx),
                'tr_seconds':tr,'target_tr_seconds':self.target_tr,'duration_seconds':img.shape[-1]*tr}


def define_dataset_dataloader_ENIGMA(root,batch_size=10,shuffling=True,num_workers=0,**kwargs):
    dataset=CleanENIGMADataset(root,**kwargs)
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffling,num_workers=num_workers)
