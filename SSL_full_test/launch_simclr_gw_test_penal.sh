#!/bin/bash
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=40G
#SBATCH --account=aip-irina
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

module load python/3.12.4
module load igraph/1.0.1
module load opencv/4.14.0

source /home/jmm1987/JMM_ENV/bin/activate

cd /scratch/jmm1987/ENIGMA_PTSD_code/SSL_full_test/

python train_test_cort_rest_regularized_cross_val_combat_age_sex_umap_epochs_model_select.py \
    --data-root /scratch/jmm1987/ENIGMA_PTSD_Clean/ \
    --output-root /scratch/jmm1987/ENIGMA_Sequential_SSL_GW_b10_penal/ \
    --learning-rate 1e-4 \
    --rs-model independent \
    --max-ssl-epochs 500 \
    --rs-crop-seconds 70 \
    --pool attention \
    --bold-variant small4 \
    --n-splits 5 \
    --fold 1 \
    --batch-size 10 \
    --eval-batch-size 10 \
    --out-dim 128 \
    --attention-heads 4 \
    --missing-covariates drop \
    --st-stride 8
