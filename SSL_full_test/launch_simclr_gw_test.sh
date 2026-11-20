#!/bin/bash
#SBATCH --partition=gpubase_h100_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=40G
#SBATCH --account=aip-irina
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

module load python/3.12.4
module load igraph/1.0.1
module load opencv/4.14.0

source /home/jmm1987/JMM_env/bin/activate

cd /scratch/jmm1987/ENIGMA_PTSD_code/SSL_full_test/

python train_test_cort_rest_regularized_cross_val_combat_age_sex_umap_epochs.py \
    --data-root /scratch/jmm1987/ENIGMA_PTSD_Clean/ \
    --output-root ./scratch/jmm1987/ENIGMA_Sequential_SSL_GW/ \
    --bold-variant small8 \
    --n-splits 5 \
    --fold 1 \
    --batch-size 4 \
    --eval-batch-size 2 \
    --out-dim 128 \
    --pool attention \
    --attention-heads 4 \
    --missing-covariates drop
    --tcn-kernel 5 \
    --tcn-blocks 3 \
    --tcn-channels 100 \
    --tcn-dilation 2 \
    --frame-chunk 8 \
    --temperature 0.1 \
    --learning-rate 0.0001 \
    --max-ssl-epochs 100 \
    --min-ssl-epochs 20 \
    --cluster-check-every 5 \
    --cluster-stability-threshold 0.90 \
    --cluster-seed-threshold 0.90 \
    --stability-patience 3 \
    --leiden-k 15 \
    --leiden-resolution 1.0 \
    --min-community-size 5 \
    --min-community-coverage 0.70 \
    --gw-epochs 15 \
    --gw-batch-size 16 \
    --gw-learning-rate 0.00002 \
    --gw-weight 0.1 \
    --gw-epsilon 0.1 \
    --anchor-weight 1.0 \
    --nuisance-weight 0.1 \
    --combat both-logsd \
    --checkpoint-every 5 \
    --seed 42
