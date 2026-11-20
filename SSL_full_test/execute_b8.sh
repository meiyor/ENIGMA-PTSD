#!/bin/bash
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=40G
#SBATCH --account=aip-irina
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j_8.out
#SBATCH --error=slurm-%x-%j_8.err

module load python/3.12.4
module load igraph/1.0.1
module load opencv/4.14.0

source /home/jmm1987/JMM_ENV/bin/activate

cd /scratch/jmm1987/ENIGMA_PTSD_code/SSL_full_test/


python train_test_cort_rest_regularized_cross_val_combat_age_sex_umap_epochs_model_select_duration_filter_with_metrics.py  --data-root /scratch/jmm1987/ENIGMA_PTSD_Clean/     --output-root /scratch/jmm1987/ENIGMA_PTSD_ssl_def_B8_DEF_EXP/     --learning-rate 1e-4  --rs-model independent    --rs-crop-seconds 200     --pool attention   --ssl-exponential-p 2.0     --bold-variant small4  --min-lr 1e-6 --ssl-anneal-epochs 500   --n-splits 5     --fold 1     --batch-size 8  --leiden-resolution 2.0   --eval-batch-size 8     --out-dim 128     --attention-heads 4     --missing-covariates drop    --st-stride 8 --max-ssl-epochs 1000 --gw-epochs 300
