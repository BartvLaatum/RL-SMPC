#!/bin/bash
#SBATCH --job-name="test_smpc"
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1G
#SBATCH --account=research-ME-dcsc
#SBATCH --output=/scratch/smsaad/RL-MPC-lettuce/delftblue_testing/slurm_outputs/smpc-%j.out

module load 2024r1
module load miniconda3

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate "/scratch/smsaad/.conda/envs/srlmpc_lettuce/"

srun python smpc.py \
    --project "smpc" \
    --env_id "LettuceGreenhouse" \
    --save_name "testing" \
    --uncertainty_value 0.1

start_time=$(date +%s)

python visualisations/smpc_performance.py \
    --project "smpc" \
    --model_names "logical-disco-90" \
    --Ns 1 \
    --mode stochastic \
    --uncertainty_value 0.1 \
    --figure_name smpc

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Script runtime: $runtime seconds"

# Deactivate conda
conda deactivate
