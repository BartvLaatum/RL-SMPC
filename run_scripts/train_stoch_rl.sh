#!/bin/bash
export PYTHONPATH=$(pwd)
echo "PYTHONPATH set to: $PYTHONPATH"
# Script to train RL agents with different uncertainty values

# Common arguments
PROJECT="SMPC"
ENV_ID="LettuceGreenhouse"
ALGORITHM="sac"
N_EVAL_EPISODES=20
N_EVALS=20
MODE="stochastic"
uncertainty_values=(0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2)

for uncertainty_value in "${uncertainty_values[@]}"; do
echo "Training with uncertainty value $uncertainty_value..."
python experiments/train_rl.py \
    --project $PROJECT \
    --env_id $ENV_ID \
    --algorithm $ALGORITHM \
    --n_eval_episodes $N_EVAL_EPISODES \
    --n_evals $N_EVALS \
    --mode $MODE \
    --uncertainty_value $uncertainty_value \
    --save_model \
    --save_env
done
