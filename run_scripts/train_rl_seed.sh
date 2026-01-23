#!/bin/bash
export PYTHONPATH=$(pwd)
echo "PYTHONPATH set to: $PYTHONPATH"
# Script to train RL agents with different uncertainty values

# Common arguments
PROJECT="SMPC"
ENV_ID="LettuceGreenhouse"
ALGORITHM="sac"
N_EVAL_EPISODES=20
N_EVALS=10
MODE="stochastic"
uncertainty_value=0.1
seeds=(43 44 45 46)

echo "Training with uncertainty value $uncertainty_value..."
for model_seed in "${seeds[@]}"; do
    python experiments/train_rl.py \
        --project $PROJECT \
        --env_id $ENV_ID \
        --algorithm $ALGORITHM \
        --n_eval_episodes $N_EVAL_EPISODES \
        --n_evals $N_EVALS \
        --mode $MODE \
        --uncertainty_value $uncertainty_value \
        --model_seed $model_seed \
        --save_model \
        --save_env
done
