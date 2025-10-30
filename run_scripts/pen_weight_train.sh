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
MODEL_SEED=42
uncertainty_value=0.1
penalty_weight_factors=(0.1 0.316 3.16 10)

echo "Training with uncertainty value $uncertainty_value..."
for penalty_weight_factor in "${penalty_weight_factors[@]}"; do
    python experiments/train_rl.py \
        --project $PROJECT \
        --env_id $ENV_ID \
        --algorithm $ALGORITHM \
        --n_eval_episodes $N_EVAL_EPISODES \
        --n_evals $N_EVALS \
        --mode $MODE \
        --uncertainty_value $uncertainty_value \
        --model_seed $MODEL_SEED \
        --penalty_weight_factor $penalty_weight_factor \
        --save_model \
        --save_env
done
