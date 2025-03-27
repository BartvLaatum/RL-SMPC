#!/bin/bash
export PYTHONPATH=$(pwd)
echo "PYTHONPATH set to: $PYTHONPATH"
# Script to train RL agents with different uncertainty values

# Common arguments
PROJECT="uncertainty-comparison"
ENV_ID="LettuceGreenhouse"
ALGORITHM="sac"
N_EVAL_EPISODES=10
N_EVALS=10
MODE="stochastic"

# Train with uncertainty value 0.05
echo "Training with uncertainty value 0.05..."
python experiments/train_rl.py \
    --project $PROJECT \
    --env_id $ENV_ID \
    --algorithm $ALGORITHM \
    --n_eval_episodes $N_EVAL_EPISODES \
    --n_evals $N_EVALS \
    --mode $MODE \
    --uncertainty_value 0.05 \
    --save_model \
    --save_env

# Train with uncertainty value 0.15
echo "Training with uncertainty value 0.15..."
python experiments/train_rl.py \
    --project $PROJECT \
    --env_id $ENV_ID \
    --n_eval_episodes $N_EVAL_EPISODES \
    --n_evals $N_EVALS \
    --algorithm $ALGORITHM \
    --mode $MODE \
    --uncertainty_value 0.15 \
    --save_model \
    --save_env

# Train with uncertainty value 0.2
echo "Training with uncertainty value 0.2..."
python experiments/train_rl.py \
    --project $PROJECT \
    --env_id $ENV_ID \
    --n_eval_episodes $N_EVAL_EPISODES \
    --n_evals $N_EVALS \
    --algorithm $ALGORITHM \
    --mode $MODE \
    --uncertainty_value 0.2 \
    --save_model \
    --save_env