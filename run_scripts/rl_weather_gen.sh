#!/bin/bash
export PYTHONPATH=$(pwd)
echo "PYTHONPATH set to: $PYTHONPATH"
# Script to train RL agents with different uncertainty values

# Common arguments
PROJECT="Weather-Generalization"
ENV_ID="LettuceGreenhouse"
ALGORITHM="sac"
N_EVAL_EPISODES=5
N_EVALS=20
MODE="stochastic"
uncertainty_values=(0.1)
training_years=(2001 2002 2003 2004 2005 2006 2007 2008 2009 2010)

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
        --training_years ${training_years[@]} \
        --save_env
        # --hyperparameter_tuning \
done
