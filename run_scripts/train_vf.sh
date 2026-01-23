#!/bin/bash
export PYTHONPATH=$(pwd)
# Script to run value function training for specified trained RL models

# Common arguments
PROJECT="SMPC"
ENV_ID="LettuceGreenhouse" 
ALGORITHM="sac"
MODE="stochastic"
UNCERTAINTY_VALUE=0.1
MODEL_NAMES=("brisk-resonance-24")

# First run value function training
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "Training value function, for model: $MODEL_NAME..."
    python RL/vf_TR_learning.py \
        --project $PROJECT \
        --env_id $ENV_ID \
        --model_name $MODEL_NAME \
        --algorithm $ALGORITHM \
        --save_name no-tightening \
        --mode $MODE \
        --uncertainty_value $UNCERTAINTY_VALUE \

    # Evaluate RL agent
    echo "Evaluating RL agent with named: $MODEL_NAME..."
    python RL/evaluate_rl.py \
        --project $PROJECT \
        --model_name $MODEL_NAME \
        --env_id $ENV_ID \
        --algorithm $ALGORITHM \
        --mode $MODE \
        --uncertainty_value $UNCERTAINTY_VALUE

done
