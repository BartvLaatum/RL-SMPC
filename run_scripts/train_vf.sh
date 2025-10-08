#!/bin/bash

# Script to run value function training followed by RL-SMPC

# Common arguments
PROJECT="SMPC"
ENV_ID="LettuceGreenhouse" 
ALGORITHM="sac"
MODE="stochastic"
UNCERTAINTY_VALUE=0.1
# MODEL_NAMEDS=("brisk-resonance-24")
MODEL_NAMES=("iconic-dust-9" "eager-bee-10" "logical-forest-11" "swift-armadillo-12")

# First run value function training
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "Training value function, for model: $MODEL_NAME..."
    python RL/vf_TR_learning.py \
        --project $PROJECT \
        --model_name $MODEL_NAME \
        --env_id $ENV_ID \
        --algorithm $ALGORITHM \
        --uncertainty_value $UNCERTAINTY_VALUE \
        --mode $MODE

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