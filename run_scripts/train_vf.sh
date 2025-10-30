#!/bin/bash
export PYTHONPATH=$(pwd)
# Script to run value function training followed by RL-SMPC

# Common arguments
PROJECT="SMPC"
ENV_ID="LettuceGreenhouse" 
ALGORITHM="sac"
MODE="stochastic"
UNCERTAINTY_VALUE=0.1
# MODEL_NAMES=("iconic-dust-9" "eager-bee-10" "logical-forest-11" "swift-armadillo-12")
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
        --use_trained_vf \
        --terminal \
        --rl_feedback

    # Run SMPC for horizons 1H-8H
    echo "Running SMPC..."
    python experiments/horizon_smpc.py \
        --project $PROJECT \
        --env_id $ENV_ID \
        --save_name no-tightening \
        --mode $MODE \
        --uncertainty_value $UNCERTAINTY_VALUE \
done
