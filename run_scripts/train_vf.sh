#!/bin/bash

# Script to run value function training followed by RL-SMPC

# Common arguments
PROJECT="SMPC"
ENV_ID="LettuceGreenhouse" 
ALGORITHM="sac"
MODE="stochastic"
UNCERTAINTY_VALUE=0.1
MODEL_NAME="brisk-resonance-24"

# First run value function training
echo "Training value function..."
python RL/vf_TR_learning.py \
    --project $PROJECT \
    --model_name $MODEL_NAME \
    --env_id $ENV_ID \
    --algorithm $ALGORITHM \
    --uncertainty_value $UNCERTAINTY_VALUE \
    --mode $MODE

# Evaluate RL agent
echo "Evaluate RL agent..."
python RL/evaluate_rl.py \
    --project $PROJECT \
    --model_name $MODEL_NAME \
    --env_id $ENV_ID \
    --algorithm $ALGORITHM \
    --mode $MODE \
    --uncertainty_value $UNCERTAINTY_VALUE

echo "Done!"
