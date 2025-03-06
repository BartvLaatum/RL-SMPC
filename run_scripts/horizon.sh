#!/bin/bash

# Script to run MPC and RL-MPC for several optimization horizons

# Common arguments
PROJECT="matching-salim"
ENV_ID="LettuceGreenhouse" 
ALGORITHM="sac"
MODE="stochastic"
UNCERTAINTY_VALUE=0.1
MODEL_NAME="likely-frost-1"

# # Evaluate RL agent
# echo "Evaluate RL agent..."
# python RL/evaluate_rl.py \
#     --project $PROJECT \
#     --model_name $MODEL_NAME \
#     --env_id $ENV_ID \
#     --algorithm $ALGORITHM \
#     --mode $MODE \
#     --uncertainty_value $UNCERTAINTY_VALUE

# # First run value function training
# echo "Training value function..."
# python RL/vf_TR_learning.py \
#     --project $PROJECT \
#     --model_name $MODEL_NAME \
#     --env_id $ENV_ID \
#     --algorithm $ALGORITHM \
#     --uncertainty_value $UNCERTAINTY_VALUE \
#     --mode $MODE


# # Run MPC for horizons 1H-6H
# echo "Running MPC..."
# python experiments/horizon_mpc.py \
#     --project $PROJECT \
#     --env_id $ENV_ID \
#     --save_name mpc \
#     --mode $MODE \
#     --uncertainty_value $UNCERTAINTY_VALUE

# Second RL-MPC for horizons 1H-6H
echo "Running RL-MPC..."
python experiments/horizon_rlmpc.py \
    --project $PROJECT \
    --env_id $ENV_ID \
    --model_name $MODEL_NAME \
    --algorithm $ALGORITHM \
    --save_name rlmpc \
    --mode $MODE \
    --uncertainty_value $UNCERTAINTY_VALUE \
    --use_trained_vf
