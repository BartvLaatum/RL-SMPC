#!/bin/bash

# Script to run value function training followed by RL-MPC

# Common arguments
PROJECT="matching-thesis"
ENV_ID="LettuceGreenhouse" 
ALGORITHM="sac"
MODEL_NAME="expert-disco-95"
MODE="stochastic"
UNCERTAINTY_VALUE=0.05

# # First run value function training
# echo "Training value function..."
# python RL/vf_TR_learning.py \
#     --project $PROJECT \
#     --model_name $MODEL_NAME \
#     --env_id $ENV_ID \
#     --algorithm $ALGORITHM \
#     --uncertainty_value $UNCERTAINTY_VALUE \
#     --mode $MODE

# Evaluate RL agent
echo "Evaluate RL agent..."
python RL/evaluate_rl.py \
    --project $PROJECT \
    --model_name $MODEL_NAME \
    --env_id $ENV_ID \
    --algorithm $ALGORITHM \
    --mode $MODE \
    --uncertainty_value $UNCERTAINTY_VALUE

# Second RL-MPC for horizons 1H-6H
echo "Running RL-MPC..."
python experiments/horizon_rlmpc.py \
    --project $PROJECT \
    --env_id $ENV_ID \
    --model_name $MODEL_NAME \
    --algorithm $ALGORITHM \
    --save_name rlmpc-bounded-states \
    --mode $MODE \
    --uncertainty_value $UNCERTAINTY_VALUE \
    --use_trained_vf


# # Then run RL-MPC
# echo "Running RL-MPC for 1H-6H prediction horizons..."
# python experiments/horizon_.py \
#     --project $PROJECT \
#     --env_id $ENV_ID \
#     --model_name $MODEL_NAME \
#     --algorithm $ALGORITHM \
#     --mode $MODE \
#     --uncertainty_scale $UNCERTAINTY_VALUE \
#     --weather_filename "outdoorWeatherWurGlas2014.csv" \
#     --use_trained_vf \

echo "Done!"
