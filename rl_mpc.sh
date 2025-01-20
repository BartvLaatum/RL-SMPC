#!/bin/bash

# Script to run value function training followed by RL-MPC

# Common arguments
PROJECT="matching-thesis"
ENV_ID="LettuceGreenhouse" 
ALGORITHM="sac"
MODEL_NAME="wobbly-brook-60"

# First run value function training
echo "Training value function..."
python RL/vf_TR_learning.py \
    --project $PROJECT \
    --model_name $MODEL_NAME \
    --env_id $ENV_ID \
    --algorithm $ALGORITHM \
    --stochastic

# Then run RL-MPC
echo "Running RL-MPC..."
python rl_mpc.py \
    --project $PROJECT \
    --env_id $ENV_ID \
    --save_name "rlmpc_run" \
    --weather_filename "outdoorWeatherWurGlas2014.csv" \
    --algorithm $ALGORITHM \
    --model_name $MODEL_NAME \
    --use_trained_vf \

echo "Done!"
