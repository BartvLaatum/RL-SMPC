#!/bin/bash
export PYTHONPATH=$(pwd)
echo "PYTHONPATH set to: $PYTHONPATH"
# Script to run MPC and RL-MPC for several optimization horizons

# Common arguments
PROJECT="SMPC"
ENV_ID="LettuceGreenhouse" 
ALGORITHM="sac"
MODE="stochastic"
UNCERTAINTY_VALUE=0.1
MODEL_NAME="restful-pyramid-7"
ORDER="first"
python rl_smpc.py \
    --project $PROJECT \
    --env_id $ENV_ID \
    --model_name $MODEL_NAME \
    --algorithm $ALGORITHM \
    --save_name $ORDER-order-terminal \
    --mode $MODE \
    --uncertainty_value $UNCERTAINTY_VALUE \
    --use_trained_vf \
    --order $ORDER
