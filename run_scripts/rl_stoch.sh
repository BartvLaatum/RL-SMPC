#!/bin/bash
export PYTHONPATH=$(pwd)
echo "PYTHONPATH set to: $PYTHONPATH"
# Script to run MPC and RL-MPC for several optimization horizons

# Common arguments
PROJECT="SMPC"
ENV_ID="LettuceGreenhouse" 
ALGORITHM="sac"
MODE="stochastic"
# UNCERTAINTY_VALUES=(0.1)
# MODEL_NAMES=("dry-serenity-22")
# UNCERTAINTY_VALUES=(0.025 0.05 0.075 0.125 0.15 0.175 0.2)
UNCERTAINTY_VALUES=(0.15)
MODEL_NAMES=("frosty-rain-50")


# Loop through uncertainty values and model names
for i in "${!UNCERTAINTY_VALUES[@]}"; do
    UNCERTAINTY_VALUE=${UNCERTAINTY_VALUES[$i]}
    MODEL_NAME=${MODEL_NAMES[$i]}

    echo "Processing uncertainty value: $UNCERTAINTY_VALUE with model: $MODEL_NAME"

    # Evaluate RL agent
    echo "Evaluate RL agent..."
    python RL/evaluate_rl.py \
        --project $PROJECT \
        --model_name $MODEL_NAME \
        --env_id $ENV_ID \
        --algorithm $ALGORITHM \
        --mode $MODE \
        --uncertainty_value $UNCERTAINTY_VALUE

    # First run value function training
    echo "Training value function..."
    python RL/vf_TR_learning.py \
        --project $PROJECT \
        --model_name $MODEL_NAME \
        --env_id $ENV_ID \
        --algorithm $ALGORITHM \
        --uncertainty_value $UNCERTAINTY_VALUE \
        --mode $MODE

    python experiments/horizon_rl_smpc.py \
        --project $PROJECT \
        --env_id $ENV_ID \
        --model_name $MODEL_NAME \
        --algorithm $ALGORITHM \
        --save_name zero-order-terminal \
        --mode $MODE \
        --uncertainty_value $UNCERTAINTY_VALUE \
        --use_trained_vf \

    # Run MPC for horizons 1H-6H
    echo "Running MPC..."
    python experiments/horizon_mpc.py \
        --project $PROJECT \
        --env_id $ENV_ID \
        --save_name warm-start \
        --mode $MODE \
        --uncertainty_value $UNCERTAINTY_VALUE

    # Run MPC for horizons 1H-6H
    echo "Running SMPC..."
    python experiments/horizon_smpc.py \
        --project $PROJECT \
        --env_id $ENV_ID \
        --save_name warm-start \
        --mode $MODE \
        --uncertainty_value $UNCERTAINTY_VALUE


done
