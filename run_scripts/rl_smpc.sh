#!/bin/bash
export PYTHONPATH=$(pwd)
echo "PYTHONPATH set to: $PYTHONPATH"
# Script to run MPC and RL-MPC for several optimization horizons

# Define arrays for uncertainty values and model names
uncertainty_values=(0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2)
model_names=("mild-rain-8" "worthy-cosmos-1" "rare-shadow-9" "restful-pyramid-7" \
            "lyric-sky-10" "volcanic-valley-2" "peach-haze-11" "blooming-glade-3")

# Common arguments
PROJECT="SMPC"
ENV_ID="LettuceGreenhouse" 
ALGORITHM="sac"
MODE="stochastic"
UNCERTAINTY_VALUE=0.1
MODEL_NAME="restful-pyramid-7"
ORDER="zero"

# Loop through uncertainty values and models
for i in "${!uncertainty_values[@]}"; do
    UNCERTAINTY_VALUE=${uncertainty_values[$i]}
    MODEL_NAME=${model_names[$i]}
    python experiments/horizon_rl_smpc.py \
        --project $PROJECT \
        --env_id $ENV_ID \
        --model_name $MODEL_NAME \
        --algorithm $ALGORITHM \
        --save_name $ORDER-order-terminal \
        --mode $MODE \
        --uncertainty_value $UNCERTAINTY_VALUE \
        --use_trained_vf \
        --order $ORDER
done