#!/bin/bash
export PYTHONPATH=$(pwd)
echo "PYTHONPATH set to: $PYTHONPATH"
# Script to run MPC and RL-SMPC with zero-order approximation for several optimization horizons

# Common arguments
PROJECT="SMPC"
ENV_ID="LettuceGreenhouse" 
ALGORITHM="sac"
MODE="stochastic"


# Define arrays for uncertainty values and model names
# uncertainty_values=(0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2)
# model_names=("mild-rain-8" "worthy-cosmos-1" "rare-shadow-9" "restful-pyramid-7" \
#             "lyric-sky-10" "volcanic-valley-2" "peach-haze-11" "blooming-glade-3")

uncertainty_values=(0.1)
model_names=("restful-pyramid-7")


# Loop through uncertainty values and models
for i in "${!uncertainty_values[@]}"; do
    UNCERTAINTY_VALUE=${uncertainty_values[$i]}
    MODEL_NAME=${model_names[$i]}
    python experiments/horizon_rl_smpc.py \
        --project $PROJECT \
        --env_id $ENV_ID \
        --model_name $MODEL_NAME \
        --algorithm $ALGORITHM \
        --save_name zero-order-terminal-box-constraints \
        --mode $MODE \
        --uncertainty_value $UNCERTAINTY_VALUE \
        --use_trained_vf \
        # --order $ORDER
done