export PYTHONPATH=$(pwd)
echo "PYTHONPATH set to: $PYTHONPATH"
# Script to run MPC and RL-MPC for several optimization horizons

# Common arguments
PROJECT="SMPC"
ENV_ID="LettuceGreenhouse" 
ALGORITHM="sac"
MODE="stochastic"
UNCERTAINTY_VALUE=0.1
MODEL_NAMES=("brisk-resonance-24")
# Uncomment to run with all trained models
# MODEL_NAMES=("brisk-resonance-24" "iconic-dust-9" "eager-bee-10" "logical-forest-11" "swift-armadillo-12")

# Run RL-SMPC for horizons 1H-8H
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "Running RL-MPC for model: $model..."
    python experiments/horizon_rl_smpc.py \
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
done
