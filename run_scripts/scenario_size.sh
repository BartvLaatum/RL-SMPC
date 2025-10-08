export PYTHONPATH=$(pwd)
echo "PYTHONPATH set to: $PYTHONPATH"
# Script to run MPC and RL-MPC for several optimization horizons

# Common arguments
PROJECT="SMPC"
ENV_ID="LettuceGreenhouse" 
ALGORITHM="sac"
MODE="stochastic"
UNCERTAINTY_VALUE=0.1
MODEL_NAME="brisk-resonance-24"

Ns=(5 15 20)

# Run RL-SMPC for horizons 1H-8H
echo "Running RL-MPC..."
for Ns_value in "${Ns[@]}"; do
    echo "Running with Ns value $Ns_value..."
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
        --rl_feedback \
        --Ns $Ns_value
done

# Run SMPC for horizons 1H-8H
echo "Running SMPC..."
for Ns_value in "${Ns[@]}"; do
    echo "... with Ns value $Ns_value..."
    python experiments/horizon_smpc.py \
        --project $PROJECT \
        --env_id $ENV_ID \
        --save_name no-tightening \
        --mode $MODE \
        --uncertainty_value $UNCERTAINTY_VALUE \
        --Ns $Ns_value
done

