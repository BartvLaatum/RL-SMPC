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

########################################
#### Run RL-SMPC for horizons 1H-8H ####
########################################

# Running RL-SMPC without value function
python experiments/horizon_rl_smpc.py \
    --project $PROJECT \
    --env_id $ENV_ID \
    --model_name $MODEL_NAME \
    --algorithm $ALGORITHM \
    --save_name no-vf \
    --mode $MODE \
    --uncertainty_value $UNCERTAINTY_VALUE \
    --terminal \
    --rl_feedback
    # --use_trained_vf \

# Running RL-SMPC without terminal state constraint
python experiments/horizon_rl_smpc.py \
    --project $PROJECT \
    --env_id $ENV_ID \
    --model_name $MODEL_NAME \
    --algorithm $ALGORITHM \
    --save_name no-terminal \
    --mode $MODE \
    --uncertainty_value $UNCERTAINTY_VALUE \
    --use_trained_vf \
    --rl_feedback
    # --terminal \


# Running RL-SMPC without terminal feedback from RL-policy
python experiments/horizon_rl_smpc.py \
    --project $PROJECT \
    --env_id $ENV_ID \
    --model_name $MODEL_NAME \
    --algorithm $ALGORITHM \
    --save_name no-feedback \
    --mode $MODE \
    --uncertainty_value $UNCERTAINTY_VALUE \
    --use_trained_vf \
    --terminal
    # --rl_feedback
