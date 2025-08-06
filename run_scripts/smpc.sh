PROJECT="SMPC"
ENV_ID="LettuceGreenhouse"
MODE="stochastic" 
UNCERTAINTY_VALUE=0.1

# Run MPC for horizons 1H-8H
echo "Running MPC..."
python experiments/horizon_mpc.py \
    --project $PROJECT \
    --env_id $ENV_ID \
    --save_name mpc \
    --mode $MODE \
    --uncertainty_value $UNCERTAINTY_VALUE

# Run SMPC for horizons 1H-8H
echo "Running SMPC..."
python experiments/horizon_smpc.py \
    --project $PROJECT \
    --env_id $ENV_ID \
    --save_name no-tightening \
    --mode $MODE \
    --uncertainty_value $UNCERTAINTY_VALUE
