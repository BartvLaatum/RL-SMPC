PROJECT="SMPC"
ENV_ID="LettuceGreenhouse" 
UNCERTAINTY_VALUE=0.1

echo "Running SMPC..."
python smpc.py \
    --project $PROJECT \
    --env_id $ENV_ID \
    --save_name smpc-tight-rh \
    --uncertainty_value $UNCERTAINTY_VALUE \


echo "Running SMPC..."
python smpc.py \
    --project $PROJECT \
    --env_id $ENV_ID \
    --save_name smpc-tight-rh-horizon-weights \
    --uncertainty_value $UNCERTAINTY_VALUE \
