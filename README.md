# Stochastic RL-MPC for Greenhouse Lettuce Production Control 🥬

## Overview

This repository provides an implementation of the integration between Reinforcement Learning (RL) and Model Predictive Control (MPC) for managing greenhouse lettuce production systems. The approach allows for controlling the system using MPC, RL, or a combination of both (RL-MPC).

## Project Structure

The project is organized as follows:

```
RL-MPC-lettuce/
│
├── README.md
├── requirements.txt
├── configs/
│   ├── models/
│   └── envs/
├── common/
│   └── utils.py
├── data/
├── train_data/
├── RL/
│   ├── rl.py
│   └── evaluate_rl.py
├── mpc.py
└── rl_mpc.py
```

- **common/utils.py**: Defines the model's differential equations and parameters.
- **configs/models/**: Contains hyperparameters for control methods.
- **configs/envs/**: Contains additional environment parameters for the greenhouse system.

## Installation

We recommend using Python 3.10+ with Anaconda for this project.

1. Clone the repository:
   ```shell
   git clone git@github.com:BartvLaatum/RL-MPC-lettuce.git
   ```

2. Install the required Python libraries:
   ```shell
   pip install -r requirements.txt
   ```

## Usage

### 1. Model Predictive Control (MPC)

Run a deterministic MPC instance and save the closed-loop trajectory results:

```shell
python mpc.py 
    --project PROJECT_NAME
    --env_id ENV_ID
    --save_name SAVE_NAME
    --weather_filename WEATHER_FILENAME
```

### 2. Reinforcement Learning (RL)

Start training and logging the RL agent. Training logs are saved to Weights and Biases (wandb), and model/environment metrics are stored in:

- `train_data/{project}/{algorithm}/{stochastic}/envs/{model_name}/`
- `train_data/{project}/{algorithm}/{stochastic}/models/{model_name}/`

```shell
python RL/rl.py
    --project PROJECT_NAME
    --env_id ENV_ID
    --algorithm RL-ALGORITHM
    --group GROUP_NAME
    --n_eval_episodes NUMBER_OF_EPISODES_TO_EVALUATE
    --n_evals NUMBER_EVALUATION_DURING_TRAINING
    --env_seed SEED_FOR_ENVIRONMENT
    --model_seed SEED_FOR_MODEL
    --stochastic STOCHASTIC_OR_DETERMINISTIC
    --device DEVICE_FOR_NN
    --save_model SAVE_MODEL
    --save_env SAVE_ENV
```

Evaluate the best-trained agent:

```shell
python RL/evaluate_rl.py
    --project PROJECT_NAME
    --env_id ENV_ID
    --model_name MODEL_NAME
    --algorithm RL-ALGORITHM
    --mode STOCHASTIC_OR_DETERMINISTIC
```

### 3. Train value function for temporal return learning.

- Saves the resulting value function in:

`train_data/{PROJECT_NAME}/{RL_ALGORITHM}/{STOCHASTIC_OR_DETERMINISTIC}/models/{MODEL_NAME}/vf.zip`

```shell
python RL/vf_TR_learning.py 
    --project PROJECT_NAME
    --env_id ENV_ID
    --algorithm RL_ALGORITHM
    --model_name MODEL_NAME
    --stochastic STOCHASTIC_OR_DETERMINISTIC
```

### 4. RL-MPC

Run a deterministic RL-MPC instance and save the closed-loop trajectory results:

```shell
python rl_mpc.py 
    --project PROJECT_NAME
    --env_id ENV_ID
    --save_name SAVE_NAME
    --weather_filename WEATHER_FILENAME
    --algorithm RL-ALGORITHM
    --model_name MODEL_NAME
    --use_trained_vf USE_TRAINED_VF
    --stochastic STOCHASTIC_OR_DETERMINISTIC
```

## Examples


### Learning value function
- Determinstic case
- Using model `resolute-darling-85`
```shell
python RL/vf_TR_learning.py 
    --project matching-thesis
    --env_id LettuceGreenhouse
    --algorithm sac
    --model_name resolute-darling-85
```

### Running RL-MPC for varying prediction horizon ranging from 1H-6H

- Deterministic case
- Using model: `resolute-darling-85`
```shell
python experiments/horizon_rlmpc.py
    --project matching-thesis
    --env_id LettuceGreenhouse
    --save_name rlmpc
    --algorithm sac
    --model_name resolute-darling-85
    --mode deterministic
    --use_trained_vf
```


### Visualize performance of algorithms over 1H-6H horizon
- Deterministic case
- For the models: `resolute-darling-85` and `salim`

```shell
python visualisations/performance_plots.py 
    --project matching-thesis 
    --model_names resolute-darling-85 salim 
    --mode deterministic
 ```

## Results

### Deterministic case
Example performance of varying horizons in the RL-MPC framework:

![Performance comparison of different prediction horizons](figures/matching-thesis/deterministic/rl-tanh-relu-thesis-mpc-v4.png)


### Stochastic case
Example performance of varying horizons in the RL-MPC framework:

![Performance comparison of different prediction horizons](figures/matching-thesis/stochastic/thesis-agent.png)
