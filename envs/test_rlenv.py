import unittest
import numpy as np
from lettuce_greenhouse import LettuceGreenhouse
import pandas as pd
import argparse
import yaml

class TestLettuceGreenhouse(unittest.TestCase):
    def setUp(self):
        # Initialize the LettuceGreenhouse environment
        
        with open(f"configs/envs/LettuceGreenhouse.yml") as f:
            env_params = yaml.safe_load(f)

        with open(f"configs/models/ppo.yml") as f:
            rl_env_params = yaml.safe_load(f)

        self.env = LettuceGreenhouse(
            weather_filename="outdoorWeatherWurGlas2014.csv",
            **env_params, **rl_env_params["LettuceGreenhouse"]
        )

        # Load control inputs from mpc-6hr (assuming it's a numpy array)
        # Load and test MPC policy with 6hr control interval
        mpc_data = pd.read_csv('data/matching-mpc/mpc-6hr.csv')
        control_colids = ["u_{0}".format(i) for i in range(self.env.nu)]
        self.controls = mpc_data[control_colids].values
        print(self.controls.shape)
        # Initialize containers for states, outputs, and econ_rewards
        self.states = []
        self.outputs = []
        self.weather = []
        self.econ_rewards = []

    def test_environment(self):
        # Test MPC policy
        self.env.reset()
        self.states.append(self.env.x)
        self.outputs.append(self.env.get_y())
        self.weather.append(self.env.get_d())
        for i in range(self.controls.shape[0]):
            obs, rew, terminated, truncated, info = self.env.step2(self.controls[i])
            self.econ_rewards.append(info['econ_rewards'])
            self.states.append(self.env.get_state())
            self.outputs.append(self.env.get_y())
            self.weather.append(self.env.get_d())
            if self.env.terminated:
                break

        # Check if the containers are not empty
        self.assertTrue(len(self.states) > 0)
        self.assertTrue(len(self.outputs) > 0)
        self.assertTrue(len(self.econ_rewards) > 0)

        # Additional checks can be added here based on expected behavior
        # Create a DataFrame to save states, outputs, and econ_rewards
        data = {}
        t = np.arange(0, self.env.L + self.env.h, self.env.h)[:-1]/86400
        data["time"] = t
        self.states  = np.array(self.states).T
        for i in range(self.states.shape[0]):
            data[f"x_{i}"] = self.states[i, :self.env.N]
        self.outputs = np.array(self.outputs).T
        for i in range(self.outputs.shape[0]):
            data[f"y_{i}"] = self.outputs[i, :self.env.N]
        self.weather = np.array(self.weather).T
        for i in range(self.weather.shape[0]):
            data[f"d_{i}"] = self.weather[i, :self.env.N]
        
        data["econ_rewards"] = np.array(self.econ_rewards)

        
        df = pd.DataFrame(data)

        # Save the DataFrame to a CSV file
        df.to_csv('data/matching-mpc/environment_results.csv', index=False)

if __name__ == '__main__':
    unittest.main()