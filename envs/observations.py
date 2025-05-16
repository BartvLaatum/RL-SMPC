from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from gymnasium import spaces


class BaseObservations(ABC):
    """
    Observer class, which gives control over the observations (aka inputs) for our RL agents.
    One can construct observations from GreenLight or from the current and future weather data.
    The model observations are computed in GreenLight,
    the weather observations are extracted from the weather data array.
    """
    def __init__(self,
                ) -> None:
        self.n_obs = None
        self.low = None
        self.high = None
        self.obs_names = None

    @abstractmethod
    def observation_space(self) -> spaces.Box:
        pass

    @abstractmethod
    def compute_obs(self) -> np.ndarray:
        """
        Compute, and retrieve observations from GreenLight and the weather.
        """
        pass

class StandardObservations(BaseObservations):
    """
    Observer class, which give us control over the observations we want to our RL algorithm to use.
    One can construct observations from GreenLight or from the current and future weather data.
    The model observations are computed in GreenLight,
    the weather observations are extracted from the weather data array.
    """
    def __init__(self,
                obs_names) -> None:
        self.obs_names = obs_names
        self.n_obs = len(obs_names)

    def observation_space(self) -> spaces.Box:
        return spaces.Box(low=-1e-4, high=1e4, shape=(self.n_obs,), dtype=np.float32)

    def compute_obs(self, yk, uk, d, timestep) -> np.ndarray:
        """
        Compute, and retrieve observations from GreenLight and the weather.
        """
        return np.concatenate([yk, uk, [timestep], d[:, timestep]])

class FutureWeatherObservations(BaseObservations):
    """
    Observer class, which give us control over the observations we want to our RL algorithm to use.
    One can construct observations from GreenLight or from the current and future weather data.
    The model observations are computed in GreenLight,
    the weather observations are extracted from the weather data array.
    """
    def __init__(self,
                obs_names) -> None:
        self.obs_names = obs_names
        self.Np = 12
        self.n_obs = len(obs_names) + self.Np*4 # 4 weather variables for Np time steps

    def observation_space(self) -> spaces.Box:
        return spaces.Box(low=-1e-4, high=1e4, shape=(self.n_obs,), dtype=np.float32)

    def compute_obs(self, yk, uk, d, timestep) -> np.ndarray:
        """
        Compute, and retrieve observations from GreenLight and the weather.
        """
        return np.concatenate([yk, uk, [timestep], d[:, timestep:timestep+self.Np+1].flatten()])
