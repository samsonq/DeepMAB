import random
import numpy as np
from bandit import Bandit


class ThompsonSampling(Bandit):
    """
    Thompson Sampling Algorithm
    """
    def __init__(self, n_arms):
        """
        Initialize a multi-armed bandit.

        Parameters
        ----------
        n_arms : int
            Number of arms.
        """
        super().__init__(n_arms)
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def pull(self, arm):
        """
        Pull arm of the bandit.

        Parameters
        ----------
        arm : int
            Arm to pull.

        Returns
        -------
        reward : float
            Reward for pulling arm.
        """
        return random.betavariate(self.alpha[arm], self.beta[arm])

    def update(self, arm, reward):
        """
        Update the bandit after pulling an arm.

        Parameters
        ----------
        arm : int
            Arm that was pulled.
        reward : float
            Reward for pulling arm.
        """
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward
        super().update(arm, reward)
        
    def reset(self):
        """
        Reset the bandit.
        """
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)
        super().reset()

    def __repr__(self):
        return f"ThompsonSampling(n_arms={self.n_arms}, n_samples={self.n_samples})"
