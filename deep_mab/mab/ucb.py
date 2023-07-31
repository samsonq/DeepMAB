import random
import numpy as np
from bandit import Bandit


class UCB(Bandit):
    """
    Upper confidence bound multi-armed bandit.
    """
    def __init__(self, n_arms, c=1):
        """
        Initialize an upper confidence bound multi-armed bandit.

        Parameters
        ----------
        n_arms : int
            Number of arms.
        c : float
            Exploration parameter.
        """
        super().__init__(n_arms)
        self.c = c
        self.n_pulls = np.zeros(n_arms)
        self.q = np.zeros(n_arms)

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
        if self.n_pulls[arm] == 0:
            return random.random()
        else:
            return self.q[arm] + self.c * np.sqrt(np.log(self.n_samples) / self.n_pulls[arm])

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
        self.n_pulls[arm] += 1
        self.q[arm] += (reward - self.q[arm]) / self.n_pulls[arm]

    def __repr__(self):
        return f"UCB(n_arms={self.n_arms}, c={self.c}, n_samples={self.n_samples})"
