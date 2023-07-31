import random
import numpy as np
from bandit import Bandit


class EpsilonGreedy(Bandit):
    """
    Epsilon-greedy multi-armed bandit.
    """
    def __init__(self, n_arms, epsilon=0.1):
        """
        Initialize an epsilon-greedy multi-armed bandit.

        Parameters
        ----------
        n_arms : int
            Number of arms.
        epsilon : float
            Probability of exploration.
        """
        super().__init__(n_arms)
        self.epsilon = epsilon
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
        if random.random() < self.epsilon:
            return random.random()
        else:
            return self.q[arm]

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
        return f"EpsilonGreedy(n_arms={self.n_arms}, epsilon={self.epsilon}, n_samples={self.n_samples})"
