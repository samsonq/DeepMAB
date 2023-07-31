import numpy as np


class Bandit:
    """
    Base class for a multi-armed bandit.
    """
    def __init__(self, n_arms):
        """
        Initialize a multi-armed bandit.

        Parameters
        ----------
        n_arms : int
            Number of arms.
        """
        self.n_arms = n_arms
        self.n_samples = 0

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
        raise NotImplementedError

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
        self.n_samples += 1

    def reset(self):
        """
        Reset the bandit.
        """
        self.n_samples = 0

    def __repr__(self):
        return f"Bandit(n_arms={self.n_arms}, n_samples={self.n_samples})"

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return self.n_arms

    def __getitem__(self, key):
        return self.pull(key)

    def __iter__(self):
        for arm in range(self.n_arms):
            yield arm, self.pull(arm)

    def __call__(self, arm):
        return self.pull(arm)

    def __eq__(self, other):
        return self.n_arms == other.n_arms and self.n_samples == other.n_samples

    def __ne__(self, other):
        return not self.__eq__(other)
