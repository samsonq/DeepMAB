import random
import numpy as np
from cbandit import CBandit


class LinUCB(CBandit):
    """
    LinUCB contextual bandit algorithm.
    """
    def __init__(self, n_arms, n_features, alpha=1.0):
        """
        Initialize LinUCB contextual bandit algorithm.
        Args:
            n_arms (int): Number of arms.
            n_features (int): Number of features.
            alpha (float): Confidence level.
        """
        super().__init__(n_arms)
        self.n_features = n_features
        self.alpha = alpha
        self.A = [np.identity(self.n_features) for _ in range(self.n_arms)]
        self.b = [np.zeros((self.n_features, 1)) for _ in range(self.n_arms)]

    def predict(self, context):
        """
        Predict the best arm.
        Args:
            context (np.ndarray): Context vector.
        Returns:
            int: Arm index.
        """
        p = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            theta = np.linalg.solve(self.A[i], self.b[i])
            p[i] = theta.T @ context + self.alpha * np.sqrt(context.T @ np.linalg.solve(self.A[i], context))
        return np.argmax(p)

    def update(self, arm, reward, context):
        """
        Update the model.
        Args:
            arm (int): Arm index.
            reward (float): Reward.
            context (np.ndarray): Context vector.
        """
        self.A[arm] += context @ context.T
        self.b[arm] += reward * context

    def __str__(self):
        return "LinUCB"

    def __repr__(self):
        return "LinUCB(n_arms={}, n_features={}, alpha={})".format(self.n_arms, self.n_features, self.alpha)

    def __eq__(self, other):
        return self.n_arms == other.n_arms and self.n_features == other.n_features and self.alpha == other.alpha
