import random
import numpy as np
from mab.bandit import Bandit


class CBandit(Bandit):
    """
    Contextual bandit class.
    """
    def __init__(self, n_arms, n_features):
        """
        Initialize the contextual bandit class.
        :param n_arms: number of arms
        :param n_features: number of features
        """
        super().__init__(n_arms)
        self.n_features = n_features
        self.theta = np.zeros((n_arms, n_features))

    def pull(self, context):
        """
        Pull arm based on context.
        :param context: context
        :return: arm
        """
        p = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            p[i] = np.dot(self.theta[i], context)
        return np.argmax(p)

    def update(self, arm, reward, context):
        """
        Update the contextual bandit.
        :param arm: arm
        :param reward: reward
        :param context: context
        """
        self.theta[arm] += reward * context

    def reset(self):
        """
        Reset the contextual bandit.
        """
        self.theta = np.zeros((self.n_arms, self.n_features))

    def __str__(self):
        return "Contextual bandit with " + str(self.n_arms) + " arms and " + str(self.n_features) + " features."

    def __repr__(self):
        return self.__str__()
