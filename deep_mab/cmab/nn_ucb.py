import random
import numpy as np
from cbandit import CBandit
import tensorflow as tf
import keras


class NeuralNetUCB(CBandit):
    """
    Neural network upper confidence bound algorithm.
    """
    def __init__(self, n_arms, n_features, hidden_size=100, epochs=100, batch_size=100, epsilon=0.1, delta=0.1, alpha=0.1):
        super(NeuralNetUCB, self).__init__(n_arms, n_features)
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.delta = delta
        self.alpha = alpha
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(self.hidden_size, activation=tf.nn.relu, input_shape=(self.n_features,)),
            keras.layers.Dense(self.n_arms)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def fit(self, X, a, r):
        self.model.fit(X, self._one_hot_encode(a, self.n_arms, self.batch_size), epochs=self.epochs, batch_size=self.batch_size, verbose=0)

    def predict(self, X):
        return self.model.predict(X)

    def _one_hot_encode(self, a, n_arms, batch_size):
        one_hot = np.zeros((batch_size, n_arms))
        one_hot[np.arange(batch_size), a] = 1
        return one_hot

    def _get_ucb(self, X):
        return self.model.predict(X) + self.alpha * np.sqrt(np.log(1 / self.delta) / self.t)

    def _get_action(self, X):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_arms - 1)
        else:
            return np.argmax(self._get_ucb(X))

    def _get_batch_action(self, X):
        return np.argmax(self._get_ucb(X), axis=1)

    def _get_reward(self, X, a):
        return self.model.predict(X)[np.arange(self.batch_size), a]

    def _get_batch_reward(self, X, a):
        return self.model.predict(X)[np.arange(self.batch_size), a]

    def _get_regret(self, X, a, r):
        return np.max(self.model.predict(X)) - r

    def _get_batch_regret(self, X, a, r):
        return np.max(self.model.predict(X), axis=1) - r

    def _get_loss(self, X, a, r):
        return self.model.evaluate(X, self._one_hot_encode(a, self.n_arms, self.batch_size), verbose=0)

    def _get_batch_loss(self, X, a, r):
        return self.model.evaluate(X, self._one_hot_encode(a, self.n_arms, self.batch_size), verbose=0)

    def _get_accuracy(self, X, a):
        return np.mean(np.argmax(self.model.predict(X), axis=1) == a)

    def _get_batch_accuracy(self, X, a):
        return np.mean(np.argmax(self.model.predict(X), axis=1) == a)
