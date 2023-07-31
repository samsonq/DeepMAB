# Contextual MAB
MAB and linear/non-linear Contextual MAB algorithms.

## Algorithms

#### Multi-Arm Bandits
* [x] Epsilon Greedy
* [x] UCB
* [x] Thompson Sampling

#### Contextual Multi-Arm Bandits
* [x] Neural Net Epsilon Greedy
* [x] LinUCB
* [x] Neural Net UCB

## Usage Instructions
* This project is published on [PyPI](https://pypi.org/project/contextual-mab/). To install package, run:

  ```
  pip install contextual-mab
  ```
* To run the algorithms, import the package and call the respective functions. For example, to run the LinUCB algorithm, run:

  ```
  from contextual_bandits.algorithms import LinUCB
  model = LinUCB(n_arms=10, alpha=1, fit_intercept=True)
  model.fit(X_train, y_train)
  model.predict(X_test)
  ```
* For more details, refer to the [documentation](https://contextual-bandits.readthedocs.io/en/latest/).
* To run the examples, clone the repository and run the following commands:

  ```
  cd contextual-bandits
  pip install -r requirements.txt
  python examples/linucb_example.py
  ```
* To run the tests, run the following commands:

  ```
  cd contextual-bandits
  pip install -r requirements.txt
  pytest
  ```
  