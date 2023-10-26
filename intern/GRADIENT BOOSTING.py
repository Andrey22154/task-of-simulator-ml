import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor


class GradientBoostingRegressor:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        loss="mse",
        verbose=False,
    ):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.verbose = verbose
        self.trees_ = []

    def _mse(self, y_true, y_pred):
        loss = np.mean((y_pred - y_true)**2)
        grad = (y_pred - y_true)/len(y_true)
        return loss, grad

    def fit(self, X, y):
        """
        Fit the model to the data.

        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples,)

        Returns:
            GradientBoostingRegressor: The fitted model.
        """
        self.base_pred_ = np.mean(y)
        residuals = y - self.base_pred_
        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X, residuals)
            self.trees_.append(tree)
            if self.verbose:
                y_pred = self.predict(X)
                mse = self._mse(y, y_pred)
                print(f"Iteration {i+1}: MSE = {mse}")
            residuals -= self.learning_rate * tree.predict(X)
        return self

    def predict(self, X):
        """Predict the target of new data.

        Args:
            X: array-like of shape (n_samples, n_features)

        Returns:
            y: array-like of shape (n_samples,)
            The predict values.

        """

        predictions = np.full(X.shape[0], self.base_pred_)
        for tree in self.trees_:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred