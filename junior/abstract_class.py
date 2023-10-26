from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


class BaseSelector(ABC):
    @abstractmethod
    def fit(self, X, y=None):
        pass

    def transform(self, X):
        return X[self.high_var_features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    @property
    def n_features_(self):
        return len(self.select_features)

    @property
    def original_features_(self):
        return self.original_features

    @property
    def selected_features_(self):
        return self.select_features

@dataclass
class PearsonSelector(BaseSelector):
    threshold: float = 0.5

    def fit(self, X, y) -> PearsonSelector:
        # Correlation between features and target
        corr = pd.concat([X, y], axis=1).corr(method="pearson")
        corr_target = corr.iloc[:-1, -1]

        self.original_features = X.columns.tolist()
        self.select_features = corr_target[
            abs(corr_target) >= self.threshold
        ].index.tolist()

        return self

@dataclass
class SpearmanSelector(BaseSelector):
    threshold: float = 0.5

    def fit(self, X, y) -> SpearmanSelector:
        corr = pd.concat([X, y], axis=1).corr(method="spearman")
        corr_target = corr.iloc[:-1, -1]
        self.original_features = X.columns.tolist()
        self.select_features = corr_target[
            abs(corr_target) >= self.threshold
        ].index.tolist()

        return self

@dataclass
class VarianceSelector(BaseSelector):
    min_var: float = 0.4

    def fit(self, X, y=None) -> VarianceSelector:
        variances = np.var(X, axis=0)
        self.original_features = X.columns.tolist()
        self.select_features = X.columns[variances > self.min_var].tolist()
        return self