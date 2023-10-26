"""Metrics."""

from typing import Any, Dict, Union, List
from dataclasses import dataclass
import datetime
from pyspark.sql.functions import to_date, lit, col, max, current_date, datediff, count, isnull

import pandas as pd
import pyspark.sql as ps


@dataclass
class Metric:
    """Base class for Metric"""

    def __call__(self, df: Union[pd.DataFrame, ps.DataFrame]) -> Dict[str, Any]:
        if isinstance(df, pd.DataFrame):
            return self._call_pandas(df)

        if isinstance(df, ps.DataFrame):
            return self._call_pyspark(df)

        msg = (
            f"Not supported type of arg 'df': {type(df)}. "
            "Supported types: pandas.DataFrame, "
            "pyspark.sql.dataframe.DataFrame"
        )
        raise NotImplementedError(msg)

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        return {}


@dataclass
class CountTotal(Metric):
    """Total number of rows in DataFrame"""

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {"total": len(df)}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        return {"total": df.count()}


@dataclass
class CountZeros(Metric):
    """Number of zeros in choosen column"""

    column: str

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = sum(df[self.column] == 0)
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:

        n = df.count()
        k = df.filter(col(self.column) == 0).count()
        return {"total": n, "count": k, "delta": k / n}

@dataclass
class CountNull(Metric):
    """Number of empty values in chosen columns"""

    columns: List[str]
    aggregation: str = "any"  # either "all", or "any"

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)  # total number of rows in the dataframe
        if self.aggregation == "any":
            k = df[self.columns].isna().any(axis=1).sum()
        elif self.aggregation == "all":
            k = df[self.columns].isna().all(axis=1).sum()
        else:
            raise ValueError("Invalid aggregation type")
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql import functions as F

        n = df.count()

        # if self.aggregation == "any":
        #     k = df
        #     for column in self.columns:
        #         k = k.where(F.col(column).isNull())
        #     k = k.count()
        # elif self.aggregation == "all":
        #     k = df
        #     for column in self.columns:
        #         k = k.filter(F.col(column).isNotNull())
        #     k = k.count()
        # if self.aggregation == "any":
        #     #df = df.filter(exists(F.col(self.columns)).isNull())
        #     k = df.filter(col(self.columns).isNull().any()).count()
        # elif self.aggregation == "all":
        #     #df = df.filter(forall(F.col(self.columns)).isNotNull())
        #     k = df.filter(col(self.columns).isNull().all()).count()
        if self.aggregation == "any":
            k = n - df.na.drop(how = 'any', subset = self.columns).count()
        elif self.aggregation == "all":
            k = n - df.na.drop(how = 'all', subset = self.columns).count()
        else:
            raise ValueError("Invalid aggregation value. Must be either 'all' or 'any'.")

        return {"total": n, "count": k, "delta": k / n}



@dataclass
class CountDuplicates(Metric):
    """Number of duplicates in choosen columns"""

    columns: List[str]

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = df.duplicated(subset=self.columns).sum()
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        n = df.count()
        k = n - df.dropDuplicates(subset=self.columns).count()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountValue(Metric):
    """Number of values in choosen column"""

    column: str
    value: Union[str, int, float]

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = sum((df[self.column] == str(self.value)) | (df[self.column] == int(self.value)) | (
                df[self.column] == float(self.value)))
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        n = df.count()
        k = df.filter((col(self.column) == str(self.value)) | (col(self.column) == int(self.value)) |
                 (col(self.column) == float(self.value))).count()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountBelowValue(Metric):
    """Number of values below threshold"""

    column: str
    value: float
    strict: bool = False

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        if self.strict:
            k = sum(df[self.column] < self.value)
        else:
            k = sum(df[self.column] <= self.value)
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        n = df.count()
        if self.strict:
            k = df.filter(col(self.column) < self.value).count()
            # k = sum(df[self.column] < self.value)
        else:
            k = df.filter(col(self.column) <= self.value).count()
            # k = sum(df[self.column] <= self.value)
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountBelowColumn(Metric):
    """Count how often column X below Y"""

    column_x: str
    column_y: str
    strict: bool = False

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        if self.strict:
            k = sum(df[self.column_x] < (df[self.column_y]))
        else:
            k = sum(df[self.column_x] <= (df[self.column_y]))
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:

        n = df.count()
        # df = df.filter(
        #     (col(self.column_x).isNotNull()| col(self.column_x).isNotNull()) |
        #     (col(self.column_y).isNotNull())
        #               )
        df = df.na.drop(subset=[self.column_x, self.column_y])

        if self.strict:
            k = df.filter(col(self.column_x) < col(self.column_y)).count()
            # k = sum(df[self.column_x] < (df[self.column_y]))
        else:
            k = df.filter(col(self.column_x) <= col(self.column_y)).count()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountRatioBelow(Metric):
    """Count how often X / Y below Z"""

    column_x: str
    column_y: str
    column_z: str
    strict: bool = False

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        if self.strict:
            k = sum((df[self.column_x] / df[self.column_y]) < (df[self.column_z]))
        else:
            k = sum((df[self.column_x] / df[self.column_y]) <= (df[self.column_z]))
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        n = df.count()
        df = df.filter(
            (col(self.column_x).isNotNull()) &
            (col(self.column_y).isNotNull()) &
            (col(self.column_z).isNotNull())
                      )
        df = df.na.drop(subset=[self.column_x, self.column_y, self.column_z])
        if self.strict:
            k = df.filter(col(self.column_x) / col(self.column_y) < col(self.column_z)).count()
        else:
            k = df.filter(col(self.column_x) / col(self.column_y) <= col(self.column_z)).count()
        return {"total": n, "count": k, "delta": k / n}

from scipy import stats
import numpy as np
@dataclass
class CountCB(Metric):
    """Calculate lower/upper bounds for N%-confidence interval"""

    column: str
    conf: float = 0.95

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        data = df[self.column].dropna()
        n = len(data)
        mean = data.mean()
        # std_err = stats.sem(data)
        std_err = np.std(data) / n ** 0.5
        ucb = np.quantile(data, 0.975)
        lcb = np.quantile(data, 0.025)
        return {"lcb": lcb, "ucb": ucb}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        data = df.select(self.column).dropna()
        n = data.count()
        # mean = data.mean()
        # #std_err = stats.sem(data)
        # std_err = np.std(data)/n**0.5
        # ucb = np.quantile(data, 0.975)
        #lcb = np.quantile(data, 0.025)
        quantiles = df.select(self.column).dropna().approxQuantile(self.column, [0.025, 0.975], 0)
        lcb = quantiles[0]
        ucb = quantiles[1]
        return {"lcb": lcb, "ucb": ucb}


from datetime import datetime
from pandas import Timestamp
from pyspark.sql.functions import to_date, lit, col, max, current_date, datediff
@dataclass
class CountLag(Metric):
    """A lag between latest date and today"""

    column: str
    fmt: str = "%Y-%m-%d"

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        dates = pd.to_datetime(df[self.column], format=self.fmt)
        max_date = dates.max()
        today = datetime.today().strftime('%Y-%m-%d')
        max_date = max_date.strftime('%Y-%m-%d')
        lag = (pd.to_datetime(today) - pd.to_datetime(max_date)).days
        return {"today": today, "last_day": max_date, "lag": lag}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]: 
        from pyspark.sql import functions as F
        from pyspark.sql.types import DateType
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import current_date

        # Создать спарк-сессию
        spark = SparkSession.builder.getOrCreate()

        max_date = df.select(max(self.column)).collect()[0][0]
        today = df.select(current_date()).collect()[0][0]
        lag = df.select(datediff(current_date(), max(self.column))).collect()[0][0]
        return {"today": today.strftime(self.fmt), "last_day": max_date, "lag": lag}

