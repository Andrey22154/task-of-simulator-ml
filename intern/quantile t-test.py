from typing import List
from typing import Tuple

import numpy as np
from scipy.stats import ttest_ind


def quantile_ttest(
    control: List[float],
    experiment: List[float],
    alpha: float = 0.05,
    quantile: float = 0.95,
    n_bootstraps: int = 1000,
) -> Tuple[float, bool]:
    """
    Bootstrapped t-test for quantiles of two samples.
    """
    bootstrap_lst_control = []
    bootstrap_lst_experiment = []

    for elem in range(n_bootstraps):
        boots_control = np.random.choice(control, size=len(control), replace=True)
        bootstrap_lst_control.append(np.median(np.percentile(boots_control, quantile)))

        boots_experiment = np.random.choice(experiment, size=len(experiment), replace=True)
        bootstrap_lst_experiment.append(np.median(np.percentile(boots_experiment, quantile)))

    p_value = ttest_ind(bootstrap_lst_control, bootstrap_lst_experiment)
    result = p_value[1] < alpha

    return p_value[1], bool(result)