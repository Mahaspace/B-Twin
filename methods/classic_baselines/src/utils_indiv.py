import os
import sys
from functools import partial
from time import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from simulations.simulation import generate_simulation


def compute_treatment_effects(df, T_0):
    """
    Computes individual, average, and treated average treatment effects.

    Parameters:
    - df: DataFrame with columns ['Y_0', 'Y_1', 'treatment', 'time', 'treated']
    -T_0: int, time step when treatment starts

    Returns:
    - A dictionary with ITE, ATE, and ATT
    """
    # Compute individual treatment effects

    post_treatment = df[df["time"] >= T_0]
    post_treatment["ITE"] = post_treatment["y_1"] - post_treatment["y_0"]

    # Average Treatment Effect (ATE)
    ATE = post_treatment["ITE"].mean()

    # Average Treatment Effect on the Treated (ATT)
    treated_units = post_treatment[post_treatment["treatment"] == 1]
    ATT = treated_units["ITE"].mean()

    return {"ITE": post_treatment["ITE"], "ATE": ATE, "ATT": ATT}


def monte_carlo_ATE(params, seed=None, n_iter=1000):
    if seed is None:
        seed = np.random.randint(1e6)
    ate = np.zeros(n_iter)
    for i in range(n_iter):
        data = generate_simulation(**params)
        ate[i] = compute_treatment_effects(data, params["treatment_time"])["ATT"]

    return ate


def run_methods(data, methods, func_params, n_jobs=1):
    ate_list = []
    execution_time_list = []

    treated_col = func_params[0]["treat_col"]
    id_col = func_params[0]["id_col"]

    # Precompute ids and control group once
    treated_ids = data[data[treated_col] == 1][id_col].unique()
    control_df = data[data[treated_col] == 0]

    for index, func in enumerate(methods):
        t1 = time()

        params = func_params[index]

        # joblib-parallelize across treated units
        ite_values = Parallel(n_jobs=n_jobs)(
            delayed(run_single_ite)(
                data=data,
                control_df=control_df,
                id=id_val,
                func=func,
                params=params,
                treated_col=treated_col,
                id_col=id_col,
            )
            for id_val in treated_ids
        )

        ate_list.append(np.mean(ite_values))
        execution_time_list.append(time() - t1)

    return ate_list, execution_time_list


def run_single_ite(data, control_df, id, func, params, treated_col, id_col):
    df_individual = pd.concat(
        [
            control_df,
            data[data[id_col] == id],  # only 1 row or few rows
        ],
        axis=0,
    )
    return func(data=df_individual, **params)


def estimate_ATE(
    methods,
    func_params,
    simulation_params,
    nb_simulations=100000,
    seed=0,
    njobs=2,  # outer parallelism
    njobs_inner=3,  # inner parallelism (per method)
    verbose=False,
):
    np.random.seed(seed)

    func_partial = partial(
        run_methods,
        methods=methods,
        func_params=func_params,
        n_jobs=njobs_inner,
    )

    ATE = Parallel(n_jobs=njobs)(
        delayed(func_partial)(generate_simulation(**simulation_params))
        for _ in tqdm(range(nb_simulations), disable=not verbose)
    )

    return np.array(ATE)
