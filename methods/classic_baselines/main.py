import os
import pickle
import time
import warnings

import numpy as np

# from .src.config import params
from .src.methods import (
    did,
    elastic_net,
    ols,
    synthetic_control,
    synthetic_control_with_dtw,
    synthetic_did,
)
from .src.utils import estimate_ATE, monte_carlo_ATE

warnings.filterwarnings("ignore")


def run_classic_baselines(
    simulation_params,
    nb_simulations=1,
    seed=0,
    njobs=2,
    save_output=True,
    output_dir="results",
    output_filename=None,
    n_units=500,
):
    # Ensure results directory exists
    simulation_params["n_units"] = n_units
    if save_output:
        os.makedirs(output_dir, exist_ok=True)

    methods = [
        did,
        ols,
        synthetic_control,
        elastic_net,
        elastic_net,
        elastic_net,
        synthetic_did,
        synthetic_control_with_dtw,
    ]

    base_params = {
        "id_col": "id",
        "index_col": "time",
        "treat_col": "treatment",
        "outcome_col": "y",
        "intervention_point": simulation_params["treatment_time"],
    }

    func_params = [
        {**base_params, "ratio": None},  # did
        {**base_params, "ratio": None},  # ols
        {**base_params, "ratio": None},  # synthetic_control
        {**base_params, "ratio": 1},  # lasso
        {**base_params, "ratio": 0},  # ridge
        {**base_params, "ratio": 0.5},  # elastic_net
        {**base_params, "ratio": None},  # synthetic_did
        {**base_params, "ratio": None},  # synthetic_control_with_dtw
    ]

    real_ATE = np.mean(monte_carlo_ATE(simulation_params, n_iter=100))

    t1 = time.time()
    output = estimate_ATE(
        methods,
        func_params,
        simulation_params,
        nb_simulations=nb_simulations,
        seed=seed,
        njobs=njobs,
        verbose=True,
    )
    t2 = time.time()

    # print(f"Simulation completed in {t2 - t1:.2f} seconds.")
    # print(f"Real ATE: {real_ATE}")

    if save_output:
        if output_filename is None:
            output_filename = (
                f"att_classic_indiv_bias={simulation_params['bias']}_"
                f"constant_effect={simulation_params['constant_effect']}_"
                f"sigma={simulation_params['sigma']}_alpha={simulation_params['alpha']}.pkl"
            )
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, "wb") as f:
            pickle.dump(output, f)
        print(f"Output saved to {output_path}")

    return output, real_ATE


if __name__ == "__main__":
    run_classic_baselines()
