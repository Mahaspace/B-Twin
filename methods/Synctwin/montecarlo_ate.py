import os
import pickle
import sys

import numpy as np

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from methods.simulations.simulation import generate_simulation
from methods.Synctwin.main_new import predict_att_synctwin, run_synctwin
from methods.Synctwin.src.config import (
    params_exp,
)


def estimate_average_treatment_effect(
    trained,
    data_dir,
    save_path: str = "results/test/",
    simulation_cfg=None,
    model_index: int = 0,
    n_iter: int = 100,
    n_units: int = 500,
    re_train=True,
    n_iter_re_train=1,
):
    # Load experiment parameters
    seed = int(params_exp["seed"])
    itr_fine_tune = int(params_exp["itr_fine_tune"])
    lam_recon = float(params_exp["lam_recon"])
    lam_prognostic = float(params_exp["lam_prognostic"])
    n_hidden = int(params_exp["n_hidden"])
    tau = float(params_exp["tau"])
    pretrain_Y = params_exp["pretrain_Y"] == "True"
    linear_decoder = params_exp["linear_decoder"] == "True"
    regular = params_exp["regular"] == "True"

    # Seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    # print(f"Running experiment with seed {seed}")

    # Adjust simulation size
    simulation_cfg["n_units"] = n_units
    avg_treatment_effect_est_list = []

    for i in range(n_iter):
        effect_est = predict_att_synctwin(
            trained=trained,
            new_data=generate_simulation(**simulation_cfg),
            data_params=simulation_cfg,
            base_path_data=data_dir,
            re_train=re_train,
            n_iter_re_train=n_iter_re_train,
        )

        avg_effect = torch.mean(effect_est).item()
        avg_treatment_effect_est_list.append(avg_effect)

    filename = f"att_synctwin_with_bias={simulation_cfg['bias']}_constant_effect={simulation_cfg['constant_effect']}_sigma={simulation_cfg['sigma']}_alpha={simulation_cfg['alpha']}.pkl"
    save_path = os.path.join(save_path, filename)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(avg_treatment_effect_est_list, f)
    print(f"Saved estimated ATEs to {save_path}")

    return avg_treatment_effect_est_list


if __name__ == "__main__":
    params = {
        "seed": "100",
        "model_id": "",
        "itr": "1",
        "itr_pretrain": "50",
        "itr_fine_tune": "20",
        "batch_size": "100",
        "pretrain_Y": "True",
        "reduced_fine_tune": "True",
        "linear_decoder": "False",
        "lam_prognostic": "1",
        "lam_recon": "1",
        "tau": "1",
        "n_hidden": "20",
        "sim_id": None,
        "regular": "True",
    }

    simulation_params = {
        "n_units": 100,
        "n_time": 168,
        "treatment_time": 84,
        "phi": 0.8,
        "sigma": 5,
        "alpha": 0.05,
        "seed": None,
        "bias": True,
        "constant_effect": True,
    }

    synctwin = run_synctwin(params_exp=params, simulation_cfg=simulation_params)
    att_list = estimate_average_treatment_effect(
        trained=synctwin,
        data_dir="./data/",
        simulation_cfg=simulation_params,
        n_iter=1,
        n_units=100,
    )
