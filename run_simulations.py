import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from methods.classic_baselines import main as classic_baselines
from methods.neural_baselines.src.experiments import (
    run_many_experiments as neural_baselines,
)
from methods.proposed_model import B_Twin, B_Twin_O
from methods.simulations.config import params
from methods.Synctwin import main as synctwin
from methods.Synctwin import montecarlo_ate

LOG_FILE = "main_script.log"

logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = "./results"
FIGURES_DIR = "./figures"


os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def run_all_models(n_iter=1, params=None):
    logger.info(f"=== Starting run with params: {params} ===")

    logger.info("[RUN] classic_baselines")
    classic_baselines.run_classic_baselines(
        nb_simulations=n_iter,
        output_dir=RESULTS_DIR,
        simulation_params=params.copy(),
        njobs=1,
        n_units=500,
    )

    for model in [B_Twin, B_Twin_O]:
        # for model in [B_Twin]:
        logger.info(f"[RUN] {model.__name__}")
        model.main(
            params=params.copy(),
            n_experiments=n_iter,
            weight_epochs=5000,
            train_epochs=100,
            save_dir=RESULTS_DIR,
            n_units_exp=500,
            lr1=1e-3,
        )
    logger.info("[RUN] Synctwin")

    synctwin_trained = synctwin.run_synctwin(
        simulation_cfg=params.copy(),
        model_dir="./methods/Synctwin/models",
        data_dir="./methods/Synctwin/data",
    )

    montecarlo_ate.estimate_average_treatment_effect(
        trained=synctwin_trained,
        simulation_cfg=params.copy(),
        data_dir="./methods/Synctwin/data",
        n_iter=n_iter,
        save_path=RESULTS_DIR,
        n_units=500,
    )
    logger.info("[RUN] Tarnet Cfrenet Dragonnet BCAUS")
    neural_baselines(
        n_experiments=n_iter,
        params=params.copy(),
        save_dir=RESULTS_DIR,
        filename="simulated_data_(sim_3)_experiment_ts",
        target="att",
        n_units=500,
    )
    print("done")
    logger.info(f"=== Finished run with params: {params} ===")


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_pickle_results(params):
    with open(
        f"results/att_classic_bias={params['bias']}_constant_effect={params['constant_effect']}_sigma={params['sigma']}_alpha={params['alpha']}.pkl",
        "rb",
    ) as f:
        output = pickle.load(f)

    with open(
        f"results/B-Twin_att_results_bias={params['bias']}_constant_effect={params['constant_effect']}_sigma={params['sigma']}_alpha={params['alpha']}.pkl",
        "rb",
    ) as f:
        att_B_twin = pickle.load(f)

    with open(
        f"results/B-Twin_C1_att_results_bias={params['bias']}_constant_effect={params['constant_effect']}_sigma={params['sigma']}_alpha={params['alpha']}.pkl",
        "rb",
    ) as f:
        att_B_twin_C1 = pickle.load(f)

    with open(
        f"results/B-Twin_C2_att_results_bias={params['bias']}_constant_effect={params['constant_effect']}_sigma={params['sigma']}_alpha={params['alpha']}.pkl",
        "rb",
    ) as f:
        att_B_twin_C2 = pickle.load(f)

    with open(
        f"results/B-Twin_O_att_results_bias={params['bias']}_constant_effect={params['constant_effect']}_sigma={params['sigma']}_alpha={params['alpha']}.pkl",
        "rb",
    ) as f:
        att_B_twin_O = pickle.load(f)
    with open(
        f"./results/att_synctwin_with_bias={params['bias']}_constant_effect={params['constant_effect']}_sigma={params['sigma']}_alpha={params['alpha']}.pkl",
        "rb",
    ) as f:
        att_Synctwin = pickle.load(f)

    att_classic = output[:, 0, :]
    return (
        att_classic,
        att_B_twin,
        att_B_twin_C1,
        att_B_twin_C2,
        att_B_twin_O,
        att_Synctwin,
    )


def tensor_list_to_mean_numpy(tensor_list):
    return [np.mean(t.cpu().detach().numpy()) for t in tensor_list]


def plot_results(att_classic, att_B, att_C1, att_C2, att_O, att_synctwin, params):
    methods_name = [
        "B-Twin-C1",
        "B-Twin-C2",
        "B-Twin-O",
        "B-Twin",
        "Synctwin",
        "DID",
        "OLS",
        "SC",
        "Lasso",
        "Ridge",
        "Elastic Net",
        "Synthetic DID",
        "Synthetic DTW",
    ]

    methods_data = {
        methods_name[0]: att_C1,
        methods_name[1]: att_C2,
        methods_name[2]: att_O,
        methods_name[3]: att_B,
        methods_name[4]: att_synctwin,
        **{methods_name[i + 5]: att_classic[:, i] for i in range(att_classic.shape[1])},
    }

    df = pd.DataFrame(methods_data)

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df)
    plt.xticks(rotation=45, fontsize=10)
    plt.axhline(0.77 * 2, color="red", linestyle="--", label="True ATE")
    plt.ylabel("Estimated ATT")
    plt.title("Model Comparison")
    plt.tight_layout()

    save_path = os.path.join(
        FIGURES_DIR,
        f"att_comparison_boxplot_simulation_params(sim_3)_bias={params['bias']}_constant_effect={params['constant_effect']}_sigma={params['sigma']}_alpha={params['alpha']}.png",
    )
    plt.savefig(save_path)
    print(f"[✓] Plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    # Step 1: Run all experiments : n_iter is the number of montecarlo runs
    for param in params:
        logger.info(f"--- Starting full pipeline for params: {param} ---")
        run_all_models(n_iter=100, params=param)

        # # Step 2: Load results
        logger.info(f"[LOAD] Loading results for params {param}")

        att_classic, att_B, att_C1, att_C2, att_O, att_synctwin = load_pickle_results(
            params=param
        )
        # # Step 3: Convert to mean ATT values
        att_B = tensor_list_to_mean_numpy(list(att_B))
        att_C1 = tensor_list_to_mean_numpy(list(att_C1))
        att_C2 = tensor_list_to_mean_numpy(list(att_C2))
        att_O = tensor_list_to_mean_numpy(list(att_O))

        # Step 4: Plot results
        logger.info(f"[PLOT] Generating comparison plot for params {param}")
        plot_results(
            att_classic, att_B, att_C1, att_C2, att_O, att_synctwin, params=param
        )
