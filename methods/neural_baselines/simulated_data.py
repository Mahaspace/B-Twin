from src.configs import params
from src.experiments import run_many_experiments

if __name__ == "__main__":
    for param in params:
        run_many_experiments(
            n_experiments=1,
            params=param,
            save_dir="/home/J31184/B-Twin-Causal_inference/results/nn_baselines",
            filename="simulated_data_(sim_3)_experiment_ts",
            target="att",
        )
