import pandas as pd
from src.bootstrap_full_training import run_att_bootstrap

synctwin_params = {
    "seed": "100",
    "model_id": "",
    "itr": "1",
    "itr_pretrain": "1000",
    "itr_fine_tune": "1000",
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
n_bootsraps = 1
tau = 0
if __name__ == "__main__":
    b_twin_parameters = {
        "batch_size": 256 * 3,
        "train_epochs": 10,
        "weight_epochs": 1000,
        "lr1": 1e-2,
        "lr2": 1e-4,
        "lambda_": 0.2,
        "L1": 1,
        "embedding_dim": 5,
        "hidden_dim": 10,
        "beta": 0.005,
        "gamma": 0.5,
        "alpha": 1,
        "epsilon": 1e-3,
        "propensity_hidden_dims": [50, 40, 30, 20, 10],
        "add_test_split": False,
    }
    data_path = "./data/mimic/processed_mimic_hourly_placebo_data_age_confounder_icu_stay_72.csv"
    results_path = "./results/bootstrap/"
    data_name = "mimic_iii"
    ## read data
    data = pd.read_csv(data_path)
    ## your data should contain the following columns: ['id', 'time', 'treatment', 'y']
    df = data[["id", "time", "y", "treatment"]]
    data_params = {
        "treatment_time": 60,
        "n_features": 1,
        "n_time": 72,
    }  ## specify treatment time, number of features and number of time points
    ## add tau to data for treaed after treatment
    mask = (df["treatment"] == 1) & (df["time"] >= data_params["treatment_time"])
    df.loc[mask, "y"] += tau

    att_estimates = run_att_bootstrap(
        data=df,
        n_bootsraps=n_bootsraps,
        data_params=data_params,
        b_twin_model_params=b_twin_parameters,
        synctwin_model_params=synctwin_params,
    )

    ## save results
    att_estimates.to_csv(
        results_path
        + f"{data_name}_att_estimates_{n_bootsraps}_tau_{tau}_re_training.csv",
        index=False,
    )

    print(f"finished bootstrap estimation {data_name}")

    # ## run on edf placebo

    b_twin_parameters = {
        "batch_size": 256 * 3,
        "train_epochs": 100,
        "weight_epochs": 5000,
        "lr1": 1e-2,
        "lr2": 1e-4,
        "lambda_": 0.7,
        "L1": 1,
        "embedding_dim": 32,
        "hidden_dim": 64,
        "beta": 0.05,
        "gamma": 0.1,
        "alpha": 1,
        "epsilon": 1e-3,
        "propensity_hidden_dims": [50, 40, 30, 20, 10],
        "add_test_split": False,
    }

    data_path = "./data/real_data/placebo_dataframe.csv"
    results_path = "./results/bootstrap/"
    data_name = "sowee_placebo"
    ## read data
    data = pd.read_csv(data_path)
    ## your data should contain the following columns: ['id', 'time', 'treatment', 'y']
    df = data[["id", "time", "y", "treatment"]]
    # map time to integer
    df["time"]
    data_params = {
        "treatment_time": 151,
        "n_features": 1,
        "n_time": 302,
    }  ## specify treatment time, number of features and number of time points
    mask = (df["treatment"] == 1) & (df["time"] >= "2022-11-01")
    df.loc[mask, "y"] += tau

    att_estimates = run_att_bootstrap(
        data=df,
        n_bootsraps=n_bootsraps,
        data_params=data_params,
        b_twin_model_params=b_twin_parameters,
        synctwin_model_params=synctwin_params,
    )

    ## save results
    att_estimates.to_csv(
        results_path
        + f"{data_name}_att_estimates_{n_bootsraps}_tau_{tau}_re_training.csv",
        index=False,
    )

    print(f"finished bootstrap estimation {data_name}")

    ## run on edf real

    data_path = "./data/real_data/treatment_dataframe.csv"
    results_path = "./results/bootstrap/"
    data_name = "sowee_real"
    ## read data
    data = pd.read_csv(data_path)
    ## your data should contain the following columns: ['id', 'time', 'treatment', 'y']
    df = data[["id", "time", "y", "treatment"]]
    data_params = {
        "treatment_time": 151,
        "n_features": 1,
        "n_time": 302,
    }  ## specify treatment time, number of features and number of time points

    att_estimates = run_att_bootstrap(
        data=df,
        n_bootsraps=n_bootsraps,
        data_params=data_params,
        b_twin_model_params=b_twin_parameters,
        synctwin_model_params=synctwin_params,
    )

    ## save results
    att_estimates.to_csv(
        results_path + f"{data_name}_att_estimates_{n_bootsraps}_re_training.csv",
        index=False,
    )
    ## save real

    print(f"finished bootstrap estimation {data_name}")
