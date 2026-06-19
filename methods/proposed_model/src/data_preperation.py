# %%
# Chargement des packages
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch


def create_paths(*args):
    for base_path in args:
        if not os.path.exists(base_path):
            os.makedirs(base_path)


def subset(data, ids, treat):
    return data[(data.id.isin(ids)) & (data["treatment"] == treat)]


def prepare_data(data, params, add_test_split=True):
    data = data[["id", "time", "y_0", "y", "treatment", "propensity_score"]]

    control_ids = data.loc[data["treatment"] == 0, "id"].unique()
    treated_ids = data.loc[data["treatment"] == 1, "id"].unique()

    control_train_ids, control_val_ids = train_test_split(
        control_ids, test_size=0.2, train_size=0.8
    )
    treated_train_ids, treated_val_ids = train_test_split(
        treated_ids, test_size=0.2, train_size=0.8
    )
    if add_test_split:
        control_val_ids, control_test_ids = train_test_split(
            control_val_ids, test_size=0.5, train_size=0.5
        )
        treated_val_ids, treated_test_ids = train_test_split(
            treated_val_ids, test_size=0.5, train_size=0.5
        )

    train = pd.concat(
        [subset(data, control_train_ids, 0), subset(data, treated_train_ids, 1)]
    )
    val = pd.concat(
        [subset(data, control_val_ids, 0), subset(data, treated_val_ids, 1)]
    )
    if add_test_split:
        test = pd.concat(
            [subset(data, control_test_ids, 0), subset(data, treated_test_ids, 1)]
        )
        folds = [train, val, test]
        fold_names = ["train", "val", "test"]
    else:
        folds = [train, val]
        fold_names = ["train", "val"]
    datasets, configs = process_dataset(folds, fold_names, params)
    return datasets, configs


## new
def process_dataset(
    folds, fold_names, params, d_type=torch.float32, counterfactual=False
):
    """Prepares dataset tensors for model training."""
    datasets = {}
    configs = {}
    for i, fold in enumerate(folds):
        fold = fold.sort_values(["treatment", "id", "time"])
        n_steps = fold["time"].nunique()
        n_control_units = int(fold[fold["treatment"] == 0]["id"].count() / n_steps)
        n_treated_units = int(fold[fold["treatment"] == 1]["id"].count() / n_steps)
        n_units = n_control_units + n_treated_units
        fold.drop("treatment", axis=1, inplace=True)
        treatment_time = params["treatment_time"]
        n_steps = fold["time"].nunique()
        # assert n_steps == params["n_time"]
        n_features = len(fold.columns) - 2
        # assert n_features == 3

        array = fold.drop(["id", "time"], axis=1).values.reshape(
            n_units, n_steps, n_features
        )
        data = np.transpose(array, (2, 1, 0))  # (n_features, n_steps, n_indiv)
        treatment_time = params["treatment_time"]

        if counterfactual:
            data = torch.tensor(data, dtype=torch.float32)
            array = data.permute((2, 1, 0))

            propensity_score = array[:, 0, -1]
            array = array[:, :, :-1]
            y_counterfactual = (
                array[:, treatment_time:, 0]
                .reshape(n_units, n_steps - treatment_time, 1)
                .clone()
            )
            array = array[:, :, 1:]
            X_r = array[:, :treatment_time, :].clone()
            X_f = array[:, treatment_time:, :-1].clone()
            y_f = (
                array[:, treatment_time:, -1]
                .reshape(n_units, n_steps - treatment_time, 1)
                .clone()
            )
            datasets[fold_names[i]] = {
                "X_r": X_r,
                "X_f": X_f,
                "y_f": y_f,
                "propensity_score": propensity_score,
                "y_counterfactual": y_counterfactual,
            }
        else:
            data = data[1:, :, :]
            data = torch.tensor(data, dtype=d_type)
            array = data.permute((2, 1, 0))

            propensity_score = array[:, 0, -1]
            array = array[:, :, :-1]
            X_r = array[:, :treatment_time, :]
            X_f = array[:, treatment_time:, :-1]
            y_f = array[:, treatment_time:, -1].reshape(
                n_units, n_steps - treatment_time, 1
            )

            datasets[fold_names[i]] = {
                "X_r": X_r,
                "X_f": X_f,
                "y_f": y_f,
                "propensity_score": propensity_score,
            }
        configs[fold_names[i]] = {
            "n_control_units": n_control_units,
            "n_treated_units": n_treated_units,
            "n_units": n_units,
            "treatment_time": treatment_time,
        }
    return datasets, configs


def prepare_real_data(data, params, add_test_split=True):
    data = data[["id", "time", "y", "treatment"]]

    control_ids = data.loc[data["treatment"] == 0, "id"].unique()
    treated_ids = data.loc[data["treatment"] == 1, "id"].unique()

    control_train_ids, control_val_ids = train_test_split(
        control_ids, test_size=0.2, train_size=0.8
    )
    treated_train_ids, treated_val_ids = train_test_split(
        treated_ids, test_size=0.2, train_size=0.8
    )
    if add_test_split:
        control_val_ids, control_test_ids = train_test_split(
            control_val_ids, test_size=0.5, train_size=0.5
        )
        treated_val_ids, treated_test_ids = train_test_split(
            treated_val_ids, test_size=0.5, train_size=0.5
        )

    train = pd.concat(
        [subset(data, control_train_ids, 0), subset(data, treated_train_ids, 1)]
    )
    val = pd.concat(
        [subset(data, control_val_ids, 0), subset(data, treated_val_ids, 1)]
    )
    if add_test_split:
        test = pd.concat(
            [subset(data, control_test_ids, 0), subset(data, treated_test_ids, 1)]
        )
        folds = [train, val, test]
        fold_names = ["train", "val", "test"]
    else:
        folds = [train, val]
        fold_names = ["train", "val"]
    datasets, configs = process_real_dataset(folds, fold_names, params)
    return datasets, configs


def process_real_dataset(
    folds,
    fold_names,
    params,
    d_type=torch.float32,
):
    """Prepares dataset tensors for model training."""
    datasets = {}
    configs = {}
    for i, fold in enumerate(folds):
        fold = fold.sort_values(["treatment", "id", "time"])
        n_steps = fold["time"].nunique()
        n_control_units = int(fold[fold["treatment"] == 0]["id"].count() / n_steps)
        n_treated_units = int(fold[fold["treatment"] == 1]["id"].count() / n_steps)
        n_units = n_control_units + n_treated_units
        fold.drop("treatment", axis=1, inplace=True)
        treatment_time = params["treatment_time"]
        n_steps = fold["time"].nunique()
        # assert n_steps == params["n_time"]
        n_features = len(fold.columns) - 2
        # assert n_features == 3

        array = fold.drop(["id", "time"], axis=1).values.reshape(
            n_units, n_steps, n_features
        )
        data = np.transpose(array, (2, 1, 0))  # (n_features, n_steps, n_indiv)
        treatment_time = params["treatment_time"]

        data = torch.tensor(data, dtype=d_type)
        array = data.permute((2, 1, 0))

        X_r = array[:, :treatment_time, :]
        X_f = array[:, treatment_time:, :-1]
        y_f = array[:, treatment_time:, -1].reshape(
            n_units, n_steps - treatment_time, 1
        )

        datasets[fold_names[i]] = {
            "X_r": X_r,
            "X_f": X_f,
            "y_f": y_f,
        }
        configs[fold_names[i]] = {
            "n_control_units": n_control_units,
            "n_treated_units": n_treated_units,
            "n_units": n_units,
            "treatment_time": treatment_time,
        }
    return datasets, configs
