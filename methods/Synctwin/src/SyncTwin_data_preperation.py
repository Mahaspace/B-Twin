# %%
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch


def create_paths(*args):
    for base_path in args:
        if not os.path.exists(base_path):
            os.makedirs(base_path)


def get_treatment_effect(
    treat_res_arr, treat_counterfactual_arr, train_step, m, sd, device
):
    m = m[-1].item()
    sd = sd[-1].item()

    return (
        torch.tensor(treat_res_arr - treat_counterfactual_arr, device=device)
        .permute((1, 2, 0))[train_step:, :, -1]
        .unsqueeze(-1)
    )


def prepare_data(data, T_0, base_path_data, device, d_type):
    """a fuction that takes as input a data of format
    columns ={'id', 'time','X','temperature', 'y_1', 'y_0', 'treatment', 'features'}
    and prepares it
    """
    data = data.copy()
    data = data[["id", "time", "y_0", "y", "treatment"]]
    n_units = data["id"].unique()

    control_ids = data.loc[data["treatment"] == 0, "id"].unique()
    treated_ids = data.loc[data["treatment"] == 1, "id"].unique()

    control_train_ids, control_val_ids = train_test_split(
        control_ids, test_size=0.2, train_size=0.8
    )
    control_val_ids, control_test_ids = train_test_split(
        control_val_ids, test_size=0.5, train_size=0.5
    )

    treated_train_ids, treated_val_ids = train_test_split(
        treated_ids, test_size=0.2, train_size=0.8
    )
    treated_val_ids, treated_test_ids = train_test_split(
        treated_val_ids, test_size=0.5, train_size=0.5
    )

    control_train = data[(data.id.isin(control_train_ids)) & (data.treatment == 0)]
    control_test = data[(data.id.isin(control_test_ids)) & (data.treatment == 0)]
    control_val = data[(data.id.isin(control_val_ids)) & (data.treatment == 0)]

    treated_train = data[(data.id.isin(treated_train_ids)) & (data.treatment == 1)]
    treated_test = data[(data.id.isin(treated_test_ids)) & (data.treatment == 1)]
    treated_val = data[(data.id.isin(treated_val_ids)) & (data.treatment == 1)]

    train = pd.concat([control_train, treated_train], axis=0, ignore_index=True)
    test = pd.concat([control_test, treated_test], axis=0)
    val = pd.concat([control_val, treated_val], axis=0)

    data_path = base_path_data + "/{}-{}.{}"

    # %%
    fold_name = ["test", "train", "val"]
    for i, fold in enumerate([test, train, val]):
        fold = fold.sort_values(["treatment", "id", "time"])

        n_control_units = fold[fold["treatment"] == 0]["id"].nunique()
        n_treated_units = fold[fold["treatment"] == 1]["id"].nunique()
        n_units = n_control_units + n_treated_units

        fold.drop("treatment", axis=1, inplace=True)

        n_steps = fold["time"].nunique()
        n_features = len(fold.columns) - 2

        data = fold.copy()
        ids_full = data.id.unique()
        ids_full = torch.tensor(ids_full, dtype=d_type).to(device)

        data.drop(["id", "time"], axis=1, inplace=True)
        array = data.values.reshape(n_units, n_steps, n_features)
        tensor = np.transpose(array, (2, 1, 0))  ## (n_features,n_step, n_indiv)
        treat_output = tensor[1, :, n_control_units:].reshape(
            1, n_steps, n_treated_units
        )  ## taking the outputs for the placebo treatment units
        treat_counterfactual = tensor[
            0, :, n_control_units:
        ].reshape(
            1, n_steps, n_treated_units
        )  ## if i was working on a simulation i would change this line of code by adding the real outcome in the feature space
        tensor = tensor[1, :, :].reshape(1, n_steps, n_units)

        covariates = torch.tensor(tensor, dtype=d_type)
        covariates = covariates.permute((1, 2, 0)).to(
            device
        )  # n_steps, n_indiv,n_features

        m = covariates.mean(dim=(0, 1))
        sd = covariates.std(dim=(0, 1))

        # features before treatment
        treatment_time = T_0
        x_full = covariates[:treatment_time, :, :]
        train_step = x_full.shape[0]

        y_full = (
            covariates[treatment_time:, :, -1].detach().clone().unsqueeze(-1)
        )  ## only the outcome of interest after treatment for all individuals
        y_full_all = covariates[
            treatment_time:, :, :
        ]  ## all the features after treatment
        y_control = covariates[treatment_time:, :n_control_units, -1].unsqueeze(
            -1
        )  ## the outcome of interest after treatment for the control units

        t_full = torch.ones_like(x_full)
        mask_full = torch.ones_like(x_full)
        batch_ind_full = torch.arange(n_units).to(device)
        y_mask_full = (batch_ind_full < n_control_units) * 1.0
        Treatment_effect = get_treatment_effect(
            treat_output, treat_counterfactual, train_step, m, sd, device
        )
        X0 = x_full[:, :n_control_units, :]
        X0 = (
            X0.permute((0, 2, 1))
            .reshape(X0.shape[0] * X0.shape[2], X0.shape[1])
            .cpu()
            .numpy()
        )

        X1 = x_full[:, n_control_units:, :]
        X1 = (
            X1.permute((0, 2, 1))
            .reshape(X1.shape[0] * X1.shape[2], X1.shape[1])
            .cpu()
            .numpy()
        )
        # print('X : ',X0.shape, X1.shape)
        Y_control = y_control[:, :, 0].cpu().numpy()
        Y_treated = y_full[:, n_control_units:, 0].cpu().numpy()
        # print('Y:', Y_control.shape, Y_treated.shape)
        Treatment_effect = Treatment_effect[:, :, 0].cpu().numpy()
        np.savetxt(data_path.format(fold_name[i], "X0", "csv"), X0, delimiter=",")
        np.savetxt(data_path.format(fold_name[i], "X1", "csv"), X1, delimiter=",")
        np.savetxt(
            data_path.format(fold_name[i], "Y_control", "csv"), Y_control, delimiter=","
        )
        np.savetxt(
            data_path.format(fold_name[i], "Y_treated", "csv"), Y_treated, delimiter=","
        )
        np.savetxt(
            data_path.format(fold_name[i], "Treatment_effect", "csv"),
            Treatment_effect,
            delimiter=",",
        )

        torch.save(x_full, data_path.format(fold_name[i], "x_full", "pth"))
        torch.save(t_full, data_path.format(fold_name[i], "t_full", "pth"))
        torch.save(mask_full, data_path.format(fold_name[i], "mask_full", "pth"))
        torch.save(
            batch_ind_full, data_path.format(fold_name[i], "batch_ind_full", "pth")
        )
        torch.save(y_full, data_path.format(fold_name[i], "y_full", "pth"))
        torch.save(y_full_all, data_path.format(fold_name[i], "y_full_all", "pth"))
        torch.save(y_control, data_path.format(fold_name[i], "y_control", "pth"))
        torch.save(y_mask_full, data_path.format(fold_name[i], "y_mask_full", "pth"))
        torch.save(m, data_path.format(fold_name[i], "m", "pth"))
        torch.save(sd, data_path.format(fold_name[i], "sd", "pth"))
        torch.save(ids_full, data_path.format(fold_name[i], "ids_full", "pth"))

        torch.save(
            Treatment_effect, data_path.format(fold_name[i], "treatment_effect", "pth")
        )

        config = {
            "n_control_units": n_control_units,
            "n_treated_units": n_treated_units,
            "n_units": n_units,
        }
        with open(data_path.format(fold_name[i], "config", "pkl"), "wb") as f:
            pickle.dump(config, file=f)


def prepare_real_data(data, T_0, base_path_data, device, d_type):
    """a fuction that takes as input a data of format
    columns ={'id', 'time','X','temperature', 'y_1', 'y_0', 'treatment', 'features'}
    and prepares it
    """
    data = data.copy()
    data = data[["id", "time", "y", "treatment"]]
    unique_ids = data["id"].unique()

    id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    data["id"] = data["id"].map(id_mapping)

    control_ids = data.loc[data["treatment"] == 0, "id"].unique()
    treated_ids = data.loc[data["treatment"] == 1, "id"].unique()

    control_train_ids, control_val_ids = train_test_split(
        control_ids, test_size=0.2, train_size=0.8
    )
    control_val_ids, control_test_ids = train_test_split(
        control_val_ids, test_size=0.5, train_size=0.5
    )

    treated_train_ids, treated_val_ids = train_test_split(
        treated_ids, test_size=0.2, train_size=0.8
    )
    treated_val_ids, treated_test_ids = train_test_split(
        treated_val_ids, test_size=0.5, train_size=0.5
    )

    control_train = data[(data.id.isin(control_train_ids)) & (data.treatment == 0)]
    control_test = data[(data.id.isin(control_test_ids)) & (data.treatment == 0)]
    control_val = data[(data.id.isin(control_val_ids)) & (data.treatment == 0)]

    treated_train = data[(data.id.isin(treated_train_ids)) & (data.treatment == 1)]
    treated_test = data[(data.id.isin(treated_test_ids)) & (data.treatment == 1)]
    treated_val = data[(data.id.isin(treated_val_ids)) & (data.treatment == 1)]

    train = pd.concat([control_train, treated_train], axis=0, ignore_index=True)
    test = pd.concat([control_test, treated_test], axis=0)
    val = pd.concat([control_val, treated_val], axis=0)

    data_path = base_path_data + "/{}-{}.{}"

    # %%
    fold_name = ["test", "train", "val"]
    for i, fold in enumerate([test, train, val]):
        fold = fold.sort_values(["treatment", "id", "time"])
        n_steps = fold["time"].nunique()
        n_control_units = int(fold[fold["treatment"] == 0]["id"].count() / n_steps)
        n_treated_units = int(fold[fold["treatment"] == 1]["id"].count() / n_steps)
        n_units = n_control_units + n_treated_units

        fold.drop("treatment", axis=1, inplace=True)

        n_features = len(fold.columns) - 2

        data = fold.copy()
        ids_full = data.id.unique()
        ids_full = torch.tensor(ids_full, dtype=d_type).to(device)

        data.drop(["id", "time"], axis=1, inplace=True)
        array = data.values.reshape(n_units, n_steps, n_features)
        tensor = np.transpose(array, (2, 1, 0))  ## (n_features,n_step, n_indiv)
        treat_output = tensor[0, :, n_control_units:].reshape(
            1, n_steps, n_treated_units
        )  ## taking the outputs for the placebo treatment units

        covariates = torch.tensor(tensor, dtype=d_type)
        covariates = covariates.permute((1, 2, 0)).to(
            device
        )  # n_steps, n_indiv,n_features

        m = covariates.mean(dim=(0, 1))
        sd = covariates.std(dim=(0, 1))

        # features before treatment
        treatment_time = T_0
        x_full = covariates[:treatment_time, :, :]
        train_step = x_full.shape[0]

        y_full = (
            covariates[treatment_time:, :, -1].detach().clone().unsqueeze(-1)
        )  ## only the outcome of interest after treatment for all individuals
        y_full_all = covariates[
            treatment_time:, :, :
        ]  ## all the features after treatment
        y_control = covariates[treatment_time:, :n_control_units, -1].unsqueeze(
            -1
        )  ## the outcome of interest after treatment for the control units

        t_full = torch.ones_like(x_full)
        mask_full = torch.ones_like(x_full)
        batch_ind_full = torch.arange(n_units).to(device)
        y_mask_full = (batch_ind_full < n_control_units) * 1.0

        X0 = x_full[:, :n_control_units, :]
        X0 = (
            X0.permute((0, 2, 1))
            .reshape(X0.shape[0] * X0.shape[2], X0.shape[1])
            .cpu()
            .numpy()
        )

        X1 = x_full[:, n_control_units:, :]
        X1 = (
            X1.permute((0, 2, 1))
            .reshape(X1.shape[0] * X1.shape[2], X1.shape[1])
            .cpu()
            .numpy()
        )
        # print('X : ',X0.shape, X1.shape)
        Y_control = y_control[:, :, 0].cpu().numpy()
        Y_treated = y_full[:, n_control_units:, 0].cpu().numpy()
        # print('Y:', Y_control.shape, Y_treated.shape)

        np.savetxt(data_path.format(fold_name[i], "X0", "csv"), X0, delimiter=",")
        np.savetxt(data_path.format(fold_name[i], "X1", "csv"), X1, delimiter=",")
        np.savetxt(
            data_path.format(fold_name[i], "Y_control", "csv"), Y_control, delimiter=","
        )
        np.savetxt(
            data_path.format(fold_name[i], "Y_treated", "csv"), Y_treated, delimiter=","
        )

        torch.save(x_full, data_path.format(fold_name[i], "x_full", "pth"))
        torch.save(t_full, data_path.format(fold_name[i], "t_full", "pth"))
        torch.save(mask_full, data_path.format(fold_name[i], "mask_full", "pth"))
        torch.save(
            batch_ind_full, data_path.format(fold_name[i], "batch_ind_full", "pth")
        )
        torch.save(y_full, data_path.format(fold_name[i], "y_full", "pth"))
        torch.save(y_full_all, data_path.format(fold_name[i], "y_full_all", "pth"))
        torch.save(y_control, data_path.format(fold_name[i], "y_control", "pth"))
        torch.save(y_mask_full, data_path.format(fold_name[i], "y_mask_full", "pth"))
        torch.save(m, data_path.format(fold_name[i], "m", "pth"))
        torch.save(sd, data_path.format(fold_name[i], "sd", "pth"))
        torch.save(ids_full, data_path.format(fold_name[i], "ids_full", "pth"))

        config = {
            "n_control_units": n_control_units,
            "n_treated_units": n_treated_units,
            "n_units": n_units,
        }
        with open(data_path.format(fold_name[i], "config", "pkl"), "wb") as f:
            pickle.dump(config, file=f)


def prepare_data_test(data, T_0, base_path_data, device, d_type):
    """a fuction that takes as input a data of format
    columns ={'id', 'time','X','temperature', 'y', 'y_0', 'treatment', 'features'}
    and prepares it
    """
    data = data.copy()
    data = data[["id", "time", "y_0", "y", "treatment"]]
    unique_ids = data["id"].unique()
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    data["id"] = data["id"].map(id_mapping)

    control_ids = data.loc[data["treatment"] == 0, "id"].unique()
    treated_ids = data.loc[data["treatment"] == 1, "id"].unique()

    control_test = data[(data.id.isin(control_ids)) & (data.treatment == 0)]

    treated_test = data[(data.id.isin(treated_ids)) & (data.treatment == 1)]

    test = pd.concat([control_test, treated_test], axis=0)

    data_path = base_path_data + "/{}-{}.{}"

    # %%
    fold_name = ["test"]
    for i, fold in enumerate([test]):
        fold = fold.sort_values(["treatment", "id", "time"])

        n_control_units = fold[fold["treatment"] == 0]["id"].nunique()
        n_treated_units = fold[fold["treatment"] == 1]["id"].nunique()
        n_units = n_control_units + n_treated_units

        fold.drop("treatment", axis=1, inplace=True)

        n_steps = fold["time"].nunique()
        n_features = len(fold.columns) - 2

        data = fold.copy()
        ids_full = data.id.unique()
        ids_full = torch.tensor(ids_full, dtype=d_type).to(device)

        data.drop(["id", "time"], axis=1, inplace=True)
        array = data.values.reshape(n_units, n_steps, n_features)
        tensor = np.transpose(array, (2, 1, 0))  ## (n_features,n_step, n_indiv)
        treat_output = tensor[1, :, n_control_units:].reshape(
            1, n_steps, n_treated_units
        )  ## taking the outputs for the placebo treatment units
        treat_counterfactual = tensor[
            0, :, n_control_units:
        ].reshape(
            1, n_steps, n_treated_units
        )  ## if i was working on a simulation i would change this line of code by adding the real outcome in the feature space
        tensor = tensor[1, :, :].reshape(1, n_steps, n_units)

        covariates = torch.tensor(tensor, dtype=d_type)
        covariates = covariates.permute((1, 2, 0)).to(device)
        m = covariates.mean(dim=(0, 1))
        sd = covariates.std(dim=(0, 1))

        # features before treatment
        treatment_time = T_0
        x_full = covariates[:treatment_time, :, :]
        train_step = x_full.shape[0]

        y_full = (
            covariates[treatment_time:, :, -1].detach().clone().unsqueeze(-1)
        )  ## only the outcome of interest after treatment for all individuals
        y_full_all = covariates[
            treatment_time:, :, :
        ]  ## all the features after treatment
        y_control = covariates[treatment_time:, :n_control_units, -1].unsqueeze(
            -1
        )  ## the outcome of interest after treatment for the control units

        t_full = torch.ones_like(x_full)
        mask_full = torch.ones_like(x_full)
        batch_ind_full = torch.arange(n_units).to(device)
        y_mask_full = (batch_ind_full < n_control_units) * 1.0
        Treatment_effect = get_treatment_effect(
            treat_output, treat_counterfactual, train_step, m, sd, device
        )
        X0 = x_full[:, :n_control_units, :]
        X0 = (
            X0.permute((0, 2, 1))
            .reshape(X0.shape[0] * X0.shape[2], X0.shape[1])
            .cpu()
            .numpy()
        )

        X1 = x_full[:, n_control_units:, :]
        X1 = (
            X1.permute((0, 2, 1))
            .reshape(X1.shape[0] * X1.shape[2], X1.shape[1])
            .cpu()
            .numpy()
        )
        # print('X : ',X0.shape, X1.shape)
        Y_control = y_control[:, :, 0].cpu().numpy()
        Y_treated = y_full[:, n_control_units:, 0].cpu().numpy()
        # print('Y:', Y_control.shape, Y_treated.shape)
        Treatment_effect = Treatment_effect[:, :, 0].cpu().numpy()
        np.savetxt(data_path.format(fold_name[i], "X0", "csv"), X0, delimiter=",")
        np.savetxt(data_path.format(fold_name[i], "X1", "csv"), X1, delimiter=",")
        np.savetxt(
            data_path.format(fold_name[i], "Y_control", "csv"), Y_control, delimiter=","
        )
        np.savetxt(
            data_path.format(fold_name[i], "Y_treated", "csv"), Y_treated, delimiter=","
        )
        np.savetxt(
            data_path.format(fold_name[i], "Treatment_effect", "csv"),
            Treatment_effect,
            delimiter=",",
        )

        torch.save(x_full, data_path.format(fold_name[i], "x_full", "pth"))
        torch.save(t_full, data_path.format(fold_name[i], "t_full", "pth"))
        torch.save(mask_full, data_path.format(fold_name[i], "mask_full", "pth"))
        torch.save(
            batch_ind_full, data_path.format(fold_name[i], "batch_ind_full", "pth")
        )
        torch.save(y_full, data_path.format(fold_name[i], "y_full", "pth"))
        torch.save(y_full_all, data_path.format(fold_name[i], "y_full_all", "pth"))
        torch.save(y_control, data_path.format(fold_name[i], "y_control", "pth"))
        torch.save(y_mask_full, data_path.format(fold_name[i], "y_mask_full", "pth"))
        torch.save(m, data_path.format(fold_name[i], "m", "pth"))
        torch.save(sd, data_path.format(fold_name[i], "sd", "pth"))
        torch.save(ids_full, data_path.format(fold_name[i], "ids_full", "pth"))

        torch.save(
            Treatment_effect, data_path.format(fold_name[i], "treatment_effect", "pth")
        )

        config = {
            "n_control_units": n_control_units,
            "n_treated_units": n_treated_units,
            "n_units": n_units,
        }
        with open(data_path.format(fold_name[i], "config", "pkl"), "wb") as f:
            pickle.dump(config, file=f)


def prepare_real_data_test(data, T_0, base_path_data, device, d_type):
    """a fuction that takes as input a data of format
    columns ={'id', 'time','X','temperature', 'y', 'treatment', 'features'}
    and prepares it
    """
    data = data.copy()
    data = data[["id", "time", "y", "treatment"]]

    unique_ids = data["id"].unique()
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    data["id"] = data["id"].map(id_mapping)

    control_ids = data.loc[data["treatment"] == 0, "id"].unique()
    treated_ids = data.loc[data["treatment"] == 1, "id"].unique()

    control_test = data[(data.id.isin(control_ids)) & (data.treatment == 0)]

    treated_test = data[(data.id.isin(treated_ids)) & (data.treatment == 1)]

    test = pd.concat([control_test, treated_test], axis=0)

    data_path = base_path_data + "/{}-{}.{}"

    # %%
    fold_name = ["test"]
    for i, fold in enumerate([test]):
        fold = fold.sort_values(["treatment", "id", "time"])
        n_steps = fold["time"].nunique()
        n_control_units = int(fold[fold["treatment"] == 0]["id"].count() / n_steps)
        n_treated_units = int(fold[fold["treatment"] == 1]["id"].count() / n_steps)
        n_units = n_control_units + n_treated_units

        fold.drop("treatment", axis=1, inplace=True)

        n_features = len(fold.columns) - 2

        data = fold.copy()
        ids_full = data.id.unique()
        ids_full = torch.tensor(ids_full, dtype=d_type).to(device)

        data.drop(["id", "time"], axis=1, inplace=True)
        array = data.values.reshape(n_units, n_steps, n_features)
        tensor = np.transpose(array, (2, 1, 0))  ## (n_features,n_step, n_indiv)

        covariates = torch.tensor(tensor, dtype=d_type)
        covariates = covariates.permute((1, 2, 0)).to(device)
        m = covariates.mean(dim=(0, 1))
        sd = covariates.std(dim=(0, 1))

        # features before treatment
        treatment_time = T_0
        x_full = covariates[:treatment_time, :, :]
        train_step = x_full.shape[0]

        y_full = (
            covariates[treatment_time:, :, -1].detach().clone().unsqueeze(-1)
        )  ## only the outcome of interest after treatment for all individuals
        y_full_all = covariates[
            treatment_time:, :, :
        ]  ## all the features after treatment
        y_control = covariates[treatment_time:, :n_control_units, -1].unsqueeze(
            -1
        )  ## the outcome of interest after treatment for the control units

        t_full = torch.ones_like(x_full)
        mask_full = torch.ones_like(x_full)
        batch_ind_full = torch.arange(n_units).to(device)
        y_mask_full = (batch_ind_full < n_control_units) * 1.0

        X0 = x_full[:, :n_control_units, :]
        X0 = (
            X0.permute((0, 2, 1))
            .reshape(X0.shape[0] * X0.shape[2], X0.shape[1])
            .cpu()
            .numpy()
        )

        X1 = x_full[:, n_control_units:, :]
        X1 = (
            X1.permute((0, 2, 1))
            .reshape(X1.shape[0] * X1.shape[2], X1.shape[1])
            .cpu()
            .numpy()
        )
        # print('X : ',X0.shape, X1.shape)
        Y_control = y_control[:, :, 0].cpu().numpy()
        Y_treated = y_full[:, n_control_units:, 0].cpu().numpy()
        # print('Y:', Y_control.shape, Y_treated.shape)

        np.savetxt(data_path.format(fold_name[i], "X0", "csv"), X0, delimiter=",")
        np.savetxt(data_path.format(fold_name[i], "X1", "csv"), X1, delimiter=",")
        np.savetxt(
            data_path.format(fold_name[i], "Y_control", "csv"), Y_control, delimiter=","
        )
        np.savetxt(
            data_path.format(fold_name[i], "Y_treated", "csv"), Y_treated, delimiter=","
        )

        torch.save(x_full, data_path.format(fold_name[i], "x_full", "pth"))
        torch.save(t_full, data_path.format(fold_name[i], "t_full", "pth"))
        torch.save(mask_full, data_path.format(fold_name[i], "mask_full", "pth"))
        torch.save(
            batch_ind_full, data_path.format(fold_name[i], "batch_ind_full", "pth")
        )
        torch.save(y_full, data_path.format(fold_name[i], "y_full", "pth"))
        torch.save(y_full_all, data_path.format(fold_name[i], "y_full_all", "pth"))
        torch.save(y_control, data_path.format(fold_name[i], "y_control", "pth"))
        torch.save(y_mask_full, data_path.format(fold_name[i], "y_mask_full", "pth"))
        torch.save(m, data_path.format(fold_name[i], "m", "pth"))
        torch.save(sd, data_path.format(fold_name[i], "sd", "pth"))
        torch.save(ids_full, data_path.format(fold_name[i], "ids_full", "pth"))

        config = {
            "n_control_units": n_control_units,
            "n_treated_units": n_treated_units,
            "n_units": n_units,
        }
        with open(data_path.format(fold_name[i], "config", "pkl"), "wb") as f:
            pickle.dump(config, file=f)
