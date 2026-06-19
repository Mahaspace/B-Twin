import pickle

import numpy as np

import torch

D_TYPE = torch.float32
DEVICE = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")


def load_config(data_path, fold="train"):
    with open(data_path.format(fold, "config", "pkl"), "rb") as f:
        config = pickle.load(file=f)
    n_control_units = config["n_control_units"]
    n_treated_units = config["n_treated_units"]
    n_units = config["n_units"]

    return n_control_units, n_treated_units, n_units


def load_tensor(data_path, fold="train"):
    x_full = torch.load(data_path.format(fold, "x_full", "pth"), map_location=DEVICE)
    t_full = torch.load(data_path.format(fold, "t_full", "pth"), map_location=DEVICE)
    mask_full = torch.load(
        data_path.format(fold, "mask_full", "pth"), map_location=DEVICE
    )
    batch_ind_full = torch.load(
        data_path.format(fold, "batch_ind_full", "pth"), map_location=DEVICE
    )
    y_full = torch.load(data_path.format(fold, "y_full", "pth"), map_location=DEVICE)
    y_control = torch.load(
        data_path.format(fold, "y_control", "pth"), map_location=DEVICE
    )
    y_mask_full = torch.load(
        data_path.format(fold, "y_mask_full", "pth"), map_location=DEVICE
    )
    m = torch.load(data_path.format(fold, "m", "pth"), map_location=DEVICE)
    sd = torch.load(data_path.format(fold, "sd", "pth"), map_location=DEVICE)
    ids_full = torch.load(
        data_path.format(fold, "ids_full", "pth"), map_location=DEVICE
    )
    treatment_effect = torch.load(
        data_path.format(fold, "treatment_effect", "pth"), map_location=DEVICE
    )
    return (
        x_full,
        t_full,
        mask_full,
        batch_ind_full,
        y_full,
        y_control,
        y_mask_full,
        m,
        sd,
        ids_full,
        treatment_effect,
    )


def load_real_tensor(data_path, fold="train"):
    x_full = torch.load(data_path.format(fold, "x_full", "pth"), map_location=DEVICE)
    t_full = torch.load(data_path.format(fold, "t_full", "pth"), map_location=DEVICE)
    mask_full = torch.load(
        data_path.format(fold, "mask_full", "pth"), map_location=DEVICE
    )
    batch_ind_full = torch.load(
        data_path.format(fold, "batch_ind_full", "pth"), map_location=DEVICE
    )
    y_full = torch.load(data_path.format(fold, "y_full", "pth"), map_location=DEVICE)
    y_control = torch.load(
        data_path.format(fold, "y_control", "pth"), map_location=DEVICE
    )
    y_mask_full = torch.load(
        data_path.format(fold, "y_mask_full", "pth"), map_location=DEVICE
    )
    m = torch.load(data_path.format(fold, "m", "pth"), map_location=DEVICE)
    sd = torch.load(data_path.format(fold, "sd", "pth"), map_location=DEVICE)
    ids_full = torch.load(
        data_path.format(fold, "ids_full", "pth"), map_location=DEVICE
    )

    return (
        x_full,
        t_full,
        mask_full,
        batch_ind_full,
        y_full,
        y_control,
        y_mask_full,
        m,
        sd,
        ids_full,
    )


def load_data_dict(version=1):
    if version == 1:
        version = ""
    else:
        version = str(version)

    val_arr1 = np.load("real_data{}/val_arr1".format(version) + ".npy")
    val_mask_arr1 = np.load("real_data{}/val_mask_arr1".format(version) + ".npy")
    ts_arr1 = np.load("real_data{}/ts_arr1".format(version) + ".npy")
    ts_mask_arr1 = np.load("real_data{}/ts_mask_arr1".format(version) + ".npy")
    patid1 = np.load("real_data{}/patid1".format(version) + ".npy")

    val_arr0 = np.load("real_data{}/val_arr0".format(version) + ".npy")
    val_mask_arr0 = np.load("real_data{}/val_mask_arr0".format(version) + ".npy")
    ts_arr0 = np.load("real_data{}/ts_arr0".format(version) + ".npy")
    ts_mask_arr0 = np.load("real_data{}/ts_mask_arr0".format(version) + ".npy")
    patid0 = np.load("real_data{}/patid0".format(version) + ".npy")

    Y0 = np.load("real_data{}/Y0".format(version) + ".npy")
    Y1 = np.load("real_data{}/Y1".format(version) + ".npy")

    data1 = {
        "val_arr": val_arr1,
        "val_mask_arr": val_mask_arr1,
        "ts_arr": ts_arr1,
        "ts_mask_arr": ts_mask_arr1,
        "patid": patid1,
        "Y": Y1,
    }

    data0 = {
        "val_arr": val_arr0,
        "val_mask_arr": val_mask_arr0,
        "ts_arr": ts_arr0,
        "ts_mask_arr": ts_mask_arr0,
        "patid": patid0,
        "Y": Y0,
    }
    return data1, data0


def get_units(d1, d0):
    n_units = d0[0].shape[0]
    n_treated = d1[0].shape[0]
    return n_units, n_treated, n_units + n_treated


def to_tensor(device, dtype, *args):
    return [torch.tensor(x, device=device, dtype=dtype) for x in args]


def get_tensors(d1_train, d0_train, device):
    x_full = np.concatenate([d0_train[0], d1_train[0]], axis=0).transpose((1, 0, 2))
    print(x_full.shape)

    mask_full = np.concatenate([d0_train[1], d1_train[1]], axis=0).transpose((1, 0, 2))
    print(mask_full.shape)

    t_full = np.concatenate([d0_train[2], d1_train[2]], axis=0)[:, :, None]
    t_full = np.tile(t_full, (1, 1, x_full.shape[-1])).transpose((1, 0, 2))
    print(t_full.shape)

    batch_ind_full = np.arange(x_full.shape[1])
    print(batch_ind_full.shape)

    y_full = np.concatenate([d0_train[-1], d1_train[-1]], axis=0).transpose((1, 0, 2))
    print(y_full.shape)

    y_control = d0_train[-1].transpose((1, 0, 2))
    print(y_control.shape)

    y_mask_full = np.ones(y_full.shape[1])
    print(y_mask_full.shape)

    patid_full = np.concatenate([d0_train[-2], d1_train[-2]], axis=0)
    print(patid_full.shape)

    x_full, t_full, mask_full, batch_ind_full, y_full, y_control, y_mask_full = (
        to_tensor(
            device,
            torch.float32,
            x_full,
            t_full,
            mask_full,
            batch_ind_full,
            y_full,
            y_control,
            y_mask_full,
        )
    )

    return (
        x_full,
        t_full,
        mask_full,
        batch_ind_full,
        y_full,
        y_control,
        y_mask_full,
        patid_full,
    )
