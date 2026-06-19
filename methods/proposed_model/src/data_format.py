## bloc for model training
import os
import random

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset


## create a custom dataset where X_r and X_f don't have the same size
## orignally i used a mesk because i wanted to predict Y_f from X_f for control units
## this need to be fixed later
class custom_dataset(Dataset):
    def __init__(self, X_r, X_f, y_f, propensity_score, n_control_units):
        self.X_r = X_r
        self.X_f = X_f
        self.y_f = y_f
        self.propensity_score = propensity_score
        self.n_control_units = n_control_units

        self.n_units = len(X_r)
        ## X_r shape : n_steps; n_units, n_features

        self.padded_X_f = torch.zeros((X_r.shape[0], X_f.shape[1], X_f.shape[2]))
        self.padded_X_f[: self.X_f.shape[0], :, :] = X_f

        self.padded_y_f = torch.zeros((X_r.shape[0], y_f.shape[1], 1))
        self.padded_y_f[: self.y_f.shape[0], :, :] = y_f

        self.mask = torch.zeros(self.n_units)
        self.mask[: self.n_control_units] = 1

    def __len__(self):
        return self.n_units

    def __getitem__(self, idx):
        return (
            self.X_r[idx],
            self.padded_X_f[idx],
            self.padded_y_f[idx],
            self.mask[idx],
            self.propensity_score[idx],
        )


class custom_real_dataset(Dataset):
    def __init__(self, X_r, X_f, y_f, n_control_units):
        self.X_r = X_r
        self.X_f = X_f
        self.y_f = y_f

        self.n_control_units = n_control_units

        self.n_units = len(X_r)
        ## X_r shape : n_steps; n_units, n_features

        self.padded_X_f = torch.zeros((X_r.shape[0], X_f.shape[1], X_f.shape[2]))
        self.padded_X_f[: self.X_f.shape[0], :, :] = X_f

        self.padded_y_f = torch.zeros((X_r.shape[0], y_f.shape[1], 1))
        self.padded_y_f[: self.y_f.shape[0], :, :] = y_f

        self.mask = torch.zeros(self.n_units)
        self.mask[: self.n_control_units] = 1

    def __len__(self):
        return self.n_units

    def __getitem__(self, idx):
        return (
            self.X_r[idx],
            self.padded_X_f[idx],
            self.padded_y_f[idx],
            self.mask[idx],
        )


def create_paths(*args):
    for base_path in args:
        if not os.path.exists(base_path):
            os.makedirs(base_path)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(datasets, configs, split, batch_size=256, seed=42, drop_last=True):
    dataset = custom_dataset(
        **datasets[split], n_control_units=configs[split]["n_control_units"]
    )

    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=0,
        drop_last=drop_last,
    )


def make_loader_from_real_data(
    datasets, configs, split, batch_size=256, seed=42, drop_last=True
):
    dataset = custom_real_dataset(
        **datasets[split], n_control_units=configs[split]["n_control_units"]
    )

    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=0,
        drop_last=drop_last,
    )
