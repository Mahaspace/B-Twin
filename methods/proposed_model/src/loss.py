import torch
import torch.nn as nn


class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, pred, target):
        diff = pred - target
        abs_diff = torch.abs(diff)
        mask = (abs_diff < self.delta).float()
        return torch.mean(
            mask * 0.5 * diff**2
            + (1 - mask) * self.delta * (abs_diff - 0.5 * self.delta)
        )


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, pred, target):
        return torch.mean((pred - target) ** 2)


def propensity_L_2_loss_objective(
    weights, p_treated_unit, p_control_units, y_treated, y_control
):
    treatment_probability = p_treated_unit.mean().detach()
    non_treatment_probability = (p_control_units).mean().detach()

    control_weighted_score = weights @ (p_control_units)
    objective = (
        (
            torch.abs(
                ((p_treated_unit.unsqueeze(1)) / treatment_probability)
                - (control_weighted_score / non_treatment_probability)
            )
        )
        ** 2
    ).mean()
    return objective


def propensity_L_1_loss_objective(
    weights, p_treated_unit, p_control_units, y_treated, y_control
):
    treatment_probability = p_treated_unit.mean().detach()

    non_treatment_probability = (p_control_units).mean().detach()

    control_weighted_score = weights @ (p_control_units)
    objective = (
        torch.abs(
            ((p_treated_unit.unsqueeze(1)) / treatment_probability)
            - (control_weighted_score / non_treatment_probability)
        )
    ).mean()
    return objective
