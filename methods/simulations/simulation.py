import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def treatment_probability(w_t):
    """Sigmoid treatment probability."""
    return 1 / (1 + np.exp(-5 * (w_t - 0.5)))


def outcome(x, W, t, alpha_=0.05, beta=0.2):
    group_effect = x * 0.5 * (W**2) + alpha_ * t * (
        W > 0.5
    )  # Time trend for specific groups
    return group_effect


def tau(w_t, treat, constant_effect=True):
    if constant_effect:
        return 2 * 0.77 * treat
    else:
        return 4 * np.log(1 + w_t) * treat


def generate_simulation(
    n_units=500,
    n_time=168 * 3,
    treatment_time=84 * 3,
    phi=0.8,
    sigma=5.0,
    alpha=0.05,
    seed=None,
    bias=True,
    constant_effect=True,
):
    """
    Generates a time series simulation where untreated outcomes follow an AR(1) process and
    treated outcomes are transformed using a highly nonlinear function.
    """
    if seed is not None:
        np.random.seed(seed)  # For reproducibility

    W = np.random.uniform(0, 1, n_units)  # Hidden variables for treatment effect

    AR_1 = np.zeros(n_time)
    Outcome = np.zeros((n_units, n_time))

    for t in range(1, n_time):
        AR_1[t] = phi * AR_1[t - 1] + np.random.normal(0, sigma)

    for i in range(n_units):
        for t in range(n_time):
            Outcome[i, t] = outcome(AR_1[t], W[i], t, alpha_=alpha) + np.random.normal(
                0, sigma
            )
    if bias:
        p = treatment_probability(W)
    else:
        p = np.ones(n_units) * 0.5

    treatment = np.random.binomial(1, p)

    # Generate outcomes
    Y_0 = np.zeros((n_units, n_time))
    Y_1 = np.zeros((n_units, n_time))
    Y = np.zeros((n_units, n_time))

    for i in range(n_units):
        for t in range(n_time):
            # if W[i] > 0.5:
            #     Y_0[i, t] = Outcome[i, t]
            # else:
            Y_0[i, t] = Outcome[i, t]
            Y_1[i, t] = Y_0[i, t] + tau(W[i], (t >= treatment_time), constant_effect)
            Y[i, t] = treatment[i] * (Y_1[i, t]) + (1 - treatment[i]) * (Y_0[i, t])

    # Create DataFrame
    time = np.arange(n_time)
    df = pd.DataFrame(
        {
            "id": np.repeat(np.arange(n_units), n_time),
            "time": np.tile(time, n_units),
            "y": Y.flatten(),
            "w": np.repeat(W, n_time),
            "y_0": Y_0.flatten(),
            "y_1": Y_1.flatten(),
            "treatment": np.repeat(treatment, n_time),
            "propensity_score": np.repeat(p, n_time),
        }
    )
    return df
