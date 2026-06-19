import warnings

import cvxpy as cp
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from dtaidistance import dtw
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")


def did(
    data, index_col, treat_col, outcome_col, intervention_point, id_col=None, ratio=None
):
    pre_treated = (
        data[(data[treat_col] == 1) & (data[index_col] < intervention_point)]
        .groupby(index_col)[outcome_col]
        .mean()
    )
    post_treated = (
        data[(data[treat_col] == 1) & (data[index_col] >= intervention_point)]
        .groupby(index_col)[outcome_col]
        .mean()
    )
    pre_control = (
        data[(data[treat_col] == 0) & (data[index_col] < intervention_point)]
        .groupby(index_col)[outcome_col]
        .mean()
    )
    post_control = (
        data[(data[treat_col] == 0) & (data[index_col] >= intervention_point)]
        .groupby(index_col)[outcome_col]
        .mean()
    )

    diff1 = (post_treated.values).mean() - (pre_treated.values).mean()
    diff2 = (post_control.values).mean() - (pre_control.values).mean()
    estimated_ate = diff1 - diff2
    return estimated_ate


def ols(
    data,
    id_col,
    index_col,
    treat_col,
    outcome_col,
    intervention_point,
    ratio=None,
    model=False,
):
    mask = data[index_col] >= intervention_point
    data["post_treatment"] = 0
    data.loc[mask, "post_treatment"] = 1
    data.set_index([index_col, id_col], inplace=True)
    Model = smf.ols(f"{outcome_col} ~ post_treatment*{treat_col}", data=data).fit()
    if model:
        return Model
    return Model.params[f"post_treatment:{treat_col}"]


def fit_unit_weights_synthetic_control(
    data, outcome_col, index_col, id_col, treat_col, intervention_point
):
    pre_data = data.loc[data[index_col] < intervention_point, :]

    y_pre_control = pre_data[pre_data[treat_col] == 0].pivot(
        index=index_col, columns=id_col, values=outcome_col
    )

    y_pre_treat_mean = (
        pre_data[pre_data[treat_col] == 1].groupby(index_col)[outcome_col].mean()
    )

    X = y_pre_control.values
    # Estimation des poids
    w = cp.Variable(X.shape[1])
    # objective = cp.Minimize(cp.sum_squares(X@w - y_pre_treat_mean.values) + T_pre*zeta**2 * cp.sum_squares(w[1:]))
    objective = cp.Minimize(cp.sum_squares(X @ w - y_pre_treat_mean.values))
    constraints = [cp.sum(w[:]) == 1, w[:] >= 1e-5]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, eps=1e-9, verbose=False)

    # print("Intercept:", w.value[0])
    return pd.Series(
        w.value[:],  # remove intercept
        name="unit_weights",
        index=y_pre_control.columns,
    )

    # joint les poids des périodes et des contrôles


def join_weights(data, unit_w, index_col, id_col, treat_col, intervention_point):
    data = data.copy()
    data.loc[data[index_col] >= intervention_point, "post_traitement"] = 1
    data.loc[data[index_col] < intervention_point, "post_traitement"] = 0
    data = (
        data.set_index([index_col, id_col])
        .join(unit_w)
        .reset_index()
        .fillna(
            {unit_w.name: 1 / data.loc[data[treat_col] == 1]["id"].nunique()}
        )  # unit_w.name: data[treat_col].mean()}) ## on remplace par la proportion des traités
        .assign(**{"weights": lambda d: d[unit_w.name].round(10)})
        .astype({treat_col: int})
    )

    return data


def synthetic_control(
    data,
    index_col,
    id_col,
    treat_col,
    outcome_col,
    intervention_point,
    ratio=None,
    model=False,
):
    unit_weights = fit_unit_weights_synthetic_control(
        data, outcome_col, index_col, id_col, treat_col, intervention_point
    )
    did_data = join_weights(
        data,
        unit_weights,
        index_col=index_col,
        id_col=id_col,
        treat_col=treat_col,
        intervention_point=intervention_point,
    )

    weighted_outcome = "weighted_" + outcome_col

    did_data[weighted_outcome] = did_data[outcome_col] * did_data["weights"]

    sc_post = (
        did_data.loc[
            (did_data[treat_col] == 0) & (did_data[index_col] >= intervention_point), :
        ]
        .groupby(index_col)[weighted_outcome]
        .sum()
    )

    traites_post = (
        did_data.loc[
            (did_data[treat_col] == 1) & (did_data[index_col] >= intervention_point), :
        ]
        .groupby(index_col)[weighted_outcome]
        .sum()
    )

    effect = (traites_post - sc_post).mean()
    if not (model):
        return effect
    else:
        return effect, did_data


def fit_unit_weights_elastic_net(
    data, outcome_col, index_col, id_col, treat_col, intervention_point, ratio
):
    # data en pré traitement
    pre_data = data.loc[data[index_col] < intervention_point, :]

    # Pivot pour que les données des contrôles soit en colonne, 1 colonne = 1 variable
    y_pre_control = pre_data[pre_data[treat_col] == 0].pivot(
        index=index_col, columns=id_col, values=outcome_col
    )

    # Variable cible la moyenne des traités par mois en pretraitement
    y_pre_treat_mean = (
        pre_data[pre_data[treat_col] == 1].groupby(index_col)[outcome_col].mean()
    )

    # Ajout d'un intercept

    X = y_pre_control.values
    # X = np.concatenate([np.ones((T_pre, 1)), y_pre_control.values], axis=1)
    ## optimisation de l'hyperparametre

    alphas = np.logspace(-8, 8, 100)
    ## calcul des poids
    if ratio == 0:
        model = ElasticNetCV(l1_ratio=ratio, alphas=alphas)
    else:
        model = ElasticNetCV(l1_ratio=ratio)
    model.fit(X, y_pre_treat_mean)

    w = model.coef_
    intercept = model.intercept_
    # print("Intercept:", w.value[0])
    return pd.Series(
        w,  # remove intercept
        name="unit_weights",
        index=y_pre_control.columns,
    ), intercept


def elastic_net(
    data,
    index_col,
    id_col,
    treat_col,
    outcome_col,
    intervention_point,
    ratio,
    model=False,
):
    unit_weights, intercept = fit_unit_weights_elastic_net(
        data, outcome_col, index_col, id_col, treat_col, intervention_point, ratio
    )
    did_data = join_weights(
        data,
        unit_weights,
        index_col=index_col,
        id_col=id_col,
        treat_col=treat_col,
        intervention_point=intervention_point,
    )

    did_data["intercept"] = intercept
    weighted_outcome = "weighted" + outcome_col

    did_data[weighted_outcome] = did_data[outcome_col] * did_data["weights"]

    sc_post = (
        did_data.loc[
            (did_data[treat_col] == 0) & (did_data[index_col] >= intervention_point), :
        ]
        .groupby(index_col)[weighted_outcome]
        .sum()
    )

    traites_post = (
        did_data.loc[
            (did_data[treat_col] == 1) & (did_data[index_col] >= intervention_point), :
        ]
        .groupby(index_col)[weighted_outcome]
        .sum()
    )

    effect = (traites_post - sc_post).mean() - intercept
    if not (model):
        return effect
    else:
        return effect, did_data, intercept


def fit_time_weights(
    data, outcome_col, index_col, id_col, treat_col, intervention_point
):
    # recupo des contrôles
    controle = data.loc[data[treat_col] == 0, :]

    # pivot the data to the (T_pre, N_co) matrix representation
    y_pre = controle[controle[index_col] < intervention_point].pivot(
        index=index_col, columns=id_col, values=outcome_col
    )

    # group post-treatment time period by units to have a (1, N_co) vector. ?? do you mean pretreatment here?
    y_post_mean = (
        controle[controle[index_col] > intervention_point]
        .groupby(id_col)[outcome_col]
        .mean()
        .values
    )

    # ajout de l'intercept.

    X = np.concatenate([np.ones((1, y_pre.shape[1])), y_pre.values], axis=0)
    N = X.shape[0] - 1

    w = cp.Variable(X.shape[0])
    objective = cp.Minimize(cp.sum_squares(w @ X - y_post_mean))
    constraints = [cp.sum(w[1:]) == 1, w[1:] >= 1e-5]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, eps=1e-9, verbose=False)

    if problem.status == cp.OPTIMAL and w.value is not None:
        time_weights_series = pd.Series(
            w.value[1:], name="time_weights", index=y_pre.index
        )
    else:
        print("Problem not solved optimally, using equal weights.")
        time_weights_series = [1 / N for i in range(N)]
        time_weights_series = pd.Series(
            time_weights_series, name="time_weights", index=y_pre.index
        )

    return time_weights_series


def calculate_regularization(
    data, outcome_col, index_col, id_col, treat_col, intervention_point
):
    n_treated_post = data.loc[
        (data[treat_col] == 1) & (data[index_col] >= intervention_point), :
    ].shape[0]

    first_diff_std = (
        data.loc[(data[treat_col] == 0) & (data[index_col] < intervention_point), :]
        .sort_values(index_col)
        .groupby(id_col)[outcome_col]
        .diff()
        .std()
    )

    return n_treated_post ** (1 / 4) * first_diff_std


def fit_unit_weights_synthetic_did(
    data, outcome_col, index_col, id_col, treat_col, intervention_point
):
    # regularisation
    zeta = calculate_regularization(
        data, outcome_col, index_col, id_col, treat_col, intervention_point
    )
    # data en pré traitement
    pre_data = data.loc[data[index_col] < intervention_point, :]

    # Pivot pour que les données des contrôles soit en colonne, 1 colonne = 1 variable
    y_pre_control = pre_data[pre_data[treat_col] == 0].pivot(
        index=index_col, columns=id_col, values=outcome_col
    )

    y_pre_treat_mean = (
        pre_data[pre_data[treat_col] == 1].groupby(index_col)[outcome_col].mean()
    )

    T_pre = y_pre_control.shape[0]
    X = np.concatenate([np.ones((T_pre, 1)), y_pre_control.values], axis=1)

    w = cp.Variable(X.shape[1])
    objective = cp.Minimize(
        cp.sum_squares(X @ w - y_pre_treat_mean.values)
        + T_pre * zeta**2 * cp.sum_squares(w[1:])
    )
    constraints = [cp.sum(w[1:]) == 1, w[1:] >= 1e-5]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, eps=1e-9, verbose=False)

    return pd.Series(
        w.value[1:],  # remove intercept
        name="unit_weights",
        index=y_pre_control.columns,
    )


def join_weights_synthetic_did(
    data, unit_w, time_w, index_col, id_col, treat_col, intervention_point
):
    data = data.copy()
    data.loc[data[index_col] >= intervention_point, "post_traitement"] = 1
    data.loc[data[index_col] < intervention_point, "post_traitement"] = 0
    data = (
        data.set_index([index_col, id_col])
        .join(time_w)
        .join(unit_w)
        .reset_index()
        .fillna(
            {
                time_w.name: 1
                / data.loc[data[index_col] > intervention_point][
                    index_col
                ].nunique(),  # time_w.name: data["post_traitement"].mean(), ## on remplace par la longeur( proportion) de la periode de postraitement
                unit_w.name: 1 / data.loc[data[treat_col] == 1][id_col].nunique(),
            }
        )  # unit_w.name: data[treat_col].mean()}) ## on remplace par la proportion des traités
        .assign(**{"weights": lambda d: (d[time_w.name] * d[unit_w.name]).round(10)})
        .astype({treat_col: int})
    )

    return data


def synthetic_did(
    data,
    index_col,
    id_col,
    treat_col,
    outcome_col,
    intervention_point,
    ratio=None,
    model=False,
):
    unit_weights = fit_unit_weights_synthetic_did(
        data, outcome_col, index_col, id_col, treat_col, intervention_point
    )
    time_weights = fit_time_weights(
        data, outcome_col, index_col, id_col, treat_col, intervention_point
    )

    did_data = join_weights_synthetic_did(
        data,
        unit_weights,
        time_weights,
        index_col=index_col,
        id_col=id_col,
        treat_col=treat_col,
        intervention_point=intervention_point,
    )

    X = did_data[[outcome_col, "post_traitement", treat_col, "weights"]]

    X.loc[X["weights"] < 1e-10, "weights"] = 1e-10
    formula = f"{outcome_col} ~ post_traitement*{treat_col}"

    did_model = smf.wls(formula, data=X, weights=X["weights"]).fit()
    did_model.params[f"post_traitement:{treat_col}"]
    try:
        did_model = smf.wls(formula, data=X, weights=X["weights"]).fit()

    except Exception:
        print("erreur SVD")
        if not (model):
            return np.inf
    else:
        if not (model):
            return did_model.params[f"post_traitement:{treat_col}"]
        else:
            return did_model, did_data


def dtw_between_series_and_array(time_series, array_of_series):
    distances = []
    if isinstance(time_series, pd.Series):
        time_series = time_series.to_numpy()
    if isinstance(array_of_series, pd.DataFrame):
        array_of_series = array_of_series.to_numpy()
    for i in range(array_of_series.shape[1]):
        distance = dtw.distance_fast(time_series, array_of_series[:, i], use_c=True)
        distances.append(distance)
    return np.array(distances)


class CustomEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, lambda_=3):
        self.lambda_ = lambda_

    def fit(self, X, y, sample_weight=None):
        n_features = X.shape[1]
        w = cp.Variable(n_features)

        dtw_distance = dtw_between_series_and_array(y, X)

        objective = cp.Minimize(
            cp.sum_squares(X @ w - y) + self.lambda_ * cp.sum(dtw_distance @ w)
        )
        constraints = [cp.sum(w) == 1, w >= 1e-5]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS, eps=1e-9, verbose=False)

        self.coef_ = w.value
        return self

    def predict(self, X):
        return X @ self.coef_


def fit_unit_weights_dtw(
    data,
    outcome_col,
    date_col,
    id_col,
    treat_col,
    intervention_point,
    lambda_values=np.logspace(-3, 3, 7),
):
    pre_data = data.loc[data[date_col] < intervention_point, :]

    y_pre_control = pre_data[data[treat_col] == 0].pivot(
        index=date_col, columns=id_col, values=outcome_col
    )
    y_pre_treat_mean = (
        pre_data[data[treat_col] == 1].groupby(date_col)[outcome_col].mean()
    )

    X = y_pre_control.values
    y = y_pre_treat_mean.values

    # Use GridSearchCV to find the best lambda value
    param_grid = {"lambda_": lambda_values}
    grid_search = GridSearchCV(
        CustomEstimator(), param_grid, cv=5, scoring="neg_mean_squared_error"
    )
    grid_search.fit(X, y)

    best_lambda = grid_search.best_params_["lambda_"]
    final_model = CustomEstimator(lambda_=best_lambda)
    final_model.fit(X, y)

    return pd.Series(
        final_model.coef_, name="unit_weights", index=y_pre_control.columns
    )


def synthetic_control_with_dtw(
    data,
    index_col,
    id_col,
    treat_col,
    outcome_col,
    intervention_point,
    ratio=False,
    model=False,
):
    unit_weights = fit_unit_weights_dtw(
        data,
        outcome_col,
        index_col,
        id_col,
        treat_col,
        intervention_point,
    )
    did_data = join_weights(
        data,
        unit_weights,
        index_col=index_col,
        id_col=id_col,
        treat_col=treat_col,
        intervention_point=intervention_point,
    )

    X = did_data[[outcome_col, "post_traitement", treat_col, "weights"]]

    X.loc[X["weights"] < 1e-10, "weights"] = 1e-10
    formula = f"{outcome_col} ~ post_traitement*{treat_col}"

    did_model = smf.wls(formula, data=X, weights=X["weights"]).fit()
    did_model.params[f"post_traitement:{treat_col}"]

    try:
        did_model = smf.wls(formula, data=X, weights=X["weights"]).fit()
    except Exception:
        print("erreur SVD")
        if not (model):
            return np.inf
    else:
        if not (model):
            return did_model.params[f"post_traitement:{treat_col}"]
        else:
            return did_model, did_data
