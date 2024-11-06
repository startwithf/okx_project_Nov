import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from arch import arch_model
import warnings

warnings.filterwarnings("ignore")


def get_modelling_data(token_name, data_full):
    df = data_full[data_full["name"] == token_name]
    df.index = df["date"]
    # Decide the size of the train, validation and test sets
    total_size = len(df)
    test_size = max(int(0.05 * total_size), 7)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size - test_size
    if test_size < 7 or val_size < 7:
        print(
            f"Test size: {test_size}, Validation size: {val_size}, Train size: {train_size}"
        )
        print(f"Data for token {token_name} is not enough for the given sizes.")

    # Convert to indeces
    train_idx = df.index[:train_size]
    val_idx = df.index[train_size : train_size + val_size]
    test_idx = df.index[-test_size:]
    # Get the target values (volatility after 7 days)
    target_train = df["daily_volatility_30_target"][train_idx]
    target_validation = df["daily_volatility_30_target"][val_idx]
    target_test = df["daily_volatility_30_target"][test_idx]
    # Get the current values (current volatility, will be the input for baseline models)
    vol_train = df["daily_volatility_30"][train_idx]
    vol_validation = df["daily_volatility_30"][val_idx]
    vol_test = df["daily_volatility_30"][test_idx]
    # Get the log return values (will be the input for GARCH model)
    lr_train = df["log_return"][train_idx]
    lr_validation = df["log_return"][val_idx]
    lr_test = df["log_return"][test_idx]
    return (
        df,
        target_train,
        target_validation,
        target_test,
        vol_train,
        vol_validation,
        vol_test,
        lr_train,
        lr_validation,
        lr_test,
    )


def RMSE(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def RMSPE(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))


def baseline_mean_model(
    token_name, modelling_data_dict_full_, val_or_test="validation"
):
    modelling_data_info = modelling_data_dict_full_[token_name]

    if val_or_test == "validation":
        mean_vol_train = modelling_data_info["vol_train"].mean()
        baseline_mean_preds = (
            np.ones(len(modelling_data_info["target_validation"])) * mean_vol_train
        )
        baseline_mean_preds = pd.Series(
            baseline_mean_preds, index=modelling_data_info["lr_validation"].index
        )
    else:
        mean_vol_pre_test = pd.concat(
            [modelling_data_info["vol_train"], modelling_data_info["vol_validation"]]
        ).mean()
        baseline_mean_preds = (
            np.ones(len(modelling_data_info["target_test"])) * mean_vol_pre_test
        )
        baseline_mean_preds = pd.Series(
            baseline_mean_preds, index=modelling_data_info["lr_test"].index
        )
    return baseline_mean_preds


def baseline_random_walk_model(
    token_name, modelling_data_dict_full_, val_or_test="validation"
):
    if val_or_test == "validation":
        return modelling_data_dict_full_[token_name]["vol_validation"]
    else:
        return modelling_data_dict_full_[token_name]["vol_test"]


def garch_model(
    token,
    modelling_data_dict_full_,
    val_or_test="validation",
    vol="GARCH",
    p=1,
    q=1,
    o=0,
    power=2,
    dist="normal",
    forecast_horizon=7,
):
    # Get the data
    token_df = modelling_data_dict_full_[token]["token_df"]

    # Set a list to store the rolling forecasts
    rolling_forecasts = []

    # Get the prediction index
    if val_or_test == "validation":
        pred_true = modelling_data_dict_full_[token]["target_validation"]
    else:
        pred_true = modelling_data_dict_full_[token]["target_test"]

    pred_idx = pred_true.index

    # Fit the GARCH model
    for i in range(0, len(pred_true), forecast_horizon):
        # Get the data at all previous time points
        idx = pred_idx[i]
        train = token_df["log_return"][:idx]

        # train model using all previous data
        model = arch_model(train, vol=vol, p=p, q=q, o=o, power=power, dist=dist)
        model_fit = model.fit(disp="off")

        # Make prediction for the next forecast_horizon days
        vaR = model_fit.forecast(
            horizon=forecast_horizon, reindex=False
        ).variance.values
        # Get the predictions of the current window
        pred = list(np.sqrt(vaR[0]))

        # Append the predictions to the rolling_forecasts list
        rolling_forecasts.extend(pred)
    rolling_forecasts = rolling_forecasts[: len(pred_true)]
    garch_pred = pd.Series(rolling_forecasts, index=pred_idx)
    return garch_pred


def get_best_pred(token_name, modelling_data_dict_full_, model_):
    if "mean" in model_:
        test_pred = baseline_mean_model(token_name, modelling_data_dict_full_, "test")
    elif "random_walk" in model_:

        test_pred = baseline_random_walk_model(
            token_name, modelling_data_dict_full_, "test"
        )
    elif "basic_garch" in model_:

        test_pred = garch_model(
            token_name, modelling_data_dict_full_, val_or_test="test"
        )
    elif "gjr" in model_:

        test_pred = garch_model(
            token_name, modelling_data_dict_full_, o=1, val_or_test="test"
        )
    elif "egarch" in model_:

        test_pred = garch_model(
            token_name,
            modelling_data_dict_full_,
            vol="EGARCH",
            forecast_horizon=1,
            val_or_test="test",
        )
    elif "tarch" in model_:

        test_pred = garch_model(
            token_name,
            modelling_data_dict_full_,
            p=1,
            q=1,
            o=1,
            power=1,
            dist="skewt",
            forecast_horizon=1,
            val_or_test="test",
        )

    return test_pred
