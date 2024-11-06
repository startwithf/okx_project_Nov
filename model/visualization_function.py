import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import numpy as np
import matplotlib.cm as cm


def plot_tokens_lineplot(data, tokens, start_date, metric):
    """
    Plot the line plot of the given metric for the given tokens.
    Parameters:
    data (DataFrame): The DataFrame containing the data.
    tokens (list): The list of tokens to plot.
    start_date (str): The start date for the plot.
    metric (str): The metric to plot.
    """
    # Filter the data
    plot_data = data[(data["name"].isin(tokens)) & (data["date"] >= start_date)]

    # Plot the data
    plt.figure(figsize=(15, 4))
    sns.lineplot(x="date", y=metric, hue="name", data=plot_data)
    plt.legend(title="Token Symbol", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xlabel("Date")
    plt.ylabel(metric)
    plt.grid(True)
    plt.show()


def plot_tokens_density(token_data, tokens, start_date, metric):
    """
    Plot the histogram for the given tokens and metric.
    Parameters:
    token_data (DataFrame): The DataFrame containing the token data.
    tokens (list): The list of tokens to plot.
    start_date (str): The start date for the plot.
    metric (str): The metric to plot.
    """
    # Filter the data
    token_data_filtered = token_data[
        (token_data["name"].isin(tokens)) & (token_data["date"] >= start_date)
    ]

    # Plot the histograms
    plt.figure(figsize=(15, 4))
    for token in tokens:
        sns.kdeplot(
            token_data_filtered[token_data_filtered["name"] == token][metric],
            label=token,
            fill=True,
            alpha=0.3,
        )

    # Plot the normal distribution
    mu, std = norm.fit(token_data_filtered[metric].dropna())
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, "k", linewidth=2, label="Normal Distribution")

    # Set the plot parameters
    plt.title(f"{metric.capitalize()} Histogram for Tokens")
    plt.xlabel(metric.capitalize())
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_tokens_boxplot(token_data, tokens, start_date, metric):

    # Filter the data
    token_data_filtered = token_data[
        (token_data["name"].isin(tokens)) & (token_data["date"] >= start_date)
    ]

    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(figsize=(15, 4))
        sns.boxplot(x=token_data_filtered["name"], y=token_data_filtered[metric], ax=ax)
        ax.set(
            xlabel="Token",
            ylabel=metric,
            title=f"{metric.capitalize()} Boxplot for Tokens",
        )
        plt.show()


def plot_token_volatility(name, df, start_date="2018-01-01"):
    """
    Plot the volatility for the given token.
    Parameters:
    symbol (str): The symbol of the token.
    data (DataFrame): The DataFrame containing the data.
    start_date (str): The start date for the plot.
    interval_lst (list): The list of intervals for the volatility calculation.
    """
    # Filter the data
    token_data = df[df["name"] == name]
    token_data = token_data[token_data["date"] >= start_date]

    # Calculate the volatility
    volatility_cols = [x for x in token_data.columns if "annualized_volatility" in x]

    # Generate a list of colors using the viridis colormap
    colors = ["lightblue", "red", "orange", "green", "gray"]

    # Plot the volatility
    plt.figure(figsize=(14, 4))
    for col, color in zip(volatility_cols, colors):
        plt.plot(
            token_data["date"],
            token_data[col],
            label=col,
            color=color,
        )
    plt.legend(loc="upper right")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.title(f"Realized Volatility Using Different Interval Windows - {name}")
    plt.grid(True)
    plt.show()


def plot_splited_date(token_name, modelling_data_dict_full_):
    if (
        token_name not in modelling_data_dict_full_
    ):  # Check if the token is in the dictionary
        print(f"Token {token_name} is not in the modelling data dictionary.")
        return
    # Get the modelling data for the token
    modelling_data_info = modelling_data_dict_full_[token_name]
    # Plot the data
    sns.set_context("paper", font_scale=1.5)
    with sns.axes_style("whitegrid"):
        _, ax = plt.subplots(figsize=(12, 4))

        ax.plot(
            modelling_data_info["token_df"]["daily_volatility_30"],
            lw=1,
            color="black",
            ls="--",
            label="Daily Volatility",
        )
        ax.plot(
            modelling_data_info["target_train"],
            color="blue",
            label="Training Volatility Target",
            lw=2,
        )
        ax.plot(
            modelling_data_info["target_validation"],
            color="orange",
            label="Validation Volatility Target",
            lw=2,
        )
        ax.plot(
            modelling_data_info["target_test"],
            color="green",
            label="Test Volatility Target",
            lw=2,
        )

        ax.legend(loc="upper right", prop={"size": 10}, frameon=True)
        plt.show()


def plot_model_prediction(x_val, y_true, y_pred, model_name):
    sns.set_context("paper", font_scale=1.7)
    plt.rcParams["axes.grid"] = False

    with sns.axes_style("whitegrid"):
        plt.figure(figsize=(18, 7))
        plt.plot(x_val, color="black", ls=":", label=f"Current Daily Volatility")

        plt.plot(y_true, color="orange", lw=2, label=f"Target Volatility")
        plt.plot(y_pred, color="darkviolet", lw=2.5, label=f"Forecasted Volatility")

        plt.title(f"{model_name}")
        plt.legend(loc="upper right", frameon=True)
