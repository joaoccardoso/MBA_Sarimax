from pathlib import Path
from typing import Any
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace import sarimax
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_dataset(filepath: Path, resample_rule: str | None):
    # Load data and set the date column as index
    df = pd.read_csv(filepath)
    df["DATA"] = pd.to_datetime(df["DATA"])
    df = df.set_index("DATA")

    # Compute the mean values between the max and min temperature from the previous hour.
    temp_columns = [
        "TEMPERATURA MÁXIMA NA HORA ANT. (°C)",
        "TEMPERATURA MÍNIMA NA HORA ANT. (°C)",
    ]

    df["TEMPERATURA"] = df[temp_columns].mean(axis=1)

    if resample_rule:
        df = df.resample(resample_rule).mean()

    return df


def plot_data_with_mean_and_std(data: pd.Series):
    plt.figure(figsize=(12, 5))
    plt.plot(data, label="Precipitação", alpha=0.5)
    plt.plot(
        data.rolling(window=52).mean(),
        label="Média Móvel (52 Semanas)",
        color="red",
    )
    plt.plot(
        data.rolling(window=52).std(),
        label="Desvio Padrao (52 Semanas)",
        color="green",
    )
    plt.ylabel("Precipitação (mm)")
    plt.grid()
    plt.legend()
    plt.title(
        "Média Móvel e Desvio Padrão dos Níveis de Precipitação por Semana entre 2015 e 2024"
    )
    plt.show()


def perform_adf_test(data: pd.Series):
    adf_result = adfuller(data)

    # Extract test results
    adf_statistic, p_value, _, _, critical_values, _ = adf_result

    print("adf_statistic: ", adf_statistic)
    print("p_value: ", p_value)
    print("critical_values: ", critical_values)

    return adf_result


def plot_acf_and_pacf(data: pd.Series):
    # Plot ACF and PACF for the weekly data
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    num_of_lags = 52  # Up to 1 years of lags

    plot_acf(data, lags=num_of_lags, ax=axes[0])
    axes[0].set_title("ACF (Dados Semanais)")
    axes[0].set_xlabel("Número de Lags")
    axes[0].set_ylabel("Correlação")
    axes[0].grid()

    plot_pacf(data, lags=num_of_lags, ax=axes[1])
    axes[1].set_title("PACF (Dados Semanais)")
    axes[1].set_xlabel("Número de Lags")
    axes[1].set_ylabel("Correlação")
    axes[1].grid()

    fig.suptitle(
        "Autocorrelação e Autocorrelação Parcial para Niveis de Precipitação Semanais para um Janela de 52 semanas (1 ano)"
    )
    fig.tight_layout()

    plt.show()


def run_model(
    endog: pd.Series,
    exog: pd.DataFrame,
    order: tuple,
    seasonal_order: tuple,
):
    model = sarimax.SARIMAX(
        endog=endog,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    # Fit the model
    result = model.fit(disp=False)
    return result


def plot_residuals(result: Any):
    residuals = result.resid
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of residuals
    axes[0].hist(residuals, bins=40, color="blue")
    axes[0].set_title("Distribuição dos Resíduos do Modelo (SARIMA Semanal)")
    axes[0].set_xlabel("Resíduos")
    axes[0].set_ylabel("Frequência")
    axes[0].grid()

    # Residuals over time
    axes[1].plot(residuals, color="red", alpha=0.7)
    axes[1].axhline(0, linestyle="--", color="black")
    axes[1].set_title("Resíduos do Modelo ao Longo do Tempo (SARIMA Semanal)")
    axes[1].set_ylabel("Resíduos")
    axes[1].grid()

    fig.suptitle("Descrição dos Resíduos do Modelo SARIMA Semanal")

    plt.show()

    # Check residuals mean and standard deviation
    residuals_mean_monthly = np.mean(residuals)
    residuals_std_monthly = np.std(residuals)

    print("Média dos Resíduos", residuals_mean_monthly)
    print("Desvio Padrão dos Resíduos", residuals_std_monthly)


def main():
    df = load_dataset("data/concat_data_2015_2024_filled.CSV", resample_rule="W")

    precip_col = "PRECIPITAÇÃO TOTAL HORÁRIO (mm)"
    precip_data = df[precip_col]

    # Plot rolling mean and standard deviation
    plot_data_with_mean_and_std(precip_data)

    # Perform ADF test
    perform_adf_test(precip_data)
    plot_acf_and_pacf(precip_data)

    features = ["PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA (mB)", "TEMPERATURA"]

    endg_train = df["PRECIPITAÇÃO TOTAL HORÁRIO (mm)"][:"2022-12-31"]
    endg_test = df["PRECIPITAÇÃO TOTAL HORÁRIO (mm)"]["2023-01-01":]

    exog_train = df[features][:"2022-12-31"]
    exog_test = df[features]["2023-01-01":]

    result = run_model(endg_train, exog_train, (3, 1, 2), (0, 0, 0, 52))
    plot_residuals(result)

    # Fazer previsões para o período de teste
    forecast = result.get_forecast(steps=len(endg_test), exog=exog_test)
    forecast_values = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()

    # Calcular métricas de avaliação
    mae = mean_absolute_error(endg_test, forecast_values)
    rmse = np.sqrt(mean_squared_error(endg_test, forecast_values))
    r2 = r2_score(endg_test, forecast_values)

    # Plotar resultados
    plt.figure(figsize=(12, 6))
    plt.plot(endg_train.index, endg_train, label="Treino", color="blue")
    plt.plot(endg_test.index, endg_test, label="Real", color="green")
    plt.plot(
        endg_test.index, forecast_values, label="Previsto", color="red", linestyle="--"
    )
    plt.fill_between(
        endg_test.index,
        confidence_intervals.iloc[:, 0],
        confidence_intervals.iloc[:, 1],
        color="pink",
        alpha=0.3,
    )
    plt.title("Previsão de Precipitação vs Valores Reais")
    plt.xlabel("Data")
    plt.ylabel("Precipitação (mm)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Imprimir métricas
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")


if __name__ == "__main__":
    main()
