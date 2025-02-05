from pathlib import Path
import pandas as pd


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
