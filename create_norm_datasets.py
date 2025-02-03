import os
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = {
    "Data": "Data",
    "Hora UTC": "Hora",
    "DATA (YYYY-MM-DD)": "Data",
    "HORA (UTC)": "Hora",
    "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)": "PRECIPITAÇÃO TOTAL HORÁRIO (mm)",
    "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)": "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA (mB)",
    "PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)": "PRESSÃO ATMOSFERICA MAX. NA HORA ANT. (mB)",
    "PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)": "PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (mB)",
    "TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)": "TEMPERATURA MÁXIMA NA HORA ANT. (°C)",
    "TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)": "TEMPERATURA MÍNIMA NA HORA ANT. (°C)",
}


def export_dataset(df: pd.DataFrame, filename: Path):
    output_name = Path(os.path.splitext(filename)[0] + "_TRATADO.CSV")
    print("Exporting to", output_name)
    df.to_csv(output_name, index=False)


def transform_dataset(old_df: pd.DataFrame):
    df = old_df.replace({",": ".", "-9999": np.nan}, regex=True)

    if "Hora UTC" in df.columns:
        time = df["Hora UTC"].str[:-4].str.zfill(4)
        datetime = pd.to_datetime(df["Data"] + " " + time.str[:2] + ":00")
    elif "HORA (UTC)" in df.columns:
        time = df["HORA (UTC)"]
        datetime = pd.to_datetime(df["DATA (YYYY-MM-DD)"] + " " + time)

    df = df.rename(columns=COLUMNS).drop(columns=["Data", "Hora"]).apply(pd.to_numeric)
    df["DATA"] = datetime
    df.insert(0, "DATA", df.pop("DATA"))

    return df


def read_dataset(filename: Path):
    dados = pd.read_csv(
        filename,
        sep=";",
        encoding="latin-1",
        skiprows=8,
        usecols=lambda x: x in list(COLUMNS.keys()),
    )

    return dados


years = ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"]

for year in years:
    filename = Path(
        "data",
        year,
        f"INMET_SE_RJ_A610_PICO DO COUTO_01-01-{year}_A_31-12-{year}.CSV",
    )

    if not filename.exists():
        print("File not found:", filename)
        continue

    print("Reading data at", filename)
    df = read_dataset(filename)

    print("Transforming dataset")
    df = transform_dataset(df)

    export_dataset(df, filename)

print("Done")
