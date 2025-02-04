import os
from pathlib import Path
from typing import Literal

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

YEARS = ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"]

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


def normalize_dataset(df: pd.DataFrame):
    features = list(df.columns)
    scaler = StandardScaler()
    norm_df = df.copy()
    norm_df[features] = scaler.fit_transform(df[features])

    return norm_df


def export_fill_gaps_candidates_figures(
    df: pd.DataFrame,
    dataset_folder: Path,
    fill_option: Literal["df_fbfill", "df_spline", "df_linear", "df_mean"],
    export_images=True,
):
    fill_gaps_candidates = {
        "df_fbfill": df.ffill().bfill(),
        "df_spline": df.interpolate(method="spline", order=1),
        "df_linear": df.interpolate(method="linear"),
        "df_mean": df.fillna(df.mean()),
    }

    if export_images:
        output_folder = dataset_folder / "fill_gaps_options"
        os.makedirs(output_folder, exist_ok=True)

        features = list(df.columns)

        for method, result_df in fill_gaps_candidates.items():
            fig, axes = plt.subplots(
                len(features), 2, figsize=(16, 3 * len(features) + 1)
            )
            for i, feature in enumerate(features):
                # Original data
                axes[i, 0].plot(
                    df.index, df[feature], label="Original", color="red", alpha=0.7
                )
                axes[i, 0].set_title(
                    f"{feature.capitalize()} - Original (com dados faltantes)"
                )
                axes[i, 0].grid()
                axes[i, 0].legend()

                # Filled data
                axes[i, 1].plot(
                    result_df.index,
                    result_df[feature],
                    label=f"Preenchido ({method})",
                    color="blue",
                    alpha=0.7,
                )
                axes[i, 1].set_title(
                    f"{feature.capitalize()} - Preenchendo Dados({method})"
                )
                axes[i, 1].grid()
                axes[i, 1].legend()

            fig.tight_layout()
            fig.savefig(output_folder / f"{method}.png")

    output_df = fill_gaps_candidates[fill_option]
    return output_df


def resample_dataset(df: pd.DataFrame):
    output_df = df.copy()
    output_df["DATA"] = pd.to_datetime(df["DATA"])
    output_df = output_df.set_index("DATA").resample("D").mean()

    return output_df


def export_dataset(df: pd.DataFrame, filename: Path, suffix: str, index: bool):
    output_name = Path(os.path.splitext(filename)[0] + f"_{suffix}.CSV")
    print("Exporting to", output_name)
    df.to_csv(output_name, index=index)


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


def main():
    for year in YEARS:
        dataset_folder = Path("data", year)
        filename = (
            dataset_folder
            / f"INMET_SE_RJ_A610_PICO DO COUTO_01-01-{year}_A_31-12-{year}.CSV"
        )

        if not filename.exists():
            print("File not found:", filename)
            continue

        print("Reading data at", filename)
        df = read_dataset(filename)

        print("Transforming dataset")
        df = transform_dataset(df)

        print("Resampling dataset to daily frequency")
        resampled_df = resample_dataset(df)

        print("Filling dataset gaps and exporting candidates")
        filled_df = export_fill_gaps_candidates_figures(
            resampled_df,
            dataset_folder,
            "df_fbfill",
            export_images=True,
        )

        print("Normalizing with z-score")
        norm_df = normalize_dataset(filled_df)

        export_dataset(
            df,
            filename,
            suffix="TRATADO",
            index=False,
        )
        export_dataset(filled_df, filename, suffix="TRATADO_MENSAL", index=True)
        export_dataset(
            filled_df.describe(), filename, suffix="DESCRITIVA_MENSAL", index=True
        )
        export_dataset(norm_df, filename, suffix="NORMALIZADO_MENSAL", index=True)

    print("Done")


if __name__ == "__main__":
    main()
