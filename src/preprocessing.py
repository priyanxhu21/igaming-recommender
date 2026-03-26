from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

KAGGLE_DATASET_HANDLE = "igormerlinicomposer/online-casino-games-dataset-1-2m-records"
APP_DATA_COLUMNS = [
    "casino",
    "game",
    "provider",
    "rtp",
    "volatility",
    "min_bet",
    "game_type",
    "free_spins_feature",
    "bonus_buy_available",
    "max_multiplier",
]


def load_data(path, usecols=None, nrows=None):
    df = pd.read_csv(path, usecols=usecols, nrows=nrows)
    return df


def resolve_dataset_path(dataset_path=None):
    if dataset_path is not None:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at: {path}")
        return path

    local_candidates = [
        Path("data/dataset.csv"),
        Path("data/online_casino_games_dataset_v2.csv"),
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return candidate

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is required to download the Kaggle dataset. "
            "Install dependencies first with `pip install -r requirements.txt`."
        ) from exc

    download_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET_HANDLE))
    csv_files = sorted(download_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files were found in the downloaded Kaggle dataset at {download_dir}"
        )

    return csv_files[0]


def load_kaggle_dataset(dataset_path=None):
    dataset_file = resolve_dataset_path(dataset_path)
    return load_data(dataset_file)


def load_app_dataset(dataset_path=None, max_rows=None):
    dataset_file = resolve_dataset_path(dataset_path)
    df = load_data(dataset_file, usecols=APP_DATA_COLUMNS, nrows=max_rows)
    return df, dataset_file


def _to_binary(series):
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0,
                "yes": 1,
                "no": 0,
            }
        )
        .fillna(0)
        .astype(int)
    )


def prepare_dataframe(df):
    df = df.copy()

    numeric_defaults = {
        "rtp": df["rtp"].median(),
        "max_multiplier": df["max_multiplier"].median(),
        "min_bet": df["min_bet"].median(),
    }

    df.fillna(
        {
            "volatility": "Medium",
            "bonus_buy_available": False,
            "free_spins_feature": False,
            **numeric_defaults,
        },
        inplace=True,
    )

    df["volatility"] = df["volatility"].astype(str).str.strip()
    df["game_type"] = df["game_type"].astype(str).str.strip()
    df["bonus_buy_available"] = _to_binary(df["bonus_buy_available"])
    df["free_spins_feature"] = _to_binary(df["free_spins_feature"])

    return df


def get_feature_mappings(df):
    return {
        "volatility": {
            value: index
            for index, value in enumerate(sorted(df["volatility"].dropna().unique()))
        },
        "game_type": {
            value: index
            for index, value in enumerate(sorted(df["game_type"].dropna().unique()))
        },
    }


def build_feature_frame(df, mappings):
    feature_df = df.copy()
    feature_df["volatility_encoded"] = feature_df["volatility"].map(
        mappings["volatility"]
    )
    feature_df["game_type_encoded"] = feature_df["game_type"].map(
        mappings["game_type"]
    )
    feature_df["feature_score"] = (
        feature_df["bonus_buy_available"] + feature_df["free_spins_feature"]
    )

    return feature_df[
        [
            "rtp",
            "volatility_encoded",
            "max_multiplier",
            "min_bet",
            "feature_score",
            "game_type_encoded",
        ]
    ]


def preprocess(df, return_mappings=False):
    df = prepare_dataframe(df)
    mappings = get_feature_mappings(df)

    df["volatility_encoded"] = df["volatility"].map(mappings["volatility"])
    df["game_type_encoded"] = df["game_type"].map(mappings["game_type"])
    df["feature_score"] = df["bonus_buy_available"] + df["free_spins_feature"]

    features = build_feature_frame(df, mappings)

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    if return_mappings:
        return df, scaled_features, scaler, mappings

    return df, scaled_features, scaler
