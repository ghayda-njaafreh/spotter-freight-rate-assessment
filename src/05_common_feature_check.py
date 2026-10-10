from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train-test.csv"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date").reset_index(drop=True)



FOLDS = [
    {
        "name": "Jul",
        "train_end": "2025-06-30",
        "valid_start": "2025-07-01",
        "valid_end": "2025-07-31",
    },
    {
        "name": "Aug",
        "train_end": "2025-07-31",
        "valid_start": "2025-08-01",
        "valid_end": "2025-08-31",
    },
    {
        "name": "Sep",
        "train_end": "2025-08-31",
        "valid_start": "2025-09-01",
        "valid_end": "2025-09-30",
    },
]


def prepare_features(frame):

    data = frame.copy()

    data["weight_missing"] = (
        data["weight"].isna().astype(int)
    )

    data["weight_negative"] = (
        (data["weight"] < 0).astype(int)
    )

    data["weight_clean"] = data["weight"].where(
        data["weight"] >= 0,
        np.nan,
    )


    data["month"] = data["date"].dt.month
    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_year"] = data["date"].dt.dayofyear

    data["day_sin"] = np.sin(
        2 * np.pi
        * data["day_of_year"]
        / 365.25
    )

    data["day_cos"] = np.cos(
        2 * np.pi
        * data["day_of_year"]
        / 365.25
    )


    data["route"] = (
        data["pickup"].astype(str)
        + "__"
        + data["delivery"].astype(str)
    )

    categorical = [
        "pickup",
        "delivery",
        "equipment",
        "route",
    ]

    for column in categorical:
        data[column] = (
            data[column]
            .fillna("__MISSING__")
            .astype(str)
        )

    features = [
        "pickup",
        "delivery",
        "route",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "distance",
        "equipment",
        "weight_clean",
        "weight_missing",
        "weight_negative",
        "month",
        "day_of_week",
        "day_of_year",
        "day_sin",
        "day_cos",
    ]

    return data[features], categorical


results = []

for fold in FOLDS:

    train = df[
        df["date"]
        <= pd.Timestamp(fold["train_end"])
    ].copy()

    valid = df[
        (
            df["date"]
            >= pd.Timestamp(fold["valid_start"])
        )
        &
        (
            df["date"]
            <= pd.Timestamp(fold["valid_end"])
        )
    ].copy()

    X_train, categorical = prepare_features(train)
    X_valid, _ = prepare_features(valid)

    cat_indices = [
        X_train.columns.get_loc(column)
        for column in categorical
    ]

    model = CatBoostRegressor(
        iterations=600,
        learning_rate=0.05,
        depth=8,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )

    model.fit(
        X_train,
        train["posted_rate"],
        cat_features=cat_indices,
        eval_set=(
            X_valid,
            valid["posted_rate"],
        ),
        early_stopping_rounds=80,
        use_best_model=True,
        verbose=False,
    )

    prediction = model.predict(X_valid)

    prediction = np.maximum(
        prediction,
        0.01,
    )

    mae = mean_absolute_error(
        valid["posted_rate"],
        prediction,
    )

    rmse = np.sqrt(
        mean_squared_error(
            valid["posted_rate"],
            prediction,
        )
    )

    r2 = r2_score(
        valid["posted_rate"],
        prediction,
    )

    results.append(
        {
            "fold": fold["name"],
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "trees": model.tree_count_,
        }
    )

    print(
        f"{fold['name']}: "
        f"MAE=${mae:.2f} | "
        f"RMSE=${rmse:.2f} | "
        f"R2={r2:.4f} | "
        f"Trees={model.tree_count_}"
    )


results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("COMMON FEATURE CV SUMMARY")
print("=" * 80)

print(results_df.to_string(index=False))

print("\nAverage:")
print(
    f"MAE=${results_df['mae'].mean():.2f}"
)

print(
    f"RMSE=${results_df['rmse'].mean():.2f}"
)

print(
    f"R2={results_df['r2'].mean():.4f}"
)

median_trees = int(
    results_df["trees"].median()
)

print(
    f"\nMedian best tree count: "
    f"{median_trees}"
)


experiment_path = (
    OUTPUT_DIR
    / "catboost_experiments.csv"
)

if experiment_path.exists():

    previous = pd.read_csv(
        experiment_path
    ).sort_values("mae")

    best = previous.iloc[0]

    print("\n" + "=" * 80)
    print("COMPARISON WITH FULL FEATURE MODEL")
    print("=" * 80)

    print(
        f"Full-feature MAE:   "
        f"${best['mae']:.2f}"
    )

    print(
        f"Common-feature MAE: "
        f"${results_df['mae'].mean():.2f}"
    )

    difference = (
        results_df["mae"].mean()
        - best["mae"]
    )

    print(
        f"MAE difference:     "
        f"${difference:.2f}"
    )


results_df.to_csv(
    OUTPUT_DIR
    / "common_feature_cv.csv",
    index=False,
)

print(
    "\nSaved:"
    "\noutputs/common_feature_cv.csv"
)
