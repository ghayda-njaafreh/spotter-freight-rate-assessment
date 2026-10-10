from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


def prepare_features(frame, weight_strategy):
    data = frame.copy()

    data["weight_missing"] = data["weight"].isna().astype(int)
    data["weight_negative"] = (data["weight"] < 0).astype(int)

    if weight_strategy == "raw":
        data["weight_clean"] = data["weight"]

    elif weight_strategy == "absolute":
        data["weight_clean"] = data["weight"].abs()

    elif weight_strategy == "missing":
        data["weight_clean"] = data["weight"].where(
            data["weight"] >= 0,
            np.nan,
        )

    else:
        raise ValueError(
            f"Unknown weight strategy: {weight_strategy}"
        )


    data["month"] = data["date"].dt.month
    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_year"] = data["date"].dt.dayofyear

    data["day_sin"] = np.sin(
        2 * np.pi * data["day_of_year"] / 365.25
    )

    data["day_cos"] = np.cos(
        2 * np.pi * data["day_of_year"] / 365.25
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
        "market_index",
        "quote_signal",
        "month",
        "day_of_week",
        "day_of_year",
        "day_sin",
        "day_cos",
    ]

    return data[features], categorical


def metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "r2": r2_score(y_true, y_pred),
    }


def run_experiment(
    target_type,
    weight_strategy,
):
    print("\n" + "=" * 80)
    print(
        f"TARGET={target_type} | "
        f"WEIGHT={weight_strategy}"
    )
    print("=" * 80)

    fold_results = []

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

        X_train, categorical = prepare_features(
            train,
            weight_strategy,
        )

        X_valid, _ = prepare_features(
            valid,
            weight_strategy,
        )

        cat_indices = [
            X_train.columns.get_loc(column)
            for column in categorical
        ]

        if target_type == "rate":
            y_train = train["posted_rate"]

        elif target_type == "rpm":
            y_train = (
                train["posted_rate"]
                / train["distance"]
            )

        else:
            raise ValueError(
                f"Unknown target: {target_type}"
            )

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
            y_train,
            cat_features=cat_indices,
            eval_set=(
                X_valid,
                (
                    valid["posted_rate"]
                    if target_type == "rate"
                    else (
                        valid["posted_rate"]
                        / valid["distance"]
                    )
                ),
            ),
            early_stopping_rounds=80,
            use_best_model=True,
            verbose=False,
        )

        prediction = model.predict(X_valid)

        if target_type == "rpm":
            prediction = (
                prediction
                * valid["distance"].to_numpy()
            )

        prediction = np.maximum(
            prediction,
            0.01,
        )

        result = metrics(
            valid["posted_rate"],
            prediction,
        )

        print(
            f"{fold['name']}: "
            f"MAE=${result['mae']:.2f} | "
            f"RMSE=${result['rmse']:.2f} | "
            f"R2={result['r2']:.4f} | "
            f"Trees={model.tree_count_}"
        )

        fold_results.append(result)

    mean_mae = np.mean(
        [r["mae"] for r in fold_results]
    )

    mean_rmse = np.mean(
        [r["rmse"] for r in fold_results]
    )

    mean_r2 = np.mean(
        [r["r2"] for r in fold_results]
    )

    print("-" * 80)
    print(
        f"AVERAGE: "
        f"MAE=${mean_mae:.2f} | "
        f"RMSE=${mean_rmse:.2f} | "
        f"R2={mean_r2:.4f}"
    )

    return {
        "target": target_type,
        "weight_strategy": weight_strategy,
        "mae": mean_mae,
        "rmse": mean_rmse,
        "r2": mean_r2,
    }


experiments = []

for target in ["rate", "rpm"]:
    for weight_strategy in [
        "raw",
        "absolute",
        "missing",
    ]:
        result = run_experiment(
            target,
            weight_strategy,
        )

        experiments.append(result)


results = pd.DataFrame(experiments)
results = results.sort_values("mae")

print("\n" + "=" * 80)
print("FINAL DEVELOPMENT RANKING")
print("=" * 80)

print(results.to_string(index=False))

results.to_csv(
    OUTPUT_DIR / "catboost_experiments.csv",
    index=False,
)

print(
    "\nSaved:"
    "\noutputs/catboost_experiments.csv"
)
