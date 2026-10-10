from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train-test.csv"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date").reset_index(drop=True)


def prepare_features(frame):
    data = frame.copy()

    data["weight_missing"] = data["weight"].isna().astype(int)
    data["weight_negative"] = (data["weight"] < 0).astype(int)

    data["weight_clean"] = data["weight"].where(
        data["weight"] >= 0,
        np.nan,
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


train = df[df["date"] <= "2025-09-30"].copy()

holdout = df[
    (df["date"] >= "2025-10-01")
    & (df["date"] <= "2025-10-31")
].copy()


print("=" * 80)
print("FINAL UNTOUCHED OCTOBER HOLDOUT")
print("=" * 80)

print(
    f"Train:   {train['date'].min().date()} -> "
    f"{train['date'].max().date()} "
    f"({len(train):,} rows)"
)

print(
    f"Holdout: {holdout['date'].min().date()} -> "
    f"{holdout['date'].max().date()} "
    f"({len(holdout):,} rows)"
)


X_train, categorical = prepare_features(train)
X_holdout, _ = prepare_features(holdout)

y_train = train["posted_rate"]
y_holdout = holdout["posted_rate"]

cat_indices = [
    X_train.columns.get_loc(column)
    for column in categorical
]


model = CatBoostRegressor(
    iterations=159,
    learning_rate=0.05,
    depth=8,
    loss_function="MAE",
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_indices,
)


pred = model.predict(X_holdout)
pred = np.maximum(pred, 0.01)

mae = mean_absolute_error(y_holdout, pred)

rmse = np.sqrt(
    mean_squared_error(y_holdout, pred)
)

median_ae = median_absolute_error(
    y_holdout,
    pred,
)

r2 = r2_score(
    y_holdout,
    pred,
)

absolute_error = np.abs(
    y_holdout.to_numpy() - pred
)

p90_error = np.percentile(
    absolute_error,
    90,
)

p95_error = np.percentile(
    absolute_error,
    95,
)


print("\nMODEL PERFORMANCE")
print("-" * 80)

print(f"MAE:             ${mae:.2f}")
print(f"Median AE:       ${median_ae:.2f}")
print(f"RMSE:            ${rmse:.2f}")
print(f"R2:               {r2:.4f}")
print(f"90th pct error:  ${p90_error:.2f}")
print(f"95th pct error:  ${p95_error:.2f}")


# --------------------------------------------------
# Error analysis table
# --------------------------------------------------

analysis = holdout[
    [
        "load_id",
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "market_index",
        "quote_signal",
        "posted_rate",
    ]
].copy()

analysis["predicted_rate"] = pred

analysis["absolute_error"] = np.abs(
    analysis["posted_rate"]
    - analysis["predicted_rate"]
)

analysis["percentage_error"] = (
    analysis["absolute_error"]
    / analysis["posted_rate"]
    * 100
)


analysis = analysis.sort_values(
    "absolute_error",
    ascending=False,
)


analysis.to_csv(
    OUTPUT_DIR / "october_holdout_errors.csv",
    index=False,
)


# --------------------------------------------------
# Feature importance
# --------------------------------------------------

importance = pd.DataFrame(
    {
        "feature": X_train.columns,
        "importance": model.get_feature_importance(),
    }
).sort_values(
    "importance",
    ascending=False,
)


importance.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False,
)


print("\nTOP 10 FEATURES")
print("-" * 80)

print(
    importance.head(10).to_string(index=False)
)


print("\nTOP 10 LARGEST ERRORS")
print("-" * 80)

print(
    analysis[
        [
            "load_id",
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "posted_rate",
            "predicted_rate",
            "absolute_error",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


print("\nSaved:")
print("outputs/october_holdout_errors.csv")
print("outputs/feature_importance.csv")
