from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train-test.csv"


df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df.sort_values("date").reset_index(drop=True)


FOLDS = [
    {
        "name": "Fold 1",
        "train_end": "2025-06-30",
        "valid_start": "2025-07-01",
        "valid_end": "2025-07-31",
    },
    {
        "name": "Fold 2",
        "train_end": "2025-07-31",
        "valid_start": "2025-08-01",
        "valid_end": "2025-08-31",
    },
    {
        "name": "Fold 3",
        "train_end": "2025-08-31",
        "valid_start": "2025-09-01",
        "valid_end": "2025-09-30",
    },
]

FINAL_HOLDOUT = {
    "name": "Final Holdout",
    "train_end": "2025-09-30",
    "valid_start": "2025-10-01",
    "valid_end": "2025-10-31",
}


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def evaluate_split(train, valid, split_name):

    print("\n" + "=" * 75)
    print(split_name)
    print("=" * 75)

    print(
        f"Train: {train['date'].min().date()} -> "
        f"{train['date'].max().date()} "
        f"({len(train):,} rows)"
    )

    print(
        f"Valid: {valid['date'].min().date()} -> "
        f"{valid['date'].max().date()} "
        f"({len(valid):,} rows)"
    )


    known_pickups = set(train["pickup"].dropna())
    known_deliveries = set(train["delivery"].dropna())

    unseen_rows = valid[
        (~valid["pickup"].isin(known_pickups))
        |
        (~valid["delivery"].isin(known_deliveries))
    ]

    print(
        f"Rows with unseen location: "
        f"{len(unseen_rows):,} "
        f"({len(unseen_rows) / len(valid) * 100:.2f}%)"
    )


    train_rpm = train["posted_rate"] / train["distance"]

    global_median_rpm = train_rpm.median()


    pred_global = valid["distance"] * global_median_rpm

    global_metrics = calculate_metrics(
        valid["posted_rate"],
        pred_global,
    )

    print("\nBaseline 1 - Global median rate-per-mile")
    print(f"Median RPM: ${global_median_rpm:.4f}")

    print(
        f"MAE:  ${global_metrics['MAE']:.2f}"
    )
    print(
        f"RMSE: ${global_metrics['RMSE']:.2f}"
    )
    print(
        f"R2:   {global_metrics['R2']:.4f}"
    )


    temp = train[["equipment"]].copy()
    temp["rate_per_mile"] = train_rpm

    rpm_by_equipment = (
        temp
        .groupby("equipment")["rate_per_mile"]
        .median()
    )

    valid_rpm = (
        valid["equipment"]
        .map(rpm_by_equipment)
        .fillna(global_median_rpm)
    )

    pred_equipment = valid["distance"] * valid_rpm

    equipment_metrics = calculate_metrics(
        valid["posted_rate"],
        pred_equipment,
    )

    print("\nBaseline 2 - Median RPM by equipment")

    print(rpm_by_equipment)

    print(
        f"\nMAE:  ${equipment_metrics['MAE']:.2f}"
    )
    print(
        f"RMSE: ${equipment_metrics['RMSE']:.2f}"
    )
    print(
        f"R2:   {equipment_metrics['R2']:.4f}"
    )

    return {
        "split": split_name,
        "global_mae": global_metrics["MAE"],
        "equipment_mae": equipment_metrics["MAE"],
        "global_rmse": global_metrics["RMSE"],
        "equipment_rmse": equipment_metrics["RMSE"],
    }


results = []

print("=" * 75)
print("TIME-BASED VALIDATION")
print("=" * 75)

for fold in FOLDS:

    train_mask = (
        df["date"]
        <= pd.Timestamp(fold["train_end"])
    )

    valid_mask = (
        (df["date"] >= pd.Timestamp(fold["valid_start"]))
        &
        (df["date"] <= pd.Timestamp(fold["valid_end"]))
    )

    train_fold = df.loc[train_mask].copy()
    valid_fold = df.loc[valid_mask].copy()

    result = evaluate_split(
        train_fold,
        valid_fold,
        fold["name"],
    )

    results.append(result)

results_df = pd.DataFrame(results)

print("\n" + "=" * 75)
print("DEVELOPMENT CV SUMMARY")
print("=" * 75)

print(results_df)

print("\nAverage Global RPM MAE:")
print(f"${results_df['global_mae'].mean():.2f}")

print("\nAverage Equipment RPM MAE:")
print(f"${results_df['equipment_mae'].mean():.2f}")


train_mask = (
    df["date"]
    <= pd.Timestamp(FINAL_HOLDOUT["train_end"])
)

valid_mask = (
    (df["date"] >= pd.Timestamp(FINAL_HOLDOUT["valid_start"]))
    &
    (df["date"] <= pd.Timestamp(FINAL_HOLDOUT["valid_end"]))
)

train_final = df.loc[train_mask].copy()
valid_final = df.loc[valid_mask].copy()

evaluate_split(
    train_final,
    valid_final,
    FINAL_HOLDOUT["name"],
)
