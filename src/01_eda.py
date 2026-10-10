from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

TRAIN_PATH = DATA_DIR / "train-test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"


train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VALIDATION_PATH)

train["date"] = pd.to_datetime(train["date"], errors="coerce")
validation["date"] = pd.to_datetime(validation["date"], errors="coerce")


print("=" * 70)
print("1. BASIC DATASET INFORMATION")
print("=" * 70)

print(f"Train shape:      {train.shape}")
print(f"Validation shape: {validation.shape}")

print("\nTrain columns:")
print(train.columns.tolist())

print("\nValidation columns:")
print(validation.columns.tolist())


print("\n" + "=" * 70)
print("2. DATE RANGES")
print("=" * 70)

print(
    f"Train:      {train['date'].min().date()} -> "
    f"{train['date'].max().date()}"
)

print(
    f"Validation: {validation['date'].min().date()} -> "
    f"{validation['date'].max().date()}"
)


print("\n" + "=" * 70)
print("3. MISSING VALUES - TRAIN")
print("=" * 70)

train_missing = train.isna().sum()
print(train_missing[train_missing > 0].sort_values(ascending=False))


print("\n" + "=" * 70)
print("4. MISSING VALUES - VALIDATION")
print("=" * 70)

validation_missing = validation.isna().sum()
print(validation_missing[validation_missing > 0].sort_values(ascending=False))


print("\n" + "=" * 70)
print("5. DUPLICATE IDs")
print("=" * 70)

print(
    "Train duplicate load_id:",
    train["load_id"].duplicated().sum()
)

print(
    "Validation duplicate load_id:",
    validation["load_id"].duplicated().sum()
)


print("\n" + "=" * 70)
print("6. NEGATIVE NUMERIC VALUES")
print("=" * 70)

numeric_columns = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
]

for column in numeric_columns:
    if column in train.columns:
        print(
            f"Train {column}: "
            f"{(train[column] < 0).sum()} negative values"
        )

    if column in validation.columns:
        print(
            f"Validation {column}: "
            f"{(validation[column] < 0).sum()} negative values"
        )


print("\n" + "=" * 70)
print("7. TARGET: posted_rate")
print("=" * 70)

print(train["posted_rate"].describe())

print(
    "\nNegative posted_rate:",
    (train["posted_rate"] <= 0).sum()
)


train["rate_per_mile"] = (
    train["posted_rate"] / train["distance"]
)

print("\n" + "=" * 70)
print("8. RATE PER MILE")
print("=" * 70)

print(train["rate_per_mile"].describe())


print("\n" + "=" * 70)
print("9. CATEGORICAL FEATURES")
print("=" * 70)

categorical_columns = [
    "pickup",
    "delivery",
    "equipment",
]

for column in categorical_columns:
    train_values = set(train[column].dropna().unique())
    validation_values = set(
        validation[column].dropna().unique()
    )

    unseen = validation_values - train_values

    print(f"\n{column}")
    print(f"Train unique:      {len(train_values)}")
    print(f"Validation unique: {len(validation_values)}")
    print(f"Unseen values:     {len(unseen)}")

    if unseen:
        print("New values:")
        print(sorted(unseen))



known_pickups = set(train["pickup"].dropna())
known_deliveries = set(train["delivery"].dropna())

unseen_location_rows = validation[
    (~validation["pickup"].isin(known_pickups))
    |
    (~validation["delivery"].isin(known_deliveries))
]

print("\n" + "=" * 70)
print("10. VALIDATION ROWS WITH UNSEEN LOCATIONS")
print("=" * 70)

print(
    f"{len(unseen_location_rows):,} / "
    f"{len(validation):,}"
)

print(
    f"Percentage: "
    f"{len(unseen_location_rows) / len(validation) * 100:.2f}%"
)


print("\n" + "=" * 70)
print("11. EQUIPMENT DISTRIBUTION")
print("=" * 70)

print("\nTrain:")
print(train["equipment"].value_counts(dropna=False))

print("\nValidation:")
print(validation["equipment"].value_counts(dropna=False))


print("\n" + "=" * 70)
print("12. MONTHLY RATE BEHAVIOR")
print("=" * 70)

monthly_summary = (
    train.assign(month=train["date"].dt.to_period("M"))
    .groupby("month")
    .agg(
        loads=("load_id", "count"),
        avg_posted_rate=("posted_rate", "mean"),
        avg_rate_per_mile=("rate_per_mile", "mean"),
        avg_market_index=("market_index", "mean"),
        avg_quote_signal=("quote_signal", "mean"),
    )
)

print(monthly_summary)


print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)