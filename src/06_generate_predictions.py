from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


TRAIN_PATH = DATA_DIR / "train-test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation-predictions-template.csv"
DECEMBER_PATH = DATA_DIR / "december-chart-inputs.csv"


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VALIDATION_PATH)
template = pd.read_csv(TEMPLATE_PATH)
december = pd.read_csv(DECEMBER_PATH)

train["date"] = pd.to_datetime(train["date"], errors="coerce")
validation["date"] = pd.to_datetime(
    validation["date"],
    errors="coerce",
)
december["date"] = pd.to_datetime(
    december["date"],
    errors="coerce",
)


# ============================================================
# FEATURE ENGINEERING - FULL MODEL
# ============================================================

def prepare_full_features(frame):

    data = frame.copy()

    # Weight-quality features
    data["weight_missing"] = (
        data["weight"].isna().astype(int)
    )

    data["weight_negative"] = (
        (data["weight"] < 0).astype(int)
    )

    # Best strategy found during development CV
    data["weight_clean"] = data["weight"].where(
        data["weight"] >= 0,
        np.nan,
    )

    # Time features
    data["month"] = data["date"].dt.month
    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_year"] = data["date"].dt.dayofyear

    data["day_sin"] = np.sin(
        2
        * np.pi
        * data["day_of_year"]
        / 365.25
    )

    data["day_cos"] = np.cos(
        2
        * np.pi
        * data["day_of_year"]
        / 365.25
    )

    # Route feature
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


# ============================================================
# FEATURE ENGINEERING - COMMON / DECEMBER MODEL
# ============================================================

def prepare_common_features(frame):

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
        2
        * np.pi
        * data["day_of_year"]
        / 365.25
    )

    data["day_cos"] = np.cos(
        2
        * np.pi
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


# ============================================================
# 1. FINAL MAIN MODEL
#
# Uses all 48,000 labeled rows.
#
# 159 trees = median best tree count from the
# development temporal CV for the winning model.
# ============================================================

print("=" * 80)
print("TRAINING FINAL FULL-FEATURE MODEL")
print("=" * 80)

X_train, categorical = prepare_full_features(train)
X_validation, _ = prepare_full_features(validation)

cat_indices = [
    X_train.columns.get_loc(column)
    for column in categorical
]

final_model = CatBoostRegressor(
    iterations=159,
    learning_rate=0.05,
    depth=8,
    loss_function="MAE",
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
)

final_model.fit(
    X_train,
    train["posted_rate"],
    cat_features=cat_indices,
)

validation_prediction = final_model.predict(
    X_validation
)

validation_prediction = np.maximum(
    validation_prediction,
    0.01,
)


# ============================================================
# Validate IDs before creating submission
# ============================================================

if len(validation) != 12000:
    raise ValueError(
        f"Expected 12,000 validation rows, "
        f"found {len(validation):,}"
    )

if validation["load_id"].duplicated().any():
    raise ValueError(
        "Duplicate validation load_id detected."
    )

prediction_map = pd.DataFrame(
    {
        "load_id": validation["load_id"],
        "predicted_rate": validation_prediction,
    }
)


# Use the supplied template order
submission = (
    template[["load_id"]]
    .merge(
        prediction_map,
        on="load_id",
        how="left",
        validate="one_to_one",
    )
)

if submission["predicted_rate"].isna().any():
    raise ValueError(
        "Some template IDs did not receive predictions."
    )

if len(submission) != 12000:
    raise ValueError(
        "Final submission does not contain 12,000 rows."
    )


SUBMISSION_PATH = ROOT / "validation_predictions.csv"

submission.to_csv(
    SUBMISSION_PATH,
    index=False,
)


print("\nMAIN PREDICTIONS")
print("-" * 80)

print(f"Rows: {len(submission):,}")
print(
    f"Minimum prediction: "
    f"${submission['predicted_rate'].min():.2f}"
)
print(
    f"Maximum prediction: "
    f"${submission['predicted_rate'].max():.2f}"
)
print(
    f"Mean prediction: "
    f"${submission['predicted_rate'].mean():.2f}"
)
print(
    f"Median prediction: "
    f"${submission['predicted_rate'].median():.2f}"
)

print(
    f"\nSaved:\n{SUBMISSION_PATH.name}"
)


# ============================================================
# 2. BUILD CITY COORDINATE LOOKUP
#
# December file does not provide lat/lon.
# We derive coordinates from the labeled data.
# ============================================================

pickup_locations = train[
    [
        "pickup",
        "pickup_lat",
        "pickup_lon",
    ]
].rename(
    columns={
        "pickup": "city",
        "pickup_lat": "lat",
        "pickup_lon": "lon",
    }
)

delivery_locations = train[
    [
        "delivery",
        "delivery_lat",
        "delivery_lon",
    ]
].rename(
    columns={
        "delivery": "city",
        "delivery_lat": "lat",
        "delivery_lon": "lon",
    }
)

locations = pd.concat(
    [
        pickup_locations,
        delivery_locations,
    ],
    ignore_index=True,
)

# Median makes the lookup robust if tiny coordinate
# inconsistencies exist for the same city.
city_coordinates = (
    locations
    .groupby("city")[["lat", "lon"]]
    .median()
)


def lookup_coordinate(city, coordinate):

    if city not in city_coordinates.index:
        raise ValueError(
            f"No training coordinates found for city: {city}"
        )

    return city_coordinates.loc[
        city,
        coordinate,
    ]


# ============================================================
# Enrich December inputs
# ============================================================

december_model_input = december.copy()

december_model_input["pickup_lat"] = (
    december_model_input["pickup"]
    .map(
        lambda city:
        lookup_coordinate(city, "lat")
    )
)

december_model_input["pickup_lon"] = (
    december_model_input["pickup"]
    .map(
        lambda city:
        lookup_coordinate(city, "lon")
    )
)

december_model_input["delivery_lat"] = (
    december_model_input["delivery"]
    .map(
        lambda city:
        lookup_coordinate(city, "lat")
    )
)

december_model_input["delivery_lon"] = (
    december_model_input["delivery"]
    .map(
        lambda city:
        lookup_coordinate(city, "lon")
    )
)


# ============================================================
# 3. DECEMBER COMMON-FEATURE MODEL
#
# market_index and quote_signal are intentionally excluded
# because those fields are not supplied in the December input.
#
# 488 trees = median best tree count from common-feature CV.
# ============================================================

print("\n" + "=" * 80)
print("TRAINING DECEMBER COMMON-FEATURE MODEL")
print("=" * 80)

X_train_common, common_categorical = (
    prepare_common_features(train)
)

X_december, _ = prepare_common_features(
    december_model_input
)

common_cat_indices = [
    X_train_common.columns.get_loc(column)
    for column in common_categorical
]

december_model = CatBoostRegressor(
    iterations=488,
    learning_rate=0.05,
    depth=8,
    loss_function="MAE",
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
)

december_model.fit(
    X_train_common,
    train["posted_rate"],
    cat_features=common_cat_indices,
)

december_prediction = december_model.predict(
    X_december
)

december_prediction = np.maximum(
    december_prediction,
    0.01,
)


# ============================================================
# Restore original December file structure exactly
# ============================================================

december_output = december[
    [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "predicted_rate",
    ]
].copy()

december_output["date"] = (
    december_output["date"]
    .dt.strftime("%Y-%m-%d")
)

december_output["predicted_rate"] = (
    december_prediction
)


DECEMBER_OUTPUT_PATH = (
    OUTPUT_DIR
    / "december_chart_inputs_completed.csv"
)

december_output.to_csv(
    DECEMBER_OUTPUT_PATH,
    index=False,
)


print("\nDECEMBER PREDICTIONS")
print("-" * 80)

print(f"Rows: {len(december_output)}")

print(
    f"Minimum prediction: "
    f"${december_output['predicted_rate'].min():.2f}"
)

print(
    f"Maximum prediction: "
    f"${december_output['predicted_rate'].max():.2f}"
)

print(
    f"Mean prediction: "
    f"${december_output['predicted_rate'].mean():.2f}"
)

print(
    "\nFirst five predictions:"
)

print(
    december_output[
        [
            "date",
            "predicted_rate",
        ]
    ]
    .head()
    .to_string(index=False)
)

print(
    f"\nSaved:\n"
    f"outputs/"
    f"{DECEMBER_OUTPUT_PATH.name}"
)


# ============================================================
# Feature importance - final main model
# ============================================================

feature_importance = pd.DataFrame(
    {
        "feature": X_train.columns,
        "importance":
            final_model.get_feature_importance(),
    }
).sort_values(
    "importance",
    ascending=False,
)

feature_importance.to_csv(
    OUTPUT_DIR
    / "final_feature_importance.csv",
    index=False,
)


print("\n" + "=" * 80)
print("FILES READY")
print("=" * 80)

print(
    "validation_predictions.csv"
)

print(
    "outputs/"
    "december_chart_inputs_completed.csv"
)

print(
    "outputs/"
    "final_feature_importance.csv"
)