# Spotter Freight Rate Prediction

Machine learning solution for the Spotter Machine Learning Engineer assessment.

The objective is to predict freight rates for 12,000 unseen loads using historical labeled freight data while maintaining a validation strategy that reflects future production inference.

## Project Overview

The labeled development dataset contains 48,000 freight loads from January through October 2025.

The final validation dataset contains 12,000 unlabeled loads from November through December 2025.

Because the evaluation period occurs after the development period, I used chronological validation rather than a random train/test split.

## Data Quality Findings

The exploratory analysis identified several data-quality considerations:

- Missing values in `weight` and `market_index`.
- Negative values in `weight`.
- No duplicated `load_id` values.
- Eight locations appear in the final validation data that do not occur in the development data.
- The target distribution contains a small number of unusually high freight rates.

Negative weights were treated as missing values after comparing multiple handling strategies during temporal cross-validation.

Geographic coordinates were retained so the model could generalize better to locations that were not represented as categorical city values during training.

## Validation Strategy

I used expanding-window temporal validation.

Development folds:

- Train January-June, validate July.
- Train January-July, validate August.
- Train January-August, validate September.

October was kept untouched as a final internal holdout.

This validation design prevents future observations from leaking into earlier training periods and more closely represents the final November-December inference scenario.

## Baseline

A historical rate-per-mile baseline was evaluated first.

The stronger baseline used the historical median rate-per-mile by equipment type.

Average temporal CV MAE:

`$226.01`

October baseline MAE:

`$231.30`

## Model Selection

CatBoost was selected because the dataset contains a combination of:

- Numerical features
- Categorical route and equipment information
- Missing numerical values
- Geographic information
- Temporal features

Experiments compared:

- Direct freight-rate prediction
- Rate-per-mile prediction
- Multiple strategies for negative weights

Direct prediction of `posted_rate` with negative weights treated as missing produced the strongest and most stable development performance.

Average development CV:

- MAE: approximately `$114.14`
- RMSE: approximately `$621.78`
- R²: approximately `0.828`

## Final Holdout Performance

The selected configuration was evaluated once on the untouched October holdout.

Results:

- MAE: `$118.50`
- Median Absolute Error: `$40.31`
- RMSE: `$651.08`
- R²: `0.8186`
- 90th percentile absolute error: `$144.24`
- 95th percentile absolute error: `$200.34`

The difference between MAE and RMSE is driven by a small heavy-tailed subset of unusually high freight rates.

These observations were retained rather than removed without a deterministic data-quality rule. An MAE training objective was used to reduce their influence.

## Feature Engineering

The model uses:

- Pickup and delivery city
- Route
- Pickup and delivery coordinates
- Distance
- Equipment
- Cleaned weight
- Weight missing/negative indicators
- Market index
- Quote signal
- Month
- Day of week
- Day of year
- Cyclical date features

The final model's feature importance indicates that distance is the dominant predictor, followed by equipment and additional market/geographic signals.

## December Prediction Chart

The supplied December input does not contain `market_index`, `quote_signal`, or geographic coordinates.

Coordinates for Lexington and Fort Wayne were recovered from the labeled development data.

A second CatBoost model was therefore trained using the common feature set available for December inference.

During development validation, removing `market_index` and `quote_signal` changed MAE from approximately `$114.14` to `$115.48`, so the reduced feature model retained similar predictive performance.

The generated December chart is available at:

`scorer_results/candidate_december.png`

## Final Validation Predictions

The final model was retrained on all 48,000 labeled development rows and used to predict every row in `validation.csv`.

The final submission file is:

`validation_predictions.csv`

It contains exactly:

```text
load_id,predicted_rate
```

for all 12,000 validation loads.

## Project Structure

```text
.
├── src/
│   ├── 01_eda.py
│   ├── 02_validation_strategy.py
│   ├── 03_catboost_experiments.py
│   ├── 04_final_holdout.py
│   ├── 05_common_feature_check.py
│   └── 06_generate_predictions.py
│
├── outputs/
│   └── december_chart_inputs_completed.csv
│
├── scorer_results/
│   └── candidate_december.png
│
├── validation_predictions.csv
├── requirements.txt
├── score.py
└── README.md
```

Raw assessment datasets are intentionally excluded from the repository.

## Setup

Create or activate a Python environment and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Place the assessment CSV files inside:

```text
data/
```

Required files:

```text
train-test.csv
validation.csv
validation-predictions-template.csv
december-chart-inputs.csv
```

## Run

Exploratory analysis:

```bash
python src/01_eda.py
```

Baseline and temporal validation:

```bash
python src/02_validation_strategy.py
```

Model experiments:

```bash
python src/03_catboost_experiments.py
```

Final October holdout:

```bash
python src/04_final_holdout.py
```

Common-feature validation:

```bash
python src/05_common_feature_check.py
```

Generate final predictions:

```bash
python src/06_generate_predictions.py
```

Validate the final files with the supplied Spotter scorer:

```bash
python score.py --predictions validation_predictions.csv --december-predictions outputs/december_chart_inputs_completed.csv
```

Successful validation confirms:

```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
```

Final validation performance on the 12,000 unlabeled loads is calculated by Spotter after submission.
