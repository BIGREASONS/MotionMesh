# Methodology Report

## 1. Problem Setup

Traffic demand prediction at geographic nodes (6-character geohashes) at 15-minute intervals.
Target is continuous (0 to 1). Evaluation metric is R-squared.

Competition training data covers Day 48 in full and Day 49 until 02:00.
Test data covers Day 49 from 02:15 to 13:45.
We use historical traffic observations to build aggregate spatiotemporal profiles.


## 2. External Data Usage and Integrity

A public historical traffic dataset contains 4.2 million observations spanning Days 1 through 61 for the same city.

Only data from Days 1-47 is used. All derived features are aggregate statistics
(means, standard deviations, quantiles, etc) representing location behavioral
profiles. The model does not perform row-level demand lookup or target recovery.


## 3. Feature Architecture

### 3.1 Historical Location Profiles (Days 1-47)

With 47 days of historical data, each geohash has roughly 2,500+ data points.
This allows construction of high-fidelity demand distributions:

- **Geohash level**: mean, std, median, min, max, Q25, Q75, Q90, skewness,
  IQR, range, coefficient of variation, count
- **Geohash + Timestamp**: mean, std, median (exact 15-minute slot profiles)
- **Geohash + Hour**: mean, std, median, min, max (hourly demand curves)
- **Geohash + Time Slot**: mean, std (96-slot daily resolution)

These features encode the structural baseline of each location. A highway
intersection that consistently handles high traffic at 8:00 AM will have
a high `hgh_mean` for hour 8, regardless of what happens on the specific
test day.

### 3.2 Day-of-Week Cyclical Features

Traffic is periodic. A Tuesday at 8 AM behaves more like previous Tuesdays
at 8 AM than like the previous day's 8 AM. By mapping each historical day
to its DOW index (day % 7), we construct per-DOW profiles:

- `same_dow_gh`: The historical average demand at this geohash, at this
  hour, on the same day of the week as Day 48.

This was consistently the highest-importance feature across all model runs
(CatBoost importance ~20). It provides a strong prior that trees can use
as a starting point, then adjust based on real-time covariates.

### 3.3 Recent Trend Features

Long-term averages can miss structural shifts. If a new road was built
on Day 40, the 47-day average would still reflect the old traffic pattern.
We compute separate aggregations from the most recent 8 days and take
the difference from the full-history average:

- `trend_geo` = recent_geo_mean - hgeo_mean
- `trend_gts` = recent_gts_mean - hgts_mean

Positive values indicate growing demand; negative values indicate decline.

### 3.4 Stability and Volatility

Not all locations are equally predictable. We compute:

- `daily_cv`: Coefficient of variation of daily mean demand per geohash
- `hourly_vol`: Standard deviation of hourly demand per (geohash, hour)

The model learns to trust historical means for stable geohashes and rely
more on other features for volatile ones.

### 3.5 Day-48 Lag Features

From competition training data (not external data), we extract the demand
observed at each (geohash, timestamp) on Day 48 and use it as a lag feature
for Day 49 predictions. A hierarchical fallback chain handles missing matches:

geohash+timestamp -> geohash+hour -> geohash -> prefix5+hour -> prefix4+hour -> global hour

To prevent circular leakage, all lag features are set to NaN for Day-48
training rows. Tree models handle NaN natively.


## 4. Why GroupKFold

In spatial datasets, random KFold allows the same geohash to appear in both
train and validation splits. Because target-encoded features and lag features
carry geohash-specific information, this creates severe information leakage:
the model effectively memorizes the geohash identity rather than learning
generalizable patterns.

GroupKFold grouped by geohash ensures complete spatial separation. Each fold
holds out approximately 250 entire geohashes. The model must predict demand
for locations it has never seen during training.

Temporal holdout benchmarks across multiple day-pairs suggest expected
performance in the 0.91-0.95 range. Optimizing against this strict standard
prevents overfitting and ensures that hyperparameter choices are robust.

### Baseline Comparison

To demonstrate that the model learns generalizable patterns beyond simple historical averages, we evaluated performance on out-of-time holdout sets:

| Method | Mean R² |
|--------|---------|
| Mean Baseline | 0.000 |
| Historical Lookup | 0.812 |
| Previous-Day Lookup| 0.827 |
| Ensemble Model | 0.913 |

The ~0.09 R² gap between lookup baselines and the ensemble proves the efficacy of the spatiotemporal feature engineering.


## 5. Ensemble Design

Three gradient boosting frameworks with different tree growth strategies:

- **CatBoost** (symmetric oblivious trees): Resistant to overfitting on
  categorical combinations. Strong with our target-encoded features.
- **LightGBM** (leaf-wise growth): Efficiently minimizes loss on complex
  interactions. Requires careful regularization.
- **XGBoost** (level-wise, histogram method): Consistent and robust.
  Provided the strongest single-model performance in our experiments.

Ensemble methods:
1. **Weighted blend**: Grid search over (CB, LGB, XGB) weight triplets.
2. **Ridge stacking**: L2-regularized linear meta-learner on OOF predictions.

### Pseudo-Labeling

The best ensemble generates predictions on the test set. These are appended
to the training data as soft targets. CatBoost and LightGBM are retrained
on the augmented dataset. This is a standard semi-supervised technique that
can reduce variance when train and test distributions are aligned.

It does not always help. Both pseudo-labeled and non-pseudo-labeled submissions
are generated for comparison.


## 6. Limitations

1. **No spatial adjacency modeling**: Geohash prefixes provide implicit spatial
   hierarchy, but no explicit graph structure or GNN is used.
2. **Performance range**: Temporal holdout benchmarks suggest 0.91-0.95 R².
   Actual competition performance depends on train/test distribution alignment.
3. **External data dependency**: The pipeline requires the historical traffic dataset to
   reproduce results.
4. **No autoregressive features**: The pipeline does not use sequential
   real-time lag features (t-1, t-2) because the test set starts at a
   different time from training, making AR features unreliable.
