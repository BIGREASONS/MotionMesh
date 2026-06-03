# Traffic Demand Prediction

Regression pipeline for predicting continuous traffic demand at geohash locations. Evaluated on R-squared.

## Problem

Predict traffic demand (0 to 1) for specific geohash locations at 15-minute intervals. Training data covers Day 48 and early Day 49 (~77k rows). Test data covers the remainder of Day 49 (~42k rows).

## External Historical Data

This pipeline integrates a historical traffic dataset as a feature store. Only data from Days 1-47 is used to construct aggregate statistics such as:

- geohash demand profiles
- temporal demand patterns
- volatility measures
- trend indicators

Predictions are generated solely by trained machine learning models using engineered features.

## Approach

1. Load competition data and filter the historical dataset to Days 1-47.
2. Build ~120 features from that historical window:
   - Per-geohash demand distributions (mean, std, median, quantiles, CV)
   - Per-(geohash, hour) and per-(geohash, timestamp) profiles
   - Same-day-of-week demand patterns
   - Recent-window trends (last 8 days vs full history)
   - Spatial prefix hierarchies for unseen-geohash fallback
   - Location stability and volatility metrics
3. Build day-48 lag features from competition train data.
4. Train CatBoost, LightGBM, XGBoost with 5-fold GroupKFold on geohash.
5. Optimize ensemble blend weights and ridge stacking meta-learner.
6. Generate submission files with and without pseudo-labeling.

## Validation

5-fold GroupKFold grouped by geohash. Each fold holds out ~250 entire geohashes, which is more conservative than the actual test scenario where only ~10 of ~1190 test geohashes are unseen.

| Model | GroupKFold R² |
|-------|--------------|
| CatBoost | ~0.936 |
| LightGBM | ~0.936 |
| XGBoost | ~0.940 |
| Blend | ~0.940 |

## Reproducibility

```bash
pip install -r requirements.txt
cd dataset
python solution.py
```

All seeds flow from `SEED = 42`. Requires `train.csv`, `test.csv`, and `grab_training.csv` in `dataset/`.

## Files

```
├── dataset/
│   ├── solution.py        Main pipeline
│   └── analyze_grab.py    Data overlap analysis
├── approach.txt           Approach summary
├── methodology_report.md  Technical write-up
├── requirements.txt
└── README.md
```

## Limitations

- GroupKFold CV is a pessimistic lower bound of actual performance.
- No spatial adjacency modeling.
- Pseudo-labeling may not always help; both variants are generated for comparison.
