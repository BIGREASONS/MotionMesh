# Traffic Demand Prediction

Spatiotemporal regression pipeline for predicting continuous traffic demand at geographic nodes.
Evaluated on R-squared.

## Problem

Predict traffic demand (float, 0 to 1) for specific geohash locations at 15-minute intervals.
The competition provides ~77k training rows spanning Day 48 and the first 2 hours of Day 49.
The test set covers the rest of Day 49 (~42k rows).

## External Historical Data Usage

A historical traffic dataset was used exclusively to construct aggregate features such as:

- geohash historical mean demand
- temporal demand profiles
- volatility statistics
- trend indicators

The model never performs row-level demand lookup for competition instances and generates
predictions solely through trained machine-learning models.

The feature-generation stage is isolated from the training stage. Only derived aggregate
features are passed downstream.

## Approach

1. Load competition train/test and the historical Grab Traffic dataset.
2. Filter historical data to the pre-competition window (Days 1-47).
3. Build ~120 features from that historical window:
   - Per-geohash demand distributions (mean, std, median, quantiles, skewness, CV)
   - Per-(geohash, timestamp) and per-(geohash, hour) demand profiles
   - Same-day-of-week demand patterns (strongest single feature)
   - Recent-window trends (last 8 days vs full history)
   - Spatial prefix hierarchies for coverage fallback
   - Location stability and volatility metrics
4. Build day-48 lag features from competition train data.
5. Train CatBoost, LightGBM, XGBoost with 5-fold GroupKFold on geohash.
6. Optimize blend weights, build ridge stacking meta-learner.
7. Run pseudo-labeling refinement pass.
8. Generate submission files.

## Validation

5-fold GroupKFold grouped by geohash. Each fold holds out ~250 entire geohashes,
which is more conservative than the actual test scenario where only ~10 of ~1190
test geohashes are unseen. CV scores therefore underestimate leaderboard performance.

Reported CV scores (GroupKFold):
- CatBoost: ~0.936
- LightGBM: ~0.936
- XGBoost:  ~0.940
- Blend:    ~0.940

## Reproducibility

All random seeds flow from a single `SEED = 42` constant.

```bash
pip install -r requirements.txt
cd dataset
python solution.py
```

Requires `train.csv`, `test.csv`, and `grab_training.csv` in the `dataset/` directory.

## File Structure

```
Traffic_Demand_Prediction/
├── dataset/
│   ├── solution.py          Main pipeline
│   ├── analyze_grab.py      Data overlap analysis utility
│   ├── train.csv            Competition data (not in git)
│   ├── test.csv             Competition data (not in git)
│   └── grab_training.csv    Historical feature store (not in git)
├── approach.txt             Submission approach document
├── methodology_report.md    Technical write-up
├── requirements.txt         Python dependencies
└── README.md                This file
```

## Limitations

- GroupKFold CV is a pessimistic lower bound of actual performance.
- No spatial adjacency modeling (no GNN or graph structure).
- Pseudo-labeling may not always help; both variants are generated for comparison.
