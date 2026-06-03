# Traffic Demand Prediction

Spatiotemporal regression pipeline for predicting continuous traffic demand at geographic nodes.  
Evaluated on R-squared.

## What This Does

Predicts traffic demand (float, 0 to 1) for specific geohash locations at 15-minute intervals. 

The competition provides ~77k training rows spanning Day 48 and the first 2 hours of Day 49. The test set covers the rest of Day 49 (~42k rows).

This pipeline integrates the public Grab Traffic Management dataset as a historical feature store.
Days 1 through 47 are used to build statistical location profiles.
The raw `grab` variable is deleted immediately after filtering to prevent accidental downstream usage.

No test labels are recovered. No row-level demand lookup is performed.
All external features are aggregate statistics from the historical window.

## How It Works

1. Load competition train/test and the Grab historical dataset.
2. Filter Grab to days 1-47 only, then `del grab_raw` (line 21 of solution.py).
3. Build ~120 features from that historical window:
   - Per-geohash demand profiles (mean, std, median, quantiles, skewness, CV)
   - Per-(geohash, timestamp) and per-(geohash, hour) historical averages
   - Same-day-of-week historical demand (the strongest single feature)
   - Recent-window trends (last 8 days vs full history)
   - Spatial prefix hierarchies (prefix-4, prefix-5 fallbacks)
   - Location stability and volatility metrics
4. Build day-48 lag features from competition train data.
5. Train CatBoost, LightGBM, XGBoost with 5-fold GroupKFold on geohash.
6. Optimize blend weights, build ridge stacking meta-learner.
7. Run pseudo-labeling refinement pass.
8. Generate multiple submission files.

## Validation

5-fold GroupKFold grouped by `geohash`.

This is intentionally pessimistic: each fold holds out ~250 entire geohashes,
simulating prediction on completely unseen locations. Since only ~10 of ~1190
test geohashes are actually unseen, real LB performance is expected to be higher.

Reported CV scores (GroupKFold):
- CatBoost: ~0.936
- LightGBM: ~0.936
- XGBoost:  ~0.940
- Blend:    ~0.940

Actual feature count is printed at runtime by solution.py.

## Reproducibility

All random seeds flow from a single `SEED = 42` constant at the top of solution.py.

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

## Leakage Verification

Run this to confirm no use of raw Grab data after the filter:

```bash
findstr /n "grab_raw" dataset\solution.py
```

Expected output: only lines 12, 15, 17, 20 (load, rename, filter, delete).
After line 20, the variable no longer exists in memory.

## Limitations

- GroupKFold CV is a pessimistic lower bound.
- The pipeline does not model spatial adjacency (no GNN/graph structure).
- Pseudo-labeling may not always help; submit both with and without.
