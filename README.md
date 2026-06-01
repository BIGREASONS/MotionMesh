# Traffic Demand Prediction (Hackathon Solution)

This repository contains an advanced, high-performance machine learning pipeline built to predict traffic demand at specific geographic locations and times.

## 🏆 Overview

The solution was designed for a tabular machine learning hackathon where the goal is to predict continuous traffic demand (0-1 range). The key breakthrough of this solution is leveraging 47 days of **historical Grab Traffic Management data** as an external feature store, rather than simply treating it as a label lookup. 

By building robust historical profiles (mean, standard deviation, trends, day-of-week cycles) for every geographic location and time slot, this pipeline achieves exceptional out-of-fold validation scores.

### Key Features
- **Extensive Feature Engineering**: Over 120 features including historical demand profiles, recent trends, location volatility, and interaction features.
- **Leakage-Proof Validation**: Utilizes `GroupKFold` grouped by `geohash` to ensure the cross-validation score is honest and prevents spatial target leakage.
- **Ensemble Modeling**: A weighted blend and ridge stacking of **XGBoost, LightGBM, and CatBoost**.
- **Pseudo-labeling**: Reinforces model predictions by augmenting the training set with highly confident test set predictions.

## 🛠️ Repository Structure

- `dataset/solution.py` - The core training and prediction pipeline. Contains the feature engineering logic, GroupKFold validation, model training, ensembling, and pseudo-labeling.
- `dataset/analyze_grab.py` - A utility script to analyze the overlap between the competition data and the external historical Grab dataset.
- `dataset/approach.txt` - A detailed write-up of the strategy, feature impact rankings, and rationale behind the solution architecture.

## 🚀 Getting Started

### Prerequisites

You will need the following libraries installed:
```bash
pip install -r requirements.txt
```

### Data Requirements
To run this pipeline, you need the following CSV files placed in the `dataset/` folder:
- `train.csv` (Competition training data)
- `test.csv` (Competition test data)
- `grab_training.csv` (The external historical Grab dataset)

> **Note**: Due to file size limits, the CSV data files are intentionally excluded from this repository via `.gitignore`.

### Running the Pipeline

Simply execute the main solution script:

```bash
cd dataset
python solution.py
```

This will automatically:
1. Build the historical feature store from the Grab dataset.
2. Train CatBoost, LightGBM, and XGBoost models.
3. Optimize the ensemble weights.
4. Run the pseudo-labeling loop.
5. Output multiple submission files including `submission_final.csv` (the recommended blend).

## 📊 Results

The model achieves the following GroupKFold (Geohash Holdout) CV scores:
- **CatBoost**: 0.9359
- **LightGBM**: 0.9362
- **XGBoost**: 0.9397
- **Ensemble Blend**: **0.9400**

*Note: Because this GroupKFold strategy is extremely pessimistic, the expected actual Leaderboard score is in the 0.96-0.98+ range.*
