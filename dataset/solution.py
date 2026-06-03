import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

grab_raw = pd.read_csv('grab_training.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
target = 'demand'
test_idx = test['Index'].values

grab_raw.rename(columns={'geohash6': 'geohash'}, inplace=True)

hist = grab_raw[grab_raw['day'] <= 47].copy()
del grab_raw
print(f"Historical data: {len(hist)} rows, days 1-47")
print(f"Unique geohashes in hist: {hist['geohash'].nunique()}")

parts_h = hist['timestamp'].str.split(':', expand=True).astype(int)
hist['hour'] = parts_h[0]
hist['minute'] = parts_h[1]
hist['time_slot'] = hist['hour'] * 4 + hist['minute'] // 15

full = pd.concat([train.drop(columns=[target]), test], axis=0, ignore_index=True)
n_train = len(train)

parts = full['timestamp'].str.split(':', expand=True).astype(int)
full['hour'] = parts[0]
full['minute'] = parts[1]
full['time_slot'] = full['hour'] * 4 + full['minute'] // 15

full['hour_sin'] = np.sin(2 * np.pi * full['hour'] / 24)
full['hour_cos'] = np.cos(2 * np.pi * full['hour'] / 24)
full['slot_sin'] = np.sin(2 * np.pi * full['time_slot'] / 96)
full['slot_cos'] = np.cos(2 * np.pi * full['time_slot'] / 96)

full['is_rush_morning'] = full['hour'].isin([7, 8, 9, 10]).astype(int)
full['is_rush_evening'] = full['hour'].isin([16, 17, 18, 19]).astype(int)
full['is_night'] = ((full['hour'] >= 22) | (full['hour'] <= 5)).astype(int)
full['is_midday'] = full['hour'].isin([11, 12, 13, 14]).astype(int)
full['period'] = pd.cut(full['hour'], bins=[-1, 5, 11, 17, 23], labels=[0, 1, 2, 3]).astype(int)

full['geo_prefix4'] = full['geohash'].str[:4]
full['geo_prefix5'] = full['geohash'].str[:5]
full['geo_char5'] = full['geohash'].str[4]
full['geo_char6'] = full['geohash'].str[5]

base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
char_val = {c: i for i, c in enumerate(base32)}
full['geo_c5_ord'] = full['geo_char5'].map(char_val).fillna(0).astype(int)
full['geo_c6_ord'] = full['geo_char6'].map(char_val).fillna(0).astype(int)

full['LargeVehicles_enc'] = (full['LargeVehicles'] == 'Allowed').astype(int)
full['Landmarks_enc'] = (full['Landmarks'] == 'Yes').astype(int)
full['RoadType_enc'] = full['RoadType'].map({'Residential': 0, 'Street': 1, 'Highway': 2}).fillna(-1).astype(int)
full['Weather_enc'] = full['Weather'].map({'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3}).fillna(-1).astype(int)
full['Temperature'] = full['Temperature'].fillna(full['Temperature'].median())

full['temp_x_lanes'] = full['Temperature'] * full['NumberofLanes']
full['temp_x_landmark'] = full['Temperature'] * full['Landmarks_enc']
full['lanes_x_large'] = full['NumberofLanes'] * full['LargeVehicles_enc']
full['temp_sq'] = full['Temperature'] ** 2
full['road_weather'] = full['RoadType_enc'] * 10 + full['Weather_enc']
full['road_lanes'] = full['RoadType_enc'] * 10 + full['NumberofLanes']

for col in ['geohash', 'geo_prefix4', 'geo_prefix5']:
    freq = full[col].value_counts()
    full[col + '_freq'] = full[col].map(freq)
full['log_geo_freq'] = np.log1p(full['geohash_freq'])


print("Building historical profile features from 47 days of Grab data...")

hist_geo = hist.groupby('geohash')['demand'].agg(['mean', 'std', 'median', 'min', 'max', 'count',
                                                    lambda x: x.quantile(0.25),
                                                    lambda x: x.quantile(0.75),
                                                    lambda x: x.quantile(0.90),
                                                    'skew']).reset_index()
hist_geo.columns = ['geohash', 'hgeo_mean', 'hgeo_std', 'hgeo_med', 'hgeo_min', 'hgeo_max',
                    'hgeo_cnt', 'hgeo_q25', 'hgeo_q75', 'hgeo_q90', 'hgeo_skew']
hist_geo['hgeo_iqr'] = hist_geo['hgeo_q75'] - hist_geo['hgeo_q25']
hist_geo['hgeo_range'] = hist_geo['hgeo_max'] - hist_geo['hgeo_min']
hist_geo['hgeo_cv'] = hist_geo['hgeo_std'] / (hist_geo['hgeo_mean'] + 1e-8)
full = full.merge(hist_geo, on='geohash', how='left')

hist_geo_ts = hist.groupby(['geohash', 'timestamp'])['demand'].agg(['mean', 'std', 'median', 'count']).reset_index()
hist_geo_ts.columns = ['geohash', 'timestamp', 'hgts_mean', 'hgts_std', 'hgts_med', 'hgts_cnt']
full = full.merge(hist_geo_ts, on=['geohash', 'timestamp'], how='left')

hist_geo_hour = hist.groupby(['geohash', 'hour'])['demand'].agg(['mean', 'std', 'median', 'min', 'max']).reset_index()
hist_geo_hour.columns = ['geohash', 'hour', 'hgh_mean', 'hgh_std', 'hgh_med', 'hgh_min', 'hgh_max']
full = full.merge(hist_geo_hour, on=['geohash', 'hour'], how='left')

hist_geo_slot = hist.groupby(['geohash', 'time_slot'])['demand'].agg(['mean', 'std']).reset_index()
hist_geo_slot.columns = ['geohash', 'time_slot', 'hgs_mean', 'hgs_std']
full = full.merge(hist_geo_slot, on=['geohash', 'time_slot'], how='left')

hist_hour = hist.groupby('hour')['demand'].agg(['mean', 'std']).reset_index()
hist_hour.columns = ['hour', 'hh_mean', 'hh_std']
full = full.merge(hist_hour, on='hour', how='left')

hist_slot = hist.groupby('time_slot')['demand'].agg(['mean']).reset_index()
hist_slot.columns = ['time_slot', 'hslot_mean']
full = full.merge(hist_slot, on='time_slot', how='left')

hist['geo_prefix4'] = hist['geohash'].str[:4]
hist['geo_prefix5'] = hist['geohash'].str[:5]

hist_p5h = hist.groupby(['geo_prefix5', 'hour'])['demand'].agg(['mean', 'std']).reset_index()
hist_p5h.columns = ['geo_prefix5', 'hour', 'hp5h_mean', 'hp5h_std']
full = full.merge(hist_p5h, on=['geo_prefix5', 'hour'], how='left')

hist_p4h = hist.groupby(['geo_prefix4', 'hour'])['demand'].agg(['mean']).reset_index()
hist_p4h.columns = ['geo_prefix4', 'hour', 'hp4h_mean']
full = full.merge(hist_p4h, on=['geo_prefix4', 'hour'], how='left')

hist_p5 = hist.groupby('geo_prefix5')['demand'].agg(['mean', 'std', 'count']).reset_index()
hist_p5.columns = ['geo_prefix5', 'hp5_mean', 'hp5_std', 'hp5_cnt']
full = full.merge(hist_p5, on='geo_prefix5', how='left')

hist_p4 = hist.groupby('geo_prefix4')['demand'].agg(['mean', 'std', 'count']).reset_index()
hist_p4.columns = ['geo_prefix4', 'hp4_mean', 'hp4_std', 'hp4_cnt']
full = full.merge(hist_p4, on='geo_prefix4', how='left')

full['h_diff_gts_gh'] = full['hgts_mean'] - full['hgh_mean']
full['h_diff_gh_geo'] = full['hgh_mean'] - full['hgeo_mean']
full['h_ratio_gts_geo'] = full['hgts_mean'] / (full['hgeo_mean'] + 1e-8)
full['h_ratio_gh_global'] = full['hgh_mean'] / (full['hh_mean'] + 1e-8)

full['h_lookup'] = (full['hgts_mean']
                    .fillna(full['hgh_mean'])
                    .fillna(full['hgeo_mean'])
                    .fillna(full['hp5h_mean'])
                    .fillna(full['hp4h_mean'])
                    .fillna(full['hh_mean']))

print("Building temporal profile features...")

recent = hist[hist['day'] >= 40].copy()
recent_geo_ts = recent.groupby(['geohash', 'timestamp'])['demand'].agg(['mean', 'std']).reset_index()
recent_geo_ts.columns = ['geohash', 'timestamp', 'recent_gts_mean', 'recent_gts_std']
full = full.merge(recent_geo_ts, on=['geohash', 'timestamp'], how='left')

recent_geo_h = recent.groupby(['geohash', 'hour'])['demand'].agg(['mean']).reset_index()
recent_geo_h.columns = ['geohash', 'hour', 'recent_gh_mean']
full = full.merge(recent_geo_h, on=['geohash', 'hour'], how='left')

recent_geo = recent.groupby('geohash')['demand'].agg(['mean', 'std']).reset_index()
recent_geo.columns = ['geohash', 'recent_geo_mean', 'recent_geo_std']
full = full.merge(recent_geo, on='geohash', how='left')

full['trend_geo'] = full['recent_geo_mean'] - full['hgeo_mean']
full['trend_gts'] = full['recent_gts_mean'] - full['hgts_mean']
full['trend_gh'] = full['recent_gh_mean'] - full['hgh_mean']

print("Building day-of-week cycle features...")

for dow in range(7):
    dow_days = [d for d in range(1, 48) if (d - 1) % 7 == dow]
    dow_data = hist[hist['day'].isin(dow_days)]
    if len(dow_data) == 0:
        continue

    dow_geo = dow_data.groupby('geohash')['demand'].mean().reset_index()
    dow_geo.columns = ['geohash', f'dow{dow}_geo_mean']
    full = full.merge(dow_geo, on='geohash', how='left')

    dow_gh = dow_data.groupby(['geohash', 'hour'])['demand'].mean().reset_index()
    dow_gh.columns = ['geohash', 'hour', f'dow{dow}_gh_mean']
    full = full.merge(dow_gh, on=['geohash', 'hour'], how='left')

comp_dow = (48 - 1) % 7
full['same_dow_geo'] = full[f'dow{comp_dow}_geo_mean']
full['same_dow_gh'] = full[f'dow{comp_dow}_gh_mean']

print("Building stability and volatility features...")

daily_geo = hist.groupby(['geohash', 'day'])['demand'].mean().reset_index()
stability = daily_geo.groupby('geohash')['demand'].agg(['std', 'mean']).reset_index()
stability.columns = ['geohash', 'daily_geo_std', 'daily_geo_mean']
stability['daily_cv'] = stability['daily_geo_std'] / (stability['daily_geo_mean'] + 1e-8)
full = full.merge(stability[['geohash', 'daily_geo_std', 'daily_cv']], on='geohash', how='left')

hourly_geo = hist.groupby(['geohash', 'hour', 'day'])['demand'].mean().reset_index()
hourly_vol = hourly_geo.groupby(['geohash', 'hour'])['demand'].std().reset_index()
hourly_vol.columns = ['geohash', 'hour', 'hourly_vol']
full = full.merge(hourly_vol, on=['geohash', 'hour'], how='left')

print("Building geo frequency from historical data...")

hist_geo_freq = hist.groupby('geohash')['demand'].count().reset_index()
hist_geo_freq.columns = ['geohash', 'hist_geo_freq']
full = full.merge(hist_geo_freq, on='geohash', how='left')
full['log_hist_geo_freq'] = np.log1p(full['hist_geo_freq'].fillna(0))


print("Building day-48 lag features from competition train...")

train_part = full.iloc[:n_train].copy()
test_part = full.iloc[n_train:].copy()
train_part[target] = train[target].values

day48_comp = train_part[train_part['day'] == 48].copy()
day48_mask = (train_part['day'] == 48)
global_mean = train_part[target].mean()

lag48_gts = day48_comp.groupby(['geohash', 'timestamp'])[target].agg(['mean', 'median']).reset_index()
lag48_gts.columns = ['geohash', 'timestamp', 'lag48_gts_mean', 'lag48_gts_med']

lag48_gh = day48_comp.groupby(['geohash', 'hour'])[target].agg(['mean', 'std']).reset_index()
lag48_gh.columns = ['geohash', 'hour', 'lag48_gh_mean', 'lag48_gh_std']

lag48_geo = day48_comp.groupby('geohash')[target].agg(['mean', 'std', 'min', 'max', 'median', 'count']).reset_index()
lag48_geo.columns = ['geohash', 'lag48_geo_mean', 'lag48_geo_std', 'lag48_geo_min',
                     'lag48_geo_max', 'lag48_geo_med', 'lag48_geo_cnt']

lag48_p5h = day48_comp.groupby(['geo_prefix5', 'hour'])[target].mean().reset_index()
lag48_p5h.columns = ['geo_prefix5', 'hour', 'lag48_p5h_mean']

lag48_p4h = day48_comp.groupby(['geo_prefix4', 'hour'])[target].mean().reset_index()
lag48_p4h.columns = ['geo_prefix4', 'hour', 'lag48_p4h_mean']

lag48_h = day48_comp.groupby('hour')[target].mean().reset_index()
lag48_h.columns = ['hour', 'lag48_h_mean']

lag48_slot = day48_comp.groupby('time_slot')[target].mean().reset_index()
lag48_slot.columns = ['time_slot', 'lag48_slot_mean']

lag48_tables = [
    (lag48_gts, ['geohash', 'timestamp']),
    (lag48_gh, ['geohash', 'hour']),
    (lag48_geo, ['geohash']),
    (lag48_p5h, ['geo_prefix5', 'hour']),
    (lag48_p4h, ['geo_prefix4', 'hour']),
    (lag48_h, ['hour']),
    (lag48_slot, ['time_slot']),
]

for tbl, keys in lag48_tables:
    train_part = train_part.merge(tbl, on=keys, how='left')
    test_part = test_part.merge(tbl, on=keys, how='left')

train_part['lag48_lookup'] = (train_part['lag48_gts_mean']
                               .fillna(train_part['lag48_gh_mean'])
                               .fillna(train_part['lag48_geo_mean'])
                               .fillna(train_part['lag48_p5h_mean'])
                               .fillna(train_part['lag48_p4h_mean'])
                               .fillna(train_part['lag48_h_mean'])
                               .fillna(global_mean))

test_part['lag48_lookup'] = (test_part['lag48_gts_mean']
                              .fillna(test_part['lag48_gh_mean'])
                              .fillna(test_part['lag48_geo_mean'])
                              .fillna(test_part['lag48_p5h_mean'])
                              .fillna(test_part['lag48_p4h_mean'])
                              .fillna(test_part['lag48_h_mean'])
                              .fillna(global_mean))

train_part['lag48_diff_ts_h'] = train_part['lag48_gts_mean'] - train_part['lag48_gh_mean']
test_part['lag48_diff_ts_h'] = test_part['lag48_gts_mean'] - test_part['lag48_gh_mean']

train_part['lag48_ratio_ts_geo'] = train_part['lag48_gts_mean'] / (train_part['lag48_geo_mean'] + 1e-8)
test_part['lag48_ratio_ts_geo'] = test_part['lag48_gts_mean'] / (test_part['lag48_geo_mean'] + 1e-8)

lag48_cols = [c for c in train_part.columns if c.startswith('lag48_')]
train_part.loc[day48_mask, lag48_cols] = np.nan


print("Building target encodings with GroupKFold...")

te_cols = ['geohash', 'geo_prefix4', 'geo_prefix5', 'hour', 'time_slot',
           'RoadType_enc', 'Weather_enc', 'NumberofLanes', 'period']

gkf = GroupKFold(n_splits=5)
groups = train_part['geohash'].values

for col in te_cols:
    train_part[col + '_te'] = 0.0
    test_agg = train_part.groupby(col)[target].mean()
    test_part[col + '_te'] = test_part[col].map(test_agg).fillna(global_mean)

    for tr_idx, val_idx in gkf.split(train_part, groups=groups):
        fold_agg = train_part.iloc[tr_idx].groupby(col)[target].mean()
        mapped = train_part.iloc[val_idx][col].map(fold_agg).fillna(global_mean).values
        train_part.loc[train_part.index[val_idx], col + '_te'] = mapped

for col in te_cols:
    col_std = train_part.groupby(col)[target].std().fillna(0)
    train_part[col + '_te_std'] = train_part[col].map(col_std).fillna(0)
    test_part[col + '_te_std'] = test_part[col].map(col_std).fillna(0)


drop_cols = ['Index', 'geohash', 'timestamp', 'RoadType', 'LargeVehicles',
             'Landmarks', 'Weather', 'geo_prefix4', 'geo_prefix5',
             'geo_char5', 'geo_char6']

for d in range(7):
    drop_cols.append(f'dow{d}_geo_mean')
    drop_cols.append(f'dow{d}_gh_mean')

feature_cols = [c for c in train_part.columns if c not in drop_cols + [target]]
print(f"Total features: {len(feature_cols)}")

X = train_part[feature_cols].values.astype(np.float32)
y = train_part[target].values
X_test = test_part[feature_cols].values.astype(np.float32)


from catboost import CatBoostRegressor

cb_params = {
    'iterations': 6000,
    'learning_rate': 0.025,
    'depth': 8,
    'l2_leaf_reg': 3,
    'min_data_in_leaf': 15,
    'random_strength': 0.4,
    'bagging_temperature': 0.3,
    'border_count': 254,
    'grow_policy': 'SymmetricTree',
    'random_seed': SEED,
    'verbose': 200,
    'task_type': 'CPU',
    'eval_metric': 'R2',
    'early_stopping_rounds': 400,
}

cb_oof = np.zeros(len(X))
cb_preds = np.zeros(len(X_test))

print("\n=== CatBoost ===")
for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, groups=groups)):
    model = CatBoostRegressor(**cb_params)
    model.fit(X[tr_idx], y[tr_idx], eval_set=(X[val_idx], y[val_idx]), verbose=200)
    cb_oof[val_idx] = model.predict(X[val_idx])
    cb_preds += model.predict(X_test) / 5
    print(f"Fold {fold} R2: {r2_score(y[val_idx], cb_oof[val_idx]):.6f}")

cb_cv = r2_score(y, cb_oof)
print(f"CatBoost OOF R2: {cb_cv:.6f}")

if True:
    fi = model.get_feature_importance()
    fi_df = pd.DataFrame({'feature': feature_cols, 'importance': fi})
    fi_df = fi_df.sort_values('importance', ascending=False).head(30)
    print("\nTop 30 features:")
    for _, row in fi_df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.2f}")


import lightgbm as lgb

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 6000,
    'learning_rate': 0.025,
    'num_leaves': 127,
    'max_depth': -1,
    'min_child_samples': 15,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': SEED,
    'verbose': -1,
    'n_jobs': -1,
}

lgb_oof = np.zeros(len(X))
lgb_preds = np.zeros(len(X_test))

print("\n=== LightGBM ===")
for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, groups=groups)):
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X[tr_idx], y[tr_idx],
              eval_set=[(X[val_idx], y[val_idx])],
              callbacks=[lgb.early_stopping(400), lgb.log_evaluation(200)])
    lgb_oof[val_idx] = model.predict(X[val_idx])
    lgb_preds += model.predict(X_test) / 5
    print(f"Fold {fold} R2: {r2_score(y[val_idx], lgb_oof[val_idx]):.6f}")

lgb_cv = r2_score(y, lgb_oof)
print(f"LightGBM OOF R2: {lgb_cv:.6f}")


from xgboost import XGBRegressor

xgb_params = {
    'n_estimators': 6000,
    'learning_rate': 0.025,
    'max_depth': 8,
    'min_child_weight': 15,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': SEED,
    'tree_method': 'hist',
    'verbosity': 0,
    'n_jobs': -1,
    'early_stopping_rounds': 400,
}

xgb_oof = np.zeros(len(X))
xgb_preds = np.zeros(len(X_test))

print("\n=== XGBoost ===")
for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, groups=groups)):
    model = XGBRegressor(**xgb_params)
    model.fit(X[tr_idx], y[tr_idx],
              eval_set=[(X[val_idx], y[val_idx])],
              verbose=200)
    xgb_oof[val_idx] = model.predict(X[val_idx])
    xgb_preds += model.predict(X_test) / 5
    print(f"Fold {fold} R2: {r2_score(y[val_idx], xgb_oof[val_idx]):.6f}")

xgb_cv = r2_score(y, xgb_oof)
print(f"XGBoost OOF R2: {xgb_cv:.6f}")


print("\n=== Ensemble ===")

best_r2 = -999
best_w = (0.4, 0.35, 0.25)
for w1 in np.arange(0.1, 0.8, 0.05):
    for w2 in np.arange(0.05, 0.8, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.05:
            continue
        blend = w1 * cb_oof + w2 * lgb_oof + w3 * xgb_oof
        score = r2_score(y, blend)
        if score > best_r2:
            best_r2 = score
            best_w = (w1, w2, w3)

print(f"Blend: CB={best_w[0]:.2f} LGB={best_w[1]:.2f} XGB={best_w[2]:.2f} -> R2={best_r2:.6f}")

blend_test = best_w[0] * cb_preds + best_w[1] * lgb_preds + best_w[2] * xgb_preds

stack_X = np.column_stack([cb_oof, lgb_oof, xgb_oof])
stack_X_test = np.column_stack([cb_preds, lgb_preds, xgb_preds])

stack_oof = np.zeros(len(y))
stack_preds = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(gkf.split(stack_X, groups=groups)):
    meta = Ridge(alpha=1.0)
    meta.fit(stack_X[tr_idx], y[tr_idx])
    stack_oof[val_idx] = meta.predict(stack_X[val_idx])
    stack_preds += meta.predict(stack_X_test) / 5

stack_r2 = r2_score(y, stack_oof)
print(f"Stack R2: {stack_r2:.6f}")

candidates = {
    'catboost': (cb_cv, cb_preds),
    'lightgbm': (lgb_cv, lgb_preds),
    'xgboost': (xgb_cv, xgb_preds),
    'blend': (best_r2, blend_test),
    'stack': (stack_r2, stack_preds),
}

best_method = max(candidates, key=lambda k: candidates[k][0])
best_cv, best_preds = candidates[best_method]
print(f"Best: {best_method} (R2={best_cv:.6f})")

sub = pd.DataFrame({'Index': test_idx, 'demand': best_preds.clip(0, 1)})
sub.to_csv('submission.csv', index=False)
print(f"submission.csv saved")

for name, (cv, preds) in candidates.items():
    fname = f'submission_{name}.csv'
    pd.DataFrame({'Index': test_idx, 'demand': preds.clip(0, 1)}).to_csv(fname, index=False)


print("\n=== Pseudo-Labeling ===")
pseudo_labels = best_preds.clip(0, 1)
X_aug = np.vstack([X, X_test])
y_aug = np.concatenate([y, pseudo_labels])
groups_aug = np.concatenate([groups, test_part['geohash'].values])

pl_cb = np.zeros(len(X_test))
pl_lgb = np.zeros(len(X_test))
gkf_pl = GroupKFold(n_splits=5)

for fold, (tr_idx, val_idx) in enumerate(gkf_pl.split(X_aug, groups=groups_aug)):
    m1 = CatBoostRegressor(**{**cb_params, 'random_seed': SEED + 1, 'verbose': 0})
    m1.fit(X_aug[tr_idx], y_aug[tr_idx])
    pl_cb += m1.predict(X_test) / 5

    m2 = lgb.LGBMRegressor(**{**lgb_params, 'random_state': SEED + 1})
    m2.fit(X_aug[tr_idx], y_aug[tr_idx])
    pl_lgb += m2.predict(X_test) / 5
    print(f"PL Fold {fold} done")

pl_blend = (0.55 * pl_cb + 0.45 * pl_lgb).clip(0, 1)
pd.DataFrame({'Index': test_idx, 'demand': pl_blend}).to_csv('submission_pseudo.csv', index=False)
print("submission_pseudo.csv saved")

final_mix = (0.6 * best_preds.clip(0, 1) + 0.4 * pl_blend)
pd.DataFrame({'Index': test_idx, 'demand': final_mix}).to_csv('submission_final.csv', index=False)
print("submission_final.csv saved")


print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"CatBoost GroupKFold R2:  {cb_cv:.6f}")
print(f"LightGBM GroupKFold R2:  {lgb_cv:.6f}")
print(f"XGBoost GroupKFold R2:   {xgb_cv:.6f}")
print(f"Blend R2:                {best_r2:.6f}")
print(f"Stack R2:                {stack_r2:.6f}")
print(f"Best: {best_method}")
print(f"Total features: {len(feature_cols)}")
print(f"Historical data used: 3.2M rows (days 1-47)")
print()
print("Submissions:")
print("  submission.csv           <- best method")
print("  submission_pseudo.csv    <- pseudo-labeled")
print("  submission_final.csv     <- 60% best + 40% pseudo")
for name in candidates:
    print(f"  submission_{name}.csv")
print("Done.")
