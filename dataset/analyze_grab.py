import pandas as pd
import numpy as np

grab = pd.read_csv(r'C:\Users\singh\Downloads\training.csv\training.csv')
comp = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

d48 = grab[grab['day'] == 48]
d49 = grab[grab['day'] == 49]
print("Grab day48:", len(d48))
print("Grab day49:", len(d49))
print("Comp train:", len(comp))
print("Comp day48:", len(comp[comp['day'] == 48]))
print("Comp day49:", len(comp[comp['day'] == 49]))
print()

merged = comp.merge(grab, left_on=['geohash', 'day', 'timestamp'],
                    right_on=['geohash6', 'day', 'timestamp'], how='inner')
print("Exact match rows:", len(merged))
if len(merged) > 0:
    diff = (merged['demand_x'] - merged['demand_y']).abs()
    print("Demand diff max:", diff.max())
    print("Demand diff mean:", diff.mean())
    print("Perfect matches:", (diff < 1e-6).sum())

print()
print("=== Grab dataset temporal coverage ===")
for d in range(45, 55):
    cnt = len(grab[grab['day'] == d])
    print(f"Day {d}: {cnt} rows, {grab[grab['day']==d]['geohash6'].nunique()} geohashes")

print()
print("=== Grab day49 full ===")
d49_ts = sorted(grab[grab['day'] == 49]['timestamp'].unique())
print("Day49 timestamps:", d49_ts)
print("Day49 rows:", len(d49))
print("Day49 geohashes:", d49['geohash6'].nunique())

print()
test_merge = test.merge(grab, left_on=['geohash', 'day', 'timestamp'],
                        right_on=['geohash6', 'day', 'timestamp'], how='inner')
print("Test rows matched in grab:", len(test_merge))
print("Test total rows:", len(test))
print("Coverage:", len(test_merge) / len(test))

print()
all_days_before_48 = grab[grab['day'] < 48]
print("Grab rows before day 48:", len(all_days_before_48))
print("Days 1-47: available for feature engineering without leakage")
