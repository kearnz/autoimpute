"""Sample DataFrames utilized for examples and tests."""

import numpy as np
import pandas as pd

# missingness lambdas
eq_miss = lambda x: np.random.choice([x, np.nan], 1)[0]
val_miss = lambda x: np.random.choice([x, np.nan], 1, p=[x/100, 1-x/100])[0]

# strategies used for imputation
num_strategies = ["mean", "median", "mode", "random", "norm", "linear"]
cat_strategies = ["mode", "categorical"]
time_strategies = ["time", "linear", "locf", "nocb"]

# Numerical DataFrame with different % missingness per column
df_num = pd.DataFrame()
df_num["A"] = np.random.choice(np.arange(90, 100), 1000)
df_num["B"] = np.random.choice(np.arange(50, 100), 1000)
df_num["C"] = np.random.choice(np.arange(1, 100), 1000)
df_num["A"] = df_num["A"].apply(eq_miss)
df_num["B"] = df_num["B"].apply(val_miss)

# Mixed DataFrame with different % missingness per column & some dependence
df_mix = pd.DataFrame()
df_mix["gender"] = np.random.choice(["Male", "Female", None], 500)
df_mix["salary"] = np.random.choice(np.arange(20, 100), 500)
df_mix["age"] = np.random.choice(
    [10, 20, 30, 40, 50, 60, 70], 500,
    p=[0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1]
)
df_mix["amm"] = np.random.choice(np.arange(0, 100), 500)

for each in df_mix.index:
    s = df_mix.loc[each, "salary"]
    if df_mix.loc[each, "age"] > 50:
        s = np.random.choice([s, np.nan], p=[0.3, 0.7])
    elif df_mix.loc[each, "age"] < 30:
        s = np.random.choice([s, np.nan], p=[0.4, 0.6])

# DataFrame with numerical feature, `np.nan` column, `None` column
df_col_miss = pd.DataFrame()
df_col_miss["A"] = np.random.choice(np.arange(90, 100), 1000)
df_col_miss["A"] = df_col_miss["A"].apply(eq_miss)
df_col_miss["B"] = np.random.choice([np.nan], 1000)
df_col_miss["C"] = np.random.choice([None], 1000)

# DataFrame to test all missing
df_all_miss = pd.DataFrame({
    "A":[None, None],
    "B": [np.nan, np.nan]
})

# DataFrame to test time series
df_ts_num = pd.DataFrame({
    "date": pd.to_datetime(['2018-01-04', '2018-01-05', '2018-01-06',
                            '2018-01-07', '2018-01-08', '2018-01-09']),
    "date_tm1": pd.to_datetime(['2017-01-04', '2017-01-05', '2017-01-06',
                                '2017-01-07', '2017-01-08', '2017-01-09']),
    "values": [np.nan, 20, np.nan, 30, 40, np.nan],
    "values_tm1": [np.nan, 15, np.nan, 25, 50, 70]
})

# DataFrame to test default with added time column and cat column
df_ts_mixed = pd.DataFrame({
    "date": pd.to_datetime(['2018-01-04', '2018-01-05', '2018-01-06',
                            '2018-01-07', '2018-01-08', '2018-01-09']),
    "values": [271238, 329285, np.nan, 260260, 263711, np.nan],
    "cats": ["red", None, "green", "green", "red", "green"]
})
