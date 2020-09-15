"""Sample DataFrames utilized for examples and tests."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
# pylint:disable=missing-docstring

# missingness lambdas
eq_miss = lambda x: np.random.choice([x, np.nan], 1)[0]
val_miss = lambda x: np.random.choice([x, np.nan], 1, p=[x/100, 1-x/100])[0]

# strategies used for imputation
num_strategies = ["mean", "median", "mode", "random", "norm", "interpolate",'normal unit variance']
cat_strategies = ["mode", "categorical"]
time_strategies = ["interpolate", "locf", "nocb"]

# Numerical DataFrame with different % missingness per column
df_num = pd.DataFrame()
df_num["A"] = np.random.choice(np.arange(90, 100), 1000)
df_num["B"] = np.random.choice(np.arange(50, 100), 1000)
df_num["C"] = np.random.choice(np.arange(1, 100), 1000)
df_num["A"] = df_num["A"].apply(eq_miss)
df_num["B"] = df_num["B"].apply(val_miss)

# Numerical DataFrame with column names for missingness classifier
df_mis_classifier = pd.DataFrame()
df_mis_classifier["a"] = np.random.choice(np.arange(90, 100), 1000)
df_mis_classifier["k"] = np.random.choice(np.arange(50, 100), 1000)
df_mis_classifier["c"] = np.random.choice(np.arange(1, 100), 1000)
df_mis_classifier["a"] = df_mis_classifier["a"].apply(eq_miss)
df_mis_classifier["k"] = df_mis_classifier["k"].apply(val_miss)

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

# bayesian regression testing
sc = lambda x, d: (x-d.mean())/d.std()
mis = lambda x: np.random.choice([x, np.nan], 1, p=[0.8, 0.2])[0]
X, y = make_regression(n_samples=1000, n_features=3, noise=0.50)
df_bayes_reg = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1],
                             "x3": X[:, 2], "y": y})
df_bayes_reg.x1 = df_bayes_reg.x1.apply(lambda x: sc(x, df_bayes_reg.x1))
df_bayes_reg.x2 = df_bayes_reg.x2.apply(lambda x: sc(x, df_bayes_reg.x2))
df_bayes_reg.x3 = df_bayes_reg.x3.apply(lambda x: sc(x, df_bayes_reg.x1))
df_bayes_reg.y = df_bayes_reg.y.apply(mis)

# bayesian logistic testing
def trans_binary(c):
    m = df_bayes_log.y.mean()
    if not pd.isnull(c):
        if c > m:
            return "male"
        return "female"

df_bayes_log = df_bayes_reg.copy()
df_bayes_log.y = df_bayes_log.y.apply(trans_binary)

# partial dependence test
df_partial_dependence = pd.DataFrame(
    {'A':np.random.uniform(0,1,100),
        'B':np.random.uniform(0,1,100)}
)
df_partial_dependence['B'][df_partial_dependence['B'] < 0.25] = np.nan
df_partial_dependence['C'] = df_partial_dependence['B'] * 2
df_partial_dependence['C'][df_partial_dependence['C'] < 0.7] = np.nan
