"""Tests for the single imputation class"""

import pytest
import numpy as np
import pandas as pd
from autoimpute.imputations.single import SingleImputer
# pylint:disable=len-as-condition

# missingness lambdas
eq_miss = lambda x: np.random.choice([x, np.nan], 1)[0]
val_miss = lambda x: np.random.choice([x, np.nan], 1, p=[x/100, 1-x/100])[0]

# Numerical DataFrame with different % missingness per column
df_num = pd.DataFrame()
df_num["A"] = np.random.choice(np.arange(90, 100), 1000)
df_num["B"] = np.random.choice(np.arange(50, 100), 1000)
df_num["C"] = np.random.choice(np.arange(1, 100), 1000)
df_num["A"] = df_num["A"].apply(eq_miss)
df_num["B"] = df_num["B"].apply(val_miss)

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

# DataFrame to test default with added time column
df_ts = pd.DataFrame({
    "date": pd.to_datetime(['2018-01-04', '2018-01-05', '2018-01-06',
                            '2018-01-07', '2018-01-08', '2018-01-09']),
    "values": [271238, 329285, np.nan, 260260, 263711, np.nan],
    "cats": ["red", None, "green", "green", "red", "green"]
})

def test_default_imputer():
    """Ensuring the _default method and values work for SingleImputer()"""
    imp = SingleImputer()
    # test numerical columns first
    imp.fit_transform(df_num)
    for strat in imp.statistics_.values():
        assert strat["strategy"] == "mean"
    for key in imp.statistics_:
        assert imp.statistics_[key]["param"] == df_num[key].mean()
    # test df_ts next
    imp.fit_transform(df_ts)
    assert imp.statistics_["date"]["strategy"] == "none"
    assert imp.statistics_["values"]["strategy"] == "mean"
    assert imp.statistics_["cats"]["strategy"] == "mode"

def test_custom_imputer():
    """Ensure custom methods when not using the _default"""
    imp_str = SingleImputer(strategy="linear")
    imp_str.fit_transform(df_num)
    for strat in imp_str.statistics_.values():
        assert strat["strategy"] == "linear"
    imp_dict = SingleImputer(strategy={"values": "mean", "cats": "mode"})
    imp_dict.fit_transform(df_ts)
    assert len(imp_dict.statistics_) == 2

def test_bad_strategy():
    """test bad imputers"""
    with pytest.raises(ValueError):
        imp = SingleImputer(strategy="not_a_strategy")
        imp.fit_transform(df_num)

def bad_imputers():
    """bad imputers"""
    bad_list = SingleImputer(strategy=["mean", "median"])
    bad_keys = SingleImputer(strategy={"X":"mean", "B":"median", "C":"mode"})
    return [bad_list, bad_keys]

@pytest.mark.parametrize("imp", bad_imputers())
def test_imputer_strategies_not_allowed(imp):
    """test bad imputers"""
    with pytest.raises(ValueError):
        imp.fit_transform(df_num)
