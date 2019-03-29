"""Tests for the SingleImputer Class."""

import pytest
from autoimpute.imputations import SingleImputer
from autoimpute.utils import dataframes
dfs = dataframes
# pylint:disable=len-as-condition
# pylint:disable=pointless-string-statement

def test_default_single_imputer():
    """Test the _default method and results for SingleImputer()."""
    imp = SingleImputer()
    # test df_num first
    # -----------------
    # all strategies should default to mean
    imp.fit_transform(dfs.df_num)
    for imputer in imp.statistics_.values():
        strat = imputer.statistics_["strategy"]
        assert strat == "mean"

    # test df_ts_mixed next
    # ---------------
    # datetime col should default to none
    # numerical col should default to mean
    # categorical col should default to mean
    imp.fit_transform(dfs.df_ts_mixed)
    date_imputer = imp.statistics_["date"]
    values_imputer = imp.statistics_["values"]
    cats_imputer = imp.statistics_["cats"]
    assert date_imputer.statistics_["strategy"] is None
    assert values_imputer.statistics_["strategy"] == "mean"
    assert cats_imputer.statistics_["strategy"] == "mode"

def test_numerical_single_imputers():
    """Test numerical methods when not using the _default."""
    for num_strat in dfs.num_strategies:
        imp = SingleImputer(strategy=num_strat)
        imp.fit_transform(dfs.df_num)
        for imputer in imp.statistics_.values():
            strat = imputer.statistics_["strategy"]
            assert strat == num_strat

def test_categorical_single_imputers():
    """Test categorical methods when not using the _default."""
    for cat_strat in dfs.cat_strategies:
        imp = SingleImputer(strategy={"cats": cat_strat})
        imp.fit_transform(dfs.df_ts_mixed)
        for imputer in imp.statistics_.values():
            strat = imputer.statistics_["strategy"]
            assert strat == cat_strat

def test_single_missing_column():
    """Test that the imputer removes columns that are fully missing."""
    with pytest.raises(ValueError):
        imp = SingleImputer()
        imp.fit_transform(dfs.df_col_miss)

def test_bad_strategy():
    """Test that strategies not supported throw a ValueError."""
    with pytest.raises(ValueError):
        imp = SingleImputer(strategy="not_a_strategy")
        imp.fit_transform(dfs.df_num)

def bad_imputers():
    """Test supported strategies but improper column specification."""
    # example with too few strategies specified for given DataFrame
    bad_list = SingleImputer(strategy=["mean", "median"])
    # example with incorrect keys for given DataFrame
    bad_keys = SingleImputer(strategy={"X":"mean", "B":"median", "C":"mode"})
    return [bad_list, bad_keys]

@pytest.mark.parametrize("imp", bad_imputers())
def test_imputer_strategies_not_allowed(imp):
    """Test bad imputers"""
    with pytest.raises(ValueError):
        imp.fit_transform(dfs.df_num)

''' HOLD OFF ON TESTS BELOW UNTIL SERIES TYPE CHECKING ERRORS IMPLEMENTED
def test_wrong_numerical_type():
    """Test supported strategies but improper column type for strategy."""
    num_for_cat = SingleImputer(strategy={"cats": "mean"})
    with pytest.raises(TypeError):
        num_for_cat.fit_transform(dfs.df_ts_mixed)

def test_wrong_categorical_type():
    """Test supported strategies but improper column type for strategy."""
    cat_for_num = SingleImputer(strategy="categorical")
    with pytest.raises(TypeError):
        cat_for_num.fit_transform(dfs.df_num)
'''
