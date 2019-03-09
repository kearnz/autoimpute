"""Tests for the SingleImputer Class."""

import pytest
from autoimpute.imputations.single_imputer import SingleImputer
from autoimpute.utils import dataframes
dfs = dataframes
# pylint:disable=len-as-condition

def test_default_single_imputer():
    """Test the _default method and results for SingleImputer()."""
    imp = SingleImputer()
    # test df_num first
    # -----------------
    # all strategies should default to mean
    imp.fit_transform(dfs.df_num)
    for strat in imp.statistics_.values():
        assert strat["strategy"] == "mean"
    for key in imp.statistics_:
        assert imp.statistics_[key]["param"] == dfs.df_num[key].mean()

    # test df_ts_mixed next
    # ---------------
    # datetime col should default to none
    # numerical col should default to mean
    # categorical col should default to mean
    imp.fit_transform(dfs.df_ts_mixed)
    assert imp.statistics_["date"]["strategy"] == "none"
    assert imp.statistics_["values"]["strategy"] == "mean"
    assert imp.statistics_["cats"]["strategy"] == "mode"

def test_numerical_single_imputers():
    """Test numerical methods when not using the _default."""
    for strat in dfs.num_strategies:
        imp_str = SingleImputer(strategy=strat)
        imp_str.fit_transform(dfs.df_num)
        for stat in imp_str.statistics_.values():
            assert stat["strategy"] == strat

def test_categorical_single_imputers():
    """Test categorical methods when not using the _default."""
    for strat in dfs.cat_strategies:
        imp_str = SingleImputer(strategy={"cats": strat})
        imp_str.fit_transform(dfs.df_ts_mixed)
        for stat in imp_str.statistics_.values():
            assert stat["strategy"] == strat

def test_column_reducer():
    """Test that the imputer removes columns that are fully missing."""
    imp_dict = SingleImputer(strategy={"values": "mean", "cats": "mode"})
    imp_dict.fit_transform(dfs.df_ts_mixed)
    assert len(imp_dict.statistics_) == 2

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
