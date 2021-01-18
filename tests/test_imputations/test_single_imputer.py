"""Tests written to ensure the SingleImputer in the imputations package works.

Tests use the pytest library. The tests in this module ensure the following:
- `test_single_missing_column` throw error when any column fully missing.
- `test_bad_strategy` throw error if strategy is not allowed.
- `test_imputer_strategies_not_allowed` test strategies misspecified.
- `test_wrong_numerical_type` test valid strategy but not for numerical.
- `test_wrong_categorical_type` test valid strategy but not for categorical.
- `test_default_single_imputer` tests the simplest implementation: defaults.
- `test_numerical_univar_imputers` test all numerical strategies.
- `test_categorical_univar_imputers` test all categorical strategies.
- `test_stochastic_predictive_imputer` test stochastic strategy.
- `test_bayesian_reg_imputer` test bayesian regression strategy.
- `test_bayesian_logistic_imputer` test bayesian logistic strategy.
- `test_pmm_lrd_imputer` test pmm and lrd strategy.
- `test_normal_unit_variance_imputer` test unit variance imputer
"""

import pytest
from autoimpute.imputations import SingleImputer
from autoimpute.utils import dataframes
dfs = dataframes
# pylint:disable=len-as-condition
# pylint:disable=pointless-string-statement

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

def test_default_single_imputer():
    """Test the _default method and results for SingleImputer()."""
    imp = SingleImputer()
    # test df_num first
    # -----------------
    # all strategies should default to pmm
    imp.fit_transform(dfs.df_num)
    for imputer in imp.statistics_.values():
        strat = imputer.statistics_["strategy"]
        assert strat == "pmm"

    # test df_ts_mixed next
    # ---------------
    # datetime col should default to none
    # numerical col should default to mean
    # categorical col should default to mean
    imp.fit_transform(dfs.df_mix)
    gender_imputer = imp.statistics_["gender"]
    salary_imputer = imp.statistics_["salary"]
    assert salary_imputer.statistics_["strategy"] == "pmm"
    assert gender_imputer.statistics_["strategy"] == "multinomial logistic"

def test_numerical_univar_imputers():
    """Test numerical methods when not using the _default."""
    for num_strat in dfs.num_strategies:
        imp = SingleImputer(strategy=num_strat)
        imp.fit_transform(dfs.df_num)
        for imputer in imp.statistics_.values():
            strat = imputer.statistics_["strategy"]
            assert strat == num_strat

def test_categorical_univar_imputers():
    """Test categorical methods when not using the _default."""
    for cat_strat in dfs.cat_strategies:
        imp = SingleImputer(strategy={"cats": cat_strat})
        imp.fit_transform(dfs.df_ts_mixed)
        for imputer in imp.statistics_.values():
            strat = imputer.statistics_["strategy"]
            assert strat == cat_strat

def test_stochastic_predictive_imputer():
    """Test stochastic works for numerical columns of PredictiveImputer."""
    # generate linear, then stochastic
    imp_p = SingleImputer(strategy={"A":"least squares"})
    imp_s = SingleImputer(strategy={"A":"stochastic"})
    # make sure both work
    _ = imp_p.fit_transform(dfs.df_num)
    _ = imp_s.fit_transform(dfs.df_num)
    assert imp_p.imputed_["A"] == imp_s.imputed_["A"]

def test_bayesian_reg_imputer():
    """Test bayesian works for numerical column of PredictiveImputer."""
    # test designed first - test kwargs and params
    imp_b = SingleImputer(strategy={"y":"bayesian least squares"},
                          imp_kwgs={"y":{"fill_value": "random",
                                         "am": 11, "cores": 2}})
    imp_b.fit_transform(dfs.df_bayes_reg)
    # test on numerical in general
    imp_n = SingleImputer(strategy="bayesian least squares")
    imp_n.fit_transform(dfs.df_num)

def test_bayesian_logistic_imputer():
    """Test bayesian works for binary column of PredictiveImputer."""
    imp_b = SingleImputer(strategy={"y":"bayesian binary logistic"},
                          imp_kwgs={"y":{"fill_value": "random"}})
    imp_b.fit_transform(dfs.df_bayes_log)

def test_pmm_lrd_imputer():
    """Test pmm and lrd work for numerical column of PredictiveImputer."""
    # test pmm first - test kwargs and params
    imp_pmm = SingleImputer(strategy={"y":"pmm"},
                            imp_kwgs={"y": {"fill_value": "random",
                                      "copy_x": False}})
    imp_pmm.fit_transform(dfs.df_bayes_reg)

    # test lrd second - test kwargs and params
    imp_lrd = SingleImputer(strategy={"y":"lrd"},
                            imp_kwgs={"y": {"fill_value": "random",
                                      "copy_x": False}})
    imp_lrd.fit_transform(dfs.df_bayes_reg)

def test_normal_unit_variance_imputer():
    """Test normal unit variance imputer for numerical column"""
    imp_pmm = SingleImputer(strategy={"y":"normal unit variance"},)
    imp_pmm.fit_transform(dfs.df_bayes_reg)

def test_partial_dependence_imputer():
    """Test to ensure that edge case for partial dependence whandled"""
    imp = SingleImputer(strategy='stochastic')
    imp.fit_transform(dfs.df_partial_dependence)
