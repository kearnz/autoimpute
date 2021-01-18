"""Tests written to ensure the MiceImputer in the imputations package works.
Note that this also tests the MultipleImputer, which really just passes to
the SingleImputer. SingleImputer has tests, some of which are the same as here.

Tests use the pytest library. The tests in this module ensure the following:
- `test_stochastic_predictive_imputer` test stochastic strategy.
- `test_bayesian_reg_imputer` test bayesian regression strategy.
- `test_bayesian_logistic_imputer` test bayesian logistic strategy.
- `test_pmm_lrd_imputer` test pmm and lrd strategy.
- `test_normal_unit_variance_imputer` test unit variance imputer
"""

import pytest
from autoimpute.imputations import MiceImputer
from autoimpute.utils import dataframes
dfs = dataframes
# pylint:disable=len-as-condition
# pylint:disable=pointless-string-statement

def test_stochastic_predictive_imputer():
    """Test stochastic works for numerical columns of PredictiveImputer."""
    # generate linear, then stochastic
    imp_p = MiceImputer(strategy={"A":"least squares"})
    imp_s = MiceImputer(strategy={"A":"stochastic"})
    # make sure both work
    _ = imp_p.fit_transform(dfs.df_num)
    _ = imp_s.fit_transform(dfs.df_num)
    assert imp_p.imputed_["A"] == imp_s.imputed_["A"]

def test_bayesian_reg_imputer():
    """Test bayesian works for numerical column of PredictiveImputer."""
    # test designed first - test kwargs and params
    imp_b = MiceImputer(strategy={"y":"bayesian least squares"},
                          imp_kwgs={"y":{"fill_value": "random",
                                         "am": 11, "cores": 2}})
    imp_b.fit_transform(dfs.df_bayes_reg)
    # test on numerical in general
    imp_n = MiceImputer(strategy="bayesian least squares")
    imp_n.fit_transform(dfs.df_num)

def test_bayesian_logistic_imputer():
    """Test bayesian works for binary column of PredictiveImputer."""
    imp_b = MiceImputer(strategy={"y":"bayesian binary logistic"},
                          imp_kwgs={"y":{"fill_value": "random"}})
    imp_b.fit_transform(dfs.df_bayes_log)

def test_pmm_lrd_imputer():
    """Test pmm and lrd work for numerical column of PredictiveImputer."""
    # test pmm first - test kwargs and params
    imp_pmm = MiceImputer(strategy={"y":"pmm"},
                            imp_kwgs={"y": {"fill_value": "random",
                                      "copy_x": False}})
    imp_pmm.fit_transform(dfs.df_bayes_reg)

    # test lrd second - test kwargs and params
    imp_lrd = MiceImputer(strategy={"y":"lrd"},
                            imp_kwgs={"y": {"fill_value": "random",
                                      "copy_x": False}})
    imp_lrd.fit_transform(dfs.df_bayes_reg)

def test_normal_unit_variance_imputer():
    """Test normal unit variance imputer for numerical column"""
    imp_pmm = MiceImputer(strategy={"y":"normal unit variance"},)
    imp_pmm.fit_transform(dfs.df_bayes_reg)

def test_partial_dependence_imputer():
    """Test to ensure that edge case for partial dependence whandled"""
    imp = MiceImputer(strategy='stochastic')
    imp.fit_transform(dfs.df_partial_dependence)
