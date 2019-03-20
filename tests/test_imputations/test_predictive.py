"""Tests for the PredictiveImputer Class."""

from autoimpute.imputations import PredictiveImputer
from autoimpute.utils import dataframes
dfs = dataframes

def test_default_predictive_imputer():
    """Test the _default method and results for PredictiveImputer()."""
    imp = PredictiveImputer()
    imp.fit_transform(dfs.df_mix)
    assert imp.statistics_["gender"]["strategy"] == "binary logistic"
    assert imp.statistics_["salary"]["strategy"] == "pmm"
    assert imp.statistics_["age"]["strategy"] == "pmm"
    assert imp.statistics_["amm"]["strategy"] == "pmm"

def test_stochastic_predictive_imputer():
    """Test stochastic works for numerical columns of PredictiveImputer."""
    # generate linear, then stochastic
    imp_p = PredictiveImputer(strategy={"A":"least squares"})
    imp_s = PredictiveImputer(strategy={"A":"stochastic"})
    # make sure both work
    _ = imp_p.fit_transform(dfs.df_num)
    _ = imp_s.fit_transform(dfs.df_num)
    assert imp_p.imputed_["A"] == imp_s.imputed_["A"]

def test_bayesian_reg_imputer():
    """Test bayesian works for numerical column of PredictiveImputer."""
    # test designed first
    imp_b = PredictiveImputer(strategy={"y":"bayesian least squares"})
    imp_b.fit_transform(dfs.df_bayes_reg)
    # test on numerical in general
    imp_n = PredictiveImputer(strategy="bayesian least squares")
    imp_n.fit_transform(dfs.df_num)

def test_bayesian_logistic_imputer():
    """Test bayesian works for binary column of PredictiveImputer."""
    imp_b = PredictiveImputer(strategy={"y":"bayesian binary logistic"},
                              fill_value="random")
    imp_b.fit_transform(dfs.df_bayes_log)
