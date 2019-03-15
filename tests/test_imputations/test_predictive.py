"""Tests for the PredictiveImputer Class."""

from autoimpute.imputations import PredictiveImputer
from autoimpute.utils import dataframes
dfs = dataframes

def test_default_predictive_imputer():
    """Test the _default method and results for PredictiveImputer()."""
    imp = PredictiveImputer()
    imp.fit_transform(dfs.df_mix)
    assert imp.statistics_["gender"]["strategy"] == "binary logistic"
    assert imp.statistics_["salary"]["strategy"] == "least squares"
    assert imp.statistics_["age"]["strategy"] == "least squares"
    assert imp.statistics_["amm"]["strategy"] == "least squares"
