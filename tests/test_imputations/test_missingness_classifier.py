"""Tests written to ensure the SingleImputer in the imputations package works.

Tests use the pytest library. The tests in this module ensure the following:
- `test_missing_classifier` tests that bug in issue 56 is fixed
"""

from autoimpute.imputations import MissingnessClassifier
from autoimpute.utils import dataframes
dfs = dataframes
# pylint:disable=len-as-condition
# pylint:disable=pointless-string-statement


def test_single_missing_column():
    """Test that the missingness classifier works correctly"""
    imp = MissingnessClassifier()
    imp.fit_predict(dfs.df_mis_classifier)
    imp.fit_predict_proba(dfs.df_mis_classifier)
