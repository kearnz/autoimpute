"""Tests for the TimeSeriesImputer Class."""

import pytest
from autoimpute.imputations import TimeSeriesImputer
from autoimpute.utils import dataframes
dfs = dataframes

def test_default_ts_imputer():
    """Test the _default method and results for SingleImputer()."""
    imp = TimeSeriesImputer()
    # test df_ts first
    # -----------------
    # all strategies should default to mean
    imp.fit_transform(dfs.df_ts_num)
    for column, imputer in imp.statistics_.items():
        strat = imputer.statistics_["strategy"]
        if column.startswith("date"):
            assert strat is None
        if column.startswith("values"):
            assert strat == "interpolate"

def test_time_index_set():
    """Test that index set using date column."""
    imp = TimeSeriesImputer()
    ts_df = imp.fit_transform(dfs.df_ts_num)
    assert ts_df.index.name == "date"
    imp_explicit = TimeSeriesImputer(index_column="date_tm1")
    ts_df_2 = imp_explicit.fit_transform(dfs.df_ts_num)
    assert ts_df_2.index.name == "date_tm1"

def test_time_missing_column():
    """Test that the imputer removes columns that are fully missing."""
    with pytest.raises(ValueError):
        timp = TimeSeriesImputer()
        timp.fit_transform(dfs.df_col_miss)

def test_time_imputers():
    """Test time-based methods when not using the _default."""
    for ts_strat in dfs.time_strategies:
        imp = TimeSeriesImputer(
            strategy={"values": ts_strat, "values_tm1": ts_strat}
        )
        imp.fit_transform(dfs.df_ts_num)
        for imputer in imp.statistics_.values():
            strat = imputer.statistics_["strategy"]
            assert strat == ts_strat
