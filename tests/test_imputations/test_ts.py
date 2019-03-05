"""Tests for the TimeSeriesImputer Class."""

from autoimpute.imputations.ts import TimeSeriesImputer
from autoimpute.utils import dataframes
dfs = dataframes

def test_default_ts_imputer():
    """Test the _default method and results for SingleImputer()."""
    imp = TimeSeriesImputer()
    # test df_ts first
    # -----------------
    # all strategies should default to mean
    imp.fit_transform(dfs.df_ts_num)
    for strat in imp.statistics_.values():
        s = strat["strategy"]
        if s.startswith("date"):
            assert s == "none"
        if s.startswith("values"):
            assert s == "linear"

def test_time_index_set():
    """Test that index set using date column."""
    imp = TimeSeriesImputer()
    ts_df = imp.fit_transform(dfs.df_ts_num)
    assert ts_df.index.name == "date"
    imp_explicit = TimeSeriesImputer(index_column="date_tm1")
    ts_df_2 = imp_explicit.fit_transform(dfs.df_ts_num)
    assert ts_df_2.index.name == "date_tm1"

def test_time_imputers():
    """Test time-based methods when not using the _default."""
    for strat in dfs.time_strategies:
        imp_str = TimeSeriesImputer(strategy={"values": strat,
                                              "values_tm1": strat})
        imp_str.fit_transform(dfs.df_ts_num)
        for stat in imp_str.statistics_.values():
            assert stat["strategy"] == strat
