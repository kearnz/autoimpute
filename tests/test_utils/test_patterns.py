"""Pytest for utils.patterns"""

import numpy as np
import pandas as pd
from autoimpute.utils.patterns import md_locations, md_pairs, md_pattern

# simulated data that matches Van Buuren 4.1 - general
# verifiable because known values for influx, outflux, etc from text itself
df_general = pd.DataFrame({
    "A": [1, 5, 9, 6, 12, 11, np.nan, np.nan],
    "B": [2, 4, 3, 6, 11, np.nan, np.nan, np.nan],
    "C": [-1, 1, np.nan, np.nan, np.nan, -1, 1, 0]
})

# hard coded, known values based on md.pattern function from Van Buuren 4.1
df_pattern = pd.DataFrame({
    "count": [2, 3, 1, 2],
    "A": [1, 1, 1, 0],
    "B": [1, 1, 0, 0],
    "C": [1, 0, 1, 1],
    "nmis": [0, 1, 1, 2]
})

# hard coded, known values based on md.pairs function from Van Buuren 4.1
dict_pairs = {}
ci = ["A", "B", "C"]
pair_df = lambda v: pd.DataFrame(v, columns=ci, index=ci)
dict_pairs["rr"] = pair_df([[6, 5, 3], [5, 5, 2], [3, 2, 5]])
dict_pairs["rm"] = pair_df([[0, 1, 3], [0, 0, 3], [2, 3, 0]])
dict_pairs["mr"] = pair_df([[0, 0, 2], [1, 0, 3], [3, 3, 0]])
dict_pairs["mm"] = pair_df([[2, 2, 0], [2, 3, 0], [0, 0, 3]])


def test_md_locations():
    """Missingness locations should equal np.isnan for each col"""
    md_loc = md_locations(df_general)
    assert isinstance(md_loc, pd.DataFrame)
    assert all(md_loc["A"] == np.isnan(df_general["A"]))
    assert all(md_loc["B"] == np.isnan(df_general["B"]))
    assert all(md_loc["C"] == np.isnan(df_general["C"]))

def test_md_pattern():
    """Missing data pattern should equal hard coded md.pattern from VB 4.1"""
    md_pat = md_pattern(df_general)
    assert isinstance(md_pat, pd.DataFrame)
    assert all(md_pat["count"] == df_pattern["count"])
    assert all(md_pat[["A", "B", "C"]] == df_pattern[["A", "B", "C"]])
    assert all(md_pat["nmis"] == df_pattern["nmis"])

def test_md_pairs():
    """Missing data pairs should equal hard coded md.pairs from VB 4.1"""
    md_pair = md_pairs(df_general)
    assert isinstance(md_pair, dict)
    assert all(md_pair["rr"] == dict_pairs["rr"])
    assert all(md_pair["mr"] == dict_pairs["mr"])
    assert all(md_pair["rm"] == dict_pairs["rm"])
    assert all(md_pair["mm"] == dict_pairs["mm"])
