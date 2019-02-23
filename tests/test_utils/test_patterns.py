"""Pytest for utils.patterns"""

import numpy as np
import pandas as pd
from autoimpute.utils.patterns import md_locations, md_pairs, md_pattern
from autoimpute.utils.patterns import inbound, outbound, flux, proportions

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

# hard coded, known values based on inbound / outbound from Van Buuren 4.1
df_inbound = pd.DataFrame({
    "A": [0, 1/3, 1],
    "B": [0, 0, 1],
    "C": [1, 1, 0]
}, index=["A", "B", "C"])

df_outbound = pd.DataFrame({
    "A": [0, 0, 0.4],
    "B": [1/6, 0, 0.6],
    "C": [0.5, 0.6, 0]
}, index=["A", "B", "C"])

# hard coded, known values based on flux / proportions from Van Buuren 4.1
df_flux = pd.DataFrame({
    "pobs": [0.75, 0.625, 0.625],
    "influx": [0.125, 0.250, 0.375],
    "outflux": [0.5, 0.375, 0.625]
}, index=["A", "B", "C"])

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

def test_inbound():
    """Assert that the inbound statistic returns expected"""
    inbound_ = inbound(df_general)
    assert isinstance(inbound_, pd.DataFrame)
    assert all(inbound_["A"] == df_inbound["A"])
    assert all(inbound_["B"] == df_inbound["B"])
    assert all(inbound_["C"] == df_inbound["C"])

def test_outbound():
    """Assert that the inbound statistic returns expected"""
    outbound_ = outbound(df_general)
    assert isinstance(outbound_, pd.DataFrame)
    assert all(outbound_["A"] == df_outbound["A"])
    assert all(outbound_["B"] == df_outbound["B"])
    assert all(outbound_["C"] == df_outbound["C"])

def test_flux():
    """Assert that columns in flux are correct"""
    flux_ = flux(df_general)
    assert isinstance(flux_, pd.DataFrame)
    assert all(flux_["pobs"] == df_flux["pobs"])
    assert all(flux_["influx"] == df_flux["influx"])
    assert all(flux_["outflux"] == df_flux["outflux"])
