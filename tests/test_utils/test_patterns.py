"""Tests written to ensure the patterns in the utils package work correctly.

The methods tested below are ports from MICE, an excellent R package for
handling missing data. The author of MICE (Van Buuren) is also the author of
Flexible Imputation of Missing Data (FIMD). The `df_general` variable below
is a simulation of the "general" pattern in section 4.1. The subsequent
dataframes are hard coded results of patterns from FIMD. They are used to
verify that this implementation in Python is working as expected. The methods
being tested (i.e. inbound, outbound, flux, etc.) all use `df_general` to
calculate their respective statistic. This allows for comparison to results
from FIMD, section 4.1.

Tests use the pytest library. The tests in this module ensure the following:
- `test_md_locations` checks missingness identified properly as 1/0.
- `test_md_pattern` checks against result from MICE md.pattern.
- `test_md_pairs` checks against result from MICE md.pairs
- `test_inbound` checks against inbound calc in 4.1 (no explicit method)
- `test_outbound` checks against outbound calc in 4.1 (no explicit method)
- `test_flux` checks against MICE flux.
"""

import numpy as np
import pandas as pd
from autoimpute.utils.patterns import md_locations, md_pairs, md_pattern
from autoimpute.utils.patterns import inbound, outbound, flux

df_general = pd.DataFrame({
    "A": [1, 5, 9, 6, 12, 11, np.nan, np.nan],
    "B": [2, 4, 3, 6, 11, np.nan, np.nan, np.nan],
    "C": [-1, 1, np.nan, np.nan, np.nan, -1, 1, 0]
})

df_pattern = pd.DataFrame({
    "count": [2, 3, 1, 2],
    "A": [1, 1, 1, 0],
    "B": [1, 1, 0, 0],
    "C": [1, 0, 1, 1],
    "nmis": [0, 1, 1, 2]
})

dict_pairs = {}
ci = ["A", "B", "C"]
create_df = lambda v: pd.DataFrame(v, columns=ci, index=ci)
dict_pairs["rr"] = create_df([[6, 5, 3], [5, 5, 2], [3, 2, 5]])
dict_pairs["rm"] = create_df([[0, 1, 3], [0, 0, 3], [2, 3, 0]])
dict_pairs["mr"] = create_df([[0, 0, 2], [1, 0, 3], [3, 3, 0]])
dict_pairs["mm"] = create_df([[2, 2, 0], [2, 3, 0], [0, 0, 3]])

df_inbound = create_df([[0, 1/3, 1], [0, 0, 1], [1, 1, 0]]).T
df_outbound = create_df([[0, 0, 0.4], [1/6, 0, 0.6], [0.5, 0.6, 0]]).T

df_flux = pd.DataFrame({
    "pobs": [0.75, 0.625, 0.625],
    "influx": [0.125, 0.250, 0.375],
    "outflux": [0.5, 0.375, 0.625]
}, index=ci)

def test_md_locations():
    """Test to ensure that missingness locations are identified.

    Missingness locations should equal np.isnan for each col.
    Assert that md_locations returns a DataFrame, and then check
    that each column equals what is expected from np.isnan.

    Args:
        None: DataFrame for testing created internally.

    Returns:
        None: asserts locations for missingness are as expected.
    """
    md_loc = md_locations(df_general)
    assert isinstance(md_loc, pd.DataFrame)
    assert all(md_loc["A"] == np.isnan(df_general["A"]))
    assert all(md_loc["B"] == np.isnan(df_general["B"]))
    assert all(md_loc["C"] == np.isnan(df_general["C"]))

def test_md_pattern():
    """Test that missing data pattern equal to expected results.

    `df_pattern` name assigned to DataFrame in module's scope that contains
    the expected pattern from VB 4.1 `md.pattern()` example in R. Result
    is hard coded, and python version tested with assertions below.

    Args:
        None: DataFrame for testing created internally.

    Returns:
        None: asserts missingness patterns are as expected.
    """
    md_pat = md_pattern(df_general)
    assert isinstance(md_pat, pd.DataFrame)
    assert all(md_pat["count"] == df_pattern["count"])
    assert all(md_pat[["A", "B", "C"]] == df_pattern[["A", "B", "C"]])
    assert all(md_pat["nmis"] == df_pattern["nmis"])

def test_md_pairs():
    """Test that missing data pairs equal to expected results.

    `dict_pairs` contains 4 keys - one for each pair expected. The pairs
    are `rr`, `mr`, `rm`, and `mm`. Pair types described in the docstrings
    of the md_pairs method in the utils.patterns module. Missing data pairs
    should equal expected pairs from VB 4.1 `md.pairs() in R.

    Args:
        None: Pairs for testing created internally.

    Returns:
        None: asserts pairs are as expected.
    """
    md_pair = md_pairs(df_general)
    assert isinstance(md_pair, dict)
    assert all(md_pair["rr"] == dict_pairs["rr"])
    assert all(md_pair["mr"] == dict_pairs["mr"])
    assert all(md_pair["rm"] == dict_pairs["rm"])
    assert all(md_pair["mm"] == dict_pairs["mm"])

def test_inbound():
    """Test that the inbound statistic equal to expected results.

    `df_inbound` contains hard-coded expected result. Tested against the
    inbound function, which takes `df_general` as an input.

    Args:
        None: DataFrame for testing created internally.

    Returns:
        None: asserts inbound statistic results are as expected.
    """
    inbound_ = inbound(df_general)
    assert isinstance(inbound_, pd.DataFrame)
    assert all(inbound_["A"] == df_inbound["A"])
    assert all(inbound_["B"] == df_inbound["B"])
    assert all(inbound_["C"] == df_inbound["C"])

def test_outbound():
    """Test that the outbound statistic equal to expected results.

    `df_outbound` contains hard-coded expected result. Tested against the
    outbound function, which takes `df_general` as an input.

    Args:
        None: DataFrame for testing created internally.

    Returns:
        None: asserts outbound statistic results are as expected.
    """
    outbound_ = outbound(df_general)
    print(df_inbound)
    assert isinstance(outbound_, pd.DataFrame)
    assert all(outbound_["A"] == df_outbound["A"])
    assert all(outbound_["B"] == df_outbound["B"])
    assert all(outbound_["C"] == df_outbound["C"])

def test_flux():
    """Test that the flux coeffs and proportions equal to expected results.

    `df_flux` contains hard-coded expected result. Tested against the
    influx, outflux, and proportions functions, which all take `df_general`
    as an input.

    Args:
        None: DataFrame for testing created internally.

    Returns:
        None: asserts pobs, influx, and outflux results are as expected.
    """
    flux_ = flux(df_general)
    assert isinstance(flux_, pd.DataFrame)
    assert all(flux_["pobs"] == df_flux["pobs"])
    assert all(flux_["influx"] == df_flux["influx"])
    assert all(flux_["outflux"] == df_flux["outflux"])
