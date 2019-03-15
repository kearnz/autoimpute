"""Private methods for handling errors throughout imputation analysis."""

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

# ERROR HANDLING
# --------------

def _not_num_series(m, s):
    """error handling for all single private methods that are numerical."""
    if not is_numeric_dtype(s):
        t = s.dtype
        e = f"{m} not appropriate for Series with type {t}. Need numeric."
        raise TypeError(e)

def _not_num_matrix(m, mat):
    """error handling for all multi private methods that are numerical."""
    try:
        for each_col in mat:
            _not_num_series(m, each_col)
    except TypeError:
        e = f"{m} not appropriate for Matrix with any non-numerical column."
        raise TypeError(e)

def _not_cat_series(m, s):
    """error handling for all single private methods that are categorical."""
    if not is_string_dtype(s):
        t = s.dtype
        e = f"{m} not appropriate for Series with type {t}. Need categorical."
        raise TypeError(e)

def _not_cat_matrix(m, mat):
    """error handling for all muilti private methods that are categorical."""
    try:
        for each_col in mat:
            _not_cat_series(m, each_col)
    except TypeError:
        e = f"{m} not appropriate for Matrix with any non-categorical column."
        raise TypeError(e)
