"""Private methods for handling errors throughout imputation analysis."""

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

# ERROR HANDLING
# --------------

def _not_num_series(m, s):
    """Private method to detect columns of Matrix that are not categorical."""
    if not is_numeric_dtype(s):
        t = s.dtype
        err = f"{m} not appropriate for Series {s.name} of type {t}."
        raise TypeError(err)

def _not_num_matrix(m, mat):
    """Private method to detect columns of Matrix that are not numerical."""
    try:
        for each_col in mat:
            c = mat[each_col]
            _not_num_series(m, c)
    except TypeError as te:
        err = f"{m} not appropriate for Matrix with non-numerical columns."
        raise TypeError(err) from te

def _not_cat_series(m, s):
    """Private method to detect Series that are not categorical."""
    if not is_string_dtype(s):
        t = s.dtype
        err = f"{m} not appropriate for Series {s.name} of type {t}."
        raise TypeError(err)

def _not_cat_matrix(m, mat):
    """Private method to detect columns of Matrix that are not categorical."""
    try:
        for each_col in mat:
            c = mat[each_col]
            _not_cat_series(m, c)
    except TypeError as te:
        err = f"{m} not appropriate for Matrix with non-categorical columns."
        raise TypeError(err) from te
