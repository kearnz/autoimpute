"""Private methods for handling errors throughout imputation analysis."""

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

# ERROR HANDLING
# --------------

def _not_num_err(m, s):
    """error hanlding for all the private methods that are numerical."""
    if not is_numeric_dtype(s):
        t = s.dtype
        e = f"{m} not appropriate for Series with type {t}. Need numeric."
        raise TypeError(e)

def _not_cat_err(m, s):
    """error handling for all the private methods that are categorical."""
    if not is_string_dtype(s):
        t = s.dtype
        e = f"{m} not appropriate for Series with type {t}. Need categorical."
        raise TypeError(e)
