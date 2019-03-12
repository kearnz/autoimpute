"""Check and validate data & methods to ensure they play well w/ imputation.

This module is a series of decorators and functions used throughout autoimpute.
The decorators perform checks or transformations on functions that accept data.
They ensure the data is well formatted prior to running a given method.
Additional functions perform data validation and error handling. They are used
in different classes or methods where the same behavior is reused.

Methods:
    check_data_structure(func)
    check_missingness(func)
    check_nan_columns(func)

Todo:
    * Support data structures other than pandas DataFrame. Consider:
        - list, tuple, and other built-in iterators and collections
        - numpy array, and pandas Series
        - scipy sparse matrices
    * If other data structures added, bring back decorator to verify 2D
        - Right now, we only accept pandas DataFrames. All pandas datafrmes
          are 2-dimensional, so no need to verify their shape.
        - If we allow for arrays or other data types, we don't get
          this verification for free and need to enforce. We can introduce
          a `check_dimensions` decorator, which checks shape of input data.
"""

import functools
import numpy as np
import pandas as pd

def check_data_structure(func):
    """Check if the data input to a function is a pandas DataFrame.

    This method acts as a decorator. It takes a function that takes data
    as its first argument and verifies that the data is a pandas DataFrame.
    Because this package has many functions which require a DataFrame
    as the first argument, this decorator makes it easy to verify this
    requirement regardless of what the rest of the function does. It
    utilizes the `functools.wrap` decorator to keep function names in tact.

    Args:
        func (function): The function that will be decorated

    Returns:
        function: decorators return function they wrap
    """
    @functools.wraps(func)
    def wrapper(d, *args, **kwargs):
        """Wrapper function that does the pandas DataFrame verification.

        The wrapper within the decorator does the actual verification. It
        is flexible in that it checks whether the first argument, d, is a
        pandas DataFrame. If it's not, it checks whether the second arg
        (which would be the first arg of *args) is a DataFrame. This flexible
        checking means this decorator can be used to validate methods
        within a class (which have self as their first argument). Class
        methods using this decorator must take a DataFrame as their second
        argument (which comes up in *args as args[0]).

        Args:
            d (any): Any data type allowed, but all data types raise an
                error unless they are a pandas DataFrame.
            *args: Any number of arguments. If d is NOT a pandas DataFrame,
                first arg MUST be a pandas DataFrame, or error raised.
            **kwargs: Keyword arguments for original function.

        Returns:
            function: Returns original function being decorated.

        Raises:
            TypeError: If one of d, args[0] not pandas DataFrame.
        """
        if isinstance(d, pd.DataFrame):
            return func(d, *args, **kwargs)
        else:
            if args:
                a = args[0]
                if isinstance(a, pd.DataFrame):
                    return func(d, *args, **kwargs)
                else:
                    err = a.__class__.__name__
                    raise TypeError(f"Type '{err}' not a DataFrame")
            else:
                err = d.__class__.__name__
                raise TypeError(f"Type '{err}' not a DataFrame'")
    return wrapper

def check_missingness(func):
    """Check if DataFrame contains all missing values or no missing values.

    This method acts as a decorator. It takes a function that takes data
    as its first argument and checks whether that data contains any
    missing or real values. This method leverages the check_data_structure
    decorator above to first verify that the data is a DataFrame, and then
    verify that the data contains real and missing values. This decorator
    also works well with classes and utilizes the `functools.wrap`
    decorator to keep function names in tact.

    Args:
        func (function): The function that will be decorated.

    Returns:
        function: decorators return functions they wrap.
    """
    @functools.wraps(func)
    @check_data_structure
    def wrapper(d, *args, **kwargs):
        """Wrap function that checks a DataFrame for missing and real values.

        This wrapper within the decorator does the actual verification. It
        checks that a DataFrame has both missing and real values. If the
        DataFrame is fully incomplete, an error is raised. If the DataFrame
        has datetime columns that are not fully complete, an error is raised.

        Args:
            d (any): Again, anything can be passed, but DataFrame required
                to continue without error from check_data_structure.
            *args: A number of arguments. First must be DataFrame if
                working within a class.
            **kwargs: Keyword arguments from original function.

        Returns:
            function: Returns original function being decorated.

        Raises:
            ValueError: If all values in DataFrame are missing.
            ValueError: If timeseries values in DataFrame are missing.
        """
        if isinstance(d, pd.DataFrame):
            n_ts = d.select_dtypes(include=(np.number, np.object))
            ts = d.select_dtypes(include=(np.datetime64,))
        else:
            a = args[0]
            n_ts = a.select_dtypes(include=(np.number, np.object))
            ts = a.select_dtypes(include=(np.datetime64,))

        # check if non-time series columns are all missing, and if so, error
        if n_ts.columns.any():
            missing_nts = pd.isnull(n_ts.values)
            if missing_nts.all():
                raise ValueError("All values missing, need some complete.")

        # check if any time series columns have missing data, and if so, error
        if ts.columns.any():
            missing_ts = pd.isnull(ts.values)
            if missing_ts.any():
                raise ValueError("Time series columns must be fully complete.")

        # return func if no missingness violations detected, then return wrap
        return func(d, *args, **kwargs)
    return wrapper

def check_nan_columns(func):
    """Removes columns inplace in DataFrame if column all missing values.

    This method acts as a decorator. It first leverages the decorator
    `check_missingness` to verify that a proper DataFrame is passed. It
    then drops the columns with all rows missing from said DataFrame.
    Dropped columns result in a warning issued with the dropped column
    name. Again, it leverages `functools.wrap` to keep func names in place.

    Args:
        func (function): The function that will be decorated.

    Returns:
        function: decorators return functions they wrap.
    """
    @functools.wraps(func)
    @check_missingness
    def wrapper(d, *args, **kwargs):
        """Wrap function that removes columns w/ all rows missing.

        The wrapper leverages the `_nan_col_dropper` helper function,
        which removes columns from a DataFrame that have all rows missing.

        Args:
            d (pd.DataFrame): DataFrame to check, if not in args.
            *args: Variable number of arguments, first of which may be df.
            **kwargs: Keyword arguments from original function.
        """
        if isinstance(d, pd.DataFrame):
            null_df = pd.isnull(d)
        else:
            null_df = pd.isnull(args[0])
        null_cols = []
        for col in null_df:
            if null_df[col].all():
                null_cols.append(col)
        if null_cols:
            raise ValueError(f"Columns {null_cols} have all values missing.")
        return func(d, *args, **kwargs)
    return wrapper
