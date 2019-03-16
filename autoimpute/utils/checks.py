"""Check and validate data & methods to ensure they play well w/ imputation.

This module is a series of decorators and functions used throughout autoimpute.
The decorators perform checks or transformations on functions that accept data.
They ensure the data is well formatted prior to running a given method.
Additional functions perform data validation and error handling. They are used
in different classes or methods where the same cheks must be performed often.

Methods:
    check_data_structure(func)
    check_missingness(func)
    check_nan_columns(func)
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
        checks whether the first argument, d, is a pandas DataFrame.
        If it's not, it checks whether the second arg (which would be the
        first arg of *args) is a DataFrame. This flexible checking means
        this decorator can validate methods within a class. Class methods
        using the decorator must take a DataFrame as the first arg after self.

        Args:
            d (self, pd.DataFrame): data to check. should be a DataFrame
                or self if an instance of a class.
            *args: Any number of arguments. If d is not a DataFrame, first arg
                must be or error raised.
            **kwargs: Keyword arguments for original function.

        Returns:
            function: Returns original function being decorated.

        Raises:
            TypeError: If one of d, args[0] not DataFrame.
        """
        d_df = isinstance(d, pd.DataFrame)
        a_df = isinstance(args[0], pd.DataFrame) if args else False
        if not any([d_df, a_df]):
            d_err = d.__class__.__name__
            a_err = args[0].__class__.__name__ if args else "first args"
            err = f"Neither {d_err} nor {a_err} are of type pd.DataFrame"
            raise TypeError(err)
        return func(d, *args, **kwargs)
    return wrapper

def check_missingness(func):
    """Check if DataFrame contains all missing values or no missing values.

    This method acts as a decorator. It takes a function that takes data
    as its first argument and checks whether that data contains any
    missing or real values. This method leverages the check_data_structure
    decorator to verify that the data is a DataFrame, and then verify that the
    data contains observed and missing values. This method utilizes the
    `functools.wrap` decorator to keep function names in tact.

    Args:
        func (function): The function that will be decorated.

    Returns:
        function: decorators return functions they wrap.
    """
    @functools.wraps(func)
    @check_data_structure
    def wrapper(d, *args, **kwargs):
        """Wrap function that checks a DataFrame's level of missingness.

        This wrapper within the decorator does the actual verification. It
        checks that a DataFrame has both missing and real values. If the
        DataFrame is fully incomplete, an error is raised. If the DataFrame
        has datetime columns that are not fully complete, an error is raised.

        Args:
            d (self, pd.DataFrame): Data to check for real and missing values.
                d can be instance of class if not pd.DataFrame.
            *args: Any number of arguments. If d is not a DataFrame, first arg
                must be or error raised.
            **kwargs: Keyword arguments from original function.

        Returns:
            function: Returns original function being decorated.

        Raises:
            ValueError: If all values in DataFrame are missing.
            ValueError: If any timeseries values in DataFrame are missing.
        """
        # b/c of check_data_structure, we know one of (d, a) is DataFrame
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
    """Checks if any column in DataFrame has all missing values.

    This method acts as a decorator. It leverages `check_missingness`
    to verify data is DataFrame with real and missing values. It then checks
    each column within the DataFrame to ensure no columns are fully missing.
    Decorator leverages `functools.wrap` to keep function names in place.

    Args:
        func (function): The function that will be decorated.

    Returns:
        function: decorators return functions they wrap.
    """
    @functools.wraps(func)
    @check_missingness
    def wrapper(d, *args, **kwargs):
        """Wrap function that checks if all rows in any one column missing.

        This wrapper does the actual verification. It ensures that each column
        within a DataFrame has at least one complete row. If any column is
        fully missing, an error is thrown. Users should remove columns from
        DataFrames if columns have no information whatsoever, as these columns
        are not useful for imputation nor analysis.

        Args:
            d (self, pd.DataFrame): DataFrame to check for missingness. If not
                DataFrame, should be instance of a class.
            *args: Variable number of arguments. If d not DataFrame, first arg
                in args should be DataFrame.
            **kwargs: Keyword arguments from original function.

        Raises:
            ValueError: If all values in any DataFrame column are missing.
        """
        # previous decorator checks ensure we are working with a DataFrame
        if isinstance(d, pd.DataFrame):
            ndf = pd.isnull(d)
        else:
            ndf = pd.isnull(args[0])
        nc = [c for c in ndf if ndf[c].all()]
        if nc:
            err = f"All values missing in column(s) {nc}. Should be removed."
            raise ValueError(err)
        return func(d, *args, **kwargs)
    return wrapper
