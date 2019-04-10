"""Check & validate data and methods to ensure they play well w/ imputation.

This module is a series of decorators and functions used throughout autoimpute.
The decorators perform checks or transformations on functions that accept data.
They ensure the data is well formatted prior to running a given method.
Additional functions perform data validation and error handling. They are used
in different classes or methods where the same checks must be performed often.
"""

import functools
import numpy as np
import pandas as pd

def check_data_structure(func):
    """Check if the data input to a function is a pandas DataFrame.

    This method acts as a decorator. It takes a function that takes data
    as its first argument and verifies that the data is a pandas DataFrame.
    Because this package has many functions which require a DataFrame as the
    first argument, the decorator makes it easy to verify this requirement
    regardless of what the rest of the function does. It utilizes the
    `functools.wrap` decorator to keep function names in tact.

    Args:
        func (function): The function that will be decorated

    Returns:
        function: decorators return function they wrap
    """
    @functools.wraps(func)
    def wrapper(d, *args, **kwargs):
        """Wrapper function for pandas DataFrame verification.

        The wrapper within the decorator does the actual verification. It
        checks whether the first argument, d, is a pandas DataFrame. If it's
        not, it checks whether the second arg (which would be the first arg of
        *args) is a DataFrame. This flexible checking means this decorator can
        validate methods in a class. Class methods using the decorator must
        take a DataFrame as the first arg after self.

        Args:
            d (self, pd.DataFrame): data to check. should be a DataFrame
                (or self if an instance of a class).
            *args: Any number of arguments. If d is not a DataFrame,
                first arg must be or error raised.
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
    """Check if accepted data contains all missing or no missing values.

    This method acts as a decorator. It takes a function that takes data
    as its first argument and checks whether that data contains any
    missing or real values. This method leverages the check_data_structure
    decorator to verify that the data is a DataFrame and then verify that
    the data contains observed and missing values. This method utilizes
    the `functools.wrap` decorator to keep function names in tact.

    Args:
        func (function): The function that will be decorated.

    Returns:
        function: decorators return functions they wrap.
    """
    @functools.wraps(func)
    @check_data_structure
    def wrapper(d, *args, **kwargs):
        """Wrap function that checks accepted data's level of missingness.

        This wrapper within the decorator does the actual verification. It
        checks that a DataFrame has both missing and real values. If the data
        is fully incomplete, an error is raised. If the data has datetime
        columns that are not fully complete, an error is raised.

        Args:
            d (self, pd.DataFrame): data to check. should be a DataFrame
                (or self if an instance of a class).
            *args: Any number of arguments. If d is not a DataFrame,
                first arg must be or error raised.
            **kwargs: Keyword arguments for original function.

        Returns:
            function: Returns original function being decorated.

        Raises:
            ValueError: If all values in data are missing.
            ValueError: If any timeseries values in data are missing.
        """
        # b/c of check_data_structure, we know 1 of (d, a) is DataFrame
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
    """Checks if any column in accepted data has all missing values.

    This method acts as a decorator. It leverages `check_missingness` to
    verify data has real and missing values. It then checks each column to
    ensure no columns are fully missing. Decorator leverages `functools.wrap`
    to keep function names in place.

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
        within data has at least one complete row. If any column is fully
        missing, an error is thrown. Users should remove columns that have no
        information whatsoever, as these columns are not useful for imputation
        nor analysis.

        Args:
            d (self, pd.DataFrame): data to check. should be a DataFrame
                (or self if an instance of a class).
            *args: Any number of arguments. If d is not a DataFrame,
                first arg must be or error raised.
            **kwargs: Keyword arguments for original function.

        Raises:
            ValueError: If all values in any column are missing.
        """
        # previous decorators ensure we are working with an accepted type
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

def check_strategy_allowed(strat_names, s):
    """Logic to determine if the strategy passed for imputation is valid.

    Imputer Classes in autoimpute have a very flexible `strategy` argument.
    The argument can be a string, an iterator, or a dictionary. In each case,
    the method(s) passed are checked against method(s) allowed.

    Args:
        strat_names (iterator): strategies allowed by given Imputer class.
        strategy (any): strategies passed as arguments.

    Returns:
        strategy (any): if string, iterator, or dictionary.

    Raises:
        ValueError: Strategies not valid (not in allowed strategies).
        TypeError: Strategy must be a string, tuple, list, or dict.
    """
    err_op = f"Strategies must be one of {list(strat_names)}."
    if isinstance(s, str):
        if s not in strat_names:
            err = f"Strategy {s} not a valid imputation method.\n"
            raise ValueError(f"{err} {err_op}")
    elif isinstance(s, (list, tuple, dict)):
        if isinstance(s, dict):
            ss = set(s.values())
        else:
            ss = set(s)
        sdiff = ss.difference(strat_names)
        if sdiff:
            err = f"Strategies {sdiff} in {s} not valid imputation.\n"
            raise ValueError(f"{err} {err_op}")
    else:
        raise TypeError("Strategy must be string, tuple, list, or dict.")
    return s

def check_strategy_fit(s, cols):
    """Check whether strategies make sense given data passed.

    An Imputer takes strategies to use for imputation. Those strategies are
    validated when instance is created. When fitting actual data, strategies
    must be validated again to verify they make sense given the columns in
    the dataset passed. For example, "mean" is fine when instance created, but
    "mean" will not work for a categorical column. This check validates
    strategy used for given column each strategy assigned to.

    Args:
        strategy (str, iter, dict): strategies passed for columns.
            String = 1 strategy, broadcast to all columns.
            Iter = multiple strategies, must match col index and length.
            Dict = multiple strategies, must match col name, but not all
            columns are mandatory. Will simply impute based on name.
        cols: columns in dataset for which strategies checked.

    Raises:
        ValueError (iter): length of columns and strategies must be equal.
        ValueError (dict): keys of strategies and columns must match.
    """
    c_l = len(cols)
    # if strategy is string, extend strategy to all cols
    if isinstance(s, str):
        sf = {c:s for c in cols}

    # if list or tuple, ensure same number of cols in X as strategies
    # note that list/tuple must have strategy specified for every column
    if isinstance(s, (list, tuple)):
        s_l = len(s)
        if s_l != c_l:
            err = "Length of columns not equal to number of strategies.\n"
            err_c = f"Length of columns: {c_l}\n"
            err_s = f"Length of strategies: {s_l}"
            raise ValueError(f"{err}{err_c}{err_s}")
        sf = {c[0]:c[1] for c in zip(cols, s)}

    # if strategy is dict, ensure keys in strategy match cols in X
    # note that dict is preferred way to impute SOME columns and not all
    if isinstance(s, dict):
        diff_s = set(s.keys()).difference(cols)
        if diff_s:
            err = "Keys of strategies and column names must match.\n"
            err_k = f"Ill-specified keys: {diff_s}"
            raise ValueError(f"{err}{err_k}")
        sf = s

    # return formatted strategy
    return sf

def check_predictors_fit(predictors, cols):
    """Checked predictors used for fitting each column.

    Similarly to `check_strategy_fit`, some Imputers use predictors to
    determine imputations to make. This check ensures those predictors
    exist within the dataset and are valid a specified imputation method.

    Args:
        predictors (str, iter, dict): predictors passed for columns.
            String = "all" or raises error.
            Iter = multiple strategies, must match col index and length.
            Dict = multiple strategies, must match col name, but not all.
            columns are mandatory. Will simply impute based on name.
        cols: columns in dataset for which predictors checked.

    Returns:
        predictors

    Raises:
        ValueError (str): string not equal to all.
        ValueError (iter): items in `predictors` not in columns of X.
        ValueError (dict, keys): keys of response must be columns in X.
        ValueError (dict, vals): vals of responses must be columns in X.
    """
    # if string, value must be `all`, or else raise an error
    if isinstance(predictors, str):
        if predictors != "all" and predictors not in cols:
            err = f"String {predictors} must be valid column in X.\n"
            err_all = "To use all columns, set predictors='all'."
            raise ValueError(f"{err}{err_all}")
        pf = {c:predictors for c in cols}

    # if list or tuple, remove nan cols and check col names
    if isinstance(predictors, (list, tuple)):
        bad_preds = [p for p in predictors if p not in cols]
        if bad_preds:
            err = f"{bad_preds} in predictors not a valid column in X."
            raise ValueError(err)
        pf = {c:predictors for c in cols}

    # if dictionary, remove nan cols and check col names
    if isinstance(predictors, dict):
        diff_s = set(predictors.keys()).difference(cols)
        if diff_s:
            err = "Keys of strategies and column names must match.\n"
            err_k = f"Ill-specified keys: {diff_s}"
            raise ValueError(f"{err}{err_k}")
        # then check the values of each key
        for k, preds in predictors.items():
            if isinstance(preds, str):
                if preds != "all" and preds not in cols:
                    err = f"Invalid column as only predictor for {k}."
                    raise ValueError(err)
            elif isinstance(preds, (tuple, list)):
                bad_preds = [p for p in preds if p not in cols]
                if bad_preds:
                    err = f"{bad_preds} for {k} not a valid column in X."
                    raise ValueError(err)
            else:
                err = "Values in predictor must be str, list, or tuple."
                raise ValueError(err)
        # finally, create predictors dict
        for c in cols:
            if c not in predictors:
                predictors[c] = "all"
        pf = predictors

    # return formatted predictors
    return pf
