"""Deletion strategies to handle the missing data in pandas DataFrame."""

from autoimpute.utils import check_nan_columns

@check_nan_columns
def listwise_delete(data, inplace=False, verbose=False):
    """Delete all rows from a DataFrame where any missing values exist.

    Deletion is one way to handle missing values. This method removes any
    records that have a missing value in any of the features. This package
    focuses on imputation, not deletion. That being said, listwise deletion
    is a necessary component of any imputation package, as its the default
    method most people (and software) use to handle missing data.

    Args:
        data (pd.DataFrame): DataFrame used to delete missing rows.
        inplace (boolean, optional): perform operation inplace.
            Defaults to False.
        verbose (boolean, optional): print information to console.
            Defaults to False.

    Returns:
        pd.DataFrame: rows with missing values removed.

    Raises:
        ValueError: columns with all data missing. Raised through decorator.
    """
    num_records_before = len(data.index)
    if inplace:
        data.dropna(inplace=True)
    else:
        data = data.dropna(inplace=False)
    num_records_after = len(data.index)
    if verbose:
        print(f"Number of records before delete: {num_records_before}")
        print(f"Number of records after delete: {num_records_after}")
    return data
