"""Deletion strategies to handle the missing data problem.

This module provides mechanisms to handle missing data by simply deleting
missing data and removing columns, rows, or both that have missingness.
The methods in this module can be successfully deployed in some use cases.
However, these strategies are generally not recommended b/c they can lead to
biased outcomes unless the dataset is MCAR and large enough that a reduction
in statistical power is negligible. Therefore, these methods are provided for
completion and thoroughness even though there are often preferred alternatives.

Methods:
    listwise_delete(data, inplace=False, verbose=False)

Todo:
    * add method for pairwise deletion
    * update docs with examples
"""

from autoimpute.utils.checks import remove_nan_columns

@remove_nan_columns
def listwise_delete(data, inplace=False, verbose=False):
    """Delete all rows from a dataframe where any missing values exist.

    Deletion one way to handle missing values. This method removes any
    records that have a missing value in any of the features. Doing so
    reduces the sample size but preserves the summary statistics (such as
    moments) if and only if the dataset is MCAR. If the dataset is not
    MCAR, listwise deletion can lead to bias and is NOT preferred method.

    Args:
        data (pd.DataFrame): DataFrame used to delete missing rows.
        inplace (boolean, optional): perform operation inplace
        verbose (boolean, optional): print information to console

    Returns:
        pd.DataFrame: rows with missing values removed. Rows of DataFrame
            will have <= the number of rows of the original
    """
    num_records_before = len(data.index)
    if inplace:
        data.dropna(inplace=inplace)
    else:
        data = data.dropna(inplace=inplace)
    num_records_after = len(data.index)
    if verbose:
        print(f"Number of records before delete: {num_records_before}")
        print(f"Number of records after delete: {num_records_after}")
    return data
