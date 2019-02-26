"""Deletion strategies to handle the missing data in pandas DataFrame.

This module provides mechanisms to handle missing data by simply deleting
missing data and removing columns, rows, or both that have missingness.
The methods in this module can be successfully deployed for some use cases.
However, these strategies are generally not recommended because they can lead
to biased outcomes unless the dataset is MCAR and large enough that a reduction
in statistical power is negligible. Therefore, these methods are provided for
completion and thoroughness even though there are often preferred alternatives.

Methods:
    listwise_delete(data, inplace=False, verbose=False)

Todo:
    * Add method for pairwise deletion.
    * Update docstrings in methods with examples.
"""

from autoimpute.utils.checks import remove_nan_columns

@remove_nan_columns
def listwise_delete(data, inplace=False, verbose=False):
    """Delete all rows from a DataFrame where any missing values exist.

    Deletion one way to handle missing values. This method removes any
    records that have a missing value in any of the features. Doing so
    reduces the sample size but preserves the summary statistics (such as
    moments) if and only if the dataset is MCAR. If the dataset is not
    MCAR, listwise deletion can lead to bias and is NOT preferred method.
    See Flexible Imputation of Missing Data, Van Buuren, for more info.

    Args:
        data (pd.DataFrame): DataFrame used to delete missing rows.
        inplace (boolean, optional): perform operation inplace.
            Defaults to False.
        verbose (boolean, optional): print information to console.
            Defaults to False.

    Returns:
        pd.DataFrame: rows with missing values removed. Number of rows
            remaining <= number of rows of the original DataFrame.
    """
    if not verbose:
        return data.dropna(inplace=inplace)
    else:
        num_records_before = len(data.index)
        if not inplace:
            data.dropna(inplace=True)
        else:
            data = data.dropna(inplace=False)
        num_records_after = len(data.index)
        print(f"Number of records before delete: {num_records_before}")
        print(f"Number of records after delete: {num_records_after}")
        return data
