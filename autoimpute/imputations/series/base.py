"""This module implements the abstract base class used for series-imputers.

The base class is quite simple but important. It specifies the contract for
how series-imputers should behave. All series-imputers should inherit from
this base class to be considered valid Imputers.
"""

import abc
from sklearn.base import BaseEstimator

class ISeriesImputer(BaseEstimator, metaclass=abc.ABCMeta):
    """ISeriesImputer implements the abstract base class for series-imputers.

    All series imputers should have a fit, impute, and fit_impute method to be
    considered valid to build imputation models. The ISeriesImputer is the
    contract series-imputers must adhere to."""

    @abc.abstractmethod
    def fit(self, X, y):
        """Contract to fit an imputation model.

        Args:
            X (pd.Series, pd.Dataframe): data used to build imputation
                model. pd.Series if univariate, pd.DataFrame if multivariate.
            y (pd.Series, None): column to impute. None if univariate,
                pd.Series if multivariate.

        Returns:
            self: instance of a class
        """

    @abc.abstractmethod
    def impute(self, X):
        """Contract to impute using a fit imputation model.

        Args:
            X (pd.Series, pd.DataFrame): data to use to generate imputations.
                pd.Series if univariate, pd.DataFrame if multivariate.

        Returns:
            imputations, scalar or array depending on imputation model.
        """

    @abc.abstractmethod
    def fit_impute(self, X, y):
        """Convenience method that implements fit & impute in one go.

         Args:
            X (pd.Series, pd.Dataframe): data used to build imputation
                model. pd.Series if univariate, pd.DataFrame if multivariate.
            y (pd.Series, None): column to impute. None if univariate,
                pd.Series if multivariate.

        Returns:
            imputations, scalar or array depending on imputation model.
        """
