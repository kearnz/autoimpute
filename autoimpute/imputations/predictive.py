"""This module uses available information in a dataset to predict imputations.

This module contains one class - the PredictiveImputer. Use this class to
predict imputations for each Series within a DataFrame using all or a subset
of the other available features. This class extends the behavior of the
SingleImputer. Unlike the SingleImputer, the supported methods in this class
are multivariate - they use more than just the series itself to determine the
best estimated values for imputaiton.

Todo:
    * class specification
    * outline strategies planned for implementation
    * create multivariate methods module with predictive strategies
"""
