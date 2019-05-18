Analysis Models
===============

This section documents analysis models within ``Autoimpute`` and their respective diagnostics. 

The ``MiLinearRegression`` and ``MiLogisticRegression`` extend linear and logistic regression to multiply imputed datasets. Under the hood, each regression class uses a ``MultipleImputer`` to handle missing data prior to supervised analysis. Users of each regression class can tweak the underlying ``MultipleImputer`` through the ``mi_kwgs`` argument or pass a pre-configured instance to the ``mi`` argument (recommended).

Users can also specify whether the classes should use ``sklearn`` or ``statsmodels`` to implement linear or logistic regression. The default is ``statsmodels``. When used, end users get more detailed parameter diagnostics for regression on multiply imputed data.

Finally, this section provides diagnostic helper methods to assess bias of parameters from a regression model.

Linear Regression for Multiply Imputed Data
-------------------------------------------

.. autoclass:: autoimpute.analysis.MiLinearRegression
    :special-members:
    :members:


Logistic Regression for Multiply Imputed Data
---------------------------------------------

.. autoclass:: autoimpute.analysis.MiLogisticRegression
    :special-members:
    :members:


Diagnostics
-----------

.. autofunction:: autoimpute.analysis.raw_bias

.. autofunction:: autoimpute.analysis.percent_bias