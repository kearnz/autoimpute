Deletion and Imputation Strategies
==================================

This section documents deletion and imputation strategies within ``Autoimpute``.

Deletion is implemented through a single function, ``listwise_delete``, documented below.

Imputation strategies are implemented as classes. The authors of this package refer to these classes as "series-imputers". Each series-imputer maps to an imputation method - either univariate or multivariate - that imputes missing values within a pandas Series or numpy array. The imputation methods are the workhorses of the DataFrame Imputers, the ``SingleImputer`` and ``MultipleImputer``. Refer to the :doc:`imputers documentation<imputers>` for more information on the DataFrame Imputers.

For more information regarding the relationship between DataFrame Imputers and series-imputers, refer to the following tutorial_. The tutorial covers series-imputers in detail as well as the design patterns behind ``AutoImpute`` Imputers.

.. _tutorial: https://kearnz.github.io/autoimpute-tutorials/imputer-mechanics-II.html

Deletion Methods
----------------

.. autofunction:: autoimpute.imputations.listwise_delete


Imputation Strategies
---------------------

.. automodule:: autoimpute.imputations.series
    :special-members:
    :members: