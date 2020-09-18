DataFrame Imputers
==================

This section documents the DataFrame Imputers within ``Autoimpute``.

DataFrame Imputers are the primary feature of the package. The ``SingleImputer`` imputes each column within a DataFrame one time, while the ``MultipleImputer`` imputes each column within a DataFrame multiple times. Under the hood, the ``MultipleImputer`` actually creates a separate instance of the ``SingleImputer``, which handles each imputation run. The ``MiceImputer`` takes the ``MultipleImputer`` one step futher, iteratively improving imputations in each column ``k`` times for each ``m`` instances the
``MultipleImputer`` creates.

Single Imputer
--------------

.. autoclass:: autoimpute.imputations.SingleImputer
    :special-members:
    :members:

Multiple Imputer
----------------

.. autoclass:: autoimpute.imputations.MultipleImputer
    :special-members:
    :members:

Mice Imputer
-------------

.. autoclass:: autoimpute.imputations.MiceImputer
   :special-members:
   :members:
