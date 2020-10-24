DataFrame Imputers
==================

This section documents the DataFrame Imputers within ``Autoimpute``.

DataFrame Imputers are the primary feature of the package. The ``SingleImputer`` imputes each column within a DataFrame one time, while the ``MultipleImputer`` imputes each column within a DataFrame multiple times using independent runs. Under the hood, the ``MultipleImputer`` actually creates separate instances of the ``SingleImputer`` to handle each run. The ``MiceImputer`` takes the ``MultipleImputer`` one step futher, iteratively improving imputations in each column ``k`` times for each ``n`` runs the
``MultipleImputer`` performs. 

The base class for all imputers is the ``BaseImputer``. While you should not use the ``BaseImputer`` directly unless you're creating your own imputer class, you should understand what it provides the other imputers. The ``BaseImputer`` also contains the strategy "key-value store", or the methods that ``autoimpute`` currently supports. 

Base Imputer
------------

.. autoclass:: autoimpute.imputations.BaseImputer
    :special-members:
    :members:

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
