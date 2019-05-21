Visualization Methods
=====================

This section documents visualization methods within ``Autoimpute``.

Visualization methods support all functionality within ``Autoimpute``, from missing data exploration to imputation analysis. The documentation below breaks down each visualization method and groups them into their respsective categories. The categories represent other modules within ``Autoimpute``.

Utility
-------

``Autoimpute`` comes with a number of :doc:`utility methods<utils>` to examine missing data before imputation takes place. This package supports these methods with a number of visualization techniques to explore patterns within missing data. The primary techniques wrap the excellent `missingno package <https://github.com/ResidentMario/missingno>`_. ``Autoimpute`` simply leverages ``missingno`` to make its offerings familiar in this packages' API design. The methods appear below:

.. autofunction:: autoimpute.visuals.plot_md_locations

.. autofunction:: autoimpute.visuals.plot_md_percent

.. autofunction:: autoimpute.visuals.plot_nullility_corr

.. autofunction:: autoimpute.visuals.plot_nullility_dendogram

Imputation
----------

Two main classes within ``Autoimpute`` are the :doc:`SingleImputer and MultipleImputer<imputers>`. The visualization module within this package contains a number of techniques to visually assess the quality and performance of these imputers. The important methods appear below:

.. autofunction:: autoimpute.visuals.helpers._validate_data

.. autofunction:: autoimpute.visuals.plot_imp_dists

.. autofunction:: autoimpute.visuals.plot_imp_boxplots

.. autofunction:: autoimpute.visuals.plot_imp_swarm

.. autofunction:: autoimpute.visuals.plot_imp_strip
