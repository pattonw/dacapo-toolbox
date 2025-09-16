.. _sec_api:

API Reference
=============

.. automodule:: dacapo_toolbox
   :no-index:

Dataset
=======

    .. autoclass:: dacapo_toolbox.dataset.iterable_dataset
        :members:

    .. autoclass:: dacapo_toolbox.dataset.SimpleAugmentConfig
        :members:

    .. autoclass:: dacapo_toolbox.dataset.DeformAugmentConfig
        :members:
    
    .. autoclass:: dacapo_toolbox.dataset.MaskedSampling
        :members:

    .. autoclass:: dacapo_toolbox.dataset.PointSampling
        :members:

Transforms
==========

Affinities
----------

    .. autoclass:: dacapo_toolbox.transforms.affs.Affs
        :members:
    
    .. autoclass:: dacapo_toolbox.transforms.affs.AffsMask
        :members:

    .. autofunction:: dacapo_toolbox.transforms.affs.compute_affs
        :members:

Distances
---------

    .. autoclass:: dacapo_toolbox.transforms.distances.SignedDistanceTransform
        :members:

    .. autoclass:: dacapo_toolbox.transforms.distances.SDTBoundaryMask
        :members:

LSDS
----

    .. autoclass:: dacapo_toolbox.transforms.lsds.LSD
        :members:

    .. autofunction:: dacapo_toolbox.transforms.lsds.get_local_shape_descriptors
    
Weight Balancing
----------------

    .. autoclass:: dacapo_toolbox.transforms.weight_balancing.BalanceLabels
        :members:

    .. autofunction:: dacapo_toolbox.transforms.weight_balancing.balance_weights

Visualizations
==============

    .. autofunction:: dacapo_toolbox.vis.preview.gif_2d

    .. autofunction:: dacapo_toolbox.vis.preview.cube

Sample Data
===========

    .. autofunction:: dacapo_toolbox.sample_datasets.cremi


