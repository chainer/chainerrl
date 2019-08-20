========
Policies
========

Policy interfaces
=================

.. autoclass:: chainerrl.policy.Policy
   :members:

   .. automethod:: __call__

Policy implementations
======================

.. autoclass:: chainerrl.policies.ContinuousDeterministicPolicy

.. autoclass:: chainerrl.policies.FCDeterministicPolicy

.. autoclass:: chainerrl.policies.FCBNDeterministicPolicy

.. autoclass:: chainerrl.policies.FCLSTMDeterministicPolicy

.. autoclass:: chainerrl.policies.FCGaussianPolicy

.. autoclass:: chainerrl.policies.FCGaussianPolicyWithStateIndependentCovariance

.. autoclass:: chainerrl.policies.FCGaussianPolicyWithFixedCovariance

.. autoclass:: chainerrl.policies.GaussianHeadWithStateIndependentCovariance

.. autoclass:: chainerrl.policies.MellowmaxPolicy

.. autoclass:: chainerrl.policies.SoftmaxPolicy

.. autoclass:: chainerrl.policies.FCSoftmaxPolicy
