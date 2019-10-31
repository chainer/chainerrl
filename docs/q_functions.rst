===========
Q-functions
===========

Q-function interfaces
=====================

.. autoclass:: chainerrl.q_function.StateQFunction
   :members:

   .. automethod:: __call__

.. autoclass:: chainerrl.q_function.StateActionQFunction
   :members:

   .. automethod:: __call__

Q-function implementations
==========================

.. autoclass:: chainerrl.q_functions.DuelingDQN

.. autoclass:: chainerrl.q_functions.DistributionalDuelingDQN

.. autoclass:: chainerrl.q_functions.SingleModelStateQFunctionWithDiscreteAction

.. autoclass:: chainerrl.q_functions.FCStateQFunctionWithDiscreteAction

.. autoclass:: chainerrl.q_functions.DistributionalSingleModelStateQFunctionWithDiscreteAction

.. autoclass:: chainerrl.q_functions.DistributionalFCStateQFunctionWithDiscreteAction

.. autoclass:: chainerrl.q_functions.FCLSTMStateQFunction

.. autoclass:: chainerrl.q_functions.FCQuadraticStateQFunction

.. autoclass:: chainerrl.q_functions.FCBNQuadraticStateQFunction

.. autoclass:: chainerrl.q_functions.SingleModelStateActionQFunction

.. autoclass:: chainerrl.q_functions.FCSAQFunction

.. autoclass:: chainerrl.q_functions.FCLSTMSAQFunction

.. autoclass:: chainerrl.q_functions.FCBNSAQFunction

.. autoclass:: chainerrl.q_functions.FCBNLateActionSAQFunction

.. autoclass:: chainerrl.q_functions.FCLateActionSAQFunction