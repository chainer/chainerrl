from chainerrl import action_value  # NOQA
from chainerrl import agent  # NOQA
from chainerrl import agents  # NOQA
from chainerrl import distribution  # NOQA
from chainerrl import env  # NOQA
from chainerrl import envs  # NOQA
from chainerrl import experiments  # NOQA
from chainerrl import explorer  # NOQA
from chainerrl import explorers  # NOQA
from chainerrl import explorers  # NOQA
from chainerrl import functions  # NOQA
from chainerrl import links  # NOQA
from chainerrl import misc  # NOQA
from chainerrl import optimizers  # NOQA
from chainerrl import policies  # NOQA
from chainerrl import policy  # NOQA
from chainerrl import q_function  # NOQA
from chainerrl import q_functions  # NOQA
from chainerrl import recurrent  # NOQA
from chainerrl import replay_buffer  # NOQA
from chainerrl import v_function  # NOQA
from chainerrl import v_functions  # NOQA

# For backward compatibility while avoiding circular import
policy.SoftmaxPolicy = policies.SoftmaxPolicy
policy.FCSoftmaxPolicy = policies.FCSoftmaxPolicy
policy.ContinuousDeterministicPolicy = policies.ContinuousDeterministicPolicy
policy.FCDeterministicPolicy = policies.FCDeterministicPolicy
policy.FCBNDeterministicPolicy = policies.FCBNDeterministicPolicy
policy.FCLSTMDeterministicPolicy = policies.FCLSTMDeterministicPolicy
policy.FCLSTMDeterministicPolicy = policies.FCLSTMDeterministicPolicy
policy.GaussianPolicy = policies.GaussianPolicy
policy.FCGaussianPolicy = policies.FCGaussianPolicy
policy.LinearGaussianPolicyWithDiagonalCovariance = \
    policies.LinearGaussianPolicyWithDiagonalCovariance
policy.LinearGaussianPolicyWithSphericalCovariance = \
    policies.LinearGaussianPolicyWithSphericalCovariance
policy.MellowmaxPolicy = policies.MellowmaxPolicy

q_function.DuelingDQN = q_functions.DuelingDQN
q_function.SingleModelStateActionQFunction = \
    q_functions.SingleModelStateActionQFunction
q_function.FCSAQFunction = q_functions.FCSAQFunction
q_function.FCLSTMSAQFunction = q_functions.FCLSTMSAQFunction
q_function.FCBNSAQFunction = q_functions.FCBNSAQFunction
q_function.FCBNLateActionSAQFunction = q_functions.FCBNLateActionSAQFunction
q_function.FCLateActionSAQFunction = q_functions.FCLateActionSAQFunction
q_function.SingleModelStateActionQFunction = \
    q_functions.SingleModelStateActionQFunction
q_function.FCStateQFunctionWithDiscreteAction = \
    q_functions.FCStateQFunctionWithDiscreteAction
q_function.FCLSTMStateQFunction = q_functions.FCLSTMStateQFunction
q_function.FCQuadraticStateQFunction = q_functions.FCQuadraticStateQFunction
q_function.FCBNQuadraticStateQFunction = \
    q_functions.FCBNQuadraticStateQFunction

v_function.SingleModelVFunction = v_functions.SingleModelVFunction
v_function.FCVFunction = v_functions.FCVFunction
