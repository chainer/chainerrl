from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import basetest_pgt as base
from chainerrl.agents.pgt import PGT


# Currently PGT does not support recurrent models
# class TestPGTOnContinuousPOABC(base._TestPGTOnContinuousPOABC):
#
#     def make_pgt_agent(self, env, model, actor_opt, critic_opt, explorer,
#                        rbuf, gpu):
#         return PGT(model, actor_opt, critic_opt, rbuf, gpu=gpu, gamma=0.9,
#                    explorer=explorer, replay_start_size=100,
#                    target_update_method='soft', target_update_interval=1,
#                    episodic_update=True, update_interval=1,
#                    act_deterministically=True)


class TestPGTOnContinuousABC(base._TestPGTOnContinuousABC):

    def make_pgt_agent(self, env, model, actor_opt, critic_opt, explorer,
                       rbuf, gpu):
        return PGT(model, actor_opt, critic_opt, rbuf, gpu=gpu, gamma=0.9,
                   explorer=explorer, replay_start_size=100,
                   target_update_method='soft', target_update_interval=1,
                   act_deterministically=True)
