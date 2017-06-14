import chainer
from chainer import cuda
import chainer.functions as F
from chainerrl.agents import DQN
from chainerrl.agents.dqn import compute_value_loss
from chainerrl.eligibility_trace import lambda_return


class LambdaReturnDQN (DQN):
    def __init__(self, *args, **kwargs):
        self.lambd = kwargs.pop('lambd')
        super().__init__(*args, **kwargs)

    def update_from_episodes(self, episodes, errors_out=None):
        weights = None
        """
        if isinstance(episodes, tuple):
            episodes, weights = episodes
            if errors_out is None:
                errors_out = []
        else:
            weights = None
        """

        if errors_out is None:
            errors_out_step = None
        else:
            del errors_out[:]
            for _ in episodes:
                errors_out.append(0.0)
            errors_out_step = []

        losses = []
        ys = []
        ts = []
        rs = []
        ixs = []
        for batch, indices in self._step_batch_generator(episodes, weights):
            y, t = self._compute_y_and_t(batch, self.gamma)
            if isinstance(t, chainer.Variable):
                t = t.data
            t = t.reshape((-1,))
            ys.append(y)
            ts.append(t)
            rs.append(batch['reward'])
            ixs.append(indices)

        # self.logger.debug('before trace:%s', ts)
        ts = lambda_return(ts, rs, lambd=self.lambd, gamma=self.gamma)
        # self.logger.debug('after trace:%s', ts)
        for y, t, indices in zip(ys, ts, ixs):
            if errors_out_step is not None:
                del errors_out_step[:]
                delta = F.sum(abs(y - t), axis=1)
                delta = cuda.to_cpu(delta.data)
                for e in delta:
                    errors_out_step.append(e)

            """
            if 'weights' in exp_batch:
                loss_step = compute_weighted_value_loss(
                    y, t, exp_batch['weights'],
                    clip_delta=self.clip_delta,
                    batch_accumulator=self.batch_accumulator)
            else:
                loss_step = compute_value_loss(
                    y, t, clip_delta=self.clip_delta,
                    batch_accumulator=self.batch_accumulator)
            """
            loss_step = compute_value_loss(
                y, t, clip_delta=self.clip_delta,
                batch_accumulator=self.batch_accumulator)
            losses.append(loss_step)
            if errors_out is not None:
                for err, index in zip(errors_out_step, indices):
                    errors_out[index] += err

        loss = F.average(F.stack(losses))

        # Update stats
        self.average_loss *= self.average_loss_decay
        self.average_loss += \
            (1 - self.average_loss_decay) * float(loss.data)

        self.model.cleargrads()
        loss.backward()
        self.optimizer.update()
        """
        if weights is not None:
            self.replay_buffer.update_errors(errors_out)
        """
