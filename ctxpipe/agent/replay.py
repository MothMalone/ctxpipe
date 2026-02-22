import numpy as np

import deterministic


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.lp_buffer = []

    def _sample_items(self, src, batch_size):
        indices = deterministic.buffer_rng.choice(len(src), batch_size, replace=False)
        return [src[int(i)] for i in indices]

    def add(self, s0, a, r, s1, done, index, fixline_id, ctx):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(
            (
                s0[None, :],
                a,
                r,
                s1[None, :],
                done,
                index,
                fixline_id,
                ctx.detach().numpy(),
            )
        )

    def sample(self, batch_size):
        s0, a, r, s1, done, index, fixline_id, ctx = zip(
            *self._sample_items(self.buffer, batch_size)
        )
        return (
            np.concatenate(s0),
            a,
            r,
            np.concatenate(s1),
            done,
            index,
            fixline_id,
            ctx,
        )

    def lp_add(self, s0, a, r, ctx):
        if len(self.lp_buffer) >= self.capacity:
            self.lp_buffer.pop(0)
        self.lp_buffer.append((s0[None, :], a, r, ctx))

    def lp_sample(self, batch_size):
        s0, a, r, ctx = zip(
            *self._sample_items(self.lp_buffer, batch_size)
        )
        return np.concatenate(s0), a, r, ctx

    def size(self):
        return len(self.buffer)

    def lp_size(self):
        return len(self.lp_buffer)
