import numpy as np
import torch


class SpecAugment:
    def __init__(self, T=8, F=8, mT=8, mF=2):
        self.T = T
        self.F = F
        self.mT = mT
        self.mF = mF

    def __call__(self, x):
        width, height = x.shape[-2:]
        mask = torch.ones_like(x, requires_grad=False)

        for _ in range(self.mT):
            t_delta = np.random.randint(low=0, high=self.T)
            t0 = np.random.randint(low=0, high=width - t_delta)
            mask[:, t0:t0 + t_delta, :] = 0

        for _ in range(self.mF):
            f_delta = np.random.randint(low=0, high=self.F)
            f0 = np.random.randint(low=0, high=height - f_delta)
            mask[:, :, f0:f0 + f_delta] = 0

        return x * mask
