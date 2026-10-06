import math

import torch
import torch.nn.functional as F


def get_positive_expectation(p_samples, measure, average=True):
    log_2 = math.log(2.0)

    if measure == 'GAN':
        ep = -F.softplus(-p_samples)
    elif measure == 'JSD':
        ep = log_2 - F.softplus(-p_samples)
    elif measure == 'X2':
        ep = p_samples ** 2
    elif measure == 'KL':
        ep = p_samples + 1.0
    elif measure == 'RKL':
        ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        ep = p_samples
    elif measure == 'H2':
        ep = 1.0 - torch.exp(-p_samples)
    elif measure == 'W1':
        ep = p_samples
    else:
        raise NotImplementedError

    if average:
        return ep.mean()
    else:
        return ep
