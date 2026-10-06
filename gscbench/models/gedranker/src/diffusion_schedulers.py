import math

import numpy as np
import torch


class CategoricalDiffusion:
    def __init__(self, steps: int) -> None:
        self.steps = int(steps)
        beta_start = 1.0e-4
        beta_end = 2.0e-2
        self.beta = np.linspace(beta_start, beta_end, self.steps)

        beta = self.beta.reshape((-1, 1, 1))
        eye = np.eye(2).reshape((1, 2, 2))
        ones = np.ones((2, 2)).reshape((1, 2, 2))

        self.q = (1.0 - beta) * eye + (beta / 2.0) * ones
        q_bar = [np.eye(2)]
        for matrix in self.q:
            q_bar.append(q_bar[-1] @ matrix)
        self.q_bar = np.stack(q_bar, axis=0)

    def sample(self, x0_onehot: torch.Tensor, timestep: np.ndarray, mapping_batch: torch.Tensor) -> torch.Tensor:
        q_bar = torch.from_numpy(self.q_bar[timestep]).float().to(x0_onehot.device).reshape(timestep.shape[0], 2, 2)
        q_bar = q_bar[mapping_batch]
        xt = torch.matmul(x0_onehot, q_bar)
        return torch.bernoulli(xt[..., 1].clamp(0.0, 1.0))


class InferenceSchedule:
    def __init__(self, *, steps: int = 1000, inference_steps: int = 1000) -> None:
        self.steps = int(steps)
        self.inference_steps = int(inference_steps)

    def __call__(self, step: int) -> tuple[int, int]:
        if step < 0 or step >= self.inference_steps:
            raise ValueError("inference step out of range")
        t1 = self.steps - int((float(step) / self.inference_steps) * self.steps)
        t1 = int(np.clip(t1, 1, self.steps))
        t2 = self.steps - int((float(step + 1) / self.inference_steps) * self.steps)
        t2 = int(np.clip(t2, 0, self.steps - 1))
        return t1, t2
