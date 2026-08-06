import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(
        self,
        x: List[List[float]],
        gamma: List[float],
        beta: List[float],
        running_mean: List[float],
        running_var: List[float],
        momentum: float,
        eps: float,
        training: bool,
    ) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)
        x = np.array(x)
        if training == True:
            x_avg = np.mean(x, axis=0)
            x_var = np.var(x, mean=x_avg, axis=0)
            x_hat = (x - x_avg) / np.sqrt(x_var + eps)
            out = gamma * x_hat + beta
            running_mean = (1.0 - momentum) * running_mean + momentum * x_avg
            running_var = (1.0 - momentum) * running_var + momentum * x_var
            return (
                np.round(out, 4),
                np.round(running_mean, 4),
                np.round(running_var, 4),
            )
        else:
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)
            out = gamma * x_hat + beta
            return (
                np.round(out, 4),
                np.round(running_mean, 4),
                np.round(running_var, 4),
            )
