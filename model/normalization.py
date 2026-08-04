import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # x: 1D feature vector
        # gamma: 1D scale parameter (same length as x)
        # beta: 1D shift parameter (same length as x)
        # eps = 1e-5
        # Normalize: x_hat = (x - mean) / sqrt(var + eps)
        # Scale and shift: out = gamma * x_hat + beta
        # return np.round(your_answer, 5)
        mean = np.mean(x)
        sqrt_diff_storage = []
        for i in range(len(x)):
            sqrt_diff = (x[i] - mean) ** 2.0
            sqrt_diff_storage.append(sqrt_diff)
        std = np.sqrt(np.mean(sqrt_diff_storage))
        xhat = []
        for j in range(len(x)):
            xhat.append((x[j] - mean) / np.sqrt((std) ** 2 + 1e-5 ))
        out = gamma * xhat + beta
        return np.round(out, 5)



