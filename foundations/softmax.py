import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        maxNum = max(z)
        shift = np.subtract(z , maxNum)
        exp_calc = np.exp(shift)
        total_exp_sum = np.sum(exp_calc)
        res = np.divide(exp_calc , total_exp_sum)
        return np.round(res, 4)
        

