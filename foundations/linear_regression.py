import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        y_pred = X @ weights
        return np.round(y_pred, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        m_diff = model_prediction - ground_truth
        non_negative_val = m_diff ** 2
        sum_val = np.sum(non_negative_val)
        return np.round(np.divide(np.sum(sum_val), len(model_prediction)), 5)
        
