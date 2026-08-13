import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        # Round to 5 decimal places.
        PE = np.zeros((seq_len, d_model), dtype=np.float64 )
        pos = np.arange(seq_len)[:, np.newaxis]

        i_2i = np.arange(0, d_model, 2)
        sin_a = np.sin(pos / 10000 ** (i_2i / d_model))
        cos_a = np.cos(pos / 10000 ** (i_2i / d_model))
        PE[:, 0::2] = sin_a
        PE[:, 1::2] = cos_a

        return np.round(PE, 5)

