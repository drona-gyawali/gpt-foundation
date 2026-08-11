import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def splitter(self, positive:List[str], negative: List[str]):
        pos = [(tex.split()) for tex in positive]
        neg = [(tex.split()) for tex in negative]
        return pos, neg

    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)

        pos, neg = self.splitter(positive, negative)
        all_senctence = pos + neg
        unique_word = sorted(list(set(((word for sentence in all_senctence for word in sentence)))))
        vocab_cache = {v:float(k+1) for k, v in enumerate(unique_word)}

        encodec_tensors = []
        for sentence in all_senctence:
            encodec_seq = []
            for word in sentence:
                encodec_seq.append(vocab_cache[word])
            encodec_tensors.append(torch.tensor(encodec_seq, dtype=torch.float32))
        return nn.utils.rnn.pad_sequence(encodec_tensors, batch_first=True, padding_value=0.0)





