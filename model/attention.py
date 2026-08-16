import torch
import torch.nn as nn
from torchtyping import TensorType


class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        key = self.key(embedded)
        query = self.query(embedded)
        value = self.value(embedded)
        q_dot_kT = query @ key.transpose(-2, -1)
        compute_attents = q_dot_kT / torch.sqrt(torch.tensor(key.shape[-1]))
        seq_len = embedded.shape[-2]
        computed_mask = torch.tril(torch.ones(seq_len,seq_len))
        masked_scores = compute_attents.masked_fill(computed_mask == 0, float('-inf'))
        W = torch.softmax(masked_scores, dim=2)
        output = W @ value
        return torch.round(output, decimals=4)
        

