import torch
import torch.nn as nn
from typing import List


class Solution:
    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        curr_act = x
        dead_fractions = []
        with torch.no_grad():
            for layer in model.children():
                curr_act = layer(curr_act)
                if isinstance(layer, nn.ReLU):
                    flattend = curr_act.view(curr_act.size(0), -1)
                    is_dead_neuron = (flattend == 0.0).all(dim=0)
                    dead_frac = is_dead_neuron.float().mean().item()
                    dead_fractions.append(round(dead_frac, 4))
        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        for i in range(len(dead_fractions)):
            if dead_fractions[i] > 0.5:
                return "use_leaky_relu"

        if dead_fractions[0] > 0.3:
            return "reinitialize"

        strictly_incr = (
            len(dead_fractions) > 1 and
            all(dead_fractions[i] < dead_fractions[i+1] for i in range(len(dead_fractions) - 1))
        )

        if strictly_incr and dead_fractions[-1] > 0.1:
            return 'reduce_learning_rate'

        max_m = max(dead_fractions)
        if float(max_m) < 0.1:
            return "healthy"

        return "healthy"
