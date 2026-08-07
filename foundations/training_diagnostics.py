import torch
import torch.nn as nn
from typing import List, Dict


class Solution:
    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []
        curr_act = x

        with torch.no_grad():
            for layer in model.children():
                curr_act = layer(curr_act)

                if isinstance(layer, nn.Linear):
                    mean = torch.mean(curr_act).item()
                    std = torch.std(curr_act).item()
                    is_dead_neuron = (curr_act <= 0.0).all(dim=0)
                    dead_frac = is_dead_neuron.float().mean().item()

                    stats.append(
                        {
                            "mean": round(mean, 4),
                            "std": round(std, 4),
                            "dead_fraction": round(dead_frac, 4),
                        }
                    )
        return stats

    def compute_gradient_stats(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        stats = []
        model.zero_grad()

        curr_act = x
        loss_fn = nn.MSELoss()
        for layer in model.children():
            curr_act = layer(curr_act)
        loss = loss_fn(curr_act, y)
        loss.backward()

        for layer in model.children():
            if isinstance(layer, nn.Linear):
                if layer.weight.grad is not None:
                    mean = torch.mean(layer.weight.grad).item()
                    std = torch.std(layer.weight.grad).item()
                    norm = torch.linalg.norm(layer.weight.grad).item()

                    stats.append(
                        {
                            "mean": round(mean, 4),
                            "std": round(std, 4),
                            "norm": round(norm, 4),
                        }
                    )
        return stats

    def diagnose(
        self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]
    ) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)

        for act in activation_stats:
            if act["dead_fraction"] > 0.5:
                return "dead_neurons"

        for act in gradient_stats:
            if act["norm"] > 1000:
                return "exploding_gradients"

        if gradient_stats and gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        for act in activation_stats:
            if act["std"] < 0.1:
                return "vanishing_gradients"

        for act in activation_stats:
            if act["std"] > 10.0:
                return "exploding_gradients"

        return "healthy"
