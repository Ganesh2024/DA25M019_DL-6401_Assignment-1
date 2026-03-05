"""
Activation Functions Module
Provides forward activations and their derivatives for backpropagation.
Supported: sigmoid, tanh, relu
"""

import numpy as np


class Activation:
    """
    Encapsulates a named activation function and its derivative.

    Usage:
        act = Activation('relu')
        z   = act.forward(pre_activation)   # forward pass
        dz  = act.backward(pre_activation)  # element-wise derivative
    """

    SUPPORTED = ("sigmoid", "tanh", "relu")

    def __init__(self, name: str):
        name = name.lower()
        if name not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported activation '{name}'. Choose from {self.SUPPORTED}."
            )
        self.name = name

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, z: np.ndarray) -> np.ndarray:
        """Apply activation function element-wise."""
        if self.name == "sigmoid":
            return self._sigmoid(z)
        elif self.name == "tanh":
            return np.tanh(z)
        elif self.name == "relu":
            return np.maximum(0.0, z)

    # ------------------------------------------------------------------
    # Derivative (for backprop) — d(activation)/d(z)
    # ------------------------------------------------------------------

    def backward(self, z: np.ndarray) -> np.ndarray:
        """
        Compute element-wise derivative of activation w.r.t. pre-activation z.
        Used during backpropagation.
        """
        if self.name == "sigmoid":
            s = self._sigmoid(z)
            return s * (1.0 - s)
        elif self.name == "tanh":
            return 1.0 - np.tanh(z) ** 2
        elif self.name == "relu":
            return (z > 0).astype(float)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        # For positive z use standard formula; for negative use exp(z)/(1+exp(z))
        pos_mask = z >= 0
        result = np.empty_like(z, dtype=float)
        result[pos_mask]  = 1.0 / (1.0 + np.exp(-z[pos_mask]))
        exp_z = np.exp(z[~pos_mask])
        result[~pos_mask] = exp_z / (1.0 + exp_z)
        return result

    def __repr__(self):
        return f"Activation('{self.name}')"