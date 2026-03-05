"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

"""
Objective (Loss) Functions Module
Provides forward loss computation and its gradient w.r.t. model output logits.
Supported: cross_entropy (with softmax), mean_squared_error
"""

import numpy as np


class ObjectiveFunction:
    """
    Encapsulates a loss function and the gradient of that loss
    with respect to the *logits* (pre-softmax outputs).

    Usage:
        loss_fn = ObjectiveFunction('cross_entropy')
        loss    = loss_fn.forward(logits, y_true)   # scalar
        grad    = loss_fn.backward(logits, y_true)  # shape (batch, num_classes)
    """

    SUPPORTED = ("cross_entropy", "mean_squared_error")

    def __init__(self, name: str):
        name = name.lower().replace("-", "_")
        if name not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported loss '{name}'. Choose from {self.SUPPORTED}."
            )
        self.name = name

    # ------------------------------------------------------------------
    # Forward — returns scalar average loss
    # ------------------------------------------------------------------

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """
        Parameters
        ----------
        logits : np.ndarray, shape (batch, num_classes)  — raw network output
        y_true : np.ndarray, shape (batch,) int labels OR (batch, num_classes) one-hot

        Returns
        -------
        loss : float  (average over batch)
        """
        y_oh = self._to_one_hot(y_true, logits.shape[1])

        if self.name == "cross_entropy":
            probs = self._softmax(logits)
            # Clip to avoid log(0)
            probs = np.clip(probs, 1e-12, 1.0)
            loss  = -np.sum(y_oh * np.log(probs)) / logits.shape[0]
        else:  # mean_squared_error
            probs = self._softmax(logits)
            loss  = np.mean(np.sum((probs - y_oh) ** 2, axis=1))
        return float(loss)

    # ------------------------------------------------------------------
    # Backward — gradient of loss w.r.t. logits
    # ------------------------------------------------------------------

    def backward(self, logits, y_true):
        batch_size = logits.shape[0]
        y_oh  = self._to_one_hot(y_true, logits.shape[1])
        probs = self._softmax(logits)

        if self.name == "cross_entropy":
            grad = (probs - y_oh) / batch_size   # ← add / batch_size
        else:
            dl_dp = 2.0 * (probs - y_oh)
            dot = np.sum(dl_dp * probs, axis=1, keepdims=True)
            grad = probs * (dl_dp - dot) / batch_size  # ← add / batch_size
        return grad

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        """Numerically stable row-wise softmax."""
        z_shifted = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    @staticmethod
    def _to_one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
        """Convert integer labels to one-hot encoding if necessary."""
        if y.ndim == 2:
            return y  # already one-hot
        one_hot = np.zeros((y.shape[0], num_classes))
        one_hot[np.arange(y.shape[0]), y.astype(int)] = 1.0
        return one_hot

    def __repr__(self):
        return f"ObjectiveFunction('{self.name}')"