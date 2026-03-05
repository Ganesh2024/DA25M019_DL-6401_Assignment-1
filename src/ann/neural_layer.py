"""
Neural Layer Module
Single fully-connected layer with forward pass, backward pass,
and weight/bias gradient storage (self.grad_W, self.grad_b).
"""

import numpy as np
from .activations import Activation


class NeuralLayer:
    """
    A single fully-connected (dense) layer.

    Attributes
    ----------
    W        : np.ndarray, shape (in_features, out_features)
    b        : np.ndarray, shape (1, out_features)
    activation : Activation | None  — None for the output layer (raw logits)
    grad_W   : np.ndarray  — gradient of loss w.r.t. W (set after backward())
    grad_b   : np.ndarray  — gradient of loss w.r.t. b (set after backward())
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str | None = "relu",
        weight_init: str = "xavier",
        weight_decay: float = 0.0,
    ):
        """
        Parameters
        ----------
        in_features  : number of input neurons
        out_features : number of output neurons
        activation   : 'sigmoid' | 'tanh' | 'relu' | None (output layer)
        weight_init  : 'xavier' | 'random'
        weight_decay : L2 regularisation coefficient (lambda)
        """
        self.in_features  = in_features
        self.out_features = out_features
        self.weight_decay = weight_decay

        # Activation (None means linear / output layer)
        if activation is not None:
            self.activation = Activation(activation)
        else:
            self.activation = None

        # Initialise weights & biases
        self.W, self.b = self._init_weights(weight_init)

        # Gradient placeholders — populated by backward()
        self.grad_W: np.ndarray = np.zeros_like(self.W)
        self.grad_b: np.ndarray = np.zeros_like(self.b)

        # Cache for forward-pass values (needed by backward)
        self._input: np.ndarray | None = None   # pre-activation input  (X)
        self._z:     np.ndarray | None = None   # pre-activation output (X @ W + b)
        self._a:     np.ndarray | None = None   # post-activation output

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self, method: str):
        """Return (W, b) according to the chosen initialisation scheme."""
        method = method.lower()
        if method == "xavier":
            # Xavier / Glorot uniform
            limit = np.sqrt(6.0 / (self.in_features + self.out_features))
            W = np.random.uniform(-limit, limit, (self.in_features, self.out_features))
        elif method == "random":
            # Small random normal weights
            W = np.random.randn(self.in_features, self.out_features) * 0.01
        elif method == "zeros":
            # All-zero initialisation (symmetry-breaking experiment)
            W = np.zeros((self.in_features, self.out_features))
        else:
            raise ValueError(
                f"Unknown weight_init '{method}'. Choose 'xavier', 'random', or 'zeros'."
            )
        b = np.zeros((1, self.out_features))
        return W, b

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute layer output.

        Parameters
        ----------
        X : np.ndarray, shape (batch, in_features)

        Returns
        -------
        a : np.ndarray, shape (batch, out_features)
            Post-activation output (or raw logits if activation is None).
        """
        self._input = X                          # cache for backprop
        self._z = X @ self.W + self.b            # linear transform
        if self.activation is not None:
            self._a = self.activation.forward(self._z)
        else:
            self._a = self._z                    # output layer: no activation
        return self._a

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        Backpropagate delta through this layer.

        Parameters
        ----------
        delta : np.ndarray, shape (batch, out_features)
            Gradient of loss w.r.t. this layer's *output* (post-activation).
            For the output layer this is dL/da_out; for hidden layers it
            arrives as the propagated delta from the layer above.

        Returns
        -------
        delta_prev : np.ndarray, shape (batch, in_features)
            Gradient to propagate to the previous layer.

        Side-effects
        ------------
        Sets self.grad_W and self.grad_b.
        """
        batch_size = self._input.shape[0]

        # If this is a hidden layer, multiply by activation derivative
        if self.activation is not None:
            delta = delta * self.activation.backward(self._z)

        # Gradient w.r.t. weights: (in_features, batch) @ (batch, out_features)
        self.grad_W = (self._input.T @ delta) / batch_size
        # L2 regularisation term
        if self.weight_decay > 0.0:
            self.grad_W += self.weight_decay * self.W

        # Gradient w.r.t. biases: mean over batch
        self.grad_b = delta.mean(axis=0, keepdims=True)

        # Gradient to pass to the previous layer
        delta_prev = delta @ self.W.T
        return delta_prev

    #------------------------------------------------------------------
    # Utilities
    #------------------------------------------------------------------

    def __repr__(self):
        act_name = self.activation.name if self.activation else "linear"
        return (
            f"NeuralLayer(in={self.in_features}, out={self.out_features}, "
            f"activation={act_name})"
        )