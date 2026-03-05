"""
Optimizers Module
Implements parameter update rules for neural network training.
Supported: sgd, momentum, nag, rmsprop, adam, nadam
"""

import numpy as np


# ======================================================================
# Base class
# ======================================================================

class BaseOptimizer:
    """Abstract base for all optimizers."""

    def __init__(self, lr: float = 0.01, weight_decay: float = 0.0):
        self.lr = lr
        self.weight_decay = weight_decay  # L2 coefficient (applied in layer grads)

    def update(self, layers):
        """Update all layer weights given their stored gradients."""
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(lr={self.lr})"


# ======================================================================
# SGD (vanilla)
# ======================================================================

class SGD(BaseOptimizer):
    """Vanilla stochastic gradient descent."""

    def update(self, layers):
        for layer in layers:
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b


# ======================================================================
# Momentum SGD
# ======================================================================

class Momentum(BaseOptimizer):
    """SGD with classical (Polyak) momentum."""

    def __init__(self, lr: float = 0.01, beta: float = 0.9, weight_decay: float = 0.0):
        super().__init__(lr, weight_decay)
        self.beta = beta
        self._vW: list = []
        self._vb: list = []
        self._initialised = False

    def _init_state(self, layers):
        self._vW = [np.zeros_like(l.W) for l in layers]
        self._vb = [np.zeros_like(l.b) for l in layers]
        self._initialised = True

    def update(self, layers):
        if not self._initialised:
            self._init_state(layers)
        for i, layer in enumerate(layers):
            self._vW[i] = self.beta * self._vW[i] + self.lr * layer.grad_W
            self._vb[i] = self.beta * self._vb[i] + self.lr * layer.grad_b
            layer.W -= self._vW[i]
            layer.b -= self._vb[i]


# ======================================================================
# Nesterov Accelerated Gradient (NAG)
# ======================================================================

class NAG(BaseOptimizer):
    """
    Nesterov Accelerated Gradient.
    The look-ahead step is approximated using the previous velocity
    (standard implementation compatible with mini-batch training).
    """

    def __init__(self, lr: float = 0.01, beta: float = 0.9, weight_decay: float = 0.0):
        super().__init__(lr, weight_decay)
        self.beta = beta
        self._vW: list = []
        self._vb: list = []
        self._initialised = False

    def _init_state(self, layers):
        self._vW = [np.zeros_like(l.W) for l in layers]
        self._vb = [np.zeros_like(l.b) for l in layers]
        self._initialised = True

    def update(self, layers):
        if not self._initialised:
            self._init_state(layers)
        for i, layer in enumerate(layers):
            vW_prev = self._vW[i].copy()
            vb_prev = self._vb[i].copy()
            self._vW[i] = self.beta * self._vW[i] + self.lr * layer.grad_W
            self._vb[i] = self.beta * self._vb[i] + self.lr * layer.grad_b
            # Nesterov correction: use updated velocity
            layer.W -= (1 + self.beta) * self._vW[i] - self.beta * vW_prev
            layer.b -= (1 + self.beta) * self._vb[i] - self.beta * vb_prev


# ======================================================================
# RMSProp
# ======================================================================

class RMSProp(BaseOptimizer):
    """RMSProp — adaptive per-parameter learning rates."""

    def __init__(
        self,
        lr: float   = 0.001,
        beta: float = 0.9,
        eps: float  = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(lr, weight_decay)
        self.beta = beta
        self.eps  = eps
        self._sW: list = []
        self._sb: list = []
        self._initialised = False

    def _init_state(self, layers):
        self._sW = [np.zeros_like(l.W) for l in layers]
        self._sb = [np.zeros_like(l.b) for l in layers]
        self._initialised = True

    def update(self, layers):
        if not self._initialised:
            self._init_state(layers)
        for i, layer in enumerate(layers):
            self._sW[i] = self.beta * self._sW[i] + (1 - self.beta) * layer.grad_W ** 2
            self._sb[i] = self.beta * self._sb[i] + (1 - self.beta) * layer.grad_b ** 2
            layer.W -= self.lr * layer.grad_W / (np.sqrt(self._sW[i]) + self.eps)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(self._sb[i]) + self.eps)


# ======================================================================
# Adam
# ======================================================================

class Adam(BaseOptimizer):
    """
    Adam optimiser (Kingma & Ba, 2015).
    Combines momentum (first moment) and RMSProp (second moment) with
    bias correction.
    """

    def __init__(
        self,
        lr:     float = 0.001,
        beta1:  float = 0.9,
        beta2:  float = 0.999,
        eps:    float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(lr, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self._t    = 0
        self._mW: list = []
        self._mb: list = []
        self._vW: list = []
        self._vb: list = []
        self._initialised = False

    def _init_state(self, layers):
        self._mW = [np.zeros_like(l.W) for l in layers]
        self._mb = [np.zeros_like(l.b) for l in layers]
        self._vW = [np.zeros_like(l.W) for l in layers]
        self._vb = [np.zeros_like(l.b) for l in layers]
        self._initialised = True

    def update(self, layers):
        if not self._initialised:
            self._init_state(layers)
        self._t += 1
        lr_t = self.lr * (np.sqrt(1 - self.beta2 ** self._t) / (1 - self.beta1 ** self._t))
        for i, layer in enumerate(layers):
            # First moment (momentum)
            self._mW[i] = self.beta1 * self._mW[i] + (1 - self.beta1) * layer.grad_W
            self._mb[i] = self.beta1 * self._mb[i] + (1 - self.beta1) * layer.grad_b
            # Second moment (squared gradient)
            self._vW[i] = self.beta2 * self._vW[i] + (1 - self.beta2) * layer.grad_W ** 2
            self._vb[i] = self.beta2 * self._vb[i] + (1 - self.beta2) * layer.grad_b ** 2
            # Parameter update
            layer.W -= lr_t * self._mW[i] / (np.sqrt(self._vW[i]) + self.eps)
            layer.b -= lr_t * self._mb[i] / (np.sqrt(self._vb[i]) + self.eps)


# ======================================================================
# Nadam
# ======================================================================

class Nadam(BaseOptimizer):
    """
    Nadam — Adam with Nesterov momentum (Dozat, 2016).
    Uses a look-ahead first moment in the parameter update step.
    """

    def __init__(
        self,
        lr:     float = 0.001,
        beta1:  float = 0.9,
        beta2:  float = 0.999,
        eps:    float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(lr, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self._t    = 0
        self._mW: list = []
        self._mb: list = []
        self._vW: list = []
        self._vb: list = []
        self._initialised = False

    def _init_state(self, layers):
        self._mW = [np.zeros_like(l.W) for l in layers]
        self._mb = [np.zeros_like(l.b) for l in layers]
        self._vW = [np.zeros_like(l.W) for l in layers]
        self._vb = [np.zeros_like(l.b) for l in layers]
        self._initialised = True

    def update(self, layers):
        if not self._initialised:
            self._init_state(layers)
        self._t += 1
        for i, layer in enumerate(layers):
            # First moment
            self._mW[i] = self.beta1 * self._mW[i] + (1 - self.beta1) * layer.grad_W
            self._mb[i] = self.beta1 * self._mb[i] + (1 - self.beta1) * layer.grad_b
            # Second moment
            self._vW[i] = self.beta2 * self._vW[i] + (1 - self.beta2) * layer.grad_W ** 2
            self._vb[i] = self.beta2 * self._vb[i] + (1 - self.beta2) * layer.grad_b ** 2
            # Bias-corrected estimates
            mW_hat = self._mW[i] / (1 - self.beta1 ** self._t)
            mb_hat = self._mb[i] / (1 - self.beta1 ** self._t)
            vW_hat = self._vW[i] / (1 - self.beta2 ** self._t)
            vb_hat = self._vb[i] / (1 - self.beta2 ** self._t)
            # Nesterov look-ahead: blend next step's momentum
            mW_nesterov = self.beta1 * mW_hat + (1 - self.beta1) * layer.grad_W / (1 - self.beta1 ** self._t)
            mb_nesterov = self.beta1 * mb_hat + (1 - self.beta1) * layer.grad_b / (1 - self.beta1 ** self._t)
            layer.W -= self.lr * mW_nesterov / (np.sqrt(vW_hat) + self.eps)
            layer.b -= self.lr * mb_nesterov / (np.sqrt(vb_hat) + self.eps)


# ======================================================================
# Factory
# ======================================================================

def get_optimizer(name: str, lr: float, weight_decay: float = 0.0, **kwargs) -> BaseOptimizer:
    """
    Factory function to instantiate an optimizer by name.

    Parameters
    ----------
    name         : 'sgd' | 'momentum' | 'nag' | 'rmsprop' | 'adam' | 'nadam'
    lr           : learning rate
    weight_decay : L2 regularisation coefficient
    **kwargs     : additional optimizer-specific hyperparameters

    Returns
    -------
    optimizer instance
    """
    name = name.lower()
    mapping = {
        "sgd":      SGD,
        "momentum": Momentum,
        "nag":      NAG,
        "rmsprop":  RMSProp,
        "adam":     Adam,
        "nadam":    Nadam,
    }
    if name not in mapping:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from {list(mapping.keys())}.")
    return mapping[name](lr=lr, weight_decay=weight_decay, **kwargs)