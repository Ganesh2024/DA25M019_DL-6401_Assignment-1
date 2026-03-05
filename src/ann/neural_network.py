"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import numpy as np
from .neural_layer import NeuralLayer
from .objective_functions import ObjectiveFunction
from .optimizers import get_optimizer


class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.

    Architecture
    ------------
    Input → [Hidden Layer × num_hidden_layers] → Output Layer (linear logits)

    The output layer has no activation; a softmax is applied only during loss
    computation (inside ObjectiveFunction) and inference.
    """

    def __init__(self, cli_args):
        """
        Build the network from parsed CLI arguments.

        Expected attributes on cli_args
        --------------------------------
        dataset        : 'mnist' | 'fashion'
        num_layers     : int — number of hidden layers
        hidden_size    : int — neurons per hidden layer
        activation     : str — 'relu' | 'sigmoid' | 'tanh'
        loss           : str — 'cross_entropy' | 'mean_squared_error'
        optimizer      : str — 'sgd' | 'momentum' | 'nag' | 'rmsprop' | 'adam' | 'nadam'
        learning_rate  : float
        weight_decay   : float
        weight_init    : str — 'xavier' | 'random'
        epochs         : int
        batch_size     : int
        """
        self.args = cli_args

        # Dataset constants
        input_dim  = 784   # 28×28 flattened
        output_dim = 10    # 10 classes

        # self.num_hidden  = cli_args.num_layers
        # self.hidden_size = cli_args.hidden_size
        
        # Support both --num_layers and --num_hidden_layers attribute names
        self.num_hidden  = getattr(cli_args, 'num_layers', None) or getattr(cli_args, 'num_hidden_layers', 3)
        self.hidden_size = getattr(cli_args, 'hidden_size', 128)
        if isinstance(self.hidden_size, list):
            self.hidden_size = self.hidden_size[0]

        self.activation  = cli_args.activation
        self.weight_init = cli_args.weight_init
        self.weight_decay = getattr(cli_args, "weight_decay", 0.0)

        # Build layer stack
        self.layers: list[NeuralLayer] = []
        prev_dim = input_dim
        for _ in range(self.num_hidden):
            self.layers.append(
                NeuralLayer(
                    in_features  = prev_dim,
                    out_features = self.hidden_size,
                    activation   = self.activation,
                    weight_init  = self.weight_init,
                    weight_decay = self.weight_decay,
                )
            )
            prev_dim = self.hidden_size

        # Output layer — no activation (logits)
        self.layers.append(
            NeuralLayer(
                in_features  = prev_dim,
                out_features = output_dim,
                activation   = None,
                weight_init  = self.weight_init,
                weight_decay = self.weight_decay,
            )
        )

        # Loss function
        self.loss_fn = ObjectiveFunction(cli_args.loss)

        # Optimizer
        self.optimizer = get_optimizer(
            name         = cli_args.optimizer,
            lr           = cli_args.learning_rate,
            weight_decay = self.weight_decay,
        )

        # Gradient arrays (populated by backward())
        self.grad_W: np.ndarray | None = None
        self.grad_b: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied).

        Parameters
        ----------
        X : np.ndarray, shape (batch, 784)

        Returns
        -------
        logits : np.ndarray, shape (batch, 10)
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out  # raw logits

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Backward propagation to compute gradients.

        Parameters
        ----------
        y_true : np.ndarray, shape (batch,) — integer class labels
        y_pred : np.ndarray, shape (batch, 10) — logits from forward()

        Returns
        -------
        grad_W : np.ndarray (object array, length = num_layers)
            grad_W[0]  = gradient for the LAST (output) layer weights
            grad_W[-1] = gradient for the FIRST (input) layer weights
        grad_b : np.ndarray (object array, same indexing)
        """
        grad_W_list = []
        grad_b_list = []

        # Gradient of loss w.r.t. logits
        delta = self.loss_fn.backward(y_pred, y_true)

        # Backprop through layers in reverse; collect grads so index 0 = last layer
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        # Store as explicit object arrays to avoid numpy broadcasting issues
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    # ------------------------------------------------------------------
    # Weight update
    # ------------------------------------------------------------------

    def update_weights(self):
        """Apply the optimizer update rule to all layers."""
        self.optimizer.update(self.layers)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray | None = None,
        y_val:   np.ndarray | None = None,
        epochs:    int = 1,
        batch_size: int = 32,
        wandb_run=None,
    ):
        """
        Mini-batch training loop.

        Parameters
        ----------
        X_train, y_train : training data
        X_val,   y_val   : optional validation data for logging
        epochs           : number of full passes over training data
        batch_size       : mini-batch size
        wandb_run        : optional W&B run object for metric logging

        Returns
        -------
        history : dict with lists 'train_loss', 'train_acc', 'val_loss', 'val_acc'
        """
        n = X_train.shape[0]
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            # Shuffle training data
            perm = np.random.permutation(n)
            X_shuf, y_shuf = X_train[perm], y_train[perm]

            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, n, batch_size):
                X_batch = X_shuf[start : start + batch_size]
                y_batch = y_shuf[start : start + batch_size]

                # Forward
                logits = self.forward(X_batch)

                # Loss
                batch_loss = self.loss_fn.forward(logits, y_batch)
                epoch_loss += batch_loss
                num_batches += 1

                # Backward
                self.backward(y_batch, logits)

                # Update
                self.update_weights()

            avg_loss = epoch_loss / num_batches
            train_metrics = self.evaluate(X_train, y_train)
            history["train_loss"].append(avg_loss)
            history["train_acc"].append(train_metrics["accuracy"])

            log_dict = {
                "epoch":      epoch,
                "train_loss": avg_loss,
                "train_acc":  train_metrics["accuracy"],
            }

            if X_val is not None and y_val is not None:
                val_metrics = self.evaluate(X_val, y_val)
                history["val_loss"].append(val_metrics["loss"])
                history["val_acc"].append(val_metrics["accuracy"])
                log_dict["val_loss"] = val_metrics["loss"]
                log_dict["val_acc"]  = val_metrics["accuracy"]
                print(
                    f"Epoch {epoch}/{epochs}  "
                    f"train_loss={avg_loss:.4f}  train_acc={train_metrics['accuracy']:.4f}  "
                    f"val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['accuracy']:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch}/{epochs}  "
                    f"train_loss={avg_loss:.4f}  train_acc={train_metrics['accuracy']:.4f}"
                )

            if wandb_run is not None:
                wandb_run.log(log_dict)

        return history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Compute loss, accuracy, precision, recall, and F1 on a dataset.

        Returns
        -------
        dict with keys: 'loss', 'accuracy', 'precision', 'recall', 'f1', 'logits'
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score
        )

        logits    = self.forward(X)
        loss      = self.loss_fn.forward(logits, y)
        y_pred    = np.argmax(logits, axis=1)
        y_true    = y.astype(int)

        accuracy  = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)

        return {
            "loss":      float(loss),
            "accuracy":  float(accuracy),
            "precision": float(precision),
            "recall":    float(recall),
            "f1":        float(f1),
            "logits":    logits,
        }

    # ------------------------------------------------------------------
    # Weight serialisation
    # ------------------------------------------------------------------

    def get_weights(self) -> dict:
        """Return a dictionary of all layer weights and biases."""
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict: dict):
        """Load weights from a dictionary (e.g. loaded from .npy file)."""
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()

    def save(self, path: str):
        """Serialise all weights to a .npy file."""
        np.save(path, self.get_weights())
        print(f"Model saved → {path}")

    def load(self, path: str):
        """Load weights from a .npy file."""
        weight_dict = np.load(path, allow_pickle=True).item()
        self.set_weights(weight_dict)
        print(f"Model loaded ← {path}")

    def __repr__(self):
        lines = ["NeuralNetwork("]
        for i, layer in enumerate(self.layers):
            lines.append(f"  ({i}): {layer}")
        lines.append(")")
        return "\n".join(lines)