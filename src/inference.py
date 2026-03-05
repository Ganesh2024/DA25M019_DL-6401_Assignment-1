"""
Inference Script
Load serialised NumPy weights, rebuild the model, and evaluate on the test set.
Outputs: Accuracy, Precision, Recall, F1-score (and optionally logs to W&B).
"""

import argparse
import json
import os
import sys

import numpy as np

# Ensure src/ is on the path when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ann import NeuralNetwork
from utils.data_loader import load_data


# -----------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------

def parse_arguments():
    """
    Parse command-line arguments for inference.

    Arguments
    ---------
    --model_path   : path to saved .npy weights (relative)
    --config_path  : optional path to config .json; if provided, other arch
                     flags can be omitted
    --dataset      : 'mnist' | 'fashion'
    --batch_size   : batch size for forward passes
    --num_layers   : number of hidden layers
    --hidden_size  : neurons per hidden layer
    --activation   : 'relu' | 'sigmoid' | 'tanh'
    --weight_init  : 'xavier' | 'random'  (needed only to build model skeleton)
    --loss         : 'cross_entropy' | 'mean_squared_error'
    --optimizer    : any valid optimizer name (not used during inference)
    --learning_rate: float (not used during inference, required by NeuralNetwork)
    --weight_decay : float
    --use_wandb    : flag to enable W&B logging
    --wandb_project: W&B project name
    """
    parser = argparse.ArgumentParser(
        description="Run inference on a saved neural network model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model & config paths
    parser.add_argument(
        "--model_path",
        type=str, default="../models/best_model.npy",
        help="Relative path to saved model weights (.npy).",
    )
    parser.add_argument(
        "--config_path",
        type=str, default=None,
        help="Optional path to config .json to auto-fill architecture flags.",
    )

    # Dataset
    parser.add_argument(
        "-d", "--dataset",
        type=str, default="mnist", choices=["mnist", "fashion"],
        help="Dataset to evaluate on.",
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int, default=256,
        help="Batch size for inference.",
    )

    # Architecture (can be overridden by --config_path)
    parser.add_argument(
        "-nhl", "--num_layers",
        type=int, default=3,
        help="Number of hidden layers.",
    )
    parser.add_argument(
        "-sz", "--hidden_size",
        type=int, default=128,
        help="Neurons per hidden layer.",
    )
    parser.add_argument(
        "-a", "--activation",
        type=str, default="relu",
        choices=["sigmoid", "tanh", "relu"],
        help="Activation function.",
    )
    parser.add_argument(
        "-wi", "--weight_init",
        type=str, default="xavier",
        choices=["random", "xavier"],
        help="Weight init method (used to build model skeleton only).",
    )
    parser.add_argument(
        "-l", "--loss",
        type=str, default="cross_entropy",
        choices=["mean_squared_error", "cross_entropy"],
        help="Loss function.",
    )
    parser.add_argument(
        "-o", "--optimizer",
        type=str, default="adam",
        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
        help="Optimizer name (not applied during inference).",
    )
    parser.add_argument(
        "-lr", "--learning_rate",
        type=float, default=0.001,
        help="Learning rate (not applied during inference).",
    )
    parser.add_argument(
        "-wd", "--weight_decay",
        type=float, default=0.0,
        help="Weight decay coefficient.",
    )

    # W&B
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Log inference results to W&B.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str, default="DA6401-Assignment1",
        help="W&B project name.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------
# Helper: overlay config.json values onto args
# -----------------------------------------------------------------------

def _apply_config(args, config_path: str):
    """Merge saved config JSON into args namespace (does not overwrite CLI values)."""
    with open(config_path, "r") as f:
        cfg = json.load(f)
    # Only set architecture/dataset keys that the user did NOT explicitly pass
    for key in ("dataset", "num_layers", "hidden_size", "activation",
                "weight_init", "loss", "optimizer", "learning_rate", "weight_decay"):
        if key in cfg:
            setattr(args, key, cfg[key])
    return args


# -----------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------

def load_model(model_path: str) -> NeuralNetwork:
    """
    Load trained model from disk.
    Reads config JSON from the same directory as the .npy file.
    """
    model_dir = os.path.dirname(os.path.abspath(model_path))
    config_path = os.path.join(model_dir, 'best_config.json')

    with open(config_path) as f:
        cfg = json.load(f)

    args = argparse.Namespace(**{k: v for k, v in cfg.items()
                                 if k in ('dataset', 'epochs', 'batch_size',
                                           'learning_rate', 'optimizer', 'num_layers',
                                           'hidden_size', 'activation', 'weight_init',
                                           'weight_decay', 'loss')})
    model = NeuralNetwork(args)
    model.load(model_path)
    return model


# -----------------------------------------------------------------------
# Evaluate model
# -----------------------------------------------------------------------

def evaluate_model(model: NeuralNetwork, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate model on test data.

    Returns
    -------
    dict with keys: 'logits', 'loss', 'accuracy', 'precision', 'recall', 'f1'
    """
    metrics = model.evaluate(X_test, y_test)
    return metrics


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> dict:
    """
    Main inference function.

    Returns
    -------
    dict with keys: logits, loss, accuracy, precision, recall, f1
    """
    args = parse_arguments()

    # Apply config file if provided
    if args.config_path is not None:
        args = _apply_config(args, args.config_path)

    # ------------------------------------------------------------------
    # 1. Load dataset (test split only)
    # ------------------------------------------------------------------
    _, _, (X_test, y_test) = load_data(args.dataset)

    # ------------------------------------------------------------------
    # 2. Build model and load weights
    # ------------------------------------------------------------------
    model = load_model(args.model_path)

    # ------------------------------------------------------------------
    # 3. Evaluate
    # ------------------------------------------------------------------
    metrics = evaluate_model(model, X_test, y_test)

    # ------------------------------------------------------------------
    # 4. Print results
    # ------------------------------------------------------------------
    print("\n── Inference Results ─────────────────────────────")
    print(f"  Dataset   : {args.dataset}")
    print(f"  Model     : {args.model_path}")
    print(f"  Loss      : {metrics['loss']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print("──────────────────────────────────────────────────\n")

    # ------------------------------------------------------------------
    # 5. Optional W&B logging
    # ------------------------------------------------------------------
    if args.use_wandb:
        try:
            import wandb
            run = wandb.init(
                project = args.wandb_project,
                job_type = "inference",
                config   = vars(args),
            )
            run.log({
                "test_loss":      metrics["loss"],
                "test_accuracy":  metrics["accuracy"],
                "test_precision": metrics["precision"],
                "test_recall":    metrics["recall"],
                "test_f1":        metrics["f1"],
            })
            run.finish()
        except ImportError:
            print("[WARNING] wandb not installed — skipping W&B logging.")

    print("Evaluation complete!")
    return metrics


if __name__ == "__main__":
    main()