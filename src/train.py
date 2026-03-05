"""
Main Training Script
Entry point for training neural networks with command-line arguments.
Logs metrics to Weights & Biases and saves the best model checkpoint.
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
    Parse command-line arguments for training.
    All flags are aligned with the assignment specification.
    """
    parser = argparse.ArgumentParser(
        description="Train a fully-connected neural network on MNIST / Fashion-MNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    parser.add_argument(
        "-d", "--dataset",
        type=str, default="mnist", choices=["mnist", "fashion"],
        help="Dataset to use: 'mnist' or 'fashion'.",
    )

    # Training schedule
    parser.add_argument(
        "-e", "--epochs",
        type=int, default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int, default=32,
        help="Mini-batch size.",
    )

    # Loss function
    parser.add_argument(
        "-l", "--loss",
        type=str, default="cross_entropy",
        choices=["mean_squared_error", "cross_entropy"],
        help="Loss function.",
    )

    # Optimizer
    parser.add_argument(
        "-o", "--optimizer",
        type=str, default="adam",
        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
        help="Gradient-descent optimizer.",
    )

    # Learning rate & regularisation
    parser.add_argument(
        "-lr", "--learning_rate",
        type=float, default=0.001,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "-wd", "--weight_decay",
        type=float, default=0.0,
        help="L2 regularisation coefficient (weight decay).",
    )

    # Architecture
    parser.add_argument(
        "-nhl", "--num_layers",
        type=int, default=3,
        help="Number of hidden layers.",
    )
    parser.add_argument(
        "-sz", "--hidden_size",
        type=int, nargs="+", default=[128],
        help="Number of neurons in each hidden layer (list or single value).",
    )
    parser.add_argument(
        "-a", "--activation",
        type=str, default="relu",
        choices=["sigmoid", "tanh", "relu"],
        help="Activation function for hidden layers.",
    )

    # Weight initialisation
    parser.add_argument(
        "-wi", "--weight_init",
        type=str, default="xavier",
        choices=["random", "xavier"],
        help="Weight initialisation strategy.",
    )

    # W&B
    parser.add_argument(
        "--wandb_project",
        type=str, default="DA6401-Assignment1",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str, default=None,
        help="W&B entity (team or username). Leave blank for default.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str, default=None,
        help="Optional display name for this W&B run.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )

    # Model output
    parser.add_argument(
        "--model_save_path",
        type=str, default="best_model.npy",
        help="Relative path to save the best model weights (.npy).",
    )
    parser.add_argument(
        "--config_save_path",
        type=str, default="best_config.json",
        help="Relative path to save the best model config (.json).",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    args = parse_arguments()
    # Flatten hidden_size list to single int (use first value)
    if isinstance(args.hidden_size, list):
        args.hidden_size = args.hidden_size[0]

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(args.dataset)

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    model = NeuralNetwork(args)
    print(model)

    # ------------------------------------------------------------------
    # 3. Set up W&B (optional)
    # ------------------------------------------------------------------
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project = args.wandb_project,
                entity  = args.wandb_entity,
                name    = args.wandb_run_name,
                config  = vars(args),
            )
        except ImportError:
            print("[WARNING] wandb not installed — skipping W&B logging.")

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    history = model.train(
        X_train    = X_train,
        y_train    = y_train,
        X_val      = X_val,
        y_val      = y_val,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        wandb_run  = wandb_run,
    )

    # ------------------------------------------------------------------
    # 5. Evaluate on test set
    # ------------------------------------------------------------------
    test_metrics = model.evaluate(X_test, y_test)
    print("\n── Test Set Results ──────────────────────────────")
    for k, v in test_metrics.items():
        if k != "logits":
            print(f"  {k:12s}: {v:.4f}")
    print("──────────────────────────────────────────────────\n")

    if wandb_run is not None:
        wandb_run.log({
            "test_loss":      test_metrics["loss"],
            "test_accuracy":  test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall":    test_metrics["recall"],
            "test_f1":        test_metrics["f1"],
        })

    # ------------------------------------------------------------------
    # 6. Save model + config
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.model_save_path) or ".", exist_ok=True)

    model.save(args.model_save_path)

    config = vars(args).copy()
    config["test_f1"]       = test_metrics["f1"]
    config["test_accuracy"] = test_metrics["accuracy"]
    with open(args.config_save_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved → {args.config_save_path}")

    if wandb_run is not None:
        wandb_run.finish()

    print("Training complete!")
    return history, test_metrics


if __name__ == "__main__":
    main()