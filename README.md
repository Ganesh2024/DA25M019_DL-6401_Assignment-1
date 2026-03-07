# DA6401 Assignment 1 — Neural Network from Scratch

**Student:** Ganesh Mula  
**Roll Number:** DA25M019  

A fully NumPy-based feedforward neural network trained on **MNIST** and **Fashion-MNIST**, with experiment tracking via **Weights & Biases**.

---

## 🔗 W&B Report

**[View Full Experiment Report on Weights & Biases](https://api.wandb.ai/links/ganeshmula02-indian-institute-of-technology-madras/yydi7753)**

The report covers:
- 2.1 Data Exploration and Class Distribution
- 2.2 Hyperparameter Sweep (100+ runs)
- 2.3 Optimizer Showdown (SGD, Momentum, NAG, RMSProp, Adam, Nadam)
- 2.4 Vanishing Gradient Analysis (Sigmoid vs ReLU)
- 2.5 Dead Neuron Investigation
- 2.6 Loss Function Comparison (MSE vs Cross-Entropy)
- 2.7 Global Performance Analysis
- 2.8 Error Analysis (Confusion Matrix)
- 2.9 Weight Initialization & Symmetry Breaking
- 2.10 Fashion-MNIST Transfer Challenge

---

## Project Structure

```
Assignment-1/
├── models/
│   ├── best_config.json          # Best model configuration
│   ├── best_model.npy            # Best model weights
│   ├── confusion_matrix.png      # Confusion matrix plot
│   ├── global_performance.png    # Train vs test accuracy overlay
│   └── misclassified.png         # Misclassified samples grid
├── notebooks/
│   └── wandb_demo.ipynb          # W&B experiment notebooks (all 10 sections)
├── src/
│   ├── ann/
│   │   ├── __init__.py
│   │   ├── activations.py        # Sigmoid, Tanh, ReLU (forward + backward)
│   │   ├── neural_layer.py       # Single fully-connected layer with grad storage
│   │   ├── neural_network.py     # Full model (forward/backward/train/evaluate)
│   │   ├── objective_functions.py# Cross-Entropy, MSE (forward + backward)
│   │   └── optimizers.py         # SGD, Momentum, NAG, RMSProp, Adam, Nadam
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_loader.py        # MNIST / Fashion-MNIST loading & preprocessing
│   ├── best_config.json          # Saved config for autograder
│   ├── best_model.npy            # Saved weights for autograder
│   ├── inference.py              # Evaluate saved model, outputs metrics
│   └── train.py                  # Train model from CLI with full argparse
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

Run from inside the `src/` folder:

```bash
cd src

python train.py \
  -d mnist \
  -e 10 \
  -b 64 \
  -l cross_entropy \
  -o adam \
  -lr 0.001 \
  -wd 0.0 \
  -nhl 3 \
  -sz 128 \
  -a relu \
  -wi xavier \
  --use_wandb \
  --wandb_project DA6401-Assignment1
```

### CLI Arguments

| Flag | Long form | Default | Description |
|------|-----------|---------|-------------|
| `-d` | `--dataset` | `mnist` | `mnist` or `fashion` |
| `-e` | `--epochs` | `10` | Number of training epochs |
| `-b` | `--batch_size` | `32` | Mini-batch size |
| `-l` | `--loss` | `cross_entropy` | `cross_entropy` or `mean_squared_error` |
| `-o` | `--optimizer` | `adam` | `sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam` |
| `-lr` | `--learning_rate` | `0.001` | Initial learning rate |
| `-wd` | `--weight_decay` | `0.0` | L2 regularisation coefficient |
| `-nhl` | `--num_layers` | `3` | Number of hidden layers |
| `-sz` | `--hidden_size` | `128` | Neurons per hidden layer (accepts list) |
| `-a` | `--activation` | `relu` | `relu`, `sigmoid`, `tanh` |
| `-wi` | `--weight_init` | `xavier` | `xavier` or `random` |

---

## Inference

Run from inside the `src/` folder:

```bash
cd src

python inference.py \
  --model_path best_model.npy \
  --config_path best_config.json \
  -d mnist
```

Outputs **Accuracy**, **Precision**, **Recall**, and **F1-score** to stdout.

---

## Design Notes

### Architecture
- Input layer: 784 neurons (28×28 flattened)
- Hidden layers: configurable count and size
- Output layer: 10 neurons (linear logits, no activation)
- Softmax applied inside loss function only

### Forward Pass
Each `NeuralLayer.forward(X)` computes `z = X @ W + b` and applies the chosen activation. The output layer returns raw logits.

### Backward Pass
`NeuralNetwork.backward()` chains `NeuralLayer.backward()` calls in reverse order. Each layer exposes `self.grad_W` and `self.grad_b` after every call for autograder verification.

### Gradient Layout
`grad_W[0]` / `grad_b[0]` → **last (output) layer**  
`grad_W[-1]` / `grad_b[-1]` → **first hidden layer**

### Weight Initialisation
- `xavier`: Glorot uniform, limit = √(6 / (fan_in + fan_out))
- `random`: Small normal weights × 0.01
- `zeros`: All zeros (symmetry experiment only)

### Optimizers
All optimizers maintain internal state (velocities, moments) and are initialised lazily on first update call.

---

## Best Model Performance

| Metric | Value |
|--------|-------|
| Dataset | MNIST |
| Architecture | 3 hidden layers × 128 neurons, ReLU |
| Optimizer | Adam (lr=0.001) |
| Batch Size | 64 |
| Epochs | 10 |
| Weight Init | Xavier |
| Test Accuracy | **98.04%** |
| Test F1-Score | **98.03%** |
