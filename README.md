# DA6401 Assignment 1 — Neural Network from Scratch

A fully NumPy-based feedforward neural network trained on **MNIST** and **Fashion-MNIST**, with experiment tracking via **Weights & Biases**.

---

## Project Structure

```
.
├── models/                    # Saved weights (.npy) and configs (.json)
│   └── .gitkeep
├── notebooks/
│   └── wandb_demo.ipynb       # W&B logging demos (sweeps, plots, analysis)
├── src/
│   ├── ann/
│   │   ├── __init__.py
│   │   ├── activations.py        # Sigmoid, Tanh, ReLU
│   │   ├── neural_layer.py       # Single fully-connected layer
│   │   ├── neural_network.py     # Full model (forward/backward/train/evaluate)
│   │   ├── objective_functions.py# Cross-Entropy, MSE
│   │   └── optimizers.py         # SGD, Momentum, NAG, RMSProp, Adam, Nadam
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_loader.py        # MNIST / Fashion-MNIST loading & preprocessing
│   ├── inference.py              # Evaluate saved model
│   └── train.py                  # Train model from CLI
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
  --wandb_project DA6401-Assignment1 \
  --model_save_path models/best_model.npy \
  --config_save_path models/best_config.json
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
| `-sz` | `--hidden_size` | `128` | Neurons per hidden layer |
| `-a` | `--activation` | `relu` | `relu`, `sigmoid`, `tanh` |
| `-wi` | `--weight_init` | `xavier` | `xavier` or `random` |

---

## Inference

```bash
cd src

python inference.py \
  --model_path models/best_model.npy \
  --config_path models/best_config.json \
  -d mnist
```

Outputs **Accuracy**, **Precision**, **Recall**, and **F1-score** to stdout.

---

## Design Notes

### Forward Pass
Each `NeuralLayer.forward(X)` computes `z = X @ W + b` and applies the chosen activation. The output layer uses no activation (raw logits). Softmax is applied inside the loss function.

### Backward Pass
`NeuralNetwork.backward()` chains `NeuralLayer.backward()` calls in reverse. Each layer stores `self.grad_W` and `self.grad_b` after every call — accessible by the autograder.

### Gradient Layout
`grad_W[0]` / `grad_b[0]` → **last (output) layer**  
`grad_W[-1]` / `grad_b[-1]` → **first hidden layer**

### Numerical Gradient Check
The backward pass is compatible with a finite-difference check at tolerance `1e-7`.

---

## W&B Report

See the public W&B report for:
- Class distribution table (Section 2.1)
- Parallel Coordinates sweep plot (Section 2.2)
- Optimizer convergence comparison (Section 2.3)
- Gradient norm plots — Sigmoid vs ReLU (Section 2.4)
- Dead neuron analysis (Section 2.5)
- MSE vs Cross-Entropy training curves (Section 2.6)
- Train vs Test accuracy overlay (Section 2.7)
- Confusion matrix + error analysis (Section 2.8)
- Weight initialisation symmetry experiment (Section 2.9)
- Fashion-MNIST transfer challenge (Section 2.10)