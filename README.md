# Pacman Neural Network Classifier

A neural network built from scratch using NumPy to classify Pacman movement directions. No ML frameworks — forward pass, backpropagation, and weight updates are all implemented manually.

## Architecture

Single hidden layer feedforward network:

| Layer | Detail |
|-------|--------|
| Input | Game state feature vector (variable size) |
| Hidden | 32 neurons, ReLU activation |
| Output | 4 neurons (North, East, South, West), Softmax |

## Training

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.01 |
| Epochs | 500 |
| L2 weight decay | 0.01 |
| Momentum | 0.9 |

- **L2 regularisation** — penalises large weights during the backward pass to reduce overfitting
- **Momentum** — accumulates a velocity for each weight to smooth updates and help escape local minima
- **Numerically stable Softmax** — subtracts the row maximum before exponentiation to prevent overflow

## Classes

**`NeuralNetwork`** — core implementation
- `forward(X)` — ReLU hidden layer → Softmax output, stores activations for backprop
- `backward(X, y)` — one-hot encodes targets, computes gradients with L2 reg, applies momentum updates
- `fit(X, y)` — runs the training loop
- `predict(X)` — returns the argmax class index

**`Classifier`** — Pacman integration wrapper
- `fit(data, target)` — converts training data to NumPy arrays and trains the network
- `predict(features, legal)` — runs inference and falls back to a random move if the predicted direction is illegal or the model hasn't been trained yet

## Requirements

```bash
pip install numpy
```
