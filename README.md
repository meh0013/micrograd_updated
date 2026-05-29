# Neural Network Engine

A scalar-valued autograd engine and multilayer perceptron (MLP) framework built from scratch in Python — no PyTorch, no TensorFlow. Implements reverse-mode automatic differentiation over a dynamically constructed DAG, with a full training pipeline applied to real multiclass classification.

Inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd), extended with vectorized training utilities, multiple gradient descent strategies, softmax + NLL loss, and early stopping.

---

## What's Inside

### `Value` — Scalar Autograd Engine
The core of the project. Every number is wrapped in a `Value` object that tracks its computation history. Calling `.backward()` performs a full reverse-mode autodiff pass over the dynamically built DAG using topological sort.

**Supported operations:** `+`, `-`, `*`, `/`, `**`, `exp`, `log`

**Activation functions:** `ReLU`, `Leaky ReLU`, `Sigmoid`, `Tanh`

Each operation defines its own `_backward()` closure — gradients accumulate correctly through the graph via the chain rule.

### `MLP` — Neural Network Library
Built on top of `Value`. Three classes: `Neuron → Layer → MLP`.

- Neuron: computes weighted sum + bias, applies activation
- Layer: collection of neurons, applies ReLU on all hidden layers
- MLP: chains layers, exposes `.parameters()` for the optimizer

### Training Utilities
Three gradient descent strategies, all implemented from scratch:

| Strategy | Implementation |
|---|---|
| Full-batch | Accumulates loss over entire dataset per epoch |
| Stochastic (SGD) | One sample per gradient update |
| Mini-batch | Shuffled batches with gradient clipping (`±1.0`) |

Additional utilities:
- Custom `train_test_split` with shuffle and random seed
- `softmax` with numerical stability (max subtraction before exp)
- `nll_loss` (negative log-likelihood for multiclass classification)
- `accuracy` evaluation on held-out test set
- Early stopping based on test accuracy threshold
- Loss curve plotting via Matplotlib
- Computation graph visualisation via Graphviz (`draw_dot`)

---

## Results on sklearn Digits Dataset

**Dataset:** 1,797 samples, 64 features (8×8 pixel images), 10 classes (digits 0–9)  
**Architecture:** MLP(64 → 16 → 10) — input layer, one hidden layer, softmax output  
**Preprocessing:** StandardScaler normalisation  

| Training Strategy | Epochs to Converge | Total Time | Test Accuracy |
|---|---|---|---|
| Full-batch GD | 50 | ~5000s | 91.6% |
| Mini-batch GD (batch=32) | 3 | ~620s | 94.9% |

Mini-batching achieved **8× faster convergence** and **3.3% higher accuracy** over full-batch gradient descent.

---

## Files

```
Neural-Network-Engine/
├── Micrograd_Practice.ipynb   # Core Value engine + MLP implementation
├── DigitsDataset.ipynb        # Full training pipeline on sklearn Digits
├── LICENSE                    # MIT License
└── LICENSE-Karpathy.txt       # License for micrograd-derived code
```

---

## Key Implementation Details

**Autograd backward pass** uses iterative topological sort to guarantee each node's gradient is fully accumulated before propagating to its children:

```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)
    self.grad = 1.0
    for node in reversed(topo):
        node._backward()
```

**Softmax with numerical stability** (avoids `exp` overflow):

```python
def softmax(logits):
    max_val = max(logits, key=lambda v: v.data)
    shifted = [x - max_val for x in logits]
    exps = [x.exp() for x in shifted]
    sum_exp = sum(exps[1:], exps[0])
    return [e / sum_exp for e in exps]
```

**Gradient clipping** in mini-batch training prevents exploding gradients:

```python
p.grad = max(min(p.grad, 1.0), -1.0)
p.data += -0.01 * p.grad
```

---

## Dependencies

```
numpy
matplotlib
scikit-learn
graphviz
torch  # used only for sanity-checking gradient values
```

---

## Motivation

This project was built to understand what happens *under the hood* in neural network training — specifically how gradients flow backwards through a computation graph, and how training dynamics change across gradient descent strategies. Rather than calling `loss.backward()` on a PyTorch tensor, every piece of the forward and backward pass here is written and understood explicitly.

---

## Acknowledgements

Core autograd architecture inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd). Training pipeline, softmax/NLL loss, mini-batch strategy, and Digits dataset experiments are original additions.
