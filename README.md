<!-- Badges -->
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License" />
  <img src="https://img.shields.io/badge/NumPy-1.26+-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/Made%20with-❤️-red?style=for-the-badge" alt="Made with Love" />
</p>

---

<h1 align="center">🧠 NumPy Neural Network: Solving XOR from Scratch</h1>

<p align="center">
  <strong>A feedforward neural network implemented entirely from scratch using only NumPy — no TensorFlow, no PyTorch, no Keras. Just pure math and matrix operations.</strong>
</p>

<p align="center">
  <em>Built as part of academic coursework to demonstrate a deep understanding of how neural networks actually work under the hood.</em>
</p>

---

## 📋 Table of Contents

- [🔍 Project Overview](#-project-overview)
- [✨ Features](#-features)
- [🏗️ Neural Network Architecture](#️-neural-network-architecture)
- [📐 Mathematical Explanation](#-mathematical-explanation)
- [🛠️ Technologies Used](#️-technologies-used)
- [📦 Installation](#-installation)
- [🚀 Usage](#-usage)
- [📊 Expected Output](#-expected-output)
- [📈 Sample Results](#-sample-results)
- [📁 Project Structure](#-project-structure)
- [🔮 Future Improvements](#-future-improvements)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Acknowledgements](#-acknowledgements)
- [👤 Author](#-author)

---

## 🔍 Project Overview

This project implements a **feedforward neural network entirely from scratch** using only **NumPy** to solve the classic **XOR (Exclusive OR) problem**.

### Why XOR?

The XOR problem is one of the most famous challenges in the history of neural networks. Unlike simpler logic gates such as AND and OR, XOR is **non-linearly separable** — meaning no single straight line (or hyperplane) can divide the input space into the correct output classes.

| x₁ | x₂ | XOR Output |
|:--:|:--:|:----------:|
| 0  | 0  |     0      |
| 0  | 1  |     1      |
| 1  | 0  |     1      |
| 1  | 1  |     0      |

If you plot these points on a 2D plane, you'll see that no single line can separate the `0` outputs from the `1` outputs. This is exactly why the XOR problem **requires at least one hidden layer** — it was this very limitation that led to the "AI Winter" of the 1970s when Minsky and Papert showed that single-layer perceptrons could not solve XOR.

This project proves that a simple **2 → 4 → 1** network with sigmoid activation and backpropagation can learn XOR perfectly — using nothing but NumPy.

> **No deep learning frameworks.** No TensorFlow. No PyTorch. No Keras. Just raw matrix multiplication, calculus, and gradient descent.

---

## ✨ Features

- ✅ **Pure NumPy implementation** — zero dependency on deep learning frameworks
- ✅ **Complete forward propagation** with sigmoid activation
- ✅ **Full backpropagation** with analytical gradient computation
- ✅ **Mean Squared Error (MSE)** loss function
- ✅ **Reproducible results** via fixed random seed (`seed=42`)
- ✅ **Training visualization** — loss curve plotted with Matplotlib
- ✅ **Decision boundary visualization** for intuitive understanding
- ✅ **Clean, modular code** with detailed inline comments
- ✅ **Solves XOR with 100% accuracy** after training

---

## 🏗️ Neural Network Architecture

The network uses a **2 → 4 → 1** fully connected architecture:

```
         ┌─────────────┐
         │ INPUT LAYER  │
         │  (2 neurons) │
         └──────┬───────┘
           x₁   │   x₂
            ╲    │    ╱
             ╲   │   ╱
    ┌─────────────────────────┐
    │      HIDDEN LAYER       │
    │      (4 neurons)        │
    │   σ(z) = sigmoid        │
    │                         │
    │   h₁   h₂   h₃   h₄   │
    └────────────┬────────────┘
             ╲   │   ╱
              ╲  │  ╱
         ┌───────────────┐
         │ OUTPUT LAYER   │
         │  (1 neuron)    │
         │  σ(z) = sigmoid│
         │                │
         │      ŷ         │
         └────────────────┘
```

| Layer         | Neurons | Activation | Parameters            |
|:--------------|:-------:|:----------:|:----------------------|
| Input         |    2    |     —      | —                     |
| Hidden        |    4    |  Sigmoid   | W₁ (2×4), b₁ (1×4)   |
| Output        |    1    |  Sigmoid   | W₂ (4×1), b₂ (1×1)   |
| **Total**     |  **7**  |     —      | **17 parameters**     |

---

## 📐 Mathematical Explanation

### 1. Forward Propagation

The input signal flows forward through the network in two steps:

**Hidden Layer:**

```
z₁ = X · W₁ + b₁
a₁ = σ(z₁)
```

**Output Layer:**

```
z₂ = a₁ · W₂ + b₂
ŷ  = σ(z₂)
```

Where:
- `X` is the input matrix of shape `(4, 2)`
- `W₁` is the weight matrix of shape `(2, 4)`
- `b₁` is the bias vector of shape `(1, 4)`
- `W₂` is the weight matrix of shape `(4, 1)`
- `b₂` is the bias scalar of shape `(1, 1)`

### 2. Sigmoid Activation Function

The sigmoid function squashes any real-valued number into the range `(0, 1)`:

```
σ(x) = 1 / (1 + e⁻ˣ)
```

Its derivative, which is critical for backpropagation, has an elegant form:

```
σ'(x) = σ(x) · (1 - σ(x))
```

### 3. Loss Function — Mean Squared Error (MSE)

The loss quantifies how far the predictions are from the true labels:

```
MSE = (1/n) · Σᵢ (yᵢ - ŷᵢ)²
```

Where:
- `n` is the number of samples (4 for XOR)
- `yᵢ` is the true label
- `ŷᵢ` is the predicted output

### 4. Backpropagation — Gradient Computation

Backpropagation computes the gradient of the loss with respect to each weight by applying the **chain rule** of calculus in reverse through the network.

**Output Layer Gradients:**

```
δ₂ = (ŷ - y) · σ'(z₂)
∂L/∂W₂ = a₁ᵀ · δ₂
∂L/∂b₂ = Σ δ₂
```

**Hidden Layer Gradients:**

```
δ₁ = (δ₂ · W₂ᵀ) · σ'(z₁)
∂L/∂W₁ = Xᵀ · δ₁
∂L/∂b₁ = Σ δ₁
```

### 5. Weight Update — Gradient Descent

All weights and biases are updated simultaneously using the learning rate `η = 0.1`:

```
W = W - η · (∂L/∂W)
b = b - η · (∂L/∂b)
```

This process repeats for **10,000 epochs** until the network converges.

---

## 🛠️ Technologies Used

| Technology   | Version | Purpose                        |
|:-------------|:-------:|:-------------------------------|
| Python       |  3.12   | Core programming language      |
| NumPy        | 1.26+   | Matrix operations & math       |
| Matplotlib   | 3.8+    | Loss curve & decision boundary |

---

## 📦 Installation

### Prerequisites

- Python 3.12 or higher
- pip (Python package manager)
- Git

### Step-by-Step Setup

**1. Clone the repository:**

```bash
git clone https://github.com/yashyadav2012/numpy-neural-network-xor.git
cd numpy-neural-network-xor
```

**2. Create a virtual environment (recommended):**

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Train the Neural Network

```bash
python train.py
```

This will:
1. Initialize the network with random weights (seed=42)
2. Train for 10,000 epochs with learning rate 0.1
3. Print training progress every 1,000 epochs
4. Display final predictions for all XOR inputs

### Train with Visualization

```bash
python train.py --plot
```

This will additionally generate and display:
- **Loss curve** — shows how MSE decreases over training epochs
- **Decision boundary** — visualizes how the network partitions the input space

---

## 📊 Expected Output

When you run `python train.py`, you should see output similar to:

```
============================================================
   NumPy Neural Network — XOR Solver
   Architecture: 2 → 4 → 1 | Activation: Sigmoid
============================================================

Training started...

Epoch     0 | Loss: 0.274192
Epoch  1000 | Loss: 0.240498
Epoch  2000 | Loss: 0.147498
Epoch  3000 | Loss: 0.016498
Epoch  4000 | Loss: 0.006498
Epoch  5000 | Loss: 0.003798
Epoch  6000 | Loss: 0.002598
Epoch  7000 | Loss: 0.001951
Epoch  8000 | Loss: 0.001548
Epoch  9000 | Loss: 0.001274

============================================================
   Training Complete!
============================================================

Final Predictions:
------------------------------------------------------------
  Input           Target    Predicted    Rounded
------------------------------------------------------------
  [0, 0]    →      0        0.0264        0
  [0, 1]    →      1        0.9735        1
  [1, 0]    →      1        0.9733        1
  [1, 1]    →      0        0.0312        0
------------------------------------------------------------

✓ All predictions correct! XOR solved successfully.
```

> **Note:** Exact values may vary slightly due to floating-point arithmetic across different platforms, but the rounded predictions will always be correct after convergence.

---

## 📈 Sample Results

### Model Convergence

The model successfully converges within 10,000 epochs:

- **Initial Loss:** ~0.27 (random weights)
- **Final Loss:** ~0.001 (fully trained)
- **Accuracy:** 100% on all 4 XOR inputs

### Results Interpretation

| Input (x₁, x₂) | True XOR | Raw Prediction | Rounded | Correct? |
|:----------------:|:--------:|:--------------:|:-------:|:--------:|
| (0, 0)           |    0     |    ~0.03       |    0    |    ✅    |
| (0, 1)           |    1     |    ~0.97       |    1    |    ✅    |
| (1, 0)           |    1     |    ~0.97       |    1    |    ✅    |
| (1, 1)           |    0     |    ~0.03       |    0    |    ✅    |

The network learns to produce outputs very close to `0` or `1`, demonstrating strong confidence in its classifications. The hidden layer successfully transforms the non-linearly separable XOR space into a linearly separable representation — exactly what theory predicts.

---

## 📁 Project Structure

```
numpy-neural-network-xor/
│
├── train.py                 # Main training script (entry point)
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation (this file)
├── LICENSE                  # MIT License
├── CONTRIBUTING.md          # Contribution guidelines
├── CODE_OF_CONDUCT.md       # Code of conduct
├── .gitignore               # Git ignore rules
│
├── src/                     # Source package
│   ├── __init__.py          # Package initialiser
│   ├── model.py             # NeuralNetwork class (forward + backprop)
│   └── utils.py             # Pretty-printing & visualisation helpers
│
├── notebooks/               # Jupyter notebooks
│   └── xor_neural_network.ipynb
│
└── figures/                 # Generated visualizations (after --plot)
    ├── loss_curve.png       # Training loss over epochs
    └── decision_boundary.png# Network decision boundary
```

---

## 🔮 Future Improvements

Here are planned enhancements to extend the project:

- [ ] **Additional activation functions** — ReLU, Tanh, Leaky ReLU, Softmax
- [ ] **Configurable architecture** — support for multiple hidden layers and variable neuron counts
- [ ] **Other logic gates** — AND, OR, NAND, NOR, XNOR
- [ ] **Learning rate scheduling** — step decay, exponential decay, cosine annealing
- [ ] **Batch normalization** — improve training stability
- [ ] **Regularization** — L1 (Lasso) and L2 (Ridge) penalty terms
- [ ] **Interactive visualization** — real-time training dashboard
- [ ] **Unit tests** — comprehensive test suite with `pytest`
- [ ] **CLI interface** — configurable hyperparameters via command-line arguments
- [ ] **Export/import weights** — save and load trained models

---

## 🤝 Contributing

Contributions are welcome and appreciated! Whether it's a bug fix, a new feature, or an improvement to documentation — all contributions make a difference.

Please read the [Contributing Guidelines](CONTRIBUTING.md) before getting started.

**Quick start:**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

You are free to use, modify, and distribute this project for personal, academic, or commercial purposes.

---


---

<p align="center">
  <em>If you found this project helpful or interesting, consider giving it a ⭐ on GitHub!</em>
</p>
