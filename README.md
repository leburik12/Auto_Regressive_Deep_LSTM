# Hierarchical Autoregressive LSTM: Multi-Layered Temporal Synthesis

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Academic Level: MIT Research](https://img.shields.io/badge/Academic-MIT%20Elite-gold.svg)](#)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 1. Abstract
This project presents a **Deep Autoregressive Recurrent Neural Network** based on a custom-implemented **Long Short-Term Memory (LSTM)** cell. By vertically stacking recurrent layers, the model extracts hierarchical features from sequential data, mapping discrete input manifolds into a high-dimensional probability space for stochastic text synthesis. The implementation emphasizes architectural transparency, utilizing raw tensor operations for gating logic rather than high-level abstractions.

---

## 2. Mathematical Foundation: The Gating Manifold

The core of this project is the `LSTMCell`, which solves the vanishing gradient problem inherent in standard RNNs through an **Additive Cell State Update**. 

### 2.1 The Four-Gate Mechanism
For a given input $x_t$ and previous hidden state $h_{t-1}$, we compute the gates $i$ (input), $f$ (forget), $g$ (cell candidate), and $o$ (output) as follows:

$$
\begin{aligned}
i_t &= \sigma(W_{ih}^{(i)} x_t + b_{ih}^{(i)} + W_{hh}^{(i)} h_{t-1} + b_{hh}^{(i)}) \\
f_t &= \sigma(W_{ih}^{(f)} x_t + b_{ih}^{(f)} + W_{hh}^{(f)} h_{t-1} + b_{hh}^{(f)}) \\
g_t &= \tanh(W_{ih}^{(g)} x_t + b_{ih}^{(g)} + W_{hh}^{(g)} h_{t-1} + b_{hh}^{(g)}) \\
o_t &= \sigma(W_{ih}^{(o)} x_t + b_{ih}^{(o)} + W_{hh}^{(o)} h_{t-1} + b_{hh}^{(o)})
\end{aligned}
$$



### 2.2 State Evolution
The cell state $c_t$ acts as a "long-term memory" carousel, updated by:
$$c_t = f_t \odot c_{t-1} + i_t \odot g_t$$

The hidden state $h_t$ is the gated, non-linear projection of the cell state:
$$h_t = o_t \odot \tanh(c_t)$$

---

## 3. Architecture Specification

### 3.1 Vertical Recurrent Stacking
The `DeepRNN` architecture employs a multi-layered approach. Each layer $L_n$ passes its temporal hidden manifold to $L_{n+1}$:
* **Lower Layers**: Extract granular n-gram structures and local character-level dependencies.
* **Higher Layers**: Synthesize abstract semantic themes and long-range coherence.

### 3.2 Inverted Dropout Regularization
To prevent co-adaptation in the deep layers, we implement **Inverted Dropout**. This scales activations by $1/(1-p)$ during training to ensure that the expected value of the signal remains constant during the evaluation phase without manual scaling.

---

## 4. Inference Engine: Boltzmann Sampling

The `generate_text` method utilizes an **Autoregressive Stochastic Decoder**. Instead of simple greedy search, we use **Temperature-Scaled Multinomial Sampling** (Boltzmann Distribution) to navigate the output logits $z$:

$$P(y_i) = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$$

* **Low $\tau$ ($< 1.0$)**: Sharpens the distribution; model becomes more deterministic.
* **High $\tau$ ($> 1.0$)**: Increases entropy; model explores more chaotic, "creative" branches of the probability manifold.



---

## 5. Technical Implementation Details

### 5.1 Optimization
* **Gradient Norm Clipping**: Applied at a threshold of $1.0$ to stabilize the deep unrolled graph and prevent exploding gradients.
* **One-Hot Manifold Projection**: Inputs are projected into a sparse high-dimensional space before being fed into the primary recurrent layer.
