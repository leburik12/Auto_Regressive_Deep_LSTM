# Stochastic-RNN-Core: Layered Autoregressive LSTM

### Abstract
This repository implements a vertically-stacked, autoregressive Long Short-Term Memory (LSTM) network from first principles. By manually engineering the gate manifolds—Input, Forget, and Output—this architecture provides a transparent lens into the mitigation of gradient decay in deep temporal graphs. The system is optimized for character-level language modeling via high-entropy stochastic inference.

---

## 1. Mathematical Architecture
The core of the system is the **Constant Error Carousel (CEC)**. Unlike vanilla RNNs, this model utilizes an additive state update mechanism to preserve the gradient signal $\nabla$ across extended temporal horizons $T$.

The state transition manifold is defined by the following gate logic:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$g_t = \tanh(W_g \cdot [h_{t-1}, x_t] + b_g)$$
$$c_t = f_t \odot c_{t-1} + i_t \odot g_t$$
$$h_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \odot \tanh(c_t)$$

### Structural Depth
The model implements **Vertical Stacking**, where the hidden state $h_t^{(l)}$ of layer $l$ serves as the input $x_t^{(l+1)}$ for the subsequent layer. This allows for hierarchical feature extraction, from character-level syntax to semantic structures.

---

## 2. Project Topology
The codebase follows a modular Separation of Concerns (SoC) paradigm:

* **`src/layers.py`**: Primitive LSTM cell gate logic and weight manifold initialization.
* **`src/model.py`**: Orchestrator for the multi-layer recurrent stack.
* **`src/trainer.py`**: Gradient descent engine featuring norm clipping and Boltzmann sampling.
* **`src/data_utils.py`**: Text-to-tensor pipeline and symbolic mapping.
* **`main.py`**: Experimental entry point for hyperparameter configuration.

---

## 3. Implementation & Execution
### Prerequisites
* **Linux Environment** (Ubuntu 22.04 LTS recommended)
* **NVIDIA Driver 535+** (LTS branch)
* **PyTorch 2.1+** with CUDA support

### Installation
```bash
# Clone the manifold
git clone [https://github.com/your-username/Stochastic-RNN-Core.git](https://github.com/your-username/Stochastic-RNN-Core.git)
cd Stochastic-RNN-Core

# Initialize environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
