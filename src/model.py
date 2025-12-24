import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class DeepRNNLayer(nn.Module):
    """
    A single recurrent unrolling layer.
    Processes the temporal dimension T of an input tensor.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = LSTMCell(input_size, hidden_size)

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        outputs = []
        
        h_t, c_t = state

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t, c_t = self.cell(x_t, (h_t, c_t))
            outputs.append(h_t)

        # Reconstruct temporal manifold: (Batch, Time, Hidden)
        return torch.stack(outputs, dim=1), (h_t, c_t)


class DeepRNN(nn.Module):
    """
    Multi-layer (Stacked) LSTM Architecture.
    Maps a discrete input manifold to a high-dimensional character-probability space.
    """
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int, 
                 output_size: int, 
                 dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # Vertical Stacking of Recurrent Layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input = input_size if i == 0 else hidden_size
            self.layers.append(DeepRNNLayer(layer_input, hidden_size))

        # Stochastic Regularization (Inverted Dropout)
        self.dropout = nn.Dropout(dropout)
        
        # Affine Projection to Vocabulary Space (Logits)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size = x.size(0)
        
        if states is None:
            states = self.init_hidden(batch_size)

        new_states = []
        current_input = x

        for i, layer in enumerate(self.layers):
            # Pass signal through recurrent layer
            current_input, layer_state = layer(current_input, states[i])
            
            # Apply dropout between layers (not on the final output)
            if i < self.num_layers - 1:
                current_input = self.dropout(current_input)
            
            new_states.append(layer_state)

        # Project the top-most hidden manifold to output space
        logits = self.fc(current_input)
        
        return logits, new_states

    def init_hidden(self, batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Initialize zero-state manifolds for all layers."""
        device = next(self.parameters()).device
        return [
            (torch.zeros(batch_size, self.hidden_size, device=device),
             torch.zeros(batch_size, self.hidden_size, device=device))
            for _ in range(self.num_layers)
        ]