class LSTMCell(nn.Module):
    """Basic LSTM Cell from scratch."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Concatenated weights for the four gates: [i, f, g, o]
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size) * 0.01)
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size) * 0.01)
        self.bias_ih = nn.Parameter(torch.zeros(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(4 * hidden_size))

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single timestep.

        Args:
            x: Input tensor of shape (batch_size, input_size)
            state: Tuple (h_prev, c_prev) where each is (batch_size, hidden_size)

        Returns:
            h_new: New hidden state (batch_size, hidden_size)
            (h_new, c_new): New state tuple for next timestep
        """
        h_prev, c_prev = state

        # Compute all gates in one matrix multiplication (efficient)
        gates = (torch.matmul(x, self.weight_ih.t()) + self.bias_ih) + \
                (torch.matmul(h_prev, self.weight_hh.t()) + self.bias_hh)

        # Split into four gates
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=1)

        i = torch.sigmoid(input_gate)    # How much new info to add
        f = torch.sigmoid(forget_gate)   # How much old cell to keep
        g = torch.tanh(cell_gate)        # Candidate new cell values
        o = torch.sigmoid(output_gate)   # How much cell to expose as hidden

        c_new = f * c_prev + i * g        # Core: additive update to cell
        h_new = o * torch.tanh(c_new)     # Hidden is gated view of cell

        return h_new, c_new

    def init_state(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and cell states with zeros."""
        device = self.weight_ih.device
        return (torch.zeros(batch_size, self.hidden_size, device=device),
                torch.zeros(batch_size, self.hidden_size, device=device))