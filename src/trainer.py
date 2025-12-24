import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class DeepRNNTrainer:

    def __init__(self, 
                 model: nn.Module, 
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module, 
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
    def train_epoch(self, data_loader: torch.utils.data.DataLoader) -> float:
        """
        Executes one full traversal of the data manifold.
        
        Note: We utilize 'Inverted Dropout' behavior via self.model.train() 
        to ensure the expected value of activations remains consistent.
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            # Hardware Projection
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            #  Embedding Projection (One-Hot Manifold)
            # We project indices into a sparse high-dimensional space
            inputs_one_hot = F.one_hot(inputs, num_classes=self.model.input_size).float()

            self.optimizer.zero_grad(set_to_none=True)

            # Forward Propagation
            # We compute predictions across the stacked LSTM layers
            outputs, _ = self.model(inputs_one_hot)

            # Tensor Reshaping (Temporal Flattening)
            # Map (Batch, Time, Classes) -> (Batch * Time, Classes) for CCE compatibility
            outputs_flat = outputs.reshape(-1, self.model.output_size)
            targets_flat = targets.reshape(-1)

            loss = self.criterion(outputs_flat, targets_flat)

            loss.backward()

            # Gradient Norm Regularization (The "Safety Governor")
            # Prevents the "Exploding Gradient" phenomenon in deep unrolled graphs
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)

        return total_loss / total_samples

    def generate_text(self, 
                     start_text: str, 
                     length: int = 100, 
                     temperature: float = 1.0,
                     vocab: Optional[object] = None) -> str:
        """
        An Autoregressive Stochastic Decoder using Boltzmann Sampling.
        
        Args:
            temperature: Scales the entropy of the prediction distribution.
                         Lower = Deterministic; Higher = Chaotic.
        """
        self.model.eval() # Disable Dropout and freeze BatchNorm
        generated = start_text
        
        # Access the vocabulary mappings from the model's attached dataset
        # In a production OOP setting, the vocab is a first-class citizen
        char_to_idx = vocab.char_to_idx
        idx_to_char = vocab.idx_to_char

        with torch.no_grad(): # Disable Gradient Tape to conserve VRAM
            # Encode seed text
            input_indices = [char_to_idx[ch] for ch in start_text]
            input_seq = torch.tensor(input_indices).unsqueeze(0).to(self.device)
            
            # Initialize the state of the Hidden Manifold
            hidden_state = self.model.init_hidden(1)
            
            for _ in range(length):
                # Prepare the current observation
                x_t = F.one_hot(input_seq[:, -1:], num_classes=self.model.input_size).float()
                
                # Forward Pass: Update Hidden State & Emit Logits
                output, hidden_state = self.model(x_t, hidden_state)
                
                # Energy-based Sampling (Temperature Scaling)
                logits = output[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
                
                # Multinomial Sampling (Monte Carlo approach)
                next_idx = np.random.choice(len(probs), p=probs)
                next_char = idx_to_char[next_idx]
                
                generated += next_char
                
                # Feedback loop: Append predicted token to the sequence
                next_idx_tensor = torch.tensor([[next_idx]]).to(self.device)
                input_seq = torch.cat([input_seq, next_idx_tensor], dim=1)
                
                # Prune context window if using a fixed sequence length constraint
                if input_seq.size(1) > self.model.layers[0].input_size: # Simplified check
                    input_seq = input_seq[:, 1:]
                    
        return generated