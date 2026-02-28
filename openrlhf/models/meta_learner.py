from torch import nn
import torch
import numpy as np


class MetaLearner(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=100, output_dim=1, beta=1):
        super(MetaLearner, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self._dtype = self.fc1.weight.dtype
        
        # Initialize weights to make initial output close to beta * input for positive values
        with torch.no_grad():
            self.fc1.weight.fill_(0.1)
            self.fc1.bias.fill_(0.0)
            self.fc2.weight.fill_(0.1)
            self.fc2.bias.fill_(0.0)
    
    def forward(self, x):
        x = x.unsqueeze(-1).to(self.fc1.weight.dtype)
        x = self.fc1(x)
        x = self.tanh(x)  # Add activation between layers
        x = self.sigmoid(self.fc2(x))
        return x.squeeze(-1)
    
    def predict(self, x):
        if isinstance(x, torch.Tensor):
            x = x.clone().detach().float()
        elif isinstance(x, (list, np.ndarray)):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported input type: {type(x)}")
        
        with torch.no_grad():
            return self(x)
    
    def save_weight(self, path):
        """Save both layer weights and biases to a txt file in append mode"""
        with open(path, 'a') as f:
            fc1_weight = self.fc1.weight.detach().cpu().float().numpy()
            fc1_bias = self.fc1.bias.detach().cpu().float().numpy()
            fc2_weight = self.fc2.weight.detach().cpu().float().numpy()
            fc2_bias = self.fc2.bias.detach().cpu().float().numpy()
            f.write(f"{fc1_weight},{fc1_bias},{fc2_weight},{fc2_bias}\n")
