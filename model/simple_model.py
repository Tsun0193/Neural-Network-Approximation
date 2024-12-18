import torch
import torch.nn as nn

class ReLU_Network(nn.Module):
    def __init__(self, input_dim, hidden_layers, hidden_units, output_dim=1):
        super(ReLU_Network, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_units))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_units, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)