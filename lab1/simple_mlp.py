import torch
import torch.nn as nn
import torch.optim as optim

class MLPWithLayerNorm(nn.Module):
    def __init__(self, input_dim):
        super(MLPWithLayerNorm, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.norm1 = nn.LayerNorm(64)  # LayerNorm after first FC layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.norm2 = nn.LayerNorm(64)  # LayerNorm after second FC layer
        self.fc3 = nn.Linear(64, 2)  # Output layer (2 classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)  # Apply LayerNorm
        x = self.relu(x)
        x = self.fc2(x)
        x = self.norm2(x)  # Apply LayerNorm again
        x = self.relu(x)
        x = self.fc3(x)
        return x