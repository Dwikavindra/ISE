import torch
import torch.nn as nn
import torch.optim as optim

class MLPWithLayerNorm(nn.Module):
    def __init__(self, input_dim):
        super(MLPWithLayerNorm, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.norm1 = nn.BatchNorm1d(64)  # LayerNorm after first FC layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        # self.norm2 = nn.LayerNorm(64)  # LayerNorm after second FC layer
        self.fc3 = nn.Linear(64, 2)  # Output layer (2 classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)  # Apply LayerNorm
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.norm2(x)  # Apply LayerNorm again
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
class DeeperMLP(nn.Module):
    def __init__(self, input_dim):
        super(DeeperMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.norm1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.norm2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 32)
        self.norm3 = nn.LayerNorm(32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.norm1(self.fc1(x)))
        x = F.relu(self.norm2(self.fc2(x)))
        x = F.relu(self.norm3(self.fc3(x)))
        x = self.fc4(x)
        return x
