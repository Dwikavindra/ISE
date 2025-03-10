import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, input_dim):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)  # Batch Normalization Layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)  # Change output layer to 2 neurons
        self.softmax = nn.Softmax(dim=1)  # Apply Softmax for multi-class classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  # Apply BatchNorm
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)  # Softmax outputs probabilities for 2 classes
