import torch
import torch.nn as nn

class AntiCFTextClassifier(nn.Module):
    def __init__(self, input_dim):
        super(AntiCFTextClassifier, self).__init__()
        
        # Frozen source model (backbone)
        self.source_fc1 = nn.Linear(input_dim, 128)
        self.source_bn1 = nn.BatchNorm1d(128)
        self.source_fc2 = nn.Linear(128, 2)

        # Adapter (trainable)
        self.adapter_fc1 = nn.Linear(input_dim, 128)
        self.adapter_bn1 = nn.BatchNorm1d(128)
        self.adapter_fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # Source model (frozen)
        with torch.no_grad():
            source_x = self.source_fc1(x)
            source_x = self.source_bn1(source_x)
            source_x = torch.relu(source_x)
            source_logits = self.source_fc2(source_x)  # No softmax — return logits

        # Adapter model (trainable)
        adapter_x = self.adapter_fc1(x)
        adapter_x = self.adapter_bn1(adapter_x)
        adapter_x = torch.relu(adapter_x)
        adapter_logits = self.adapter_fc2(adapter_x)  # No softmax — return logits

        return source_logits, adapter_logits  # Both are raw logits
