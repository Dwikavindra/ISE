import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from simple_mlp import MLPWithLayerNorm
import TextDataset
import tent
import importlib
import copy

# Reload custom modules
importlib.reload(TextDataset)
importlib.reload(tent)

# Constants
DATASETS = ['caffe', 'incubator-mxnet', 'keras', 'pytorch', 'tensorflow']
RESULTS = []
EPOCHS = 50
BATCH_SIZE = 32
VAL_RATIO = 0.2
INPUT_DIM = 1000

# Helper: train model
def train_model(train_loader):

    model = MLPWithLayerNorm(input_dim=INPUT_DIM)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return model

# Helper: evaluate model
def evaluate_model(model, data_loader, base, testset, iteration):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_preds, all_probs, all_targets = [], [], []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X=batch_X.to(device)
            batch_y=batch_y.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='binary')
    recall = recall_score(all_targets, all_preds, average='binary')
    f1 = f1_score(all_targets, all_preds, average='binary')
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = None

    name = f"{base}->{testset}"
    return {
        "iteration":iteration,
        "name": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }


