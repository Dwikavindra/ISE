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
    model.eval()
    all_preds, all_probs, all_targets = [], [], []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
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

# Run automation
# for base_dataset in DATASETS:
#     all_base_results = []
#     for _ in range(50):
#         model, val_loader = train_model(base_dataset)
#         result = evaluate_model(model, val_loader, base_dataset, base_dataset)
#         all_base_results.append(result)

#         # Save base model state
#         torch.save(model.state_dict(), f'models/{base_dataset}.pt')

#         for test_dataset in DATASETS:
#             if test_dataset == base_dataset:
#                 continue

#             test_data = TextDataset.TextDatasetTFIDF(f'datasets/{test_dataset}.csv')
#             test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

#             result_infer = evaluate_model(model, test_loader, base_dataset, test_dataset)
#             RESULTS.append(result_infer)

#             # Tent evaluation
#             model_state = copy.deepcopy(model.state_dict())
#             for name, param in model.named_parameters():
#                 if "norm" not in name:
#                     param.requires_grad = False

#             tent.Tent(model, test_data)
#             result_tented = evaluate_model(model, test_loader, base_dataset, test_dataset, tented=True)
#             RESULTS.append(result_tented)
#             tent.reset(model, model_state)

#     # Add base dataset internal validation
#     RESULTS.extend(all_base_results)

# Save to CSV
# df_results = pd.DataFrame(RESULTS)
# os.makedirs("results", exist_ok=True)
# df_results.to_csv("results/evaluation_results.csv", index=False)

