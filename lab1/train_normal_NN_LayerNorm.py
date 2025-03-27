
import pandas as pd
import numpy as np
import os
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import torch
import torch.nn as nn
import simple_mlp
import TextDataset
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
import helper
from torch.utils.data import random_split, DataLoader

def check_and_clear_existing_file(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        print(f"⚠️  The file '{file_path}' already exists and is not empty.")
        response = input("Do you want to delete and start over? Type 'yes' to confirm: ").strip().lower()
        if response != 'yes':
            print("❌ Operation cancelled by user. No changes made.")
            exit(1)
        else:
            os.remove(file_path)
            print(f"✅ File '{file_path}' has been deleted. Starting fresh.")
def write_row_to_csv(file_path, columns, values):

    assert len(columns) == len(values), "Columns and values must be the same length."
    folder = os.path.dirname(file_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    df = pd.DataFrame([dict(zip(columns, values))])
    df.to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))


for i in range(20):
    print(f"In iteration {i+1}")
    full_dataset = TextDataset.TextDatasetTFIDF('datasets/tensorflow.csv')

    # Step 2: Define train/val split sizes
    val_ratio = 0.2
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    # Step 3: Split the dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Step 4: Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = simple_mlp.MLPWithLayerNorm(input_dim=1000).to('cpu') 
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=1e-2) 
    model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs=100* len(train_loader)
# ========== Training Loop ==========

    for epoch in range(epochs):
        model.to(device)
        model.train()
        total_train_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X=batch_X.to(device)
            batch_y=batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        print(f"Epoch {epoch} until {epochs}")
    
        
    # "iteration":iteration,
    # "name": name,
    # "accuracy": accuracy,
    # "precision": precision,
    # "recall": recall,
    # "f1": f1,
    # "auc": auc
    data=helper.evaluate_model(model,val_loader,"tensorflow","tensorflow",i+1)
    file_path=f'base_line_tf/base_data_tensorflow.csv'
    if(i==0):
        check_and_clear_existing_file(file_path)
    write_row_to_csv(
                file_path=file_path,
                columns=['iteration', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'],
                values=[i+1, data["accuracy"],data["precision"],data["recall"],data["f1"], data["auc"]]
            )
    model_file_path=f'models/baseline_model_tensorflow_{i+1}_iteration.pt'
    torch.save(model, model_file_path)
    print(f"Finish iteration {i+1} model saved in {model_file_path}")
