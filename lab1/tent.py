import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

def entropy(tensor):
    return -(F.softmax(tensor, dim=-1) * F.log_softmax(tensor, dim=-1)).sum(dim=-1)

def Tent(args, model, dataset):
    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()  # Allow model to update normalization layers (if any)
    
    all_predictions = []
    all_labels = []
    
    for batch, label_batch in tqdm(eval_loader):
        batch, label_batch = batch.to('mps'), label_batch.to('mps')
        
        # Forward pass
        outputs = model(batch)

        # Compute entropy loss
        loss = entropy(outputs).mean()
        loss.backward()
        optimizer.step()
        model.zero_grad()
        
        # Store predictions
        predictions = torch.argmax(outputs, dim=1)  # Convert logits to class predictions
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(label_batch.cpu().numpy())

    # Compute accuracy
    correct = sum(1 for pred, true in zip(all_predictions, all_labels) if pred == true)
    accuracy = correct / len(all_labels) * 100

    print(f"Test-Time Adaptation (Tent) Accuracy: {accuracy:.2f}%")
    return all_predictions
