import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
device=("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def reset(model, model_state):
    model.load_state_dict(model_state)

def entropy(logits):
    probs = F.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)

def kl(p_logits, q_logits):
    p_log_probs = F.log_softmax(p_logits, dim=-1)
    q_probs = F.softmax(q_logits, dim=-1)
    return F.kl_div(p_log_probs, q_probs, reduction='none').sum(dim=-1)

def AntiCF( model, dataset):
    """
    AntiCF for TF-IDF + classification with accuracy tracking.
    Args:
        - args: contains batch_size, lr, device, alpha
        - model: AntiCFTextClassifier
        - dataset: TensorDataset(X, y)
    """
    eval_loader = DataLoader(dataset, shuffle=True, batch_size=64)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    model.to(device)
    model.train()
    
    all_preds = []
    all_labels = []

    for batch in tqdm(eval_loader, desc="AntiCF Training"):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        source_logits, adapter_logits = model(inputs)

        # Loss
        entropy_loss = entropy(adapter_logits).mean()
        kl_loss = kl(adapter_logits, source_logits).mean()
        loss = entropy_loss * (1 - 0.2) + kl_loss * 0.2

        # Backward pass
        loss.backward()
        optimizer.step()
        model.zero_grad()

        # Predictions
        preds = adapter_logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    # Compute final accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n✅ Final Accuracy: {accuracy:.4f}")
    
    return all_preds, all_labels, accuracy
