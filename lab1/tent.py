import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def reset(model, model_state):
    model.load_state_dict(model_state)

def entropy(tensor):
    return -(F.softmax(tensor, dim=-1) * F.log_softmax(tensor, dim=-1)).sum(dim=-1)
def configureTent(model):
    for name, param in model.named_parameters():
        if "norm" not in name:
            param.requires_grad = False
    return model

def Tent(model, eval_loader,origin_dataset_name,target_dataset_name,iteration,):
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=0.3
    )
   

    all_predictions = []
    all_labels = []
    all_probs = []

    for batch, label_batch in tqdm(eval_loader, desc="Adapting with Tent"):
        batch, label_batch = batch.to(device), label_batch.to(device)

        outputs = model(batch)  # Raw logits
        probs = F.softmax(outputs, dim=1)

        loss = entropy(outputs).mean()
        loss.backward()
        optimizer.step()
        model.zero_grad()

        predictions = torch.argmax(probs, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(label_batch.cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())  # For binary AUC

    # === Metrics ===
    acc = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='binary')
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = None

    print(f"\n📊 Test-Time Adaptation (Tent) Metrics:")
    print(f"  Accuracy : {acc * 100:.2f}%")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    if auc is not None:
        print(f"  ROC AUC  : {auc:.4f}")
    else:
        print("  ROC AUC  : Not computed (requires binary classification and both classes present)")

    return {
        "iteration":iteration,
        "name": f"{origin_dataset_name}->{target_dataset_name}",
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    
    }
