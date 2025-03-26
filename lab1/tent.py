import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def reset(model, model_state):
    model.load_state_dict(model_state)

def entropy(tensor):
    return -(F.softmax(tensor, dim=-1) * F.log_softmax(tensor, dim=-1)).sum(dim=-1)
def configureTent(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model

def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm1d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def Tent(model, eval_loader,origin_dataset_name,target_dataset_name,iteration,):
    model.to('cpu')
    model=configureTent(model)
    params, param_names = collect_params(model)
    optimizer  = torch.optim.Adam(params, lr=1e-2)
    all_predictions = []
    all_labels = []
    all_probs = []

    for batch, label_batch in tqdm(eval_loader, desc="Adapting with Tent"):
        for i in range(1):
            batch, label_batch = batch.to('cpu'), label_batch.to('cpu')
            outputs = model(batch)  # Raw logits
            probs = torch.softmax(outputs, dim=1)

            loss = entropy(outputs).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

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


    return {
        "iteration":iteration,
        "name": f"{origin_dataset_name}->{target_dataset_name}",
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    
    }


def configureTentLayerNorm(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
    return model

def collect_params_layer_norm(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.LayerNorm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def TentLayerNorm(model, eval_loader,origin_dataset_name,target_dataset_name,iteration,):
    print("Here")
    model.to('cpu')
    model=configureTentLayerNorm(model)
    params, param_names = collect_params_layer_norm(model)
    optimizer  = torch.optim.Adam(params, lr=1e-2)
    all_predictions = []
    all_labels = []
    all_probs = []

    for batch, label_batch in tqdm(eval_loader, desc="Adapting with Tent"):
        for i in range(1):
            batch, label_batch = batch.to('cpu'), label_batch.to('cpu')
            outputs = model(batch)  # Raw logits
            probs = torch.softmax(outputs, dim=1)

            loss = entropy(outputs).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

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

    return {
        "iteration":iteration,
        "name": f"{origin_dataset_name}->{target_dataset_name}",
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    
    }
