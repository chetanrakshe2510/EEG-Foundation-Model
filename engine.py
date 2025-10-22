# engine.py (Updated)
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    balanced_accuracy_score,
)

def train_one_epoch(dataloader, model, loss_fn, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training Epoch", leave=False)

    # ✨ FIX: The third item is now 'correct_labels_batch', a tensor of floats/nans
    for eeg_batch, target_batch, correct_labels_batch in progress_bar:
        eeg_batch = eeg_batch.to(device).float()
        target_batch = target_batch.to(device).float().unsqueeze(1)
        
        preds = model(eeg_batch)
        loss = loss_fn(preds, target_batch)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    if scheduler:
        scheduler.step()
    return total_loss / len(dataloader)


@torch.no_grad()
def valid_model(dataloader, model, device):
    model.eval()
    all_preds, all_targets, all_correct = [], [], []

    progress_bar = tqdm(dataloader, desc="Validating", leave=False)

    # ✨ FIX: The third item is now 'correct_labels_batch'
    for eeg_batch, target_batch, correct_labels_batch in progress_bar:
        eeg_batch = eeg_batch.to(device).float()
        
        preds = model(eeg_batch)
        
        all_preds.append(preds.cpu().numpy())
        all_targets.append(target_batch.numpy())
        
        # correct_labels_batch is already a tensor of numbers, so we just append it
        all_correct.append(correct_labels_batch.numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    all_correct = np.concatenate(all_correct).astype(float)

    # --- Regression Metrics ---
    valid_mask = ~np.isnan(all_targets)
    y_true_reg, y_pred_reg = all_targets[valid_mask], all_preds[valid_mask]
    mae = mean_absolute_error(y_true_reg, y_pred_reg)
    r2 = r2_score(y_true_reg, y_pred_reg)
    rmse = np.sqrt(np.mean((y_true_reg - y_pred_reg)**2))
    
    # --- Classification Metrics ---
    valid_mask_clf = ~np.isnan(all_correct)
    y_true_clf = all_correct[valid_mask_clf].astype(int)
    y_score_clf = -all_preds[valid_mask_clf]
    
    auc_roc, bal_acc = 0.0, 0.0
    if len(np.unique(y_true_clf)) > 1:
        auc_roc = roc_auc_score(y_true_clf, y_score_clf)
        threshold = np.median(y_score_clf)
        y_pred_clf = (y_score_clf > threshold).astype(int)
        bal_acc = balanced_accuracy_score(y_true_clf, y_pred_clf)

    return {'val_rmse': rmse, 'val_mae': mae, 'val_r2': r2, 
            'val_auc_roc': auc_roc, 'val_bal_acc': bal_acc}