
import torch
import numpy as np
import copy
import warnings
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

from braindecode.preprocessing import preprocess, create_windows_from_events
from braindecode.datasets import BaseConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

# Import from our custom modules
from config import get_config
import data_loader
import preprocessing as preproc
import model as model_zoo
import engine
from lr_finder import LRFinder  # Import the new LRFinder class

warnings.filterwarnings("ignore", category=UserWarning, module="pandas.core.computation.expressions")


# =====================================================================
# --- Custom Dataset for Regression ---
# =====================================================================
class RegressionWrapperDataset(Dataset):
    """PyTorch Dataset wrapper for regression with an additional 'correct' label."""
    def __init__(self, braindecode_dataset, target_name='rt_from_stimulus'):
        self.braindecode_dataset = braindecode_dataset
        self.target_name = target_name
        self.metadata = self.braindecode_dataset.get_metadata()

    def __len__(self):
        return len(self.braindecode_dataset)

    def _safe_float_conversion(self, value):
        """Safely converts boolean, None, or number to float."""
        if value is None:
            return np.nan
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        return float(value)

    def __getitem__(self, index):
        X, _, _ = self.braindecode_dataset[index]
        metadata_row = self.metadata.iloc[index]
        y_regression = torch.tensor(metadata_row[self.target_name], dtype=torch.float32)
        correct_value = metadata_row.get('correct', np.nan)
        correct_label = self._safe_float_conversion(correct_value)
        return X, y_regression, correct_label


# =====================================================================
# --- Main Fine-tuning Script ---
# =====================================================================
def main():
    Config = get_config(test_mode=False)
    print(f"Using device: {Config.DEVICE.upper()} ⚙️")

    # Ensure artifacts directory exists
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # STAGE 2: FINE-TUNING ON CCD
    print("\n--- STAGE 2: Fine-tuning on Contrast Change Detection (CCD) ---")
    ccd_dataset = data_loader.load_bids_data(Config.TRAIN_RELEASES, Config.FINETUNE_TASK)
    if not ccd_dataset.datasets:
        print("Error: No data loaded.")
        return

    print("\nApplying preprocessing steps...")
    base_preprocessors = preproc.get_base_preprocessors(Config.SFREQ, Config.L_FREQ, Config.H_FREQ)
    preprocess(ccd_dataset, base_preprocessors, n_jobs=4)

    annotation_preprocessors = [
        preproc.Preprocessor(
            preproc.annotate_trials_with_target, target_field="rt_from_stimulus",
            epoch_length=Config.EPOCH_LEN_SECONDS, apply_on_array=False
        ),
        preproc.Preprocessor(preproc.add_aux_anchors, apply_on_array=False),
    ]
    preprocess(ccd_dataset, annotation_preprocessors, n_jobs=4)
    dataset_with_anchors = preproc.keep_only_recordings_with(Config.ANCHOR_EVENT, ccd_dataset)
    print(f"Found {len(dataset_with_anchors.datasets)} recordings with the anchor event.")

    print("\nCreating windows from events...")
    start_offset_samples = int(Config.TMIN * Config.SFREQ)
    stop_offset_samples = int(Config.TMAX * Config.SFREQ)
    windows_dataset = create_windows_from_events(
        dataset_with_anchors, mapping={Config.ANCHOR_EVENT: 0},
        trial_start_offset_samples=start_offset_samples,
        trial_stop_offset_samples=stop_offset_samples,
        preload=True, verbose=False,
    )
    windows_dataset = preproc.add_extras_columns(
        windows_dataset, dataset_with_anchors, desc="contrast_trial_start"
    )

    print("\nSplitting data into train and validation sets by subject...")
    subjects = windows_dataset.get_metadata()["subject"].unique()
    rng = check_random_state(Config.SEED)
    train_subj, valid_subj = train_test_split(subjects, test_size=Config.VALID_FRAC, random_state=rng)

    subject_wise_windows = windows_dataset.split('subject')
    train_concat_set = BaseConcatDataset([subject_wise_windows[s] for s in train_subj if s in subject_wise_windows])
    valid_concat_set = BaseConcatDataset([subject_wise_windows[s] for s in valid_subj if s in subject_wise_windows])

    train_set = RegressionWrapperDataset(train_concat_set)
    valid_set = RegressionWrapperDataset(valid_concat_set)
    print(f"Data split successful: Train items={len(train_set)}, Valid items={len(valid_set)}")

    train_loader = DataLoader(train_set, batch_size=Config.BATCH_SIZE, shuffle=True,
                              num_workers=Config.NUM_WORKERS, drop_last=True, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=Config.BATCH_SIZE, shuffle=False,
                              num_workers=Config.NUM_WORKERS, pin_memory=True)

    X_shape, _, _ = train_set[0]
    n_chans, n_times = X_shape.shape
    model = model_zoo.create_model(n_chans=n_chans, n_times=n_times, sfreq=Config.SFREQ)

    try:
        print("\nAttempting to load pre-trained backbone weights from 'pretrained_backbone.pth'...")
        model.backbone.load_state_dict(torch.load("pretrained_backbone.pth", map_location=Config.DEVICE))
        print("✅ Pre-trained weights loaded successfully!")
    except FileNotFoundError:
        print("⚠️ WARNING: 'pretrained_backbone.pth' not found. Training backbone from scratch.")

    model.to(Config.DEVICE)
    print(f"\nModel '{Config.MODEL_NAME}' initialized on {Config.DEVICE}.")

    # --- Optional Non-blocking LR Finder ---
    if getattr(Config, "USE_LR_FINDER", False):
        print("\n--- Running Learning Rate Finder (Non-blocking) ---")
        optimizer_lr_find = AdamW(model.parameters(), lr=1e-7)
        loss_fn_for_find = MSELoss()

        lr_finder = LRFinder(model, optimizer_lr_find, loss_fn_for_find, Config.DEVICE)
        lr_finder.range_test(train_loader, start_lr=1e-6, end_lr=1e-1, num_iter=100)
        lr_finder.plot()

        save_path = artifacts_dir / "lr_find_finetune.png"
        plt.savefig(save_path)
        plt.close()
        print(f"📈 LR Finder plot saved to: {save_path}")
        print("You can inspect the plot and update LEARNING_RATE in config.py accordingly.\n")
    else:
        print("⚙️ Skipping LR Finder (Config.USE_LR_FINDER=False).")

    # --- Starting Fine-Tuning Loop ---
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.N_EPOCHS_FINETUNE - 1)
    loss_fn = MSELoss()
    best_composite_score, epochs_no_improve, best_state, best_epoch = -float("inf"), 0, None, None

    print(f"\n--- Starting Fine-Tuning Loop ---")
    for epoch in range(1, Config.N_EPOCHS_FINETUNE + 1):
        train_loss = engine.train_one_epoch(train_loader, model, loss_fn, optimizer, scheduler, Config.DEVICE)
        val_metrics = engine.valid_model(valid_loader, model, Config.DEVICE)

        score_mae = 1 / (1 + val_metrics['val_mae'])
        score_r2 = val_metrics['val_r2']
        composite_score = (0.67 * score_mae) + (0.33 * score_r2)

        print(f"\n--- Epoch {epoch}/{Config.N_EPOCHS_FINETUNE} Summary ---")
        print(f"Train Loss: {train_loss:.4f} | Val Composite Score: {composite_score:.4f}")
        print(f"Val MAE: {val_metrics['val_mae']:.4f} | Val R²: {val_metrics['val_r2']:.4f}")
        print(f"Val AUC-ROC: {val_metrics['val_auc_roc']:.4f} | Valid Bal Acc: {val_metrics['val_bal_acc']:.4f}\n")

        if composite_score > best_composite_score + Config.MIN_DELTA:
            best_composite_score = composite_score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
            print(f"✨ New best validation score: {best_composite_score:.4f}. Saving model state.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= Config.EARLY_STOPPING_PATIENCE:
                print(f"🛑 Early stopping triggered after {epochs_no_improve} epochs with no improvement.")
                break

    print("\n--- Training Finished ---")
    if best_epoch is not None:
        print(f"🏆 Best Validation Score: {best_composite_score:.4f} (achieved at epoch {best_epoch})")
        model.load_state_dict(best_state)
        torch.save(best_state, artifacts_dir / "best_finetuned_model.pth")
        print(f"✅ Best fine-tuned model state saved to '{artifacts_dir / 'best_finetuned_model.pth'}'.")


if __name__ == "__main__":
    main()
