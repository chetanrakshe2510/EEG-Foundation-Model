import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from braindecode.preprocessing import create_fixed_length_windows, preprocess
from braindecode.augmentation import ChannelsDropout, GaussianNoise, TimeReverse

# Import from our custom modules
from config import get_config
import data_loader
import preprocessing as preproc
from model import EEGNeX_Backbone  # Use our corrected custom backbone
from lr_finder import LRFinder     # Import the LRFinder class

# Use the get_config function for consistency
Config = get_config(test_mode=False)


# ==============================================================================
# 1. CONTRASTIVE LEARNING COMPONENTS
# ==============================================================================
class ContrastiveTransform:
    """Applies two different sets of augmentations to the same input EEG window."""
    def __init__(self):
        self.augmentation_one = nn.Sequential(
            TimeReverse(probability=0.5),
            GaussianNoise(probability=0.25, std=0.1),
            ChannelsDropout(probability=0.3, p_drop=0.2),
        )
        self.augmentation_two = nn.Sequential(
            TimeReverse(probability=0.5),
            GaussianNoise(probability=0.25, std=0.1),
            ChannelsDropout(probability=0.3, p_drop=0.2),
        )

    def __call__(self, x):
        view1 = self.augmentation_one(x)
        view2 = self.augmentation_two(x)
        return view1, view2


def nt_xent_loss(z1, z2, temperature=0.5):
    """Normalized Temperature-scaled Cross-Entropy loss for contrastive learning."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.size(0)
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    l_pos = torch.diag(similarity_matrix, batch_size)
    r_pos = torch.diag(similarity_matrix, -batch_size)
    positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

    mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
    negatives = similarity_matrix[mask].view(2 * batch_size, -1)

    logits = torch.cat([positives, negatives], dim=1) / temperature
    labels = torch.zeros(2 * batch_size, device=z1.device, dtype=torch.long)
    loss = F.cross_entropy(logits, labels)
    return loss


class ContrastiveModel(nn.Module):
    """Wraps a backbone model with a projection head for contrastive learning."""
    def __init__(self, backbone, feature_size, projection_dim=128):
        super().__init__()
        self.backbone = backbone
        self.projection_head = nn.Sequential(
            nn.Linear(feature_size, feature_size // 2),
            nn.ReLU(),
            nn.Linear(feature_size // 2, projection_dim)
        )

    def forward(self, x):
        features = self.backbone.forward_features(x)
        features = features.view(features.size(0), -1)
        projection = self.projection_head(features)
        return projection


# ==============================================================================
# 2. PRE-TRAINING ENGINE
# ==============================================================================
def pretrain_one_epoch(dataloader, model, loss_fn, optimizer, transform, device):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Pre-training Epoch")

    for X, _, _ in progress_bar:
        X = X.to(device).float()
        view1, view2 = transform(X)
        z1 = model(view1)
        z2 = model(view2)
        loss = loss_fn(z1, z2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


# ==============================================================================
# 3. MAIN PRE-TRAINING SCRIPT
# ==============================================================================
def pretrain_main():
    """Main function to run the self-supervised pre-training pipeline."""
    print(f"Using device: {Config.DEVICE.upper()} ⚙️")
    print("--- Starting Self-Supervised Pre-training on SuS Task ---")

    # Ensure artifacts directory exists
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    sus_dataset = data_loader.load_bids_data(Config.TRAIN_RELEASES, Config.PRETRAIN_TASK)

    # 2. Preprocess Data
    print("\nApplying preprocessing steps...")
    base_preprocessors = preproc.get_base_preprocessors(Config.SFREQ, Config.L_FREQ, Config.H_FREQ)
    preprocess(sus_dataset, base_preprocessors, n_jobs=1)

    # 3. Create Unlabeled Windows
    windows_dataset = create_fixed_length_windows(
        sus_dataset,
        window_size_samples=int(Config.EPOCH_LEN_SECONDS * Config.SFREQ),
        window_stride_samples=int(Config.EPOCH_LEN_SECONDS * Config.SFREQ / 2),
        drop_last_window=True,
        preload=True,
    )
    print(f"Created {len(windows_dataset)} windows for pre-training.")

    # 4. Create DataLoader and Model
    pretrain_loader = DataLoader(
        windows_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
        num_workers=Config.NUM_WORKERS, drop_last=True, pin_memory=True
    )

    n_chans = windows_dataset[0][0].shape[0]
    n_times = windows_dataset[0][0].shape[1]
    backbone = EEGNeX_Backbone(n_chans=n_chans, n_outputs=1, n_times=n_times, sfreq=Config.SFREQ).to(Config.DEVICE)

    dummy_input = torch.randn(1, n_chans, n_times, device=Config.DEVICE)
    with torch.no_grad():
        features = backbone.forward_features(dummy_input)
    feature_size = features.view(features.size(0), -1).shape[1]
    print(f"Backbone feature size determined to be: {feature_size}")

    model = ContrastiveModel(backbone, feature_size=feature_size).to(Config.DEVICE)
    transform = ContrastiveTransform()

    # --- Optional Non-blocking LR Finder ---
    if getattr(Config, "USE_LR_FINDER", False):
        print("\n--- Running Learning Rate Finder (Non-blocking) ---")
        optimizer_lr_find = AdamW(model.parameters(), lr=1e-7)
        lr_finder = LRFinder(model, optimizer_lr_find, nt_xent_loss, Config.DEVICE)
        lr_finder.range_test(pretrain_loader, start_lr=1e-6, end_lr=1e-1, num_iter=100, is_pretrain=True)
        lr_finder.plot()

        save_path = artifacts_dir / "lr_find_pretrain.png"
        plt.savefig(save_path)
        plt.close()
        print(f"📈 LR Finder plot saved to: {save_path}")
        print("You can inspect the plot and update LEARNING_RATE in config.py accordingly.\n")
    else:
        print("⚙️ Skipping LR Finder (Config.USE_LR_FINDER=False).")

    # 6. Training Loop
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    print("\n--- Starting Pre-training Loop ---")

    for epoch in range(1, Config.N_EPOCHS_PRETRAIN + 1):
        avg_loss = pretrain_one_epoch(pretrain_loader, model, nt_xent_loss, optimizer, transform, Config.DEVICE)
        print(f"Epoch {epoch}/{Config.N_EPOCHS_PRETRAIN} | Average Loss: {avg_loss:.4f}")

    print("\n--- Pre-training Finished ---")

    # 7. Save the Backbone
    torch.save(model.backbone.state_dict(), artifacts_dir / "pretrained_backbone.pth")
    print(f"✅ Pre-trained backbone weights saved to '{artifacts_dir / 'pretrained_backbone.pth'}'.")


if __name__ == "__main__":
    pretrain_main()
