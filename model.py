# model.py
import torch
from torch import nn
from braindecode.models import EEGNeX

class EEGNeX_Backbone(EEGNeX):
    """
    A modified EEGNeX model that exposes a `forward_features` method.
    This version correctly extracts features by programmatically removing the
    final classification layers, making it robust to braindecode updates.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ✨ FIX: Instead of relying on an internal attribute like '.convs',
        # we define the feature extractor as all layers EXCEPT the final
        # pooling and linear classification layers. This is more robust.
        # The original EEGNeX is a Sequential model, so we can access its children.
        modules = list(self.children())
        self.feature_extractor = nn.Sequential(*modules[:-2])

    def forward_features(self, x):
        return self.feature_extractor(x)

class TransferLearner(nn.Module):
    """
    A model for transfer learning that combines a backbone feature extractor
    with a prediction head and optional demographic data.
    """
    def __init__(self, backbone, n_demographics=0, freeze_backbone=True):
        super().__init__()
        self.backbone = backbone
        self.n_demographics = n_demographics
        
        # Dynamically determine the feature size after the backbone
        self._determine_feature_size()
        
        in_features = self.feature_size + self.n_demographics
        out_features = 1 # Final output is a single regression value
        
        # Define the prediction head
        self.head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, out_features)
        ).to(next(backbone.parameters()).device)

        if freeze_backbone:
            self.freeze_backbone()

    def _determine_feature_size(self):
        # Create a dummy input on the same device as the model
        dummy_input = torch.randn(
            1, self.backbone.n_chans, self.backbone.n_times, 
            device=next(self.backbone.parameters()).device
        )
        with torch.no_grad():
            features = self.backbone.forward_features(dummy_input)
        
        # ✨ FIX: Change the double negative '--1' to a single '-1' to flatten the tensor.
        self.feature_size = features.view(features.size(0), -1).shape[1]

    def freeze_backbone(self):
        print("Freezing backbone weights.")
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        print("Unfreezing backbone weights.")
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, eeg_data, demographics=None):
        features = self.backbone.forward_features(eeg_data)
        features = features.view(features.size(0), -1) # Flatten

        if self.n_demographics > 0:
            if demographics is None:
                raise ValueError("Model expects demographic data, but none was provided.")
            combined_features = torch.cat([features, demographics], dim=1)
        else:
            combined_features = features
            
        return self.head(combined_features)

def create_model(n_chans, n_times, sfreq, n_demographics=0):
    """Helper function to create the backbone and final model."""
    backbone = EEGNeX_Backbone(
        n_chans=n_chans, n_outputs=1, n_times=n_times, sfreq=sfreq
    )

    model = TransferLearner(
        backbone, 
        n_demographics=n_demographics, 
        freeze_backbone=True
    )
    return model

