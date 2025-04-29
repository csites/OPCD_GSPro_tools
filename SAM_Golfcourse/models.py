# In your models.py file

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        # Initial convolutional layers (based on the absence of errors for conv1)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1) # e.g., 1 -> 32 channels, spatial size same
        self.relu = nn.ReLU()

        # --- CORRECTED LAYERS to match the likely SAVED structure ---
        # The saved checkpoint indicates a structure that ends with a Linear layer
        # that takes 32 input features (from conv1 output after pooling) and outputs num_classes.

        # Add Global Average Pooling to reduce spatial dimensions
        # This pools B x 32 x H x W down to B x 32 x 1 x 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Add a Linear layer that takes the pooled features (after squeezing)
        # It takes 32 input features (matching the channels after conv1/relu/pool/squeeze)
        # It outputs num_classes features (the class logits)
        self.fc = nn.Linear(32, num_classes) # Shape of fc.weight will be [num_classes, 32] - matching the checkpoint!
        # --- End CORRECTED LAYERS ---

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: B x 1 x 256 x 256 (SAM's low-res mask)
        x = self.conv1(x) # Output shape: B x 32 x 256 x 256
        x = self.relu(x)  # Output shape: B x 32 x 256 x 256

        # --- CORRECTED forward pass to use the pool and linear layer ---
        x = self.pool(x) # Output shape: B x 32 x 1 x 1
        x = x.squeeze(-1).squeeze(-1) # Squeeze spatial dimensions to B x 32
        x = self.fc(x) # Pass through the linear layer - Output shape: B x num_classes
        # --- End CORRECTED forward pass ---

        return x
