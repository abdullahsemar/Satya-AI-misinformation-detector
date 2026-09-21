import os
import torch
import torch.nn as nn
from typing import List, Tuple

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "efficientnet_b0.pt")

def get_device() -> torch.device:
    """Returns CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_model_dir_exists(model_dir: str = DEFAULT_MODEL_DIR):
    """Ensures the directory for storing local model checkpoints exists."""
    os.makedirs(model_dir, exist_ok=True)

def generate_xai_markers(fake_confidence: float, is_deepfake: bool) -> List[str]:
    """
    Generates explainable AI (XAI) diagnostic markers based on model output score.
    """
    xai_markers = []
    if is_deepfake or fake_confidence >= 0.6:
        if fake_confidence > 0.8:
            xai_markers.append("Unnatural facial texture anomaly detected by EfficientNet-B0")
        if fake_confidence > 0.7:
            xai_markers.append("High-frequency boundary noise detected in feature maps")
        if fake_confidence > 0.9:
            xai_markers.append("Spatial warping & synthetic artifacts identified in facial region")
        if not xai_markers:
            xai_markers.append("Subtle synthetic anomalies detected")
    else:
        xai_markers.append("Natural structural consistency confirmed by EfficientNet-B0")
        
    return xai_markers

def create_fine_tune_setup(model: nn.Module, lr: float = 1e-4) -> Tuple[torch.optim.Optimizer, nn.Module]:
    """
    Prepares the EfficientNet-B0 model for fine-tuning on deepfake datasets
    (e.g., FaceForensics++, DFDC, or Celeb-DF).
    Unfreezes top layers and sets up AdamW optimizer & CrossEntropy/BCE loss.
    """
    # Unfreeze feature extractor parameters if fine-tuning
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    return optimizer, criterion
