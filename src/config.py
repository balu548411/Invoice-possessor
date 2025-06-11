import os
from pathlib import Path

# Base paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / "training_data"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "lables"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Data configuration
IMAGE_SIZE = (1280, 1280)  # Increased resolution for better detail capture
TRAIN_VAL_SPLIT = 0.85  # Increased training percentage
BATCH_SIZE = 4  # Reduced to accommodate larger model
NUM_WORKERS = 4

# Model configuration
MODEL_CONFIG = {
    "backbone": "swin_base_patch4_window7_224",  # Upgraded to Swin Transformer base model
    "hidden_dim": 384,  # Increased hidden dimension
    "encoder_layers": 8,  # More encoder layers
    "encoder_heads": 12,  # More attention heads
    "decoder_layers": 8,  # More decoder layers
    "decoder_heads": 12,  # More attention heads
    "dim_feedforward": 3072,  # Larger feedforward dimension
    "dropout": 0.2,  # Increased dropout for regularization
    "activation": "gelu",  # GELU activation (better than ReLU)
    "num_queries": 150,  # More queries to handle complex documents
    "entity_classes": {
        "InvoiceId": 0,
        "InvoiceDate": 1,
        "DueDate": 2,
        "VendorName": 3,
        "VendorAddress": 4,
        "CustomerName": 5,
        "CustomerAddress": 6,
        "InvoiceTotal": 7,
        "TaxAmount": 8,
        "ItemDescription": 9,
        "ItemQuantity": 10,
        "ItemPrice": 11,
        "ItemAmount": 12,
    },
}

# Training configuration
TRAIN_CONFIG = {
    "epochs": 150,  # More training epochs
    "lr": 2e-4,  # Slightly higher base learning rate
    "lr_backbone": 2e-5,  # Slightly higher backbone learning rate
    "weight_decay": 1e-4,
    "warmup_epochs": 5,  # Learning rate warmup
    "lr_scheduler": "cosine",  # Cosine annealing scheduler
    "min_lr": 1e-6,  # Minimum learning rate for cosine scheduler
    "clip_max_norm": 0.1,
    "early_stopping_patience": 15,  # More patience for early stopping
    "mixed_precision": True,  # Enable mixed precision training
}

# Augmentation configuration
AUG_CONFIG = {
    "rotate_limit": 5,
    "scale_limit": 0.1,
    "shift_limit": 0.1,
    "brightness_contrast_limit": 0.2,
    "gaussian_noise_limit": 25.0,
    "blur_limit": 3,
    "grid_distortion_prob": 0.2,  # Add controlled distortion
    "random_shadow_prob": 0.2,  # Simulate shadows
    "cutout_prob": 0.1,  # Cutout augmentation
    "elastic_transform_prob": 0.2,  # Elastic transform for realistic document deformation
}

# Inference configuration
INFERENCE_CONFIG = {
    "confidence_threshold": 0.7,
    "nms_threshold": 0.5,
}

# Logging configuration
LOGGING_CONFIG = {
    "use_wandb": False,
    "wandb_project": "document-parsing-enhanced",
    "log_interval": 50,  # More frequent logging
} 