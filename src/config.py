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
IMAGE_SIZE = (1024, 1024)  # Input size for the model (height, width)
TRAIN_VAL_SPLIT = 0.8  # 80% training, 20% validation
BATCH_SIZE = 8
NUM_WORKERS = 4

# Model configuration
MODEL_CONFIG = {
    "backbone": "resnet50",  # Visual backbone ["resnet50", "efficientnet_b3", "swin_tiny"]
    "hidden_dim": 256,
    "encoder_layers": 6,
    "encoder_heads": 8,
    "decoder_layers": 6,
    "decoder_heads": 8,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "activation": "relu",
    "num_queries": 100,  # Max number of entities to extract
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
    "epochs": 100,
    "lr": 1e-4,
    "lr_backbone": 1e-5,
    "weight_decay": 1e-4,
    "lr_drop": 50,
    "clip_max_norm": 0.1,
    "early_stopping_patience": 10,
}

# Augmentation configuration
AUG_CONFIG = {
    "rotate_limit": 5,
    "scale_limit": 0.1,
    "shift_limit": 0.1,
    "brightness_contrast_limit": 0.2,
    "gaussian_noise_limit": 25.0,
}

# Inference configuration
INFERENCE_CONFIG = {
    "confidence_threshold": 0.7,
    "nms_threshold": 0.5,
}

# Logging configuration
LOGGING_CONFIG = {
    "use_wandb": False,
    "wandb_project": "document-parsing",
    "log_interval": 100,
} 