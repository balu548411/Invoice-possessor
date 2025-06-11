import os

# Data paths
DATA_ROOT = '../training_data'
IMAGE_DIR = os.path.join(DATA_ROOT, 'images')
LABEL_DIR = os.path.join(DATA_ROOT, 'lables')  # Match the original directory spelling
OUTPUT_DIR = './outputs'
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, 'models')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')

# Dataset settings
TRAIN_VAL_TEST_SPLIT = [0.8, 0.1, 0.1]
MAX_SEQ_LENGTH = 512
MAX_IMAGE_SIZE = (800, 800)  # (height, width)
RANDOM_SEED = 42

# Model architecture
MODEL_CONFIG = {
    'vision_encoder': 'resnet50',  # or 'vit_base_patch16_224', 'efficientnet_b3'
    'text_decoder': 'bert',  # or 'roberta', 't5'
    'vision_encoder_pretrained': True,
    'text_decoder_pretrained': True,
    'hidden_dim': 768,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'nhead': 8,
    'dim_feedforward': 2048,
    'dropout': 0.1,
}

# Training settings
BATCH_SIZE = 8
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10
WARMUP_EPOCHS = 3
USE_AMP = True  # Automatic Mixed Precision
EARLY_STOPPING_PATIENCE = 10
SAVE_CHECKPOINT_STEPS = 100

# Special tokens
PAD_TOKEN = '<pad>'
START_TOKEN = '<start>'
END_TOKEN = '<end>'
UNK_TOKEN = '<unk>'

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True) 