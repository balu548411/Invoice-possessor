# Intelligent Document Parser

This project implements a deep learning model for intelligent document parsing, extracting structured information (key-value pairs) from document images.

## Project Overview

The Intelligent Document Parser uses a combination of computer vision and natural language processing techniques to:

1. Extract text and layout information from document images
2. Identify important fields and their values (e.g., invoice numbers, dates, amounts)
3. Output structured JSON data similar to Azure's Document Intelligence services

## Features

- **Advanced Visual Processing**: Uses state-of-the-art visual backbones (ResNet, EfficientNet, or Swin Transformer)
- **Document Understanding**: Transformer encoder-decoder architecture for document comprehension
- **Entity Recognition**: Identifies document entities with bounding boxes and classifications
- **Structured Output**: Returns well-structured JSON with key-value pairs
- **API Service**: FastAPI implementation for easy integration

## Project Structure

```
.
├── requirements.txt           # Python dependencies
├── train.py                   # Training script
├── infer.py                   # Inference script
├── api_service.py             # FastAPI server for document parsing
├── src/                       # Source code
│   ├── config.py              # Configuration parameters
│   ├── data/                  # Data handling
│   │   └── dataset.py         # Document dataset implementation
│   ├── model/                 # Model implementation
│   │   ├── backbones.py       # Visual backbones (ResNet, EfficientNet, Swin)
│   │   ├── document_parser.py # Main document parsing model
│   │   ├── transformer.py     # Transformer encoder-decoder
│   │   └── losses.py          # Loss functions
│   ├── evaluation/            # Evaluation metrics
│   │   └── metrics.py         # Evaluation metrics for document parsing
│   ├── inference/             # Inference utilities
│   │   └── predictor.py       # Document parsing predictor
│   └── utils/                 # Utility functions
├── training_data/             # Training data
│   ├── images/                # Document images
│   └── lables/                # JSON label files
└── outputs/                   # Training outputs
    ├── models/                # Saved models
    └── logs/                  # Training logs
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd document-parser
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

To train the model:

```bash
python train.py
```

The training script:
- Loads the document dataset from `training_data/`
- Splits data into training and validation sets
- Trains the model with the configured parameters
- Saves checkpoints to `outputs/models/`
- Tracks metrics (optionally with WandB)

## Inference

To run inference on a document image:

```bash
python infer.py --image path/to/document.jpg --output results.jpg --json_output results.json
```

Options:
- `--image`: Path to the input document image (required)
- `--model`: Path to model weights (default: best model from training)
- `--output`: Path to save visualization image (optional)
- `--json_output`: Path to save JSON results (optional)
- `--threshold`: Confidence threshold (default: from config)

## API Service

Run the API service with:

```bash
python api_service.py
```

The service provides:
- `/parse`: Parse documents and return structured JSON
- `/visualize`: Parse documents and return visualized results
- `/model_info`: Get information about the loaded model

Access the API documentation at http://localhost:8000/docs

## Model Architecture

The model has three main components:

1. **Visual Backbone**: Extracts visual features from the document image
   - Options: ResNet, EfficientNet, Swin Transformer

2. **Transformer Encoder-Decoder**:
   - Encoder: Processes document visual features with self-attention
   - Decoder: Generates entity representations with cross-attention to the document

3. **Prediction Heads**:
   - Classification: Identifies entity types
   - Bounding Box: Localizes entities in the document

## License

[MIT License]

## Acknowledgements

- This project was inspired by the DETR (Detection Transformer) architecture
- Uses pre-trained backbone models from torchvision and timm 