# Invoice Possessor: Document Understanding Model

This project implements a deep learning model for extracting structured data from invoice images, similar to Azure's DocumentIntelligenceClient but built from scratch using a hybrid CNN+Transformer architecture.

## Overview

Invoice Possessor is an end-to-end document understanding system that converts invoice images into structured JSON data. It uses a powerful neural network model that combines computer vision and natural language processing techniques to accurately extract key information from diverse invoice formats.

## Key Features

- **Deep Learning Architecture**: Leverages a hybrid CNN+Transformer model for robust document understanding
- **End-to-End Pipeline**: Complete solution from data preparation to model training and inference
- **Structured Output**: Extracts key-value pairs from invoices in clean JSON format
- **Custom Tokenization**: Specialized tokenizer for document data representation
- **Comprehensive Evaluation**: Multiple metrics to assess extraction quality
- **Easy Inference**: Simple API for processing new invoice images

## Project Structure

The main implementation is in the `project` directory:

```
project/
├── config.py               # Configuration settings
├── data_preprocess.py      # Data loading and preprocessing
├── model_arch.py           # Model architecture definition
├── train.py                # Training script
├── eval.py                 # Evaluation script
├── inference.py            # Inference script
├── run.py                  # Sample demo script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Getting Started

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/Invoice-possessor.git
   cd Invoice-possessor
   ```

2. Install dependencies:
   ```
   cd project
   pip install -r requirements.txt
   ```

### Quick Start

The easiest way to run the complete pipeline is using the `run.py` script:

```
python run.py --mode demo
```

This will:
1. Check for dependencies
2. Look for trained models
3. Run inference on sample images if models are available

### Training a Model

To train the model on your invoice dataset:

```
python run.py --mode train
```

or directly:

```
python train.py
```

Training parameters can be configured in `config.py`.

### Evaluating the Model

To evaluate the model performance:

```
python eval.py --model_path ./outputs/models/best_model.pt --split test --visualize
```

### Inference on New Invoices

To process new invoice images:

```
python inference.py --model_path ./outputs/models/best_model.pt --tokenizer_path ./outputs/models/tokenizer.json --input_path /path/to/invoice.jpg --visualize
```

## Model Architecture

The model uses a hybrid architecture:

1. **Vision Encoder**: Processes invoice images into high-dimensional features
   - Supports ResNet50, EfficientNet, or Vision Transformer backbones

2. **Transformer Decoder**: Interprets visual features to generate structured data
   - Uses self-attention and cross-attention mechanisms
   - Autoregressive generation for output sequences

3. **Custom Tokenizer**: Handles conversion between JSON and token sequences

## Dataset Structure

The model expects training data in this structure:
- `training_data/images/`: Contains invoice images (JPG format)
- `training_data/lables/`: Contains corresponding JSON labels

## Using as a Library

```python
from project.inference import InvoiceProcessor

processor = InvoiceProcessor(
    model_path="./project/outputs/models/best_model.pt",
    tokenizer_path="./project/outputs/models/tokenizer.json"
)

# Process a single image
result = processor.process_image("path/to/invoice.jpg")
print(result["json"])
```

## Custom Training

For custom training on your own invoice dataset:

1. Organize your data as described in the Dataset Structure section
2. Adjust parameters in `config.py` as needed
3. Run the training script: `python train.py`

## License

[License information]

## Acknowledgments

- This project draws inspiration from document AI research
- Thanks to Azure's DocumentIntelligenceClient for the approach 