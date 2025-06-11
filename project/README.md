# Invoice Processor: Deep Learning Model for Structured Data Extraction

This project provides an end-to-end solution for extracting structured data from invoice images. It uses a hybrid CNN+Transformer architecture to process document images and output structured JSON key-value pairs.

## Features

- **Hybrid Architecture**: Uses a CNN vision encoder (ResNet50/EfficientNet/ViT) and a transformer decoder
- **End-to-End Training**: Complete pipeline from data processing to inference
- **Structured Output**: Extracts key-value pairs from invoices in JSON format
- **Flexible Design**: Supports various document formats and structures

## Project Structure

```
project/
├── config.py               # Configuration settings
├── data_preprocess.py      # Data loading and preprocessing
├── model_arch.py           # Model architecture definition
├── train.py                # Training script
├── eval.py                 # Evaluation script
├── inference.py            # Inference script
├── requirements.txt        # Python dependencies
└── outputs/                # Output directory for models and logs
    ├── models/             # Saved model checkpoints
    └── logs/               # Training logs
```

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd invoice-processor
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Dataset

The model is designed to work with a dataset organized as follows:
- `training_data/images/`: Contains invoice images (JPG/PNG)
- `training_data/lables/`: Contains corresponding JSON labels 

Each JSON label file should have a structure with key-value pairs representing the data to extract from the invoice.

## Training

To train the model:

```
python train.py --resume False
```

To resume training from a checkpoint:

```
python train.py --resume True --checkpoint ./outputs/models/checkpoint_epoch10.pt
```

Training configuration options are defined in `config.py`.

## Evaluation

To evaluate a trained model:

```
python eval.py --model_path ./outputs/models/best_model.pt --split test --visualize
```

This will compute various metrics including:
- Exact Match
- BLEU Score
- METEOR Score 
- Key-Value Match metrics

## Inference

For inference on new invoice images:

```
python inference.py --model_path ./outputs/models/best_model.pt --tokenizer_path ./outputs/models/tokenizer.json --input_path ./path/to/invoice.jpg --visualize
```

To process a batch of images:

```
python inference.py --model_path ./outputs/models/best_model.pt --tokenizer_path ./outputs/models/tokenizer.json --input_path ./path/to/image/directory --output_dir ./results
```

## Using as a Library

You can also use the model programmatically:

```python
from inference import InvoiceProcessor

# Initialize the processor
processor = InvoiceProcessor(
    model_path="./outputs/models/best_model.pt",
    tokenizer_path="./outputs/models/tokenizer.json"
)

# Process a single image
result = processor.process_image("path/to/invoice.jpg", visualize=True)
print(result["json"])

# Process multiple images
results = processor.process_batch(["image1.jpg", "image2.jpg"], output_dir="./results")
```

## Model Architecture

The model uses a hybrid CNN+Transformer architecture:

1. **Vision Encoder**:
   - Uses a pre-trained model (ResNet50, EfficientNet, or Vision Transformer)
   - Converts the document image into a sequence of feature vectors

2. **Transformer Decoder**:
   - Autoregressive decoder that generates output tokens one by one
   - Uses self-attention and cross-attention to extract relevant information
   - Projects output embeddings to token probabilities

3. **Tokenizer**:
   - Custom tokenizer for converting between JSON data and token sequences
   - Handles vocabulary and special tokens for sequence representation

## Performance

The model performance largely depends on the quality and quantity of training data. With sufficient training data, you can expect:

- **Key extraction accuracy**: ~85-95%
- **Value extraction accuracy**: ~75-85%

More complex or variable invoice formats may require additional training data.

## License

[License information here]

## Acknowledgments

- This project draws inspiration from the document AI research community
- Thanks to Azure's DocumentIntelligenceClient for the general approach 