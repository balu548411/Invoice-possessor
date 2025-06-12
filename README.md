# Invoice Processor Model

A deep learning model for automated invoice processing that extracts structured data from invoice images and PDFs, similar to Azure's Form Recognizer for invoices.

## Features

- Processes invoice images (JPG, PNG) and PDFs
- Extracts key invoice fields such as:
  - Invoice number
  - Date
  - Due date
  - Total amount
  - Vendor name
  - Customer name
  - Tax amount
  - Subtotal
  - Payment terms
  - Description
- Uses a hybrid architecture combining:
  - CNN (EfficientNetB3) for visual feature extraction
  - BERT for text understanding 
  - Multi-head output for field extraction

## Project Structure

```
.
├── training_data/       # Training dataset
│   ├── images/          # Invoice images
│   └── labels/          # JSON label files
├── src/                 # Source code
│   ├── data_processor.py  # Data processing utilities
│   ├── model.py         # Model architecture
│   ├── train.py         # Training script
│   └── inference.py     # Inference script
├── models/              # Saved model checkpoints
├── logs/                # Training logs
└── inference_results/   # Results from inference
```

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

The model requires:
- TensorFlow 2.x
- OpenCV
- pytesseract
- pdf2image
- transformers
- Other utilities like numpy, matplotlib, etc.

## Usage

### Training the Model

To train the model on your invoice dataset:

```bash
python src/train.py \
  --image_dir training_data/images \
  --label_dir training_data/labels \
  --epochs 20 \
  --batch_size 8
```

Parameters:
- `--image_dir`: Directory containing invoice images
- `--label_dir`: Directory containing corresponding JSON labels
- `--model_dir`: Directory to save the trained model (default: '../models')
- `--log_dir`: Directory to save logs and plots (default: '../logs')
- `--img_height`: Image height for model input (default: 800)
- `--img_width`: Image width for model input (default: 800)
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Training batch size (default: 8)

### Running Inference

To extract information from new invoices using the trained model:

```bash
python src/inference.py \
  --input_path /path/to/invoice.pdf \
  --model_path models/final_model.h5
```

Or for processing a directory of invoices:

```bash
python src/inference.py \
  --input_path /path/to/invoice_directory \
  --model_path models/final_model.h5
```

Parameters:
- `--input_path`: Path to input file or directory containing files
- `--output_dir`: Directory to save output JSON files (default: '../inference_results')
- `--model_path`: Path to trained model weights (default: '../models/final_model.h5')

## Model Architecture

The model uses a hybrid architecture:

1. **Image Processing Branch**:
   - EfficientNetB3 as the backbone for feature extraction
   - Global average pooling to get fixed-size feature vectors
   - Dense layers for feature transformation

2. **Text Processing Branch**:
   - BERT for understanding the textual content of invoices
   - Uses the CLS token output as document representation

3. **Multi-head Output**:
   - Separate prediction heads for each invoice field
   - Character-level sequence prediction for flexibility

## Dataset Format

The training data consists of:

1. **Images**: Invoice images in JPG/PNG format
2. **Labels**: JSON files with field annotations including:
   - Document text content
   - Word bounding boxes
   - Field values

## Performance Optimization

- Model weights are saved at checkpoints during training
- Early stopping to prevent overfitting
- Learning rate scheduling for better convergence
- Appropriate batch size for memory efficiency

## Future Improvements

- Add table extraction capabilities for line items
- Implement more sophisticated field extraction using NER
- Support for multiple languages
- Attention visualization for better explainability
- Fine-tuning with domain-specific invoice types

## License

This project is licensed under the MIT License - see the LICENSE file for details.