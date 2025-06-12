# Invoice Processor - Deep Learning Model

This project implements a deep learning model for invoice processing, similar to Azure's pre-trained invoice model. It can extract key fields from invoice images and PDFs, such as invoice numbers, dates, amounts, and vendor information.

## Architecture

The model uses a hybrid architecture:

- **Vision Component**: Vision Transformer (ViT) for visual understanding of the document layout
- **Text Component**: Transformer-based text encoder for processing OCR text
- **Layout Understanding**: Special positional encoding for 2D layout information
- **Multimodal Fusion**: Fusion transformer to combine visual and textual features
- **Field Extraction**: Specialized heads for extracting specific invoice fields

## Project Structure

- `src/`: Source code directory
  - `data_processor.py`: Invoice data processing utilities
  - `dataset.py`: PyTorch dataset for loading and batching invoice data
  - `model.py`: PyTorch model definition
  - `trainer.py`: Training and evaluation routines
  - `train.py`: Main script for training the model
  - `inference.py`: Script for performing inference with the trained model
- `training_data/`: Directory containing training data
  - `images/`: Invoice images
  - `labels/`: JSON annotation files
- `processed_data/`: Directory for processed data (created during training)
- `checkpoints/`: Directory for model checkpoints (created during training)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have Tesseract OCR installed (for OCR capability):
```bash
# For Ubuntu
sudo apt-get install tesseract-ocr

# For macOS
brew install tesseract
```

## Data Format

The training data consists of invoice images and corresponding JSON annotation files. The JSON files should have the following structure:

```json
{
  "pages": [
    {
      "page_number": 1,
      "width": 1224.0,
      "height": 1584.0,
      "lines": [
        {
          "content": "Invoice #12345",
          "polygon": [
            {"x": 86.0, "y": 76.0}, 
            {"x": 296.0, "y": 77.0}, 
            {"x": 296.0, "y": 103.0}, 
            {"x": 86.0, "y": 102.0}
          ]
        },
        // More lines...
      ],
      // Other page info...
    }
  ]
}
```

## Usage

### 1. Process Data and Train the Model

```bash
python src/train.py --images_dir training_data/images --labels_dir training_data/labels
```

Additional training options:
```bash
python src/train.py --images_dir training_data/images \
                    --labels_dir training_data/labels \
                    --batch_size 16 \
                    --epochs 30 \
                    --learning_rate 1e-4 \
                    --image_size 384
```

### 2. Process Invoices with the Trained Model

```bash
python src/inference.py --model_path checkpoints/best_model_epoch_10.pth \
                       --document_path your_invoice.jpg \
                       --output_path results.json
```

## Customization

You can customize the model architecture in `model.py` and training parameters in `train.py`.

## Performance

The model's performance depends significantly on training data quality and quantity. For best results:

1. Use a diverse set of invoice images
2. Ensure accurate annotations, especially for key fields
3. Tune hyperparameters like learning rate, batch size, and image resolution

## Extending the Model

To extract additional fields or support more document types:

1. Update the `extract_key_fields` method in `data_processor.py`
2. Modify the field extractor in `model.py`
3. Add new field types to the training and evaluation routines

## License

This project is provided for educational and research purposes only.