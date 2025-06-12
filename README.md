# Invoice Processing Deep Learning Model 🧾🤖

A powerful deep learning model for invoice processing and entity extraction, similar to Azure Form Recognizer but more customizable and powerful. This project combines computer vision and natural language processing to extract structured information from invoice images.

## 🌟 Features

- **Multi-modal Architecture**: Combines vision and text understanding using state-of-the-art models
- **Spatial Attention**: Understanding of document layout and spatial relationships
- **Entity Extraction**: Extracts key invoice fields (invoice numbers, dates, amounts, vendor info, etc.)
- **High Accuracy**: Advanced transformer-based architecture with attention mechanisms
- **Production Ready**: Includes inference API similar to Azure Form Recognizer
- **Flexible Training**: Supports various vision backbones and text models
- **Advanced Training**: Mixed precision, gradient accumulation, learning rate scheduling

## 🏗️ Architecture

The model consists of several key components:

1. **Multi-modal Encoder**: Combines EfficientNet (vision) + LayoutLM (text/layout)
2. **Spatial Attention**: Understands relationships between text elements
3. **Entity Classifier**: BIO tagging for entity recognition
4. **Confidence Estimation**: Provides confidence scores for extractions

## 📋 Supported Entity Types

- Invoice Number
- Invoice Date
- Due Date
- Vendor Name & Address
- Customer Name & Address
- Total Amount
- Subtotal
- Tax Amount
- Line Items
- Payment Terms

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Invoice-possessor

# Install dependencies
pip install -r requirements.txt
```

### Data Structure

Your training data should be organized as follows:
```
training_data/
├── images/
│   ├── invoice1.jpg
│   ├── invoice2.jpg
│   └── ...
└── labels/
    ├── invoice1.json
    ├── invoice2.json
    └── ...
```

The JSON files should contain OCR annotations in Azure Form Recognizer format with bounding boxes and text content.

### Training

```bash
# Basic training
python train_model.py

# Custom configuration
python train_model.py \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --max_epochs 100 \
    --vision_model efficientnet_b4 \
    --experiment_name "my_experiment"

# Resume from checkpoint
python train_model.py --resume_from path/to/checkpoint.ckpt
```

### Inference

```bash
# Process single invoice
python inference_demo.py \
    --model_path path/to/model.ckpt \
    --image_path path/to/invoice.jpg \
    --visualize

# Batch processing
python inference_demo.py \
    --model_path path/to/model.ckpt \
    --image_path path/to/invoice_dir/ \
    --output_dir results/
```

## 📊 Training Configuration

Key training parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 8 | Training batch size |
| `learning_rate` | 2e-4 | Initial learning rate |
| `max_epochs` | 50 | Maximum training epochs |
| `vision_model` | efficientnet_b3 | Vision backbone |
| `text_model` | layoutlm-base | Text/layout model |
| `image_size` | 512x512 | Input image dimensions |
| `precision` | 16-mixed | Training precision |

## 🔧 Model Architecture Details

### Vision Backbone
- **EfficientNet**: Scalable and efficient CNN architecture
- **Feature Extraction**: Multi-scale features for different text sizes
- **ROI Pooling**: Extract features for specific text regions

### Text Understanding
- **LayoutLM**: Pre-trained model understanding text and layout
- **Positional Encoding**: 2D positional embeddings for spatial relationships
- **Attention Mechanisms**: Multi-head attention for text relationships

### Fusion Layer
- **Spatial Attention**: Custom attention mechanism for layout understanding
- **Multi-task Learning**: Joint training on multiple objectives
- **Confidence Estimation**: Uncertainty quantification

## 📈 Performance Metrics

The model tracks several metrics:

- **Entity F1 Score**: Primary metric for entity extraction quality
- **Precision/Recall**: Per-entity type performance
- **Confidence Calibration**: How well confidence scores reflect accuracy
- **Spatial Accuracy**: Correctness of bounding box predictions

## 🛠️ API Usage

### Python API

```python
from src.inference import InvoiceProcessor

# Initialize processor
processor = InvoiceProcessor("path/to/model.ckpt")

# Process invoice
result = processor.process_invoice("invoice.jpg")

# Extract specific fields
invoice_number = result['pages'][0]['extracted_fields']['INVOICE_NUM']['value']
total_amount = result['pages'][0]['extracted_fields']['TOTAL']['value']
```

### Batch Processing

```python
from src.inference import InvoiceProcessor

processor = InvoiceProcessor("model.ckpt")

# Process multiple invoices
image_paths = ["inv1.jpg", "inv2.jpg", "inv3.jpg"]
results = processor.batch_process(image_paths, output_dir="results/")
```

## 📂 Project Structure

```
Invoice-possessor/
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # Data loading and preprocessing
│   ├── model_architecture.py   # Model definitions
│   ├── training.py            # Training logic
│   └── inference.py           # Inference and API
├── training_data/
│   ├── images/                # Invoice images
│   └── labels/                # JSON annotations
├── train_model.py             # Main training script
├── inference_demo.py          # Inference demonstration
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🔄 Data Preprocessing

The data processing pipeline:

1. **Image Loading**: Supports various formats (JPG, PNG)
2. **OCR Integration**: Extracts text and bounding boxes
3. **Entity Mapping**: Maps text to entity types using pattern matching
4. **Augmentation**: Random transformations for robustness
5. **Normalization**: Coordinate and feature normalization

## 🎯 Training Process

### Data Splitting
- **Training**: 80% of data
- **Validation**: 20% of data
- **Stratification**: Optional stratification by document type

### Loss Functions
- **Entity Loss**: Cross-entropy for entity classification
- **Spatial Loss**: Regression for bounding box refinement
- **Confidence Loss**: MSE for confidence estimation

### Optimization
- **Optimizer**: AdamW with weight decay
- **Scheduler**: OneCycleLR for learning rate scheduling
- **Regularization**: Dropout, weight decay, gradient clipping

## 📊 Monitoring and Logging

### Weights & Biases Integration
- Real-time metric tracking
- Model artifact logging
- Hyperparameter sweeps
- Experiment comparison

### Metrics Tracked
- Training/validation loss
- Per-entity F1 scores
- Learning rate progression
- GPU utilization
- Training time per epoch

## 🔧 Advanced Configuration

### Custom Vision Backbones
```python
# Use different vision models
configs = {
    'vision_model': 'resnet50',  # or 'efficientnet_b5', 'vit_base_patch16_224'
    'd_model': 768,
    'num_layers': 8
}
```

### Multi-GPU Training
```bash
# Automatic multi-GPU detection
python train_model.py --batch_size 32  # Will use all available GPUs
```

### Mixed Precision Training
```bash
# Enable for faster training with lower memory usage
python train_model.py --precision 16-mixed
```

## 🚀 Production Deployment

### Model Export
```python
# Export trained model
from src.inference import InvoiceProcessor

processor = InvoiceProcessor("model.ckpt")
# Model is automatically optimized for inference
```

### API Server
```python
from src.inference import InvoiceAPI

api = InvoiceAPI("model.ckpt")
result = api.analyze_invoice(image_bytes)
```

## 🧪 Validation and Testing

### Cross-validation
```bash
# K-fold cross-validation
python train_model.py --cv_folds 5
```

### Test Set Evaluation
```python
from src.training import evaluate_model

metrics = evaluate_model(model, test_dataloader)
print(f"Test F1: {metrics['f1']:.3f}")
```

## 📈 Performance Optimization

### Training Speed
- **Mixed Precision**: 2x faster training
- **Gradient Accumulation**: Effective larger batch sizes
- **Multi-GPU**: Linear scaling with GPU count
- **Optimized Data Loading**: Multi-worker data loading

### Memory Optimization
- **Gradient Checkpointing**: Trade compute for memory
- **Dynamic Batching**: Efficient memory usage
- **Model Sharding**: For very large models

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size or enable gradient accumulation
   python train_model.py --batch_size 4 --accumulate_grad_batches 8
   ```

2. **Poor Entity Extraction**
   ```bash
   # Try different vision backbone or increase model size
   python train_model.py --vision_model efficientnet_b5 --d_model 1024
   ```

3. **Slow Training**
   ```bash
   # Enable mixed precision and increase batch size
   python train_model.py --precision 16-mixed --batch_size 16
   ```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Azure Form Recognizer**: Inspiration for API design
- **LayoutLM**: Pre-trained layout understanding model
- **EfficientNet**: Efficient computer vision backbone
- **PyTorch Lightning**: Training framework
- **Weights & Biases**: Experiment tracking

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the documentation
- Review the example notebooks

---

**Happy Invoice Processing! 🧾✨** 