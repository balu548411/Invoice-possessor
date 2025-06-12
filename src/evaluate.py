import os
import argparse
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

from data_processor import InvoiceDataProcessor
from model import InvoiceProcessor

def calculate_field_accuracy(pred_value, true_value, field_name):
    """Calculate accuracy for a specific field"""
    # This is a simplified accuracy calculation
    # In a real-world scenario, you would use more sophisticated metrics based on field type
    if pred_value and true_value:
        pred_value = pred_value.lower().strip()
        true_value = true_value.lower().strip()
        
        if field_name in ['invoice_number', 'total_amount']:
            # For numeric fields, remove non-alphanumeric chars before comparing
            pred_clean = ''.join(c for c in pred_value if c.isalnum())
            true_clean = ''.join(c for c in true_value if c.isalnum())
            return 1.0 if pred_clean == true_clean else 0.0
        elif field_name in ['date', 'due_date']:
            # For dates, allow more flexibility in comparison
            # This is simplified - in practice you'd use date parsing
            pred_clean = ''.join(c for c in pred_value if c.isdigit())
            true_clean = ''.join(c for c in true_value if c.isdigit())
            return 1.0 if pred_clean == true_clean else 0.0
        else:
            # For text fields, use string comparison
            return 1.0 if pred_value == true_value else 0.0
    else:
        return 0.0

def main(args):
    # Create directories for results
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Initialize data processor
    print("Initializing data processor...")
    data_processor = InvoiceDataProcessor(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        img_size=(args.img_height, args.img_width)
    )
    
    # Create dataset
    print("Loading evaluation dataset...")
    _, (test_images, test_labels), _ = data_processor.create_dataset(split_ratio=0.0)
    
    # Extract text for the text branch
    test_text = [label.get('document_text', '') for label in test_labels]
    
    # Initialize model
    print("Initializing model...")
    model = InvoiceProcessor(
        img_size=(args.img_height, args.img_width),
        max_text_length=args.max_text_length,
        num_fields=args.num_fields
    )
    
    # Load trained weights
    if os.path.exists(args.model_path):
        print(f"Loading model weights from {args.model_path}...")
        model.load_model(args.model_path)
    else:
        print(f"Error: Model weights not found at {args.model_path}")
        return
    
    # Evaluate model on test set
    print("Evaluating model...")
    results = []
    field_metrics = {
        'invoice_number': {'correct': 0, 'total': 0},
        'date': {'correct': 0, 'total': 0},
        'due_date': {'correct': 0, 'total': 0},
        'total_amount': {'correct': 0, 'total': 0},
        'vendor_name': {'correct': 0, 'total': 0},
        'customer_name': {'correct': 0, 'total': 0},
        'tax_amount': {'correct': 0, 'total': 0},
        'subtotal': {'correct': 0, 'total': 0},
        'payment_terms': {'correct': 0, 'total': 0},
        'description': {'correct': 0, 'total': 0}
    }
    
    for i, (img, text, label) in enumerate(tqdm(zip(test_images, test_text, test_labels), total=len(test_images))):
        # Get model prediction
        prediction = model.predict(img, text)
        
        # Compare prediction with ground truth for each field
        for field in field_metrics.keys():
            if field in prediction and field in label:
                pred_value = prediction[field]
                true_value = label.get(field, '')
                
                # Calculate field accuracy
                accuracy = calculate_field_accuracy(pred_value, true_value, field)
                
                # Update metrics
                field_metrics[field]['total'] += 1
                field_metrics[field]['correct'] += accuracy
        
        # Store result for further analysis
        results.append({
            'prediction': prediction,
            'ground_truth': label
        })
    
    # Calculate overall accuracy for each field
    overall_metrics = {}
    for field, metrics in field_metrics.items():
        if metrics['total'] > 0:
            accuracy = metrics['correct'] / metrics['total']
            overall_metrics[field] = {
                'accuracy': accuracy,
                'samples': metrics['total']
            }
    
    # Calculate average accuracy across all fields
    valid_accuracies = [m['accuracy'] for m in overall_metrics.values() if m['samples'] > 0]
    avg_accuracy = np.mean(valid_accuracies) if valid_accuracies else 0.0
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Average accuracy across all fields: {avg_accuracy:.4f}")
    print("\nField-wise accuracy:")
    for field, metrics in overall_metrics.items():
        print(f"  {field}: {metrics['accuracy']:.4f} (samples: {metrics['samples']})")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    fields = [field for field in overall_metrics.keys() if overall_metrics[field]['samples'] > 0]
    accuracies = [overall_metrics[field]['accuracy'] for field in fields if overall_metrics[field]['samples'] > 0]
    
    plt.bar(fields, accuracies, color='skyblue')
    plt.axhline(y=avg_accuracy, color='r', linestyle='-', label=f'Average: {avg_accuracy:.4f}')
    plt.xlabel('Fields')
    plt.ylabel('Accuracy')
    plt.title('Field-wise Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(args.results_dir, 'evaluation_results.png'))
    
    # Save evaluation metrics
    with open(os.path.join(args.results_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump({
            'average_accuracy': avg_accuracy,
            'field_metrics': overall_metrics
        }, f, indent=2)
    
    # Save detailed results
    with open(os.path.join(args.results_dir, 'detailed_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate invoice processor model')
    parser.add_argument('--image_dir', type=str, default='../training_data/images', 
                        help='Directory containing test image files')
    parser.add_argument('--label_dir', type=str, default='../training_data/labels', 
                        help='Directory containing test JSON label files')
    parser.add_argument('--model_path', type=str, default='../models/final_model.h5', 
                        help='Path to trained model weights')
    parser.add_argument('--results_dir', type=str, default='../evaluation_results', 
                        help='Directory to save evaluation results')
    parser.add_argument('--img_height', type=int, default=800, 
                        help='Image height for model input')
    parser.add_argument('--img_width', type=int, default=800, 
                        help='Image width for model input')
    parser.add_argument('--max_text_length', type=int, default=512, 
                        help='Maximum text length for BERT input')
    parser.add_argument('--num_fields', type=int, default=10, 
                        help='Number of invoice fields to extract')
    
    args = parser.parse_args()
    main(args)