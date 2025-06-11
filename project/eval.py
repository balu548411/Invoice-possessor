import os
import torch
import json
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from PIL import Image
import difflib
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import nltk
import re

from config import *
from data_preprocess import prepare_dataset, get_dataloaders, visualize_sample
from model_arch import InvoiceTransformer


# Download required NLTK data
try:
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('omw-1.4')
except:
    print("NLTK downloads failed. Some metrics may not work.")


def load_model(model_path, vocab_size, device):
    """Load a trained model from checkpoint"""
    model = InvoiceTransformer(vocab_size=vocab_size)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def normalize_text(text):
    """Normalize text for evaluation"""
    # Convert to lowercase
    text = text.lower()
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation (keep = sign for key-value pairs)
    text = re.sub(r'[^\w\s=]', '', text)
    # Strip leading and trailing spaces
    text = text.strip()
    return text


def compute_exact_match(pred_text, target_text):
    """Compute exact match score between prediction and target"""
    pred_text = normalize_text(pred_text)
    target_text = normalize_text(target_text)
    return int(pred_text == target_text)


def compute_bleu_score(pred_text, target_text):
    """Compute BLEU score between prediction and target"""
    pred_tokens = nltk.word_tokenize(normalize_text(pred_text))
    target_tokens = nltk.word_tokenize(normalize_text(target_text))
    return sentence_bleu([target_tokens], pred_tokens)


def compute_meteor_score(pred_text, target_text):
    """Compute METEOR score between prediction and target"""
    pred_tokens = nltk.word_tokenize(normalize_text(pred_text))
    target_tokens = nltk.word_tokenize(normalize_text(target_text))
    try:
        return meteor_score([target_tokens], pred_tokens)
    except:
        return 0.0  # Fallback in case meteor_score fails


def compute_key_match_score(pred_text, target_text):
    """Compute percentage of matching keys between prediction and target"""
    # Extract keys from text (assuming "key = value" format)
    pred_keys = set(re.findall(r'(\w+)\s*=', normalize_text(pred_text)))
    target_keys = set(re.findall(r'(\w+)\s*=', normalize_text(target_text)))
    
    if len(target_keys) == 0:
        return 0.0
    
    # Calculate precision, recall, and F1 score
    if len(pred_keys) == 0:
        precision = 0.0
    else:
        precision = len(pred_keys.intersection(target_keys)) / len(pred_keys)
        
    recall = len(pred_keys.intersection(target_keys)) / len(target_keys)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
        
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def compute_value_match_score(pred_text, target_text):
    """Compute percentage of matching values for matching keys"""
    pred_text = normalize_text(pred_text)
    target_text = normalize_text(target_text)
    
    # Extract key-value pairs (assuming "key = value" format)
    pred_pairs = dict(re.findall(r'(\w+)\s*=\s*([^;]+)', pred_text))
    target_pairs = dict(re.findall(r'(\w+)\s*=\s*([^;]+)', target_text))
    
    # Find common keys
    common_keys = set(pred_pairs.keys()).intersection(set(target_pairs.keys()))
    
    if len(common_keys) == 0:
        return 0.0
        
    # Count matching values
    matching_values = 0
    for key in common_keys:
        pred_value = pred_pairs[key].strip()
        target_value = target_pairs[key].strip()
        if pred_value == target_value:
            matching_values += 1
            
    return matching_values / len(common_keys)


def evaluate_model(model, dataloader, tokenizer, device, num_samples=None):
    """Evaluate model on validation/test set"""
    model.eval()
    
    # Metrics
    exact_match_scores = []
    bleu_scores = []
    meteor_scores = []
    key_precision_scores = []
    key_recall_scores = []
    key_f1_scores = []
    value_match_scores = []
    
    # Sample predictions for visualization
    samples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if num_samples is not None and batch_idx >= num_samples:
                break
                
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            batch_size = images.size(0)
            
            for i in range(batch_size):
                # Generate prediction for single image
                image = images[i:i+1]
                label = labels[i:i+1]
                
                # Get actual label text
                target_text = tokenizer.decode_ids(label[0].cpu().tolist())
                
                # Generate prediction
                pred_tokens = model.generate(image, max_length=MAX_SEQ_LENGTH, temperature=0.7)
                pred_text = tokenizer.decode_ids(pred_tokens)
                
                # Compute metrics
                exact_match = compute_exact_match(pred_text, target_text)
                bleu = compute_bleu_score(pred_text, target_text)
                meteor = compute_meteor_score(pred_text, target_text)
                key_match = compute_key_match_score(pred_text, target_text)
                value_match = compute_value_match_score(pred_text, target_text)
                
                exact_match_scores.append(exact_match)
                bleu_scores.append(bleu)
                meteor_scores.append(meteor)
                key_precision_scores.append(key_match["precision"])
                key_recall_scores.append(key_match["recall"])
                key_f1_scores.append(key_match["f1"])
                value_match_scores.append(value_match)
                
                # Save sample for visualization
                if len(samples) < 5:
                    image_path = batch["image_path"][i]
                    label_path = batch["label_path"][i]
                    samples.append({
                        "image_path": image_path,
                        "label_path": label_path,
                        "target_text": target_text,
                        "pred_text": pred_text,
                        "metrics": {
                            "exact_match": exact_match,
                            "bleu": bleu,
                            "meteor": meteor,
                            "key_match": key_match,
                            "value_match": value_match
                        }
                    })
    
    # Compute average metrics
    metrics = {
        "exact_match": np.mean(exact_match_scores),
        "bleu": np.mean(bleu_scores),
        "meteor": np.mean(meteor_scores),
        "key_precision": np.mean(key_precision_scores),
        "key_recall": np.mean(key_recall_scores),
        "key_f1": np.mean(key_f1_scores),
        "value_match": np.mean(value_match_scores)
    }
    
    return metrics, samples


def visualize_predictions(samples, output_dir=None):
    """Visualize model predictions"""
    for i, sample in enumerate(samples):
        image_path = sample["image_path"]
        target_text = sample["target_text"]
        pred_text = sample["pred_text"]
        metrics = sample["metrics"]
        
        # Display image
        plt.figure(figsize=(16, 16))
        plt.imshow(Image.open(image_path))
        plt.title(f"Sample {i+1} - F1: {metrics['key_match']['f1']:.4f}, Value Match: {metrics['value_match']:.4f}")
        plt.axis('off')
        
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"sample_{i+1}_image.png"))
        
        plt.show()
        
        # Display prediction vs ground truth
        print(f"Sample {i+1} - Key F1: {metrics['key_match']['f1']:.4f}, Value Match: {metrics['value_match']:.4f}")
        print("\nPrediction:")
        print(pred_text[:1000] + ("..." if len(pred_text) > 1000 else ""))
        print("\nGround Truth:")
        print(target_text[:1000] + ("..." if len(target_text) > 1000 else ""))
        print("\n" + "-"*80 + "\n")
        
        if output_dir is not None:
            # Save text comparison to file
            with open(os.path.join(output_dir, f"sample_{i+1}_comparison.txt"), "w") as f:
                f.write(f"Sample {i+1} - Key F1: {metrics['key_match']['f1']:.4f}, Value Match: {metrics['value_match']:.4f}\n\n")
                f.write("Prediction:\n")
                f.write(pred_text + "\n\n")
                f.write("Ground Truth:\n")
                f.write(target_text + "\n\n")


def main(args):
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare dataset
    print("Preparing datasets...")
    datasets = prepare_dataset(split=True)
    dataloaders = get_dataloaders(datasets, batch_size=1, num_workers=1)  # Use batch size 1 for evaluation
    tokenizer = dataloaders["tokenizer"]
    
    # Choose dataset split for evaluation
    if args.split == "val":
        eval_dataloader = dataloaders["val"]
        print(f"Evaluating on validation set ({len(datasets['val'])} samples)")
    elif args.split == "test":
        eval_dataloader = dataloaders["test"]
        print(f"Evaluating on test set ({len(datasets['test'])} samples)")
    elif args.split == "train":
        eval_dataloader = dataloaders["train"]
        print(f"Evaluating on training set ({len(datasets['train'])} samples)")
    else:
        raise ValueError(f"Unknown split: {args.split}")
    
    # Load the model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, tokenizer.vocab_size, device)
    
    # Evaluate
    print("Evaluating model...")
    metrics, samples = evaluate_model(
        model, eval_dataloader, tokenizer, device, 
        num_samples=args.num_samples
    )
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"Exact Match: {metrics['exact_match']:.4f}")
    print(f"BLEU Score: {metrics['bleu']:.4f}")
    print(f"METEOR Score: {metrics['meteor']:.4f}")
    print(f"Key Precision: {metrics['key_precision']:.4f}")
    print(f"Key Recall: {metrics['key_recall']:.4f}")
    print(f"Key F1 Score: {metrics['key_f1']:.4f}")
    print(f"Value Match: {metrics['value_match']:.4f}")
    
    # Save metrics to file
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Visualize predictions
    if args.visualize:
        print("\nVisualizing predictions...")
        visualize_predictions(samples, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate invoice processing model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], 
                        help="Dataset split to evaluate on")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="./outputs/evaluation", 
                        help="Directory to save evaluation results")
    parser.add_argument("--visualize", action="store_true", help="Visualize predictions")
    
    args = parser.parse_args()
    
    main(args) 