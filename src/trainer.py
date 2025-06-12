import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

class InvoiceModelTrainer:
    """
    Trainer class for the invoice processor model
    """
    def __init__(self, 
                 model: nn.Module,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 checkpoints_dir: str = "checkpoints"):
        """
        Initialize the trainer
        
        Args:
            model: Model to train
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            device: Device to use for training
            learning_rate: Learning rate
            weight_decay: Weight decay
            checkpoints_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Create scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
        )
        
        # Define losses
        self.field_confidence_loss = nn.BCEWithLogitsLoss()
        
        print(f"Initialized trainer on {device}")
        
    def _run_epoch(self, epoch: int, is_training: bool) -> Dict:
        """
        Run a single epoch
        
        Args:
            epoch: Epoch number
            is_training: Whether to train or evaluate
            
        Returns:
            Dictionary of metrics
        """
        if is_training:
            self.model.train()
            dataloader = self.train_dataloader
            desc = f"Epoch {epoch} (Train)"
        else:
            self.model.eval()
            dataloader = self.val_dataloader
            desc = f"Epoch {epoch} (Val)"
        
        # Track metrics
        epoch_loss = 0.0
        field_losses = {}
        field_accuracies = {}
        
        # Track field extraction metrics
        field_predictions = []
        field_targets = []
        
        progress_bar = tqdm(dataloader, desc=desc)
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            images = batch['image'].to(self.device)
            tokens = batch['tokens'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            boxes = batch['boxes'].to(self.device)
            
            # Compute attention mask for tokens
            # 1 for padding (to be ignored), 0 for valid tokens
            token_mask = attention_mask.eq(0)
            
            # Forward pass
            with torch.set_grad_enabled(is_training):
                outputs = self.model(images, tokens, boxes, token_mask)
                
                # Calculate losses
                batch_loss = 0.0
                
                # Simplified training loss - in a real implementation, we would extract
                # ground truth labels from the key_fields for each field
                # For now, we'll just use a dummy focal loss for demonstration
                doc_features = outputs['doc_features']
                word_features = outputs['word_features']
                field_preds = outputs['field_predictions']
                
                # Placeholder for field extraction supervision
                # This would be replaced with the actual field labels in a real implementation
                
                # Calculate loss for each field
                for field_name, field_data in field_preds.items():
                    # Dummy targets for demonstration - in reality, these would come from annotations
                    # We're just creating random targets here
                    batch_size = images.size(0)
                    dummy_indices = torch.randint(0, word_features.size(1), (batch_size,)).to(self.device)
                    dummy_targets = torch.zeros(batch_size, word_features.size(1)).to(self.device)
                    
                    # Create one-hot targets
                    for i, idx in enumerate(dummy_indices):
                        dummy_targets[i, idx] = 1.0
                    
                    # Calculate field confidence loss
                    # Generate dummy confidence scores with requires_grad=True
                    dummy_scores = torch.randn(batch_size, word_features.size(1), device=self.device, requires_grad=True)
                    field_loss = self.field_confidence_loss(dummy_scores, dummy_targets)
                    
                    # Add to total loss
                    batch_loss += field_loss
                    
                    # Track field-specific metrics
                    if field_name not in field_losses:
                        field_losses[field_name] = []
                    field_losses[field_name].append(field_loss.item())
            
            # Backpropagate and optimize
            if is_training:
                self.optimizer.zero_grad()
                # Make sure batch_loss requires grad
                if not batch_loss.requires_grad:
                    # Create a dummy tensor that requires grad and add it to batch_loss
                    dummy_tensor = torch.tensor(1.0, device=self.device, requires_grad=True)
                    batch_loss = batch_loss + 0.0 * dummy_tensor
                batch_loss.backward()
                self.optimizer.step()
            
            # Track overall loss
            epoch_loss += batch_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": batch_loss.item()})
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / len(dataloader)
        
        # Calculate field-specific metrics
        field_metrics = {}
        for field_name, losses in field_losses.items():
            field_metrics[field_name] = {
                'loss': np.mean(losses),
            }
        
        # Combine metrics
        metrics = {
            'loss': avg_loss,
            'field_metrics': field_metrics,
        }
        
        return metrics
    
    def train(self, num_epochs: int, early_stopping_patience: int = 5) -> Dict:
        """
        Train the model
        
        Args:
            num_epochs: Number of epochs to train for
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Dictionary of training history
        """
        print(f"Starting training for {num_epochs} epochs")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'best_epoch': 0,
            'best_val_loss': float('inf'),
            'field_metrics': {}
        }
        
        # Early stopping counter
        no_improvement = 0
        
        # Train for the specified number of epochs
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self._run_epoch(epoch, is_training=True)
            train_loss = train_metrics['loss']
            
            # Validate
            val_metrics = self._run_epoch(epoch, is_training=False)
            val_loss = val_metrics['loss']
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Save field metrics
            for field_name, metrics in val_metrics['field_metrics'].items():
                if field_name not in history['field_metrics']:
                    history['field_metrics'][field_name] = []
                history['field_metrics'][field_name].append(metrics)
            
            # Print epoch summary
            print(f"Epoch {epoch}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # Check for improvement
            if val_loss < history['best_val_loss']:
                history['best_val_loss'] = val_loss
                history['best_epoch'] = epoch
                no_improvement = 0
                
                # Save checkpoint
                self.save_checkpoint(f"best_model_epoch_{epoch}.pth")
                print(f"Saved checkpoint at epoch {epoch} (val_loss={val_loss:.4f})")
            else:
                no_improvement += 1
                print(f"No improvement for {no_improvement} epochs")
            
            # Check for early stopping
            if no_improvement >= early_stopping_patience:
                print(f"Early stopping after {epoch} epochs")
                break
            
            # Save checkpoint every few epochs
            if epoch % 5 == 0:
                self.save_checkpoint(f"model_epoch_{epoch}.pth")
        
        # Save final model
        self.save_checkpoint("final_model.pth")
        
        # Save training history
        self.save_history(history)
        
        print(f"Training completed. Best epoch: {history['best_epoch']} (val_loss={history['best_val_loss']:.4f})")
        
        return history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoints_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint_path = self.checkpoints_dir / filename
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint file {checkpoint_path} does not exist")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def save_history(self, history: Dict):
        """Save training history"""
        history_path = self.checkpoints_dir / "training_history.json"
        
        # Convert to serializable format
        serializable_history = {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'best_epoch': history['best_epoch'],
            'best_val_loss': float(history['best_val_loss']),
            'field_metrics': {}
        }
        for field_name, metrics_list in history['field_metrics'].items():
            serializable_history['field_metrics'][field_name] = []
            for metrics in metrics_list:
                serializable_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        serializable_metrics[k] = float(v)
                    else:
                        serializable_metrics[k] = v
                serializable_history['field_metrics'][field_name].append(serializable_metrics)
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=2)
    
    def evaluate(self, test_dataloader: DataLoader) -> Dict:
        """
        Evaluate the model on a test set
        
        Args:
            test_dataloader: DataLoader for test data
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        # Track metrics
        total_loss = 0.0
        field_metrics = {
            'accuracy': {},
            'precision': {},
            'recall': {},
            'f1': {}
        }
        
        # Ground truth and predictions for each field
        field_preds = {}
        field_targets = {}
        
        for field_name in self.model.field_extractor.fields:
            field_preds[field_name] = []
            field_targets[field_name] = []
        
        # Process batches
        progress_bar = tqdm(test_dataloader, desc="Evaluating")
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                images = batch['image'].to(self.device)
                tokens = batch['tokens'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                boxes = batch['boxes'].to(self.device)
                texts = batch['text']
                key_fields = batch['key_fields']
                
                # Compute attention mask for tokens
                token_mask = attention_mask.eq(0)
                
                # Forward pass
                outputs = self.model(images, tokens, boxes, token_mask)
                
                # Extract predicted fields
                # In a real implementation, this would compare the model's predictions to ground truth
                # For now, we'll calculate dummy metrics
                
                # TODO: Implement real field comparison for metrics
                pass
        
        # Calculate metrics for each field
        for field_name in field_preds:
            if len(field_preds[field_name]) > 0:
                # Field-specific metrics
                # Note: In a real implementation, these would be calculated using sklearn
                field_metrics['accuracy'][field_name] = 0.5  # Dummy value
                field_metrics['precision'][field_name] = 0.5  # Dummy value
                field_metrics['recall'][field_name] = 0.5  # Dummy value
                field_metrics['f1'][field_name] = 0.5  # Dummy value
        
        # Calculate overall metrics (average across fields)
        overall_metrics = {
            'accuracy': np.mean([v for v in field_metrics['accuracy'].values()]),
            'precision': np.mean([v for v in field_metrics['precision'].values()]),
            'recall': np.mean([v for v in field_metrics['recall'].values()]),
            'f1': np.mean([v for v in field_metrics['f1'].values()])
        }
        
        # Combine metrics
        evaluation = {
            'loss': total_loss / len(test_dataloader),
            'field_metrics': field_metrics,
            'overall_metrics': overall_metrics
        }
        
        return evaluation 