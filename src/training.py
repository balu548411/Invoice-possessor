import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import logging

from .model_architecture import InvoiceProcessingModel
from .data_processing import create_data_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InvoiceProcessingLightningModule(pl.LightningModule):
    """PyTorch Lightning module for training the invoice processing model"""
    
    def __init__(self,
                 vision_model: str = 'efficientnet_b3',
                 text_model: str = 'microsoft/layoutlm-base-uncased',
                 d_model: int = 768,
                 num_layers: int = 6,
                 num_heads: int = 12,
                 learning_rate: float = 2e-4,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 1000,
                 max_steps: int = 10000,
                 entity_weight: float = 1.0,
                 key_value_weight: float = 0.5,
                 confidence_weight: float = 0.3):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Model
        self.model = InvoiceProcessingModel(
            vision_model=vision_model,
            text_model=text_model,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads
        )
        
        # Loss functions
        self.entity_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        self.key_value_criterion = nn.CrossEntropyLoss()
        self.confidence_criterion = nn.MSELoss()
        
        # Metrics storage
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
    def forward(self, batch):
        return self.model(batch)
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                    batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss"""
        
        # Entity classification loss
        entity_logits = outputs['entity_logits']
        entity_targets = self._create_entity_targets(batch)
        entity_loss = self.entity_criterion(
            entity_logits.view(-1, entity_logits.size(-1)),
            entity_targets.view(-1)
        )
        
        # Key-value classification loss (simplified - using random targets for demo)
        key_logits = outputs['key_logits']
        value_logits = outputs['value_logits']
        batch_size, num_boxes = key_logits.shape[:2]
        
        # Create dummy key-value targets (in real scenario, these would come from annotations)
        key_targets = torch.randint(0, 2, (batch_size, num_boxes), device=self.device)
        value_targets = torch.randint(0, 2, (batch_size, num_boxes), device=self.device)
        
        key_loss = self.key_value_criterion(
            key_logits.view(-1, 2), key_targets.view(-1)
        )
        value_loss = self.key_value_criterion(
            value_logits.view(-1, 2), value_targets.view(-1)
        )
        
        # Confidence loss (using dummy targets)
        confidence_scores = outputs['confidence_scores']
        confidence_targets = torch.rand_like(confidence_scores.squeeze(-1))
        confidence_loss = self.confidence_criterion(
            confidence_scores.squeeze(-1), confidence_targets
        )
        
        # Combined loss
        total_loss = (
            self.hparams.entity_weight * entity_loss +
            self.hparams.key_value_weight * (key_loss + value_loss) +
            self.hparams.confidence_weight * confidence_loss
        )
        
        return {
            'total_loss': total_loss,
            'entity_loss': entity_loss,
            'key_loss': key_loss,
            'value_loss': value_loss,
            'confidence_loss': confidence_loss
        }
    
    def _create_entity_targets(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Create entity targets from batch (simplified version)"""
        batch_size, num_boxes = batch['boxes'].shape[:2]
        
        # For demo purposes, create random entity targets
        # In real scenario, these would be derived from the annotations
        entity_targets = torch.randint(
            0, len(self.model.entity_classifier.entity_types),
            (batch_size, num_boxes),
            device=self.device
        )
        
        # Mask padded boxes
        valid_mask = batch['boxes'].sum(dim=-1) > 0
        entity_targets = entity_targets * valid_mask.long()
        
        return entity_targets
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        losses = self.compute_loss(outputs, batch)
        
        # Log losses
        for loss_name, loss_value in losses.items():
            self.log(f'train_{loss_name}', loss_value, prog_bar=True)
        
        # Also log train_loss as an alias for train_total_loss for consistency
        self.log('train_loss', losses['total_loss'], prog_bar=True)
        
        # Store outputs for epoch end
        self.training_step_outputs.append({
            'loss': losses['total_loss'].detach(),
            'entity_logits': outputs['entity_logits'].detach(),
            'entity_targets': self._create_entity_targets(batch).detach()
        })
        
        return losses['total_loss']
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        losses = self.compute_loss(outputs, batch)
        
        # Log losses
        for loss_name, loss_value in losses.items():
            self.log(f'val_{loss_name}', loss_value, prog_bar=True)
        
        # Also log val_loss as an alias for val_total_loss for compatibility
        self.log('val_loss', losses['total_loss'], prog_bar=True)
        
        # Store outputs for epoch end
        self.validation_step_outputs.append({
            'loss': losses['total_loss'].detach(),
            'entity_logits': outputs['entity_logits'].detach(),
            'entity_targets': self._create_entity_targets(batch).detach()
        })
        
        return losses['total_loss']
    
    def on_training_epoch_end(self):
        # Compute epoch metrics
        avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
        self.log('train_epoch_loss', avg_loss)
        
        # Compute entity classification metrics
        all_logits = torch.cat([x['entity_logits'] for x in self.training_step_outputs])
        all_targets = torch.cat([x['entity_targets'] for x in self.training_step_outputs])
        
        preds = torch.argmax(all_logits, dim=-1).cpu().numpy().flatten()
        targets = all_targets.cpu().numpy().flatten()
        
        # Filter out padding (label 0)
        mask = targets != 0
        if mask.sum() > 0:
            preds = preds[mask]
            targets = targets[mask]
            
            accuracy = accuracy_score(targets, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                targets, preds, average='weighted', zero_division=0
            )
            
            self.log('train_accuracy', accuracy)
            self.log('train_precision', precision)
            self.log('train_recall', recall)
            self.log('train_f1', f1)
        
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        # Compute epoch metrics
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        self.log('val_epoch_loss', avg_loss)
        
        # Compute entity classification metrics
        all_logits = torch.cat([x['entity_logits'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['entity_targets'] for x in self.validation_step_outputs])
        
        preds = torch.argmax(all_logits, dim=-1).cpu().numpy().flatten()
        targets = all_targets.cpu().numpy().flatten()
        
        # Filter out padding (label 0)
        mask = targets != 0
        if mask.sum() > 0:
            preds = preds[mask]
            targets = targets[mask]
            
            accuracy = accuracy_score(targets, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                targets, preds, average='weighted', zero_division=0
            )
            
            self.log('val_accuracy', accuracy)
            self.log('val_precision', precision)
            self.log('val_recall', recall)
            self.log('val_f1', f1)
        
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        # Create optimizer
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Create learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=self.hparams.max_steps,
            pct_start=self.hparams.warmup_steps / self.hparams.max_steps,
            anneal_strategy='cos'
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }


class InvoiceTrainer:
    """High-level trainer class for invoice processing model"""
    
    def __init__(self,
                 train_df,
                 val_df,
                 config: Optional[Dict] = None):
        
        self.train_df = train_df
        self.val_df = val_df
        
        # Default configuration
        self.config = {
            'batch_size': 8,
            'num_workers': 4,
            'max_epochs': 50,
            'learning_rate': 2e-4,
            'weight_decay': 0.01,
            'image_size': (512, 512),
            'precision': '16-mixed',
            'accumulate_grad_batches': 4,
            'gradient_clip_val': 1.0,
            'project_name': 'invoice-processing',
            'experiment_name': 'multimodal-v1',
            'output_dir': './outputs'
        }
        
        if config:
            self.config.update(config)
    
    def setup_logging(self):
        """Setup Weights & Biases logging"""
        wandb_logger = WandbLogger(
            project=self.config['project_name'],
            name=self.config['experiment_name'],
            log_model=True
        )
        return wandb_logger
    
    def setup_callbacks(self):
        """Setup training callbacks"""
        callbacks = [
            ModelCheckpoint(
                monitor='val_f1',
                mode='max',
                save_top_k=3,
                filename='invoice-model-{epoch:02d}-{val_f1:.3f}',
                save_last=True
            ),
            EarlyStopping(
                monitor='val_total_loss',
                patience=10,
                mode='min'
            ),
            LearningRateMonitor(logging_interval='step')
        ]
        return callbacks
    
    def train(self):
        """Main training function"""
        logger.info("Starting invoice processing model training...")
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            self.train_df,
            self.val_df,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            image_size=self.config['image_size']
        )
        
        # Create model
        model = InvoiceProcessingLightningModule(
            learning_rate=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Setup logging and callbacks
        wandb_logger = self.setup_logging()
        callbacks = self.setup_callbacks()
        
        # Create trainer
        # Determine strategy based on available devices
        import torch
        if torch.cuda.device_count() > 1:
            from pytorch_lightning.strategies import DDPStrategy
            strategy = DDPStrategy(find_unused_parameters=True)
        else:
            strategy = 'auto'
        
        trainer = pl.Trainer(
            max_epochs=self.config['max_epochs'],
            precision=self.config['precision'],
            accumulate_grad_batches=self.config['accumulate_grad_batches'],
            gradient_clip_val=self.config['gradient_clip_val'],
            logger=wandb_logger,
            callbacks=callbacks,
            accelerator='auto',
            devices='auto',
            strategy=strategy
        )
        
        # Train model
        trainer.fit(model, train_loader, val_loader)
        
        logger.info("Training completed!")
        
        return model, trainer
    
    def resume_training(self, checkpoint_path: str):
        """Resume training from checkpoint"""
        logger.info(f"Resuming training from {checkpoint_path}")
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            self.train_df,
            self.val_df,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            image_size=self.config['image_size']
        )
        
        # Load model from checkpoint
        model = InvoiceProcessingLightningModule.load_from_checkpoint(checkpoint_path)
        
        # Setup logging and callbacks
        wandb_logger = self.setup_logging()
        callbacks = self.setup_callbacks()
        
        # Create trainer
        # Determine strategy based on available devices
        import torch
        if torch.cuda.device_count() > 1:
            from pytorch_lightning.strategies import DDPStrategy
            strategy = DDPStrategy(find_unused_parameters=True)
        else:
            strategy = 'auto'
        
        trainer = pl.Trainer(
            max_epochs=self.config['max_epochs'],
            precision=self.config['precision'],
            accumulate_grad_batches=self.config['accumulate_grad_batches'],
            gradient_clip_val=self.config['gradient_clip_val'],
            logger=wandb_logger,
            callbacks=callbacks,
            accelerator='auto',
            devices='auto',
            strategy=strategy
        )
        
        # Resume training
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)
        
        return model, trainer


def train_invoice_model(train_df, val_df, config: Optional[Dict] = None):
    """Convenience function to train the invoice model"""
    trainer = InvoiceTrainer(train_df, val_df, config)
    return trainer.train() 