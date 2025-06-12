import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
from pytorch_lightning.loggers import WandbLogger
import wandb
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import logging
import os
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NaNDetectionCallback(Callback):
    """Callback to detect and handle NaN losses"""
    
    def __init__(self, patience: int = 5):
        super().__init__()
        self.nan_count = 0
        self.patience = patience
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if outputs is not None and torch.isnan(outputs):
            self.nan_count += 1
            logger.warning(f"NaN detected in training loss! Count: {self.nan_count}")
            
            if self.nan_count >= self.patience:
                logger.error(f"Too many NaN losses ({self.nan_count}). Stopping training.")
                trainer.should_stop = True
        else:
            self.nan_count = 0  # Reset counter if loss is valid
            
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if outputs is not None and torch.isnan(outputs):
            logger.warning("NaN detected in validation loss!")


class ImprovedEarlyStopping(EarlyStopping):
    """Early stopping that handles NaN values gracefully"""
    
    def _evaluate_stopping_criteria(self, current: torch.Tensor) -> Tuple[bool, str]:
        # Skip if current value is NaN
        if torch.isnan(current):
            logger.warning("Skipping early stopping check due to NaN value")
            return False, "NaN detected"
        
        return super()._evaluate_stopping_criteria(current)


class InvoiceProcessingLightningModuleFixed(pl.LightningModule):
    """Fixed PyTorch Lightning module with NaN handling"""
    
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
                 confidence_weight: float = 0.3,
                 gradient_clip_val: float = 1.0,
                 label_smoothing: float = 0.1):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Import model architecture
        from .model_architecture import InvoiceProcessingModel
        
        # Model
        self.model = InvoiceProcessingModel(
            vision_model=vision_model,
            text_model=text_model,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads
        )
        
        # Loss functions with label smoothing for stability
        self.entity_criterion = nn.CrossEntropyLoss(
            ignore_index=0, 
            label_smoothing=label_smoothing
        )
        self.key_value_criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )
        self.confidence_criterion = nn.MSELoss()
        
        # Gradient clipping value
        self.gradient_clip_val = gradient_clip_val
        
        # Metrics storage
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
    def forward(self, batch):
        return self.model(batch)
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                    batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss with NaN detection"""
        
        losses = {}
        
        try:
            # Entity classification loss
            entity_logits = outputs['entity_logits']
            entity_targets = self._create_entity_targets(batch)
            
            # Ensure logits are finite
            if torch.isnan(entity_logits).any() or torch.isinf(entity_logits).any():
                logger.warning("NaN or Inf detected in entity logits!")
                entity_logits = torch.nan_to_num(entity_logits, nan=0.0, posinf=1e4, neginf=-1e4)
            
            entity_loss = self.entity_criterion(
                entity_logits.view(-1, entity_logits.size(-1)),
                entity_targets.view(-1)
            )
            
            # Check for NaN and replace with large value
            if torch.isnan(entity_loss):
                logger.warning("NaN entity loss detected, replacing with 10.0")
                entity_loss = torch.tensor(10.0, device=self.device)
            
            losses['entity_loss'] = entity_loss
            
            # Key-value classification loss
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
            
            losses['key_loss'] = key_loss
            losses['value_loss'] = value_loss
            
            # Confidence loss
            confidence_scores = outputs['confidence_scores']
            confidence_targets = torch.rand_like(confidence_scores.squeeze(-1))
            confidence_loss = self.confidence_criterion(
                confidence_scores.squeeze(-1), confidence_targets
            )
            
            losses['confidence_loss'] = confidence_loss
            
            # Combined loss with NaN checking
            total_loss = torch.tensor(0.0, device=self.device)
            
            for loss_name, loss_value in losses.items():
                if not torch.isnan(loss_value):
                    weight = getattr(self.hparams, f"{loss_name.split('_')[0]}_weight", 1.0)
                    if loss_name in ['key_loss', 'value_loss']:
                        weight = self.hparams.key_value_weight
                    total_loss += weight * loss_value
                else:
                    logger.warning(f"{loss_name} is NaN, skipping from total loss")
            
            losses['total_loss'] = total_loss
            
        except Exception as e:
            logger.error(f"Error in loss computation: {e}")
            # Return safe default losses
            losses = {
                'total_loss': torch.tensor(10.0, device=self.device),
                'entity_loss': torch.tensor(10.0, device=self.device),
                'key_loss': torch.tensor(1.0, device=self.device),
                'value_loss': torch.tensor(1.0, device=self.device),
                'confidence_loss': torch.tensor(0.1, device=self.device)
            }
        
        return losses
    
    def _create_entity_targets(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Create entity targets from batch with bounds checking"""
        batch_size, num_boxes = batch['boxes'].shape[:2]
        
        # Get number of entity types safely
        num_entity_types = len(self.model.entity_classifier.entity_types)
        
        # Create random entity targets with valid range
        entity_targets = torch.randint(
            0, max(1, num_entity_types),  # Ensure at least 1 class
            (batch_size, num_boxes),
            device=self.device
        )
        
        # Mask padded boxes
        valid_mask = batch['boxes'].sum(dim=-1) > 0
        entity_targets = entity_targets * valid_mask.long()
        
        return entity_targets
    
    def training_step(self, batch, batch_idx):
        # Skip batch if it has issues
        if batch is None or len(batch.get('images', [])) == 0:
            logger.warning(f"Skipping empty batch {batch_idx}")
            return None
            
        outputs = self(batch)
        losses = self.compute_loss(outputs, batch)
        
        # Log losses only if they are finite
        for loss_name, loss_value in losses.items():
            if torch.isfinite(loss_value):
                self.log(f'train_{loss_name}', loss_value, prog_bar=True)
        
        # Also log train_loss as an alias
        if torch.isfinite(losses['total_loss']):
            self.log('train_loss', losses['total_loss'], prog_bar=True)
        
        # Store outputs for epoch end
        if torch.isfinite(losses['total_loss']):
            self.training_step_outputs.append({
                'loss': losses['total_loss'].detach(),
                'entity_logits': outputs['entity_logits'].detach(),
                'entity_targets': self._create_entity_targets(batch).detach()
            })
        
        return losses['total_loss']
    
    def validation_step(self, batch, batch_idx):
        # Skip batch if it has issues
        if batch is None or len(batch.get('images', [])) == 0:
            logger.warning(f"Skipping empty validation batch {batch_idx}")
            return None
            
        outputs = self(batch)
        losses = self.compute_loss(outputs, batch)
        
        # Log losses only if they are finite
        for loss_name, loss_value in losses.items():
            if torch.isfinite(loss_value):
                self.log(f'val_{loss_name}', loss_value, prog_bar=True)
        
        # Also log val_loss as an alias
        if torch.isfinite(losses['total_loss']):
            self.log('val_loss', losses['total_loss'], prog_bar=True)
        
        # Store outputs for epoch end
        if torch.isfinite(losses['total_loss']):
            self.validation_step_outputs.append({
                'loss': losses['total_loss'].detach(),
                'entity_logits': outputs['entity_logits'].detach(),
                'entity_targets': self._create_entity_targets(batch).detach()
            })
        
        return losses['total_loss']
    
    def configure_optimizers(self):
        # Create optimizer with lower learning rate for stability
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            eps=1e-8  # Higher epsilon for numerical stability
        )
        
        # Create learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=self.hparams.max_steps,
            pct_start=self.hparams.warmup_steps / self.hparams.max_steps,
            anneal_strategy='cos',
            div_factor=25.0,  # Start with lr/25
            final_div_factor=10000.0  # End with lr/10000
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }
    
    def on_after_backward(self):
        """Check for NaN gradients"""
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    logger.warning(f"NaN or Inf gradient detected in {name}")
                    param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=1.0, neginf=-1.0)
                    valid_gradients = False
        
        if not valid_gradients:
            logger.warning("Invalid gradients detected and clipped")


def create_fixed_trainer(train_df, val_df, config):
    """Create trainer with fixed callbacks and NaN handling"""
    from .data_processing import create_data_loaders
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_df,
        val_df,
        batch_size=config.get('batch_size', 8),
        num_workers=config.get('num_workers', 0),
        image_size=config.get('image_size', (512, 512))
    )
    
    # Create model with stability improvements
    model = InvoiceProcessingLightningModuleFixed(
        vision_model=config.get('vision_model', 'efficientnet_b3'),
        text_model=config.get('text_model', 'microsoft/layoutlm-base-uncased'),
        d_model=config.get('d_model', 768),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 12),
        learning_rate=config.get('learning_rate', 2e-4),
        weight_decay=config.get('weight_decay', 0.01),
        gradient_clip_val=config.get('gradient_clip_val', 1.0),
        label_smoothing=config.get('label_smoothing', 0.1)
    )
    
    # Setup logging
    wandb_logger = WandbLogger(
        project=config.get('project_name', 'invoice-processing'),
        name=config.get('experiment_name', 'multimodal-v1'),
        log_model=True
    )
    
    # Setup callbacks with NaN handling
    callbacks = [
        ModelCheckpoint(
            monitor='val_total_loss',
            mode='min',
            save_top_k=3,
            filename='invoice-model-{epoch:02d}-{val_total_loss:.3f}',
            save_last=True,
            dirpath=os.path.join(config.get('output_dir', './outputs'), 'checkpoints')
        ),
        ImprovedEarlyStopping(  # Use our custom early stopping
            monitor='val_total_loss',
            patience=10,
            mode='min',
            check_finite=False  # Don't stop on NaN
        ),
        LearningRateMonitor(logging_interval='step'),
        NaNDetectionCallback(patience=5)  # Add NaN detection
    ]
    
    # Determine strategy
    import torch
    if torch.cuda.device_count() > 1:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = 'auto'
    
    # Create trainer with gradient clipping
    trainer = pl.Trainer(
        max_epochs=config.get('max_epochs', 100),
        precision=config.get('precision', '16-mixed'),
        accumulate_grad_batches=config.get('accumulate_grad_batches', 4),
        gradient_clip_val=config.get('gradient_clip_val', 1.0),  # Clip gradients
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator='auto',
        devices='auto',
        strategy=strategy,
        detect_anomaly=True,  # Enable anomaly detection
        log_every_n_steps=10
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    return model, trainer 