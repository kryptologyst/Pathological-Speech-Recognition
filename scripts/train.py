#!/usr/bin/env python3
"""Training script for pathological speech recognition models."""

import os
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb

from src.pathological_speech_recognition.models import Wav2Vec2PathologicalSpeechModel, ConformerPathologicalSpeechModel
from src.pathological_speech_recognition.data import AudioPreprocessor, PathologicalSpeechAugmentation
from src.pathological_speech_recognition.metrics import PathologicalSpeechMetrics
from src.pathological_speech_recognition.utils.common import (
    get_device, set_seed, setup_logging, create_output_dir, EarlyStopping
)

logger = logging.getLogger(__name__)


class PathologicalSpeechTrainer:
    """Trainer for pathological speech recognition models."""
    
    def __init__(self, config: DictConfig):
        """Initialize trainer.
        
        Args:
            config: Training configuration.
        """
        self.config = config
        
        # Setup
        set_seed(config.seed)
        setup_logging(config.log_level)
        
        self.device = get_device(config.device)
        logger.info(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir = create_output_dir(config.output_dir, config.experiment_name)
        logger.info(f"Output directory: {self.output_dir}")
        
        # Save configuration
        OmegaConf.save(config, os.path.join(self.output_dir, "config.yaml"))
        
        # Initialize components
        self.model = self._initialize_model()
        self.preprocessor = AudioPreprocessor(config.data)
        self.augmentation = PathologicalSpeechAugmentation(config.data.augmentation)
        self.metrics = PathologicalSpeechMetrics()
        
        # Training components
        self.optimizer = self._initialize_optimizer()
        self.scheduler = self._initialize_scheduler()
        self.criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.get("early_stopping_patience", 5),
            min_delta=config.training.get("early_stopping_min_delta", 0.001)
        )
        
        # Initialize wandb if configured
        if config.get("wandb", {}).get("enabled", False):
            wandb.init(
                project=config.wandb.project,
                name=config.experiment_name,
                config=OmegaConf.to_container(config, resolve=True)
            )
    
    def _initialize_model(self) -> nn.Module:
        """Initialize the model.
        
        Returns:
            Initialized model.
        """
        model_name = self.config.model.name.lower()
        
        if model_name == "wav2vec2":
            model = Wav2Vec2PathologicalSpeechModel(self.config.model)
        elif model_name == "conformer":
            model = ConformerPathologicalSpeechModel(self.config.model)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = model.to(self.device)
        logger.info(f"Initialized {model_name} model")
        
        return model
    
    def _initialize_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer.
        
        Returns:
            Initialized optimizer.
        """
        optimizer_name = self.config.training.optimizer.lower()
        
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=(
                    self.config.training.get("adam_beta1", 0.9),
                    self.config.training.get("adam_beta2", 0.999)
                ),
                eps=self.config.training.get("adam_epsilon", 1e-8)
            )
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        logger.info(f"Initialized {optimizer_name} optimizer")
        return optimizer
    
    def _initialize_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Initialize learning rate scheduler.
        
        Returns:
            Initialized scheduler or None.
        """
        scheduler_type = self.config.training.get("lr_scheduler_type", "linear")
        
        if scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.training.warmup_steps,
                num_training_steps=self.config.training.num_epochs * 1000  # Approximate
            )
        elif scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs
            )
        else:
            scheduler = None
        
        if scheduler:
            logger.info(f"Initialized {scheduler_type} scheduler")
        
        return scheduler
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            dataloader: Training data loader.
            epoch: Current epoch number.
            
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_values = batch["input_values"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs["loss"]
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )
            
            # Optimizer step
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            total_samples += input_values.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (batch_idx + 1):.4f}"
            })
            
            # Logging
            if batch_idx % self.config.training.logging_steps == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                )
                
                if self.config.get("wandb", {}).get("enabled", False):
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        "batch": batch_idx
                    })
        
        avg_loss = total_loss / len(dataloader)
        
        return {
            "train_loss": avg_loss,
            "train_samples": total_samples
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model.
        
        Args:
            dataloader: Evaluation data loader.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        self.model.eval()
        self.metrics.reset()
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                input_values = batch["input_values"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs["loss"]
                total_loss += loss.item()
                total_samples += input_values.size(0)
                
                # Generate predictions for metrics
                predicted_ids = self.model.generate(
                    input_values=input_values,
                    attention_mask=attention_mask
                )
                
                # Decode predictions
                if hasattr(self.model, 'processor'):
                    predictions = [
                        self.model.processor.decode(ids) 
                        for ids in predicted_ids
                    ]
                else:
                    predictions = ["prediction"] * predicted_ids.size(0)
                
                # Get references
                references = batch.get("text", ["reference"] * predicted_ids.size(0))
                
                # Update metrics
                self.metrics.update(references, predictions)
        
        # Compute final metrics
        eval_metrics = self.metrics.compute()
        eval_metrics["eval_loss"] = total_loss / len(dataloader)
        eval_metrics["eval_samples"] = total_samples
        
        return eval_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch.
            metrics: Current metrics.
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config
        }
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        """Train the model.
        
        Args:
            train_dataloader: Training data loader.
            val_dataloader: Validation data loader.
        """
        logger.info("Starting training...")
        
        best_metric = float('inf')
        
        for epoch in range(1, self.config.training.num_epochs + 1):
            logger.info(f"Epoch {epoch}/{self.config.training.num_epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_dataloader, epoch)
            
            # Evaluation
            if epoch % self.config.training.eval_strategy == 0:
                eval_metrics = self.evaluate(val_dataloader)
                
                # Log metrics
                logger.info(f"Epoch {epoch} - Train Loss: {train_metrics['train_loss']:.4f}, "
                          f"Val Loss: {eval_metrics['eval_loss']:.4f}, "
                          f"WER: {eval_metrics.get('wer', 0.0):.4f}")
                
                # Wandb logging
                if self.config.get("wandb", {}).get("enabled", False):
                    wandb.log({
                        "epoch": epoch,
                        **train_metrics,
                        **eval_metrics
                    })
                
                # Check if best model
                metric_name = self.config.training.metric_for_best_model
                current_metric = eval_metrics.get(metric_name, eval_metrics['eval_loss'])
                
                is_best = current_metric < best_metric
                if is_best:
                    best_metric = current_metric
                
                # Save checkpoint
                if epoch % self.config.training.save_strategy == 0:
                    self.save_checkpoint(epoch, eval_metrics, is_best)
                
                # Early stopping
                if self.early_stopping(current_metric, self.model):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        logger.info("Training completed!")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train pathological speech recognition model")
    parser.add_argument("--config-name", type=str, default="config", help="Configuration name")
    parser.add_argument("--config-path", type=str, default="configs", help="Configuration path")
    parser.add_argument("--checkpoint", type=str, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = os.path.join(args.config_path, f"{args.config_name}.yaml")
    config = OmegaConf.load(config_path)
    
    # Override with command line arguments
    if args.checkpoint:
        config.checkpoint = args.checkpoint
    
    # Initialize trainer
    trainer = PathologicalSpeechTrainer(config)
    
    # TODO: Initialize data loaders
    # This would require implementing dataset classes and data loading
    # For now, we'll create placeholder loaders
    
    logger.info("Note: Data loaders need to be implemented for full training functionality")
    
    # Placeholder for data loaders
    # train_dataloader = create_train_dataloader(config)
    # val_dataloader = create_val_dataloader(config)
    
    # trainer.train(train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
