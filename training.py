"""
Training module for the Bike VAE project.
This module handles the training loop, validation, and model checkpointing.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Optional
import time
import json

from src.vae_model import VAE


class VAETrainer:
    """
    Trainer class for the VAE model.
    
    Handles training loop, validation, and model checkpointing with CUDA support.
    
    Attributes:
        model (VAE): The VAE model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        optimizer (torch.optim.Optimizer): Optimizer for training
        device (torch.device): Device to use for training (CPU or CUDA)
        log_dir (str): Directory for saving logs
        checkpoint_dir (str): Directory for saving model checkpoints
    """
    
    def __init__(self, 
                 model: VAE, 
                 train_loader: DataLoader, 
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 log_dir: str = 'logs',
                 checkpoint_dir: str = 'checkpoints'):
        """
        Initialize the trainer.
        
        Args:
            model: VAE model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: Optimizer for training
            device: Device to use for training (CPU or CUDA)
            log_dir: Directory for saving logs
            checkpoint_dir: Directory for saving model checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        
        # Create directories if they don't exist
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Move model to device
        self.model.to(device)
        
        # Initialize best validation loss for model checkpointing
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        # Set model to training mode
        self.model.train()
        
        # Initialize metrics
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        
        # Create progress bar with tqdm
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        # Iterate over batches
        for batch_idx, data in enumerate(progress_bar):
            # Move data to device
            data = data.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            recon_batch, original_batch, mu, log_var = self.model(data)
            
            # Calculate loss
            loss_dict = self.model.calculate_loss(recon_batch, original_batch, mu, log_var)
            loss = loss_dict['loss']
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_recon_loss += loss_dict['reconstruction_loss'].item()
            total_kl_loss += loss_dict['kl_loss'].item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item() / len(data),
                'recon_loss': loss_dict['reconstruction_loss'].item() / len(data),
                'kl_loss': loss_dict['kl_loss'].item() / len(data)
            })
        
        # Calculate average metrics
        avg_loss = total_loss / len(self.train_loader.dataset)
        avg_recon_loss = total_recon_loss / len(self.train_loader.dataset)
        avg_kl_loss = total_kl_loss / len(self.train_loader.dataset)
        
        # Log metrics to TensorBoard
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        self.writer.add_scalar('ReconLoss/train', avg_recon_loss, epoch)
        self.writer.add_scalar('KLLoss/train', avg_kl_loss, epoch)
        
        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model on the validation set.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        
        # Create progress bar with tqdm
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        
        # No gradient computation for validation
        with torch.no_grad():
            # Iterate over batches
            for batch_idx, data in enumerate(progress_bar):
                # Move data to device
                data = data.to(self.device)
                
                # Forward pass
                recon_batch, original_batch, mu, log_var = self.model(data)
                
                # Calculate loss
                loss_dict = self.model.calculate_loss(recon_batch, original_batch, mu, log_var)
                loss = loss_dict['loss']
                
                # Update metrics
                total_loss += loss.item()
                total_recon_loss += loss_dict['reconstruction_loss'].item()
                total_kl_loss += loss_dict['kl_loss'].item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'val_loss': loss.item() / len(data),
                    'val_recon_loss': loss_dict['reconstruction_loss'].item() / len(data),
                    'val_kl_loss': loss_dict['kl_loss'].item() / len(data)
                })
        
        # Calculate average metrics
        avg_loss = total_loss / len(self.val_loader.dataset)
        avg_recon_loss = total_recon_loss / len(self.val_loader.dataset)
        avg_kl_loss = total_kl_loss / len(self.val_loader.dataset)
        
        # Log metrics to TensorBoard
        self.writer.add_scalar('Loss/val', avg_loss, epoch)
        self.writer.add_scalar('ReconLoss/val', avg_recon_loss, epoch)
        self.writer.add_scalar('KLLoss/val', avg_kl_loss, epoch)
        
        # Save sample reconstructions
        if epoch % 5 == 0:
            self.save_reconstructions(epoch)
        
        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss
        }
    
    def save_reconstructions(self, epoch: int, num_samples: int = 8):
        """
        Save sample reconstructions to TensorBoard.
        
        Args:
            epoch: Current epoch number
            num_samples: Number of samples to reconstruct
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Get a batch of validation data
        data_iter = iter(self.val_loader)
        data = next(data_iter)
        
        # Select a subset of samples
        data = data[:num_samples].to(self.device)
        
        # Reconstruct images
        with torch.no_grad():
            recon_batch, _, _, _ = self.model(data)
        
        # Denormalize images (assuming normalization with ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        data = data * std + mean
        recon_batch = recon_batch * std + mean
        
        # Clamp values to [0, 1]
        data = torch.clamp(data, 0, 1)
        recon_batch = torch.clamp(recon_batch, 0, 1)
        
        # Create comparison grid
        comparison = torch.cat([data, recon_batch])
        
        # Add to TensorBoard
        self.writer.add_images(f'Reconstructions/epoch_{epoch}', comparison, 0)
        
        # Also generate some random samples
        with torch.no_grad():
            samples = self.model.sample(num_samples, self.device)
        
        # Denormalize samples
        samples = samples * std + mean
        samples = torch.clamp(samples, 0, 1)
        
        # Add to TensorBoard
        self.writer.add_images(f'Samples/epoch_{epoch}', samples, 0)
    
    def save_checkpoint(self, epoch: int, val_metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            val_metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_metrics['loss'],
            'val_recon_loss': val_metrics['recon_loss'],
            'val_kl_loss': val_metrics['kl_loss']
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth'))
        
        # Save checkpoint for current epoch
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        # Save best model if this is the best so far
        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best_model.pth'))
    
    def train(self, num_epochs: int, early_stopping_patience: int = 10) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train for
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            
        Returns:
            Dictionary of training and validation metrics for each epoch
        """
        # Initialize metrics history
        metrics_history = {
            'train_loss': [],
            'train_recon_loss': [],
            'train_kl_loss': [],
            'val_loss': [],
            'val_recon_loss': [],
            'val_kl_loss': []
        }
        
        # Initialize early stopping counter
        patience_counter = 0
        
        # Train for specified number of epochs
        for epoch in range(1, num_epochs + 1):
            # Train for one epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update metrics history
            metrics_history['train_loss'].append(train_metrics['loss'])
            metrics_history['train_recon_loss'].append(train_metrics['recon_loss'])
            metrics_history['train_kl_loss'].append(train_metrics['kl_loss'])
            metrics_history['val_loss'].append(val_metrics['loss'])
            metrics_history['val_recon_loss'].append(val_metrics['recon_loss'])
            metrics_history['val_kl_loss'].append(val_metrics['kl_loss'])
            
            # Check if this is the best model so far
            is_best = val_metrics['loss'] < self.best_val_loss
            
            if is_best:
                self.best_val_loss = val_metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Print epoch summary
            print(f"Epoch {epoch}/{num_epochs} - "
                  f"Train Loss: {train_metrics['loss']:.4f} - "
                  f"Val Loss: {val_metrics['loss']:.4f}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Save metrics history
        with open(os.path.join(self.log_dir, 'metrics_history.json'), 'w') as f:
            json.dump(metrics_history, f)
        
        # Close TensorBoard writer
        self.writer.close()
        
        return metrics_history


def train_vae(model: VAE, 
              train_loader: DataLoader, 
              val_loader: DataLoader,
              config: Dict[str, Any]) -> VAETrainer:
    """
    Train a VAE model with the specified configuration.
    
    Args:
        model: VAE model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Training configuration
        
    Returns:
        Trained VAETrainer object
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_cuda', True) else 'cpu')
    print(f"Using device: {device}")
    
    # Set optimizer
    optimizer_name = config.get('optimizer', 'adam').lower()
    lr = config.get('learning_rate', 1e-3)
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.get('weight_decay', 1e-5))
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=config.get('momentum', 0.9), weight_decay=config.get('weight_decay', 1e-5))
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Create trainer
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        log_dir=config.get('log_dir', 'logs'),
        checkpoint_dir=config.get('checkpoint_dir', 'checkpoints')
    )
    
    # Train model
    trainer.train(
        num_epochs=config.get('num_epochs', 100),
        early_stopping_patience=config.get('early_stopping_patience', 10)
    )
    
    return trainer


def load_model(checkpoint_path: str, model: VAE, device: torch.device) -> VAE:
    """
    Load a model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: VAE model to load weights into
        device: Device to load the model onto
        
    Returns:
        Loaded model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device
    model.to(device)
    
    return model
