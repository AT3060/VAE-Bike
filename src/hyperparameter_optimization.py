"""
Hyperparameter optimization module for the Bike VAE project.
This module uses Optuna to find the optimal hyperparameters for the VAE model.
"""

import os
import torch
import optuna
from optuna.trial import Trial
import numpy as np
import json
from typing import Dict, Any, List, Tuple, Optional
import logging
from torch.utils.data import DataLoader

from src.vae_model import VAE
from src.training import train_vae


class VAEHyperparameterOptimizer:
    """
    Hyperparameter optimizer for the VAE model using Optuna.
    
    Attributes:
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        img_size (Tuple[int, int]): Input image size (height, width)
        img_channels (int): Number of input image channels
        study_name (str): Name of the Optuna study
        n_trials (int): Number of trials to run
        storage (str): Storage URL for the Optuna study
        direction (str): Direction of optimization ('minimize' or 'maximize')
    """
    
    def __init__(self,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 img_size: Tuple[int, int] = (128, 128),
                 img_channels: int = 3,
                 study_name: str = 'vae_optimization',
                 n_trials: int = 20,
                 storage: Optional[str] = None,
                 direction: str = 'minimize'):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            img_size: Input image size (height, width)
            img_channels: Number of input image channels
            study_name: Name of the Optuna study
            n_trials: Number of trials to run
            storage: Storage URL for the Optuna study
            direction: Direction of optimization ('minimize' or 'maximize')
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.img_size = img_size
        self.img_channels = img_channels
        self.study_name = study_name
        self.n_trials = n_trials
        self.storage = storage
        self.direction = direction
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('VAEHyperparameterOptimizer')
        
        # Create results directory
        os.makedirs('optuna_results', exist_ok=True)
    
    def objective(self, trial: Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation loss (to be minimized)
        """
        # Sample hyperparameters
        config = self.sample_hyperparameters(trial)
        
        # Log hyperparameters
        self.logger.info(f"Trial {trial.number}: {config}")
        
        # Create model with sampled hyperparameters
        model = VAE(
            img_channels=self.img_channels,
            hidden_dims=config['hidden_dims'],
            latent_dim=config['latent_dim'],
            img_size=self.img_size,
            kl_weight=config['kl_weight']
        )
        
        # Set up training configuration
        train_config = {
            'optimizer': config['optimizer'],
            'learning_rate': config['learning_rate'],
            'weight_decay': config['weight_decay'],
            'num_epochs': config['num_epochs'],
            'early_stopping_patience': config['early_stopping_patience'],
            'use_cuda': True,
            'log_dir': f'logs/trial_{trial.number}',
            'checkpoint_dir': f'checkpoints/trial_{trial.number}'
        }
        
        # Create directories
        os.makedirs(train_config['log_dir'], exist_ok=True)
        os.makedirs(train_config['checkpoint_dir'], exist_ok=True)
        
        # Train model
        try:
            trainer = train_vae(model, self.train_loader, self.val_loader, train_config)
            
            # Get best validation loss
            best_val_loss = trainer.best_val_loss
            
            # Save trial results
            self.save_trial_results(trial, config, best_val_loss)
            
            return best_val_loss
        except Exception as e:
            self.logger.error(f"Error in trial {trial.number}: {e}")
            # Return a high loss value to indicate failure
            return float('inf')
    
    def sample_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        """
        Sample hyperparameters for the current trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of sampled hyperparameters
        """
        # Sample network architecture hyperparameters
        latent_dim = trial.suggest_int('latent_dim', 32, 256, log=True)
        
        # Sample hidden dimensions for the convolutional layers
        n_layers = trial.suggest_int('n_layers', 3, 5)
        base_filters = trial.suggest_int('base_filters', 16, 64, log=True)
        hidden_dims = [base_filters * (2**i) for i in range(n_layers)]
        
        # Sample loss function hyperparameters
        kl_weight = trial.suggest_float('kl_weight', 0.001, 0.1, log=True)
        
        # Sample optimizer hyperparameters
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        
        # Sample training hyperparameters
        num_epochs = trial.suggest_int('num_epochs', 30, 100)
        early_stopping_patience = trial.suggest_int('early_stopping_patience', 5, 15)
        
        return {
            'latent_dim': latent_dim,
            'hidden_dims': hidden_dims,
            'kl_weight': kl_weight,
            'optimizer': optimizer,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'num_epochs': num_epochs,
            'early_stopping_patience': early_stopping_patience
        }
    
    def save_trial_results(self, trial: Trial, config: Dict[str, Any], val_loss: float):
        """
        Save trial results to a JSON file.
        
        Args:
            trial: Optuna trial object
            config: Hyperparameter configuration
            val_loss: Validation loss
        """
        # Create results dictionary
        results = {
            'trial_number': trial.number,
            'hyperparameters': config,
            'val_loss': val_loss
        }
        
        # Save to JSON file
        with open(f'optuna_results/trial_{trial.number}.json', 'w') as f:
            json.dump(results, f, indent=4)
    
    def optimize(self) -> optuna.Study:
        """
        Run the hyperparameter optimization.
        
        Returns:
            Completed Optuna study
        """
        # Create Optuna study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=self.direction,
            load_if_exists=True
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Log best trial
        self.logger.info(f"Best trial: {study.best_trial.number}")
        self.logger.info(f"Best value: {study.best_value}")
        self.logger.info(f"Best hyperparameters: {study.best_params}")
        
        # Save best hyperparameters
        self.save_best_hyperparameters(study)
        
        return study
    
    def save_best_hyperparameters(self, study: optuna.Study):
        """
        Save the best hyperparameters to a JSON file.
        
        Args:
            study: Completed Optuna study
        """
        # Get best trial
        best_trial = study.best_trial
        
        # Create best hyperparameters dictionary
        best_hyperparameters = {
            'trial_number': best_trial.number,
            'val_loss': best_trial.value,
            'hyperparameters': best_trial.params
        }
        
        # Save to JSON file
        with open('optuna_results/best_hyperparameters.json', 'w') as f:
            json.dump(best_hyperparameters, f, indent=4)


def create_model_from_best_hyperparameters(img_channels: int = 3, img_size: Tuple[int, int] = (128, 128)) -> VAE:
    """
    Create a VAE model with the best hyperparameters found by Optuna.
    
    Args:
        img_channels: Number of input image channels
        img_size: Input image size (height, width)
        
    Returns:
        VAE model with the best hyperparameters
    """
    # Load best hyperparameters
    try:
        with open('optuna_results/best_hyperparameters.json', 'r') as f:
            best_hyperparameters = json.load(f)
        
        params = best_hyperparameters['hyperparameters']
        
        # Reconstruct hidden_dims
        n_layers = params['n_layers']
        base_filters = params['base_filters']
        hidden_dims = [base_filters * (2**i) for i in range(n_layers)]
        
        # Create model
        model = VAE(
            img_channels=img_channels,
            hidden_dims=hidden_dims,
            latent_dim=params['latent_dim'],
            img_size=img_size,
            kl_weight=params['kl_weight']
        )
        
        return model
    except FileNotFoundError:
        print("Best hyperparameters file not found. Using default hyperparameters.")
        
        # Create model with default hyperparameters
        model = VAE(
            img_channels=img_channels,
            hidden_dims=[32, 64, 128, 256, 512],
            latent_dim=128,
            img_size=img_size,
            kl_weight=0.005
        )
        
        return model
