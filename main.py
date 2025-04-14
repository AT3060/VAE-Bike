"""
Bike VAE Project - Main Script

This script provides a command-line interface for training and using the Variational Autoencoder
for bike image generation. It includes options for training, hyperparameter optimization,
and generating new images.

Usage:
    python main.py train --data_dir /path/to/data --epochs 50
    python main.py optimize --data_dir /path/to/data --trials 20
    python main.py generate --model_path /path/to/model.pth --num_images 16
"""

import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.data_loading import get_data_loaders
from src.vae_model import VAE
from src.training import train_vae, load_model
from src.hyperparameter_optimization import VAEHyperparameterOptimizer, create_model_from_best_hyperparameters
from src.image_generation import ImageGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Bike VAE Project')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the VAE model')
    train_parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory')
    train_parser.add_argument('--img_size', type=int, nargs=2, default=[128, 128], help='Image size (height, width)')
    train_parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--kl_weight', type=float, default=0.005, help='KL divergence weight')
    train_parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize hyperparameters')
    optimize_parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory')
    optimize_parser.add_argument('--img_size', type=int, nargs=2, default=[128, 128], help='Image size (height, width)')
    optimize_parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    optimize_parser.add_argument('--trials', type=int, default=20, help='Number of Optuna trials')
    optimize_parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate images')
    generate_parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    generate_parser.add_argument('--num_images', type=int, default=16, help='Number of images to generate')
    generate_parser.add_argument('--img_size', type=int, nargs=2, default=[128, 128], help='Image size (height, width)')
    generate_parser.add_argument('--output_dir', type=str, default='generated_images', help='Output directory')
    
    return parser.parse_args()


def train(args):
    """Train the VAE model."""
    print(f"Training VAE model with data from {args.data_dir}")
    
    # Create output directories
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        img_size=tuple(args.img_size)
    )
    
    # Create model
    model = VAE(
        img_channels=3,
        hidden_dims=[32, 64, 128, 256, 512],
        latent_dim=args.latent_dim,
        img_size=tuple(args.img_size),
        kl_weight=args.kl_weight
    )
    
    # Train model
    train_config = {
        'optimizer': 'adam',
        'learning_rate': args.lr,
        'weight_decay': 1e-5,
        'num_epochs': args.epochs,
        'early_stopping_patience': 10,
        'use_cuda': torch.cuda.is_available(),
        'log_dir': os.path.join(args.output_dir, 'logs'),
        'checkpoint_dir': os.path.join(args.output_dir, 'checkpoints')
    }
    
    trainer = train_vae(model, train_loader, val_loader, train_config)
    
    print(f"Training complete. Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Model saved to {os.path.join(args.output_dir, 'checkpoints', 'best_model.pth')}")


def optimize(args):
    """Optimize hyperparameters using Optuna."""
    print(f"Optimizing hyperparameters with {args.trials} trials")
    
    # Create output directories
    os.makedirs(os.path.join(args.output_dir, 'optuna_results'), exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        img_size=tuple(args.img_size)
    )
    
    # Create hyperparameter optimizer
    optimizer = VAEHyperparameterOptimizer(
        train_loader=train_loader,
        val_loader=val_loader,
        img_size=tuple(args.img_size),
        img_channels=3,
        study_name='vae_optimization',
        n_trials=args.trials
    )
    
    # Run optimization
    study = optimizer.optimize()
    
    print(f"Optimization complete. Best trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.4f}")
    print(f"Best hyperparameters: {study.best_params}")
    print(f"Results saved to {os.path.join(args.output_dir, 'optuna_results')}")


def generate(args):
    """Generate images using a trained model."""
    print(f"Generating {args.num_images} images using model from {args.model_path}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = VAE(
        img_channels=3,
        hidden_dims=[32, 64, 128, 256, 512],
        latent_dim=128,
        img_size=tuple(args.img_size),
        kl_weight=0.005
    )
    
    # Load model
    model = load_model(args.model_path, model, device)
    
    # Create image generator
    generator = ImageGenerator(model, device, tuple(args.img_size))
    
    # Generate images
    generator.generate_and_save_images(
        args.num_images,
        filename=os.path.join(args.output_dir, 'generated.png'),
        nrow=4
    )
    
    print(f"Images saved to {os.path.join(args.output_dir, 'generated.png')}")


def main():
    """Main function."""
    args = parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'optimize':
        optimize(args)
    elif args.command == 'generate':
        generate(args)
    else:
        print("Please specify a command: train, optimize, or generate")


if __name__ == '__main__':
    main()
