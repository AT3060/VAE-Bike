"""
Data loading module for the Bike VAE project.
This module handles loading and preprocessing of bike images with data augmentation.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional, Dict, Any
import random


class BikeDataset(Dataset):
    """
    Dataset class for loading and preprocessing bike images.
    
    Attributes:
        image_paths (List[str]): List of paths to bike images
        transform (A.Compose): Albumentations transformations for data augmentation
        img_size (Tuple[int, int]): Target image size (height, width)
    """
    
    def __init__(self, 
                 data_dir: str, 
                 img_size: Tuple[int, int] = (128, 128),
                 transform: Optional[A.Compose] = None,
                 split: str = 'train'):
        """
        Initialize the BikeDataset.
        
        Args:
            data_dir: Directory containing bike images
            img_size: Target image size (height, width)
            transform: Optional albumentations transformations
            split: Dataset split ('train' or 'val')
        """
        self.img_size = img_size
        
        # Get all image files from the data directory
        self.image_paths = []
        valid_extensions = ['.jpg', '.jpeg', '.png']
        for file in os.listdir(data_dir):
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                self.image_paths.append(os.path.join(data_dir, file))
        
        # Split dataset into train and validation sets (80/20 split)
        random.seed(42)  # For reproducibility
        random.shuffle(self.image_paths)
        
        if split == 'train':
            self.image_paths = self.image_paths[:int(0.8 * len(self.image_paths))]
        else:  # validation set
            self.image_paths = self.image_paths[int(0.8 * len(self.image_paths)):]
        
        # Set up transformations
        if transform is None:
            if split == 'train':
                # More aggressive augmentation for training
                self.transform = A.Compose([
                    A.Resize(height=img_size[0], width=img_size[1]),
                    A.HorizontalFlip(p=0.5),  # Flip image horizontally with 50% probability
                    A.RandomBrightnessContrast(p=0.3),  # Adjust brightness/contrast
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),  # Geometric transformations
                    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.3),  # Color jittering
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize to ImageNet stats
                    ToTensorV2(),  # Convert to PyTorch tensor
                ])
            else:
                # Only resize and normalize for validation
                self.transform = A.Compose([
                    A.Resize(height=img_size[0], width=img_size[1]),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
        else:
            self.transform = transform
    
    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single image from the dataset.
        
        Args:
            idx: Index of the image to retrieve
            
        Returns:
            Transformed image tensor
        """
        # Load image
        img_path = self.image_paths[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
            
        return image


def get_data_loaders(data_dir: str, 
                     batch_size: int = 32, 
                     img_size: Tuple[int, int] = (128, 128),
                     num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and validation.
    
    Args:
        data_dir: Directory containing bike images
        batch_size: Batch size for training
        img_size: Target image size (height, width)
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = BikeDataset(data_dir, img_size=img_size, split='train')
    val_dataset = BikeDataset(data_dir, img_size=img_size, split='val')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Speeds up data transfer to CUDA
        drop_last=True,   # Drop the last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def visualize_augmentations(data_dir: str, img_size: Tuple[int, int] = (128, 128), num_samples: int = 5):
    """
    Utility function to visualize data augmentations.
    This can be used in a Jupyter notebook to check the augmentation effects.
    
    Args:
        data_dir: Directory containing bike images
        img_size: Target image size (height, width)
        num_samples: Number of sample images to visualize
    """
    import matplotlib.pyplot as plt
    
    # Get a few random images
    all_images = []
    valid_extensions = ['.jpg', '.jpeg', '.png']
    for file in os.listdir(data_dir):
        if any(file.lower().endswith(ext) for ext in valid_extensions):
            all_images.append(os.path.join(data_dir, file))
    
    if len(all_images) == 0:
        print("No images found in the directory.")
        return
    
    # Select random images
    random.seed(42)
    sample_images = random.sample(all_images, min(num_samples, len(all_images)))
    
    # Create the augmentation pipeline
    transform = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.3),
    ])
    
    # Visualize original and augmented images
    plt.figure(figsize=(15, 5 * num_samples))
    
    for i, img_path in enumerate(sample_images):
        # Load original image
        original_img = np.array(Image.open(img_path).convert('RGB'))
        
        # Create 3 augmented versions
        augmented_imgs = [transform(image=original_img)["image"] for _ in range(3)]
        
        # Plot original and augmented images
        plt.subplot(num_samples, 4, i * 4 + 1)
        plt.imshow(original_img)
        plt.title(f"Original {i+1}")
        plt.axis('off')
        
        for j, aug_img in enumerate(augmented_imgs):
            plt.subplot(num_samples, 4, i * 4 + j + 2)
            plt.imshow(aug_img)
            plt.title(f"Augmented {i+1}-{j+1}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()
