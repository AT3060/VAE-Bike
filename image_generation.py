"""
Image generation module for the Bike VAE project.
This module handles sampling from the latent space and generating new bike images.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Optional
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
import logging

from src.vae_model import VAE


class ImageGenerator:
    """
    Image generator for the VAE model.
    
    Handles sampling from the latent space and generating new bike images.
    
    Attributes:
        model (VAE): Trained VAE model
        device (torch.device): Device to use for generation
        img_size (Tuple[int, int]): Image size (height, width)
        latent_dim (int): Dimension of the latent space
    """
    
    def __init__(self, 
                 model: VAE, 
                 device: torch.device = None,
                 img_size: Tuple[int, int] = (128, 128)):
        """
        Initialize the image generator.
        
        Args:
            model: Trained VAE model
            device: Device to use for generation (CPU or CUDA)
            img_size: Image size (height, width)
        """
        self.model = model
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Move model to device
        self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Store image size and latent dimension
        self.img_size = img_size
        self.latent_dim = model.latent_dim
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('ImageGenerator')
        
        # Create output directory
        os.makedirs('generated_images', exist_ok=True)
    
    def sample_latent_vectors(self, num_samples: int, seed: Optional[int] = None) -> torch.Tensor:
        """
        Sample random vectors from the latent space.
        
        Args:
            num_samples: Number of vectors to sample
            seed: Random seed for reproducibility
            
        Returns:
            Tensor of sampled latent vectors
        """
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Sample from standard normal distribution
        z = torch.randn(num_samples, self.latent_dim).to(self.device)
        
        return z
    
    def generate_images(self, 
                        num_samples: int, 
                        seed: Optional[int] = None,
                        denormalize: bool = True) -> torch.Tensor:
        """
        Generate images by sampling from the latent space.
        
        Args:
            num_samples: Number of images to generate
            seed: Random seed for reproducibility
            denormalize: Whether to denormalize the images
            
        Returns:
            Tensor of generated images
        """
        # Sample latent vectors
        z = self.sample_latent_vectors(num_samples, seed)
        
        # Generate images
        with torch.no_grad():
            images = self.model.decoder(z)
        
        # Denormalize images if requested
        if denormalize:
            # Assuming normalization with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            
            images = images * std + mean
            images = torch.clamp(images, 0, 1)
        
        return images
    
    def interpolate_images(self, 
                           img1: torch.Tensor, 
                           img2: torch.Tensor, 
                           steps: int = 10,
                           denormalize: bool = True) -> torch.Tensor:
        """
        Interpolate between two images in the latent space.
        
        Args:
            img1: First input image
            img2: Second input image
            steps: Number of interpolation steps
            denormalize: Whether to denormalize the images
            
        Returns:
            Tensor of interpolated images
        """
        # Move images to device
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        
        # Generate interpolated images
        with torch.no_grad():
            interp_images = self.model.interpolate(img1, img2, steps)
        
        # Denormalize images if requested
        if denormalize:
            # Assuming normalization with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            
            interp_images = interp_images * std + mean
            interp_images = torch.clamp(interp_images, 0, 1)
        
        return interp_images
    
    def save_images(self, 
                    images: torch.Tensor, 
                    filename: str, 
                    nrow: int = 8,
                    padding: int = 2):
        """
        Save generated images to a file.
        
        Args:
            images: Tensor of images to save
            filename: Output filename
            nrow: Number of images per row in the grid
            padding: Padding between images
        """
        # Create grid of images
        grid = make_grid(images, nrow=nrow, padding=padding, normalize=False)
        
        # Save grid
        save_image(grid, filename)
        
        self.logger.info(f"Saved {len(images)} images to {filename}")
    
    def generate_and_save_images(self, 
                                 num_samples: int, 
                                 filename: str = 'generated_images/generated.png',
                                 nrow: int = 8,
                                 seed: Optional[int] = None):
        """
        Generate and save images in one step.
        
        Args:
            num_samples: Number of images to generate
            filename: Output filename
            nrow: Number of images per row in the grid
            seed: Random seed for reproducibility
        """
        # Generate images
        images = self.generate_images(num_samples, seed)
        
        # Save images
        self.save_images(images, filename, nrow)
    
    def generate_variations(self, 
                            img: torch.Tensor, 
                            num_variations: int = 10,
                            std_dev: float = 0.1,
                            filename: str = 'generated_images/variations.png',
                            seed: Optional[int] = None):
        """
        Generate variations of an input image by adding noise in the latent space.
        
        Args:
            img: Input image
            num_variations: Number of variations to generate
            std_dev: Standard deviation of the noise
            filename: Output filename
            seed: Random seed for reproducibility
        """
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Move image to device
        img = img.to(self.device)
        
        # Encode image to get latent representation
        with torch.no_grad():
            mu, log_var = self.model.encoder(img.unsqueeze(0))
        
        # Create variations by adding noise to the latent vector
        variations = []
        for _ in range(num_variations):
            # Add random noise
            noise = torch.randn_like(mu) * std_dev
            z = mu + noise
            
            # Decode to get variation
            with torch.no_grad():
                variation = self.model.decoder(z)
            
            variations.append(variation)
        
        # Concatenate variations
        variations = torch.cat(variations, dim=0)
        
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        variations = variations * std + mean
        variations = torch.clamp(variations, 0, 1)
        
        # Save variations
        self.save_images(variations, filename)
    
    def explore_latent_space(self, 
                             num_samples: int = 100,
                             perplexity: int = 30,
                             filename: str = 'generated_images/latent_space.png',
                             seed: Optional[int] = None):
        """
        Explore the latent space by generating samples and visualizing them with t-SNE.
        
        Args:
            num_samples: Number of samples to generate
            perplexity: Perplexity parameter for t-SNE
            filename: Output filename
            seed: Random seed for reproducibility
        """
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.error("scikit-learn is required for t-SNE visualization")
            return
        
        # Sample latent vectors
        z = self.sample_latent_vectors(num_samples, seed)
        
        # Generate images
        with torch.no_grad():
            images = self.model.decoder(z)
        
        # Denormalize images
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        images = images * std + mean
        images = torch.clamp(images, 0, 1)
        
        # Convert latent vectors to numpy for t-SNE
        z_np = z.cpu().numpy()
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed)
        z_tsne = tsne.fit_transform(z_np)
        
        # Plot t-SNE visualization
        plt.figure(figsize=(10, 10))
        plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c='blue', alpha=0.5)
        plt.title('t-SNE visualization of latent space')
        plt.savefig(filename)
        plt.close()
        
        # Save a subset of images
        subset_size = min(36, num_samples)
        subset_indices = np.random.choice(num_samples, subset_size, replace=False)
        subset_images = images[subset_indices]
        
        self.save_images(subset_images, filename.replace('.png', '_samples.png'), nrow=6)
    
    def generate_image_grid(self, 
                            dim1_range: Tuple[float, float] = (-3.0, 3.0),
                            dim2_range: Tuple[float, float] = (-3.0, 3.0),
                            grid_size: int = 8,
                            latent_dims: Tuple[int, int] = (0, 1),
                            filename: str = 'generated_images/latent_grid.png'):
        """
        Generate a grid of images by varying two dimensions of the latent space.
        
        Args:
            dim1_range: Range for the first dimension
            dim2_range: Range for the second dimension
            grid_size: Size of the grid (grid_size x grid_size)
            latent_dims: Indices of the two latent dimensions to vary
            filename: Output filename
        """
        # Create grid of latent vectors
        dim1_values = np.linspace(dim1_range[0], dim1_range[1], grid_size)
        dim2_values = np.linspace(dim2_range[0], dim2_range[1], grid_size)
        
        # Create empty grid for images
        images = []
        
        # Generate images for each grid point
        for i, val1 in enumerate(dim1_values):
            row_images = []
            for j, val2 in enumerate(dim2_values):
                # Create latent vector with zeros
                z = torch.zeros(1, self.latent_dim).to(self.device)
                
                # Set the two dimensions we're varying
                z[0, latent_dims[0]] = val1
                z[0, latent_dims[1]] = val2
                
                # Generate image
                with torch.no_grad():
                    img = self.model.decoder(z)
                
                # Denormalize
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
                
                img = img * std + mean
                img = torch.clamp(img, 0, 1)
                
                row_images.append(img)
            
            # Concatenate row
            row = torch.cat(row_images, dim=0)
            images.append(row)
        
        # Concatenate all rows
        grid = torch.cat(images, dim=0)
        
        # Save grid
        self.save_images(grid, filename, nrow=grid_size)
    
    def generate_animation_frames(self,
                                  start_vector: torch.Tensor,
                                  end_vector: torch.Tensor,
                                  num_frames: int = 30,
                                  output_dir: str = 'generated_images/animation'):
        """
        Generate frames for an animation by interpolating between two latent vectors.
        
        Args:
            start_vector: Starting latent vector
            end_vector: Ending latent vector
            num_frames: Number of frames to generate
            output_dir: Output directory for frames
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate frames
        for i in range(num_frames):
            # Interpolate between start and end vectors
            alpha = i / (num_frames - 1)
            z = (1 - alpha) * start_vector + alpha * end_vector
            
            # Generate image
            with torch.no_grad():
                img = self.model.decoder(z)
            
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
            
            # Save frame
            save_image(img, os.path.join(output_dir, f'frame_{i:03d}.png'))
        
        self.logger.info(f"Generated {num_frames} animation frames in {output_dir}")


def load_and_preprocess_image(image_path: str, img_size: Tuple[int, int] = (128, 128)) -> torch.Tensor:
    """
    Load and preprocess an image for the VAE model.
    
    Args:
        image_path: Path to the image file
        img_size: Target image size (height, width)
        
    Returns:
        Preprocessed image tensor
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Resize image
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations
    img_tensor = transform(img)
    
    return img_tensor
