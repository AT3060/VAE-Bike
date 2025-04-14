"""
VAE architecture module for the Bike VAE project.
This module defines the encoder, decoder, and complete VAE model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any, Optional


class Encoder(nn.Module):
    """
    Encoder network for the VAE.
    
    Transforms input images into parameters of the latent distribution (mean and log variance).
    
    Attributes:
        img_channels (int): Number of input image channels (3 for RGB)
        hidden_dims (List[int]): List of hidden dimensions for the convolutional layers
        latent_dim (int): Dimension of the latent space
    """
    
    def __init__(self, 
                 img_channels: int = 3, 
                 hidden_dims: List[int] = None, 
                 latent_dim: int = 128,
                 img_size: Tuple[int, int] = (128, 128)):
        """
        Initialize the encoder network.
        
        Args:
            img_channels: Number of input image channels (3 for RGB)
            hidden_dims: List of hidden dimensions for the convolutional layers
            latent_dim: Dimension of the latent space
            img_size: Input image size (height, width)
        """
        super(Encoder, self).__init__()
        
        # Set default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        self.img_channels = img_channels
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        
        # Build encoder convolutional layers
        modules = []
        in_channels = img_channels
        
        # Create a sequence of Conv2d -> BatchNorm -> LeakyReLU blocks
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, 
                              out_channels=h_dim,
                              kernel_size=3, 
                              stride=2, 
                              padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2)
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Calculate the size of the feature maps after the convolutional layers
        # This is needed to determine the input size for the fully connected layers
        h, w = img_size
        for _ in range(len(hidden_dims)):
            h = (h + 2 - 3) // 2 + 1  # Formula: (H + 2*padding - kernel_size) // stride + 1
            w = (w + 2 - 3) // 2 + 1
        
        self.feature_size = hidden_dims[-1] * h * w
        
        # Fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(self.feature_size, latent_dim)
        self.fc_log_var = nn.Linear(self.feature_size, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input image tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Tuple of (mean, log_variance) of the latent distribution
        """
        # Pass input through convolutional layers
        x = self.encoder(x)
        
        # Flatten the output for the fully connected layers
        x = torch.flatten(x, start_dim=1)
        
        # Get mean and log variance
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        
        return mu, log_var


class Decoder(nn.Module):
    """
    Decoder network for the VAE.
    
    Transforms samples from the latent space back into images.
    
    Attributes:
        img_channels (int): Number of output image channels (3 for RGB)
        hidden_dims (List[int]): List of hidden dimensions for the convolutional layers (in reverse)
        latent_dim (int): Dimension of the latent space
    """
    
    def __init__(self, 
                 img_channels: int = 3, 
                 hidden_dims: List[int] = None, 
                 latent_dim: int = 128,
                 img_size: Tuple[int, int] = (128, 128)):
        """
        Initialize the decoder network.
        
        Args:
            img_channels: Number of output image channels (3 for RGB)
            hidden_dims: List of hidden dimensions for the convolutional layers
            latent_dim: Dimension of the latent space
            img_size: Target image size (height, width)
        """
        super(Decoder, self).__init__()
        
        # Set default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        self.img_channels = img_channels
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        
        # Calculate the size of the feature maps after the convolutional layers in the encoder
        # This is needed to determine the input size for the first transposed convolutional layer
        h, w = img_size
        for _ in range(len(hidden_dims)):
            h = (h + 2 - 3) // 2 + 1
            w = (w + 2 - 3) // 2 + 1
        
        self.feature_size = hidden_dims[-1] * h * w
        self.init_h, self.init_w = h, w
        
        # Fully connected layer to convert latent vector to feature map
        self.decoder_input = nn.Linear(latent_dim, self.feature_size)
        
        # Build decoder transposed convolutional layers
        modules = []
        
        # Reverse the hidden dimensions for the decoder
        hidden_dims = hidden_dims[::-1]
        
        # Create a sequence of ConvTranspose2d -> BatchNorm -> LeakyReLU blocks
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                      hidden_dims[i + 1],
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(0.2)
                )
            )
        
        # Final layer to produce the output image
        self.decoder = nn.Sequential(*modules)
        
        # Final transposed convolution to get the right number of channels
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                              hidden_dims[-1],
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dims[-1], 
                     out_channels=img_channels,
                     kernel_size=3, 
                     padding=1),
            nn.Tanh()  # Output values between -1 and 1
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            z: Latent vector of shape [batch_size, latent_dim]
            
        Returns:
            Reconstructed image tensor
        """
        # Convert latent vector to feature map
        z = self.decoder_input(z)
        
        # Reshape to feature map
        z = z.view(-1, self.hidden_dims[-1], self.init_h, self.init_w)
        
        # Pass through transposed convolutional layers
        z = self.decoder(z)
        
        # Final layer to get the output image
        x_recon = self.final_layer(z)
        
        return x_recon


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model.
    
    Combines the encoder and decoder networks and implements the VAE loss function.
    
    Attributes:
        encoder (Encoder): Encoder network
        decoder (Decoder): Decoder network
        latent_dim (int): Dimension of the latent space
    """
    
    def __init__(self, 
                 img_channels: int = 3, 
                 hidden_dims: List[int] = None, 
                 latent_dim: int = 128,
                 img_size: Tuple[int, int] = (128, 128),
                 kl_weight: float = 0.005):
        """
        Initialize the VAE model.
        
        Args:
            img_channels: Number of image channels (3 for RGB)
            hidden_dims: List of hidden dimensions for the convolutional layers
            latent_dim: Dimension of the latent space
            img_size: Input/output image size (height, width)
            kl_weight: Weight for the KL divergence term in the loss function
        """
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        
        # Create encoder and decoder
        self.encoder = Encoder(img_channels, hidden_dims, latent_dim, img_size)
        self.decoder = Decoder(img_channels, hidden_dims, latent_dim, img_size)
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from the latent distribution.
        
        Args:
            mu: Mean of the latent distribution
            log_var: Log variance of the latent distribution
            
        Returns:
            Sampled latent vector
        """
        # Calculate standard deviation from log variance
        std = torch.exp(0.5 * log_var)
        
        # Sample from standard normal distribution
        eps = torch.randn_like(std)
        
        # Reparameterization trick: z = mu + std * eps
        z = mu + std * eps
        
        return z
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            x: Input image tensor
            
        Returns:
            Tuple of (reconstructed_image, input_image, mean, log_variance)
        """
        # Encode input to get latent distribution parameters
        mu, log_var = self.encoder(x)
        
        # Sample from the latent distribution using the reparameterization trick
        z = self.reparameterize(mu, log_var)
        
        # Decode the latent vector to get the reconstructed image
        x_recon = self.decoder(z)
        
        return x_recon, x, mu, log_var
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample images from the latent space.
        
        Args:
            num_samples: Number of images to sample
            device: Device to use for sampling
            
        Returns:
            Tensor of sampled images
        """
        # Sample from standard normal distribution
        z = torch.randn(num_samples, self.latent_dim).to(device)
        
        # Decode the latent vectors to get images
        samples = self.decoder(z)
        
        return samples
    
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """
        Interpolate between two images in the latent space.
        
        Args:
            x1: First input image
            x2: Second input image
            steps: Number of interpolation steps
            
        Returns:
            Tensor of interpolated images
        """
        # Encode both images to get their latent representations
        mu1, log_var1 = self.encoder(x1.unsqueeze(0))
        mu2, log_var2 = self.encoder(x2.unsqueeze(0))
        
        # Create interpolation steps in the latent space
        alphas = torch.linspace(0, 1, steps).to(mu1.device)
        
        # Interpolate between the two latent vectors
        z_interp = torch.zeros(steps, self.latent_dim).to(mu1.device)
        for i, alpha in enumerate(alphas):
            z_interp[i] = (1 - alpha) * mu1 + alpha * mu2
        
        # Decode the interpolated latent vectors
        interp_images = self.decoder(z_interp)
        
        return interp_images
    
    def calculate_loss(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate the VAE loss function.
        
        The loss consists of two terms:
        1. Reconstruction loss (how well the decoder reconstructs the input)
        2. KL divergence (how close the latent distribution is to a standard normal)
        
        Args:
            recon_x: Reconstructed image
            x: Original input image
            mu: Mean of the latent distribution
            log_var: Log variance of the latent distribution
            
        Returns:
            Dictionary containing the total loss and its components
        """
        # Reconstruction loss (mean squared error)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        # For a normal distribution with mean mu and std exp(log_var/2), the KL divergence with
        # a standard normal distribution is:
        # KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        return {
            'loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss
        }
