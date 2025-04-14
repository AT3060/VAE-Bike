"""
Bike VAE Project - README

This file provides instructions for setting up and using the project in a GitHub repository.
"""

# Bike Image Variational Autoencoder

This repository contains a complete implementation of a Variational Autoencoder (VAE) for generating new bike images from a learned latent space. The model is designed to work with a dataset of bike images and includes features for data augmentation, CUDA acceleration, and hyperparameter optimization.

## Setup Instructions

1. Clone this repository:
```bash
git clone https://github.com/yourusername/bike-vae.git
cd bike-vae
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
   - Place your bike images in the `data` directory
   - The code expects 411 bike images as mentioned in the original requirements

## Project Structure

```
bike_vae_project/
├── data/                   # Data directory (place your bike images here)
├── notebooks/              # Jupyter notebooks for training and generation
│   ├── 01_data_exploration.ipynb
│   ├── 02_vae_training.ipynb
│   ├── 03_image_generation.ipynb
├── src/                    # Source code for the VAE implementation
│   ├── data_loading.py     # Data loading and augmentation
│   ├── vae_model.py        # VAE architecture
│   ├── training.py         # Training loop with CUDA support
│   ├── hyperparameter_optimization.py  # Optuna integration
│   ├── image_generation.py # Image generation utilities
│   ├── main.py             # Command-line interface
├── requirements.txt        # Required Python packages
└── README.md               # Project documentation
```

## Features

- **Variational Autoencoder**: Implementation of a VAE architecture specifically designed for bike images
- **Data Augmentation**: Enhanced training with augmented images using the albumentations library
- **CUDA Support**: GPU acceleration for faster training
- **Hyperparameter Optimization**: Automated tuning using Optuna
- **Progress Tracking**: Training progress visualization with tqdm
- **Image Generation**: Tools for sampling from the latent space to generate new bike images

## Usage

### Using Jupyter Notebooks

The easiest way to use this project is through the provided Jupyter notebooks:

1. Start Jupyter:
```bash
jupyter notebook
```

2. Navigate to the `notebooks` directory and open the notebooks in order:
   - `01_data_exploration.ipynb`: Explore and visualize the bike image dataset
   - `02_vae_training.ipynb`: Train the VAE model with hyperparameter optimization
   - `03_image_generation.ipynb`: Generate new bike images from the learned latent space

### Using Command Line Interface

The project also provides a command-line interface for training, optimization, and generation:

1. Train the VAE model:
```bash
python src/main.py train --data_dir data --epochs 50
```

2. Optimize hyperparameters:
```bash
python src/main.py optimize --data_dir data --trials 20
```

3. Generate new images:
```bash
python src/main.py generate --model_path output/checkpoints/best_model.pth --num_images 16
```

## Running on a Computer Cluster

This implementation is designed to be run on a high-performance computing cluster. The code includes configurations for utilizing CUDA-enabled GPUs efficiently.

To run on a cluster:

1. Make sure CUDA is available on the cluster
2. Adjust batch size and other parameters based on available GPU memory
3. Use the command-line interface for headless execution

Example cluster submission script (adjust according to your cluster's job scheduler):

```bash
#!/bin/bash
#SBATCH --job-name=bike_vae
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=16G

module load cuda/11.3
module load python/3.8

cd $SLURM_SUBMIT_DIR
python src/main.py train --data_dir data --epochs 100 --batch_size 64
```

## Customization

You can customize various aspects of the model:

- **Image Size**: Adjust the `img_size` parameter (default: 128x128)
- **Latent Dimension**: Change the `latent_dim` parameter (default: 128)
- **Network Architecture**: Modify the `hidden_dims` list in the VAE model
- **KL Weight**: Adjust the `kl_weight` parameter to balance reconstruction vs. KL loss

## License

[MIT License](LICENSE)
