# Bike Image Variational Autoencoder

This project implements a Variational Autoencoder (VAE) for generating new bike images from a learned latent space. The implementation includes data augmentation, CUDA support, hyperparameter optimization with Optuna, and progress tracking with tqdm.

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
├── GITHUB_README.md        # GitHub repository instructions
└── README.md               # This file
```

## Features

- **Variational Autoencoder**: Implementation of a VAE architecture specifically designed for bike images
- **Data Augmentation**: Enhanced training with augmented images using the albumentations library
- **CUDA Support**: GPU acceleration for faster training
- **Hyperparameter Optimization**: Automated tuning using Optuna
- **Progress Tracking**: Training progress visualization with tqdm
- **Image Generation**: Tools for sampling from the latent space to generate new bike images

## Getting Started

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Place your bike images in the `data` directory

3. Explore the Jupyter notebooks in the `notebooks` directory:
   - `01_data_exploration.ipynb`: Explore and visualize the bike image dataset
   - `02_vae_training.ipynb`: Train the VAE model with hyperparameter optimization
   - `03_image_generation.ipynb`: Generate new bike images from the learned latent space

## Running on a Computer Cluster

This implementation is designed to be run on a high-performance computing cluster. The code includes configurations for utilizing CUDA-enabled GPUs efficiently.

To run on a cluster:

1. Clone this repository to your cluster environment
2. Install the required dependencies
3. Use the Jupyter notebooks or command-line interface to train and generate images

## Command-Line Interface

The project provides a command-line interface for training, optimization, and generation:

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

## GitHub Repository

For detailed instructions on setting up and using this project in a GitHub repository, please refer to the `GITHUB_README.md` file.

## Code Documentation

All code is thoroughly documented with docstrings and comments explaining the implementation details. The Jupyter notebooks provide step-by-step examples of how to use the code.

## License

[MIT License](LICENSE)
