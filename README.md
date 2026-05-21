# Ship Classification Using Spectrograms and Deep Learning

## Overview

This project focuses on underwater ship classification using spectrogram representations generated from raw `.wav` audio recordings. The workflow converts underwater acoustic signals into image-based spectrograms and trains multiple deep learning models to classify different vessel and noise types.

The project explores:

* Spectrogram generation from underwater acoustic recordings
* CNN and Vision Transformer (ViT) classification models
* Variational Autoencoder (VAE) data augmentation
* Diffusion-based spectrogram generation
* Model comparison using classification metrics and confusion matrices

The notebook was developed and tested in Google Colab using PyTorch and torchvision.

---

# Dataset

The dataset consists of underwater acoustic recordings from four different classes:

| Class | Description      |
| ----- | ---------------- |
| 1     | Kai boat         |
| 2     | Background noise |
| 3     | UUV              |
| 4     | Speedboat        |

Audio recordings are stored as `.wav` files and later converted into spectrogram images for model training.

---

# Project Workflow

## 1. Audio Loading and Extraction

The notebook begins by mounting Google Drive and loading compressed dataset archives.

Zip files are extracted into separate folders for each class:

* `kai`
* `noise`
* `uuv`
* `speedboat`

Folder validation is also performed to verify that all files are loaded correctly before preprocessing begins.

---

## 2. Spectrogram Generation

A reusable spectrogram generation function is implemented using:

* `scipy.signal.spectrogram`
* `numpy`
* `matplotlib`
* `PIL`

The generated spectrograms are saved as image files for later use during model training.

### Spectrogram Parameters

| Parameter         | Value   |
| ----------------- | ------- |
| Window Size       | 1024    |
| Overlap           | 800     |
| FFT Size          | 4096    |
| Minimum Frequency | 10 Hz   |
| Maximum Frequency | 2500 Hz |

Each class receives its own spectrogram output directory:

* `kai_spectrograms`
* `noise_spectrograms`
* `uuv_spectrograms`
* `speedboat_spectrograms`

The notebook also visualizes example spectrograms from each class to verify preprocessing quality.

---

# Data Preparation

The spectrogram dataset is split into:

* Training set
* Validation set

Folder structures are automatically created and organized using class labels compatible with `torchvision.datasets.ImageFolder`.

Image transformations include:

* Resize to 224x224
* Tensor conversion
* Normalization

Separate transforms are also defined for VAE training.

---

# Deep Learning Models

## EfficientNet-B0

A CNN-based classifier built using the pretrained `EfficientNet-B0` architecture from torchvision.

The final classification layer is modified to output predictions for four classes.

### Why EfficientNet?

EfficientNet provides:

* Strong image classification performance
* Lower computational cost compared to larger CNNs
* Good generalization on spectrogram-based datasets

---

## Vision Transformer (ViT)

A Vision Transformer classifier is also implemented using torchvision pretrained weights.

The final classification head is replaced to support the four target classes.

### Why ViT?

ViT models are useful for evaluating transformer-based image classification performance compared to traditional CNN architectures.

---

# Training Pipeline

A reusable training function is implemented to:

* Train models across multiple epochs
* Compute validation metrics
* Store predictions and labels
* Generate evaluation outputs

The following metrics are calculated:

* Accuracy
* Precision
* Recall

Confusion matrices are also generated for performance analysis.

### Training Hyperparameters

| Parameter         | Value |
| ----------------- | ----- |
| Batch Size        | 16    |
| Epochs            | 10    |
| Learning Rate     | 0.001 |
| Number of Classes | 4     |

GPU acceleration is automatically enabled when CUDA is available.

---

# Variational Autoencoder (VAE)

A conditional Variational Autoencoder is implemented to generate synthetic spectrogram images.

The VAE uses:

* Encoder-decoder architecture
* Latent vector sampling
* Class embeddings for conditional generation

Synthetic images are generated for each class and saved into:

`/content/augmented_spectrograms`

The augmented dataset is then combined with the original training data using `ConcatDataset`.

---

# Diffusion Model

A simplified U-Net style diffusion generator is also implemented.

The diffusion model is trained to generate additional synthetic spectrograms for each class.

Generated outputs are saved into:

`/content/diffusion_augmented_spectrograms`

These generated samples are later merged with the original dataset for additional training experiments.

---

# Augmentation Experiments

The notebook evaluates multiple training scenarios:

| Experiment          | Description                                     |
| ------------------- | ----------------------------------------------- |
| Original Dataset    | Training using only original spectrograms       |
| VAE Augmented       | Training using VAE-generated spectrograms       |
| Diffusion Augmented | Training using diffusion-generated spectrograms |

Both EfficientNet and ViT are evaluated under each augmentation strategy.

---

# Results and Observations

## General Findings

* EfficientNet consistently produced the strongest overall performance.
* ViT performance improved with augmentation but remained less stable than EfficientNet.
* VAE augmentation helped increase dataset diversity, but generated samples occasionally introduced noise artifacts.
* Diffusion-generated spectrograms produced more realistic samples compared to the VAE outputs.

---

# Evaluation Outputs

The notebook includes:

* Classification metrics
* Confusion matrices
* Spectrogram visualizations
* Synthetic image visualizations
* Model comparison summaries

These outputs are used to compare:

* CNN vs Transformer performance
* Original vs augmented datasets
* VAE vs diffusion augmentation quality

---

# Libraries and Frameworks

## Core Libraries

* Python
* PyTorch
* torchvision
* NumPy
* SciPy
* Matplotlib
* PIL
* scikit-learn

## Environment

The notebook was primarily designed for:

* Google Colab
* CUDA-enabled GPU execution

---

# File Structure

Example folder structure:

```text
project/
│
├── kai/
├── noise/
├── uuv/
├── speedboat/
│
├── kai_spectrograms/
├── noise_spectrograms/
├── uuv_spectrograms/
├── speedboat_spectrograms/
│
├── augmented_spectrograms/
├── diffusion_augmented_spectrograms/
│
└── Ship_Classification.ipynb
```

---

# Running the Notebook

## Recommended Environment

Google Colab is recommended due to:

* GPU support
* Google Drive integration
* Faster training performance

## Steps

1. Upload dataset zip files to Google Drive.
2. Update dataset paths in the notebook if necessary.
3. Run cells sequentially:

   * Dataset extraction
   * Spectrogram generation
   * Dataset splitting
   * Model training
   * Augmentation experiments
   * Evaluation

---

# Future Improvements

Potential future improvements for this project include:

* Longer model training
* Hyperparameter optimization
* Advanced diffusion architectures
* Additional transformer variants
* Real-time ship detection pipelines
* Improved synthetic sample quality

---

# Summary

This project demonstrates a complete deep learning pipeline for underwater acoustic ship classification using spectrogram-based image representations.

The notebook combines:

* Signal preprocessing
* Computer vision techniques
* CNN and Transformer models
* Generative augmentation methods
* Classification evaluation

The experiments show that EfficientNet performs strongly on spectrogram classification tasks, while synthetic data augmentation using VAEs and diffusion models can improve dataset diversity and help support model training.

