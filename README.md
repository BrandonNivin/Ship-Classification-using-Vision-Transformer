# Ship Classification Using Spectrograms and Deep Learning

## Overview

This project focuses on underwater acoustic ship classification using spectrogram representations generated from raw `.wav` audio recordings. The workflow converts underwater audio into spectrogram images and trains multiple deep learning models to classify different vessel and noise types.

The notebook includes:

* Spectrogram generation from `.wav` files
* EfficientNet and Vision Transformer classification
* VAE-based augmentation
* Diffusion-based spectrogram generation
* Model comparison and evaluation

The project was developed and tested primarily in Google Colab using PyTorch.

---

# Dataset Access

The dataset files are too large to upload directly to GitHub.

Download the dataset zip files here:

[Google Drive Dataset Folder](https://drive.google.com/drive/folders/1aV_n8aEMuKqn92XevdWePbSL4pgiNd9J?usp=drive_link)

After downloading, upload the zip files into your own Google Drive before running the notebook.

---

# Expected Dataset Structure

The notebook expects the dataset zip files to be available in Google Drive.

Example structure:

```text
MyDrive/
└── ShipClassificationDataset/
    ├── kai.zip
    ├── noise.zip
    ├── uuv.zip
    └── speedboat.zip
```

---

# Updating Dataset Paths

Inside the notebook, update the zip file paths to match your Google Drive folder location.

Locate the dataset extraction section and modify the paths similar to this:

```python
kai_zip = "/content/drive/MyDrive/ShipClassificationDataset/kai.zip"
noise_zip = "/content/drive/MyDrive/ShipClassificationDataset/noise.zip"
uuv_zip = "/content/drive/MyDrive/ShipClassificationDataset/uuv.zip"
speedboat_zip = "/content/drive/MyDrive/ShipClassificationDataset/speedboat.zip"
```

If your dataset folder has a different name or location, update the paths accordingly.

---

# Running the Notebook

## Recommended Environment

Google Colab is recommended because the notebook was developed and tested there with GPU acceleration enabled.

---

## Setup Steps

### 1. Mount Google Drive

Run the Google Drive mount cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### 2. Upload Dataset Files

Upload all dataset zip files into your chosen Google Drive folder.

---

### 3. Update Dataset Paths

Modify the dataset zip paths in the notebook to match your Google Drive folder structure.

---

### 4. Run Notebook Cells Sequentially

Run the notebook from top to bottom:

1. Dataset extraction
2. Spectrogram generation
3. Dataset splitting
4. Model training
5. VAE augmentation
6. Diffusion augmentation
7. Evaluation and visualization

---

# Dataset Classes

The dataset contains four classes:

| Class | Description      |
| ----- | ---------------- |
| 1     | Kai boat         |
| 2     | Background noise |
| 3     | UUV              |
| 4     | Speedboat        |

---

# Spectrogram Generation

The notebook converts raw audio recordings into spectrogram images using:

* `scipy.signal.spectrogram`
* `numpy`
* `matplotlib`

Generated spectrograms are saved into separate class directories for training.

### Spectrogram Parameters

| Parameter       | Value           |
| --------------- | --------------- |
| Window Size     | 1024            |
| Overlap         | 800             |
| FFT Size        | 4096            |
| Frequency Range | 10 Hz – 2500 Hz |

---

# Deep Learning Models

## EfficientNet-B0

A pretrained EfficientNet-B0 CNN model is used for spectrogram classification.

The final classification layer is modified for four output classes.

---

## Vision Transformer (ViT)

A pretrained Vision Transformer model is also implemented to compare transformer-based image classification performance against CNNs.

---

# Data Augmentation

## Variational Autoencoder (VAE)

A conditional VAE is used to generate synthetic spectrogram images for each class.

Generated samples are combined with the original dataset for additional training experiments.

---

## Diffusion Model

A simplified U-Net style diffusion model is also implemented to generate synthetic spectrograms.

These generated samples are used for additional augmentation experiments and model evaluation.

---

# Training Configuration

| Parameter         | Value |
| ----------------- | ----- |
| Batch Size        | 16    |
| Epochs            | 10    |
| Learning Rate     | 0.001 |
| Number of Classes | 4     |

CUDA is automatically enabled when available.

---

# Evaluation

The notebook includes:

* Accuracy
* Precision
* Recall
* Confusion matrices
* Spectrogram visualizations
* Synthetic image visualizations

The goal was to compare:

* CNN vs Transformer performance
* Original vs augmented datasets
* VAE vs diffusion augmentation quality

---

# Libraries Used

* Python
* PyTorch
* torchvision
* NumPy
* SciPy
* Matplotlib
* PIL
* scikit-learn

---

# File Structure

```text
project/
│
├── Ship_Classification.ipynb
├── README.md
│
├── kai_spectrograms/
├── noise_spectrograms/
├── uuv_spectrograms/
├── speedboat_spectrograms/
│
├── augmented_spectrograms/
└── diffusion_augmented_spectrograms/
```

---

# Summary

This project builds a complete deep learning pipeline for underwater ship classification using spectrogram-based image representations.

The notebook combines signal preprocessing, computer vision models, and generative augmentation methods to evaluate different approaches for underwater acoustic classification tasks.
