# Galaxy Morphology Classification with Vision Transformer (ViT)

## Overview
This project implements a galaxy morphology classification system using the Vision Transformer (ViT) model to classify galaxies into 10 distinct morphological categories from the Galaxy10 DECALS dataset. The system leverages GPU acceleration, data augmentation, and advanced visualization techniques to achieve robust performance. The notebook (`cvt-galaxy-morphology-classification.ipynb`) runs efficiently on a single GPU (e.g., NVIDIA T4).

## Key Features
- **Model**: Vision Transformer (ViT-B/16) fine-tuned for galaxy morphology classification.
- **Dataset**: Galaxy10 DECALS, containing 15,962 training and 1,774 test images across 10 classes (e.g., Disturbed, Merging, Round Smooth, Barred Spiral).
- **Performance**: Achieves high accuracy with a test set evaluation, optimized via data augmentation and class-balanced sampling.
- **Visualization**: Includes confusion matrices, class distribution plots, and sample image displays using Matplotlib and Seaborn.
- **Tools**: Built with PyTorch, Transformers, Datasets, Scikit-learn, Matplotlib, and Weights & Biases (WandB) for logging.

## Key Findings
- **Accuracy**: The ViT model demonstrates strong performance in distinguishing complex galaxy morphologies, particularly for classes like Barred Spiral and Round Smooth Galaxies.
- **Class Imbalance**: Addressed through data augmentation and balanced sampling, improving performance on underrepresented classes (e.g., Cigar-Shaped Smooth Galaxies).
- **Visual Insights**: Confusion matrices reveal misclassifications primarily between similar morphologies (e.g., Unbarred Tight vs. Loose Spiral Galaxies).
- **Runtime**: Total notebook execution time is approximately 600 seconds on a T4 GPU, with inference times of ~0.1 seconds per image.

## Methodology
1. **Data Preparation**: Loads the Galaxy10 DECALS dataset using the `datasets` library, with images resized to 256x256 pixels and labels cast to `ClassLabel` for 10 morphological classes.
2. **Data Augmentation**: Applies random flips, rotations, and color jitter to enhance model generalization.
3. **Model Training**: Fine-tunes ViT-B/16 with a custom classification head, using cross-entropy loss and AdamW optimizer.
4. **Evaluation**: Computes accuracy, precision, recall, and F1-score on the test set, visualized via confusion matrices.
5. **Visualization**: Generates plots for class distribution, sample images, and performance metrics.

## Requirements
- **Hardware**: GPU (e.g., NVIDIA T4) for training and inference.
- **Software**:
  - Python 3.10
  - PyTorch, Transformers, Datasets, Scikit-learn, Matplotlib, Seaborn, WandB
  - Install dependencies: `pip install datasets transformers accelerate torch torchvision scikit-learn matplotlib wandb`

## Usage
1. Clone the repository and navigate to the project directory.
2. Install required packages (see Requirements).
3. Run the Jupyter notebook `cvt-galaxy-morphology-classification.ipynb`.
4. Monitor training progress and visualizations via WandB integration.

## Visualizations
- **Sample Images**: Displays example galaxies for each class.
- **Confusion Matrix**: Visualizes model performance, identifying misclassification patterns.
- ![classes](https://i.postimg.cc/WpQgvq6z/Untitled.png)
- ![Confusion_Matrix](https://i.postimg.cc/CLQtYNrD/Untitlaed.png)

## Future Work
- Experiment with larger ViT variants (e.g., ViT-L/16) for improved accuracy.
- Incorporate advanced augmentation techniques (e.g., CutMix, MixUp).
- Extend the dataset with additional galaxy images for better generalization.

---

*This project provides a scalable framework for galaxy morphology classification, combining state-of-the-art deep learning with comprehensive visualizations for astronomical research.*
