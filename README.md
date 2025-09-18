# Stealthy Hijacking: A Semi-Honest Reconstruction Attack with Adversarially Perturbed Gradients in Split Learning

## Introduction
This repository contains the official implementation of SAM (Stealthy Adversarial Mimicry), a novel data reconstruction attack in Split Learning. SAM combines the strengths of active and passive attacks, achieving high reconstruction quality while maintaining stealth.

## Supported Datasets
CIFAR-10, EMNIST, Fashion-MNIST, MNIST, STL-10, TinyImageNet, CelebA, UTKFace

## Installation

1. Clone the repository.

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Execution

1. Run SAM:
   ```sh
   python main.py
   ```

## Notes
- Ensure you have the required datasets downloaded and placed in the appropriate directories.
- For CelebA, please manually download the dataset to the `./data/celeba` directory.
