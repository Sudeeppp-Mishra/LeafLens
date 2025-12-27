# 🌿 LeafLens

**AI-Based Plant Disease Detection and Recommendation System**

## Overview

LeafLens is an AI-powered desktop application that detects plant leaf diseases using deep learning–based image classification. The system identifies diseases from leaf images and provides actionable recommendations along with audio feedback to support farmers and agricultural users.

## Features

- CNN-based plant disease classification
- ResNet18 model trained on PlantVillage dataset
- Displays disease name and prediction confidence
- Text-to-Speech (TTS) for audio output
- Rule-based recommendations (treatment, prevention, urgency)
- Offline desktop application using PySide6

## Dataset

- **PlantVillage Dataset**
- Image classification format (no bounding boxes)
- Contains healthy and diseased leaf images across multiple crops

## Model

- Architecture: ResNet18 (Pretrained)
- Framework: PyTorch
- Input size: 224 × 224 RGB images

## Technology Stack

- Python
- PyTorch, Torchvision
- PySide6
- OpenCV, PIL, NumPy
- pyttsx3 (TTS)

## Status

🚧 Project Under Construction
