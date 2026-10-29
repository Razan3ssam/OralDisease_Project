# Oral Disease Classification

![Python](https://img.shields.io/badge/Python-3.10--3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![Model](https://img.shields.io/badge/Model-EfficientNetB0-purple)

A computer vision project that classifies oral images into six conditions using deep learning and transfer learning. It includes data preprocessing, model comparison, hyperparameter tuning, evaluation, and deployment using FastAPI.

**Disclaimer:** This project is for educational purposes only and is not intended for medical diagnosis.

## Classes

- Calculus
- Caries
- Gingivitis
- Hypodontia
- Tooth Discoloration
- Ulcer

## Dataset

- **Source:** [Oral Diseases Dataset – Kaggle](https://www.kaggle.com/datasets/salmansajid05/oral-diseases)
- **Images:** 7,462
- **Classes:** 6
- **Test set:** 1,052 images

The dataset was cleaned to remove conflicting labels, and a fixed train/validation/test split was used for consistent model comparison.

## Methodology

- Data cleaning and preprocessing
- Custom CNN baseline
- Transfer learning using ResNet50V2, EfficientNetB0, and MobileNetV2
- Two-stage training and fine-tuning
- Hyperparameter tuning using validation Macro-F1
- Evaluation on a held-out test set
- Deployment using FastAPI

## Results

| Model | Test Accuracy | Macro-F1 |
|---|---:|---:|
| **EfficientNetB0** | **83.2%** | **0.794** |
| MobileNetV2 | 81.4% | 0.783 |
| ResNet50V2 | 77.8% | 0.720 |
| Custom CNN | — | ~0.62 |

EfficientNetB0 achieved the best overall test performance, while MobileNetV2 offered faster inference and fewer parameters.

## Features

- Six-class image classification
- Prediction confidence and class probabilities
- Web interface for image uploads
- Low-confidence warning for predictions below 50%
- REST API with interactive Swagger documentation
- Input validation and file-size limits

## Getting Started

### Requirements

- Python 3.10–3.12
- pip

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Razan3ssam/OralDisease_Project.git
cd OralDisease_Project
```

2. Download `tuned_EfficientNetB0.keras` from the [Releases](https://github.com/Razan3ssam/OralDisease_Project/releases) page and place it inside the `models/` directory.

3. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

4. Start the application:

```bash
uvicorn main:app --reload
```

5. Open the application at http://127.0.0.1:8000.

API documentation: http://127.0.0.1:8000/docs

## Project Structure

```text
OralDisease_Project/
├── main.py
├── class_names.json
├── requirements.txt
├── models/
│   └── tuned_EfficientNetB0.keras
└── static/
    ├── index.html
    └── image.jpeg
```

## Limitations

- Performance has not been validated on external clinical datasets.
- Class imbalance and near-duplicate images may affect evaluation results.
- Calculus has limited test samples and lower classification performance.
- The model may produce predictions for non-oral images.
- Confidence scores do not guarantee prediction correctness.

The model requires further validation before any potential clinical application.

## Future Improvements

- Model ensembling
- Out-of-distribution detection
- Confidence calibration
- Grad-CAM explainability
- Evaluation across multiple random seeds
- Docker and cloud deployment

## Links

- [GitHub Repository](https://github.com/Razan3ssam/OralDisease_Project)
- [Kaggle Notebook](https://www.kaggle.com/code/razan3ssam/oraldisease-project)
- [Dataset](https://www.kaggle.com/datasets/salmansajid05/oral-diseases)

## Author

**Razan Essam**

[GitHub](https://github.com/Razan3ssam)
