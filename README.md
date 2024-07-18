# DermAssist: Skin Disease Classification with MobileNetV2

## Overview
DermAssist is an advanced machine learning project focused on classifying various skin diseases using state-of-the-art deep learning techniques. This project leverages the power of transfer learning with the MobileNetV2 architecture to achieve high accuracy in skin disease classification. 

## Table of Contents
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Preprocessing](#preprocessing)
- [Models Used](#models-used)
  - [MobileNetV2](#mobilenetv2)
  - [Other Models](#other-models)
- [Training Process](#training-process)
- [Results](#results)
- [Usage](#usage)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## Project Structure
DermAssist/
│
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── training.ipynb
│   └── evaluation.ipynb
│
├── models/
│   ├── mobilenetv2_model.keras
│   ├── ...
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
│
├── README.md
│
└── requirements.txt


## Datasets
The dataset used in this project consists of images of various skin diseases. The dataset is split into training, validation, and testing sets. Each image is labeled with the corresponding skin disease class.

You can download the dataset from the following link:
- [Skin Diseases Image Dataset](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset)

Ensure that you place the dataset in the `data/` directory after downloading and extracting it.

## Preprocessing
The preprocessing steps include:
- Image resizing to 224x224 pixels.
- Data augmentation to improve model generalization.
- Normalization of pixel values.

## Models Used
### MobileNetV2
MobileNetV2 is a lightweight deep learning model designed for mobile and embedded vision applications. It is used here due to its balance between accuracy and computational efficiency.

### Other Models
In addition to MobileNetV2, other models were experimented with to compare their performance:
- **ResNet50**: A deeper network that showed promising results but required more computational resources.
- **VGG16**: Known for its simplicity and effectiveness, but it was not as efficient as MobileNetV2 in this task.

## Training Process
The training process involves:
- Freezing the base layers of the pre-trained MobileNetV2 model.
- Adding custom dense layers for classification.
- Compiling the model with the Adam optimizer and categorical cross-entropy loss.
- Training the model with early stopping and learning rate reduction callbacks.

## Results
The best results were achieved using MobileNetV2 with the following performance:
- **Accuracy**: 70%
- **Validation Accuracy**: Consistently high with minimal overfitting.

### Other Models' Performance
- **ResNet50**: Accuracy ~62%
- **VGG16**: Accuracy ~65%

## Usage
To use this project, follow these steps:
1. Clone the repository.
2. Install the required packages using `pip install -r requirements.txt`.
3. Download and place the dataset in the `data/` directory.
4. Run the preprocessing notebook to prepare the data.
5. Train the model using the training notebook.
6. Evaluate the model using the evaluation notebook.

## Installation
```bash
git clone https://github.com/yourusername/DermAssist.git
cd DermAssist
pip install -r requirements.txt

Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

