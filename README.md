# Traffic-Sign-Classification-Using-CNN

This repository provides a comprehensive implementation of a machine learning pipeline for classifying traffic signs using a Convolutional Neural Network (CNN). The project demonstrates various steps involved in data preparation, model development, and evaluation to create an efficient traffic sign classification system. 

## Features
- **Data Loading and Exploration**:
  - The notebook begins by loading the traffic sign dataset, which consists of labeled images representing various traffic sign categories.
  - Exploratory Data Analysis (EDA) is performed to understand the distribution of traffic sign classes, image sizes, and potential imbalances in the dataset.

- **Data Preprocessing**:
  - Images are resized to a uniform dimension to ensure compatibility with the CNN model.
  - Pixel values are normalized to fall within a range of [0, 1] for faster and more stable model training.
  - The dataset is split into training, validation, and testing subsets to evaluate the model's generalization.

- **Model Development**:
  - A CNN architecture is designed specifically for image classification tasks, including layers such as:
    - Convolutional layers to extract spatial features.
    - Pooling layers to reduce spatial dimensions and computational complexity.
    - Dense (fully connected) layers for final classification.
  - Dropout layers are included to reduce overfitting by randomly deactivating neurons during training.

- **Model Training**:
  - The model is trained using a categorical cross-entropy loss function and an optimizer like Adam.
  - Training is conducted over several epochs with a defined batch size, and the learning rate is adjusted dynamically if needed.
  - Metrics such as accuracy and loss are recorded for both training and validation sets to monitor performance.

- **Evaluation**:
  - The model’s performance is evaluated on the test dataset.
  - Detailed plots of training and validation accuracy/loss over epochs are generated to visualize the learning process.
  - Confusion matrices are used to assess class-wise performance and identify potential areas of improvement.

- **Prediction and Visualization**:
  - The trained model makes predictions on unseen test data.
  - Sample predictions are visualized alongside their corresponding true labels to illustrate the model’s accuracy.

## Prerequisites
To run this project, ensure you have the following Python libraries installed:
- TensorFlow: For building and training the CNN.
- NumPy: For numerical computations and array manipulation.
- Pandas: For handling dataset-related operations.
- Matplotlib: For creating visualizations of data and model performance.

You can install these libraries using pip:
```bash
pip install tensorflow numpy pandas matplotlib
```

## Usage
1. Clone this repository to your local machine using the following command:
   ```bash
   git clone https://github.com/yourusername/traffic-sign-classification.git
   ```
2. Ensure you have the required dependencies installed as mentioned above.
3. Open the Jupyter Notebook file (`Traffic_Sign_Classification.ipynb`) in your preferred environment (e.g., Jupyter Notebook, VSCode, Google Colab).
4. Follow the notebook's cells sequentially to load the data, preprocess it, train the model, and evaluate its performance.

## Dataset
The traffic sign dataset used in this project contains labeled images of various traffic signs. The dataset should be structured in a folder format, with each folder representing a specific traffic sign class. If using a custom dataset, ensure that it is properly organized and compatible with the preprocessing steps defined in the notebook.

## Results
The CNN model achieves a high level of accuracy on the test dataset, demonstrating its effectiveness in classifying traffic signs. Key results include:
- Training and validation accuracy/loss trends that indicate successful learning without overfitting.
- Accurate predictions for most traffic sign categories, with a few misclassifications highlighted for analysis.
- Visualizations that provide insights into the model’s strengths and potential areas for improvement.

## Future Work
- **Model Improvement**:
  - Experiment with deeper or more complex architectures to further enhance accuracy.
  - Implement transfer learning using pretrained models such as ResNet or VGGNet for better performance.

- **Data Augmentation**:
  - Apply advanced augmentation techniques such as random rotations, zooming, and horizontal flipping to improve robustness.

- **Deployment**:
  - Develop a user-friendly web or mobile application that leverages the trained model for real-time traffic sign recognition.

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute the code as per the terms of the license.

## Developer
## Muhammad Bilal
## Muhammad Huzaifa Zeb
## Muhammad Talha
