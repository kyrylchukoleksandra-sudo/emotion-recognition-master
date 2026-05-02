# Building and Optimizing a Neural Network for Emotion Recognition on Images (FER2013)

**Variant 3**

---

## 1. Problem Statement

**Goal:**  
Develop and optimize a neural network model for the task of multiclass image classification, specifically for emotion recognition from facial images.

**Input Data:**  
- Grayscale facial images, size 64x64 pixels  
- 7 emotion classes: angry, disgust, fear, happy, sad, surprise, neutral

---

## 2. Task Type

Multiclass image classification.

---

## 3. Transformation Object (Image Type)

Grayscale facial images, size 64x64 pixels, 1 channel.

---

## 4. Neural Network Modifications

### 4.1. Basic Architecture

A basic CNN typically consists of sequential Conv2D layers, MaxPooling2D, a single Dropout layer, and a Dense (fully connected) layer before the output. It does not include residual connections or attention mechanisms.

### 4.2. Modifications

- **ResNet-like blocks (residual connections):** Add direct paths for gradients, enabling deeper networks and preventing vanishing gradients. Improve the model’s ability to learn from complex data.
- **SE blocks (Squeeze-and-Excitation, attention mechanism):** Add channel-wise attention, allowing the network to focus on the most informative features. Increase classification accuracy.
- **Batch Normalization after each convolutional layer:** Stabilizes and accelerates training, reduces overfitting risk.
- **Dropout before the final layer:** Reduces overfitting by randomly disabling some neurons during training.
- **L2 regularization:** Added to all Conv2D and Dense layers for additional weight regularization.
- **Hyperparameter optimization:** Automatic tuning of learning_rate, l2, dropout, dense_units, batch_size using KerasTuner.
- **EarlyStopping:** Automatically stops training if validation loss does not improve, restoring the best weights.
- **Custom data generator:** Efficiently loads and processes large volumes of data in batches.

---

## 5. Rationale for Task Selection

Emotion recognition from faces is a relevant task for many fields: security, education, medicine, marketing, human-computer interfaces. Modern neural network approaches achieve high accuracy even under challenging conditions (varying lighting, noise, angles).

---

## 6. Technical Specification (Requirements)

**Project Goal:**  
Develop a neural network technology for automatic emotion recognition from facial images, robust to changes in lighting, quality, and angles.

**Main requirements:**
- Use a modern convolutional neural network architecture (e.g., ResNet, VGG-Face, or custom CNN).
- Implement one or more modifications to improve accuracy (e.g., attention layer, data augmentation, model ensembling).
- Train and test the model on an open dataset (e.g., FER2013, AffectNet, CK+).
- Evaluate model performance using standard metrics (accuracy, F1-score).
- Compare results of basic and modified models.
- Present results as a report, presentation, and code archive.

**Software functionality:**
- Loads facial images
- Performs preprocessing (normalization, noise removal, augmentation)
- Uses a convolutional neural network with attention mechanism for emotion recognition
- Outputs recognition results and quality metrics
- Allows comparison of different architectures and modifications

---

## 7. Usage

1. Install dependencies:
    ```bash
    pip install tensorflow keras-tuner matplotlib scikit-learn seaborn
    ```
2. Prepare the FER2013 dataset in the required folder structure.
3. Run the main script:
    ```bash
    python advanced_cnn_model.py
    ```
4. Results (plots, metrics, model) are saved to corresponding files.

---

## 8. Author / Contact

**Author:**  
Oleksandra Kyryrlchuk

**Contact:**  
kyrylchuk.oleksandra@gmail.com

---

