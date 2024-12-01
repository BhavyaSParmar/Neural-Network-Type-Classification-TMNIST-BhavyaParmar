# **Typeface MNIST Classification with CNN**

This repository contains the implementation of a **Convolutional Neural Network (CNN)** designed to classify digit images from the **Typeface MNIST (TMNIST)** dataset. The TMNIST dataset features digit representations (0–9) rendered in diverse font styles, providing a challenging and unique classification task.

## **Dataset Overview**
The **TMNIST** dataset is inspired by MNIST but introduces variability with 2,990 unique font styles. It contains:
- **29,900 samples** of 28x28 grayscale images.
- Digit labels (0–9) and pixel intensity values (784 features per image).

---

## **Objective**
- Build an efficient CNN capable of accurately classifying digit images across a variety of font styles.
- Analyze the model's performance metrics, including **accuracy** and **loss**, to evaluate its learning and generalization capabilities.

---

## **Model Architecture**
The CNN is built using TensorFlow/Keras and includes:
1. **Convolutional Layers**: Extract spatial features using filters (3x3) with ReLU activation.
2. **Max Pooling Layers**: Reduce dimensionality while retaining important features.
3. **Dropout Layers**: Prevent overfitting by randomly dropping neurons during training.
4. **Dense Layers**: Perform classification, ending with a **softmax** layer for multi-class probability.

---

## **Training Process**
- **Optimizer**: Adam (adaptive learning rate for faster convergence).
- **Loss Function**: Categorical Crossentropy (suitable for multi-class classification).
- **Metric**: Accuracy (proportion of correct predictions).
- **Epochs**: 6
- **Batch Size**: 64

### **Performance**
- **Validation Accuracy**: 98.76%
- **Test Accuracy**: 98.92%
- **Test Loss**: 0.035

The model exhibits strong generalization, achieving high accuracy on unseen data with minimal loss.

---

## **Key Features**
- **Preprocessing**:
  - Pixel normalization (0–1 range).
  - One-hot encoding for multi-class labels.
  - Dataset reshaped to include a channel dimension for CNN compatibility.
- **Visualization**: Sample images and performance trends plotted using Matplotlib.
- **Mathematical Foundations**:
  - Feedforward computation using ReLU and softmax activation functions.
  - Backpropagation to optimize weights and biases.
  - Categorical Crossentropy to measure loss.

---

## **Results**
- The CNN effectively learns and generalizes patterns in the TMNIST dataset.
- Training and validation losses steadily decrease while accuracies improve, showcasing the model's stability and robustness.
- The final model achieves **99% classification accuracy**, demonstrating its capability for real-world applications.

---

## **How to Use**
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/tmnist-cnn-classification.git
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python train_model.py
   ```
4. View predictions and performance metrics in the output logs and visualizations.

---

## **References**
- [TensorFlow CNN Tutorial](https://www.tensorflow.org/tutorials/images/cnn)
- [Understanding Convolutional Neural Networks](https://learnopencv.com/understanding-convolutional-neural-networks-cnn/)
- [Towards Data Science – CNN Basics](https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939)

---
