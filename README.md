# 🧠 CIFAR-10 Object Recognition using ResNet50

**Repository Name:** `cifar10-object-recognition-resnet50`
**Description:** Deep Learning project for **object recognition** using **ResNet50** on the **CIFAR-10 dataset**. Built with **TensorFlow** and **Keras**, achieving high accuracy through transfer learning and fine-tuning.

---

## 🧩 Overview

This project demonstrates how to use **ResNet50**, a powerful Convolutional Neural Network, for **multi-class image classification** on the **CIFAR-10** dataset.
It covers the **complete training pipeline**, including preprocessing, model architecture setup, transfer learning, evaluation, and visualization of predictions.

---

## 📊 Dataset

Dataset used:
🔗 [CIFAR-10 Dataset (Kaggle / TensorFlow Datasets)](https://www.cs.toronto.edu/~kriz/cifar.html)

* Contains **60,000 RGB images** (32×32 pixels) across **10 object categories**.
* Classes include: *airplane, automobile, bird, cat, deer, dog, frog, horse, ship,* and *truck*.
* Split into **50,000 training** and **10,000 testing** images.

---

## ⚙️ Project Structure

```
📂 CIFAR10-Object-Recognition
│
├── Object_Recognition(Cifar-10).ipynb   # Jupyter Notebook (Training + Evaluation)
└── README.md                            # Project documentation
```

---

## 🚀 How to Run

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/cifar10-object-recognition-resnet50.git
cd cifar10-object-recognition-resnet50
```

### 2️⃣ Install Dependencies

```bash
pip install tensorflow numpy matplotlib pandas scikit-learn
```

### 3️⃣ Run the Notebook

```bash
jupyter notebook Object_Recognition(Cifar-10).ipynb
```

This will train the **ResNet50** model and save the trained weights as:

```
/kaggle/working/cifar10_trained_model.h5
```

---

## 🧠 Model Architecture

| Component         | Description                                |
| ----------------- | ------------------------------------------ |
| Base Model        | ResNet50 (pre-trained on ImageNet)         |
| Top Layers        | Custom Dense + Dropout layers for CIFAR-10 |
| Input Shape       | 32×32×3 RGB images                         |
| Optimizer         | Adam                                       |
| Loss Function     | Categorical Crossentropy                   |
| Metrics           | Accuracy                                   |
| Accuracy Achieved | ~91% on test data                          |

---

## 🌟 Features

✅ Transfer Learning using ResNet50 backbone
✅ Fine-tuning for optimal feature extraction
✅ Evaluation metrics and confusion matrix visualization
✅ Model saving for reuse or deployment
✅ End-to-end workflow in a single Jupyter Notebook

---

## 📈 Results

* **Training Accuracy:** ~95%
* **Validation Accuracy:** ~92%
* **Test Accuracy:** ~91%
* **Loss:** Consistently decreasing — stable training curve

---

## 🧪 Example Workflow

1. Load and normalize CIFAR-10 images.
2. Import **ResNet50** (without top layers) for feature extraction.
3. Add custom classification layers.
4. Train and evaluate the model.
5. Save trained model for deployment or inference.

---

## 📦 Dependencies

```
tensorflow
numpy
matplotlib
pandas
scikit-learn
```

---

## 🧾 Example Predictions

After training, you can visualize test predictions:

```python
# Example
plt.imshow(X_test[0])
print("Predicted:", class_names[np.argmax(prediction[0])])
```

Output:

```
Predicted: airplane ✈️
```

---

## 👨‍💻 Author

**Hassan Saleem**
📧 [chhassan1041@gmail.com](mailto:chhassan1041@gmail.com)

---

