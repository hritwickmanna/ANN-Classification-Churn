# 🏦 Bank Customer Retention Prediction using ANN

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Deployed App](https://img.shields.io/badge/Live%20Demo-Streamlit-brightgreen?logo=streamlit)](https://ann-classification-churn-hritwick.streamlit.app/)

---

## 📌 Project Overview

This project uses an **Artificial Neural Network (ANN)** to predict whether a bank customer is likely to **stay** or **leave (churn)**. The model is built with TensorFlow/Keras and deployed as an interactive web application using Streamlit.

-   **Current PoC deployment**: [Streamlit Web App](https://ann-classification-churn-hritwick.streamlit.app/)
-   **Model Performance**: Achieved **~87% accuracy** with a **validation loss of ~0.35**.
-   **Training**: Early stopping was applied after ~18 epochs to prevent overfitting.

---

## 📊 Features & Prediction Example

The model predicts customer churn based on the following input features.

### Input Features

-   `CreditScore`
-   `Geography` (e.g., France, Spain, Germany)
-   `Gender` (Male/Female)
-   `Age`
-   `Tenure` (in years)
-   `Balance`
-   `NumOfProducts`
-   `HasCrCard` (1 if yes, 0 if no)
-   `IsActiveMember` (1 if yes, 0 if no)
-   `EstimatedSalary`

### Example

**Input:**
```python
{
    'CreditScore': 700,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 5,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}
```

**Output Prediction:**
```
Prediction: 0 → Customer is likely to Stay
```

### 🎯 Target Variable

-   **1** → Customer Leaves (Churn)
-   **0** → Customer Stays

---

## 🧠 Model & Training

### Model Architecture

-   **Input Layer**: Encodes numerical and one-hot encoded categorical features.
-   **Hidden Layers**: Multiple `Dense` layers using the **ReLU** activation function.
-   **Output Layer**: A single `Dense` neuron with a **Sigmoid** activation function for binary classification.

### Training Configuration

-   **Optimizer**: `Adam`
-   **Loss Function**: `BinaryCrossentropy`
-   **Metrics**: `Accuracy`
-   **Early Stopping**: Training is configured to stop when the validation loss no longer improves.

### ⚡ Training Highlights

-   **Max Epochs**: 100
-   **Early Stopping Triggered**: ~18 epochs
-   **Final Accuracy**: ~87%
-   **Validation Loss**: ~0.35

---

## 🚀 How to Run Locally

1️⃣ **Clone the repository**
```bash
git clone https://github.com/hritwickmanna/ANN-Classification-Churn.git
cd ANN-Classification-Churn
```

2️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

3️⃣ **Run the Streamlit app**
```bash
streamlit run app.py
```
4️⃣ Open the local URL provided in your terminal to interact with the app.

---

## 📦 Project Structure

```graphql
├── app.py                     # Main Streamlit web application
├── model.h5                   # Saved Keras/TensorFlow model
├── label_encoder_gender.pkl   # Saved scikit-learn LabelEncoder for Gender
├── onehot_encoder_geo.pkl     # Saved scikit-learn OneHotEncoder for Geography
├── Churn_Modelling.csv        # The dataset used for training
├── requirements.txt           # Project dependencies
├── experiments.ipynb          # Jupyter notebook for model experimentation
├── prediction.ipynb           # Jupyter notebook for testing predictions
├── README.md                  # This README file
└── LICENSE                    # Project license file
```

---

## 📌 Next Steps

-   **Flask Deployment**: Expose the model as a REST API for production-level inference.
-   **Monitoring & Logging**: Implement tools to track model predictions and performance over time.
-   **Cloud Hosting**: Deploy the Flask API on a cloud platform like AWS, Azure, GCP, or Heroku.

---

## 📜 License

This project is licensed under the MIT License.
