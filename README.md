
# Uruti AgriTech – Machine Learning for Startup Evaluation

### 🔬 Problem Statement

Rwanda’s agri-sector, while critical, lacks structured evaluation for startup support. Uruti is a platform that leverages **machine learning models** to predict whether an agri-startup is:

- ✅ **Funding Eligible**
- 🤝 **Mentorship Suitable**
- ❌ **Rejected**

The project integrates real-world application data with public datasets to train and deploy predictive models.

---

## 📊 Model Overview

We implemented:

- ✅ **Logistic Regression, XGBoost, SVM, Random Forest** (Traditional ML)
- ✅ **A simple feedforward Neural Network**
- ✅ **Five optimized Neural Network variants**

---

## 🧠 Neural Network Architecture

![NN Architecture](https://github.com/user-attachments/assets/7fe74302-5003-44ff-b291-0ece525b9025)


- **Input Layer:** 64 features
- **Hidden Layers:** 32 → 16 nodes with ReLU activation
- **Output Layer:** 3 nodes with Softmax
- **Task:** Multiclass classification

---

## 🧪 Neural Network Evaluation Table

| Instance | Optimizer | Regularizer | Epochs | Early Stopping | Dropout | Learning Rate | Accuracy | F1 Score | Recall | Precision |
|----------|-----------|-------------|--------|----------------|---------|----------------|----------|----------|--------|-----------|
| 1        | Adam (default) | None        | 700    | No             | 0.00    | Default        | 0.4920   | 0.5016   | 0.4920 | 0.5125    |
| 2        | Adagrad   | None        | 1000   | Yes(50 Epochs)             | 0.30    | 0.001          | 0.3973   | 0.4335   | 0.3973 | 0.5222    |
| 3        | RMSprop   | L2          | 1500   | Yes(50 Epochs)            | 0.20    | 0.005          | 0.3187   | 0.2985   | 0.3187 | 0.5508    |
| 4        | Adam      | L1          | 5000   | Yes(50 Epochs)            | 0.00    | 0.0001         | 0.5093   | 0.5141   | 0.5093 | 0.5473    |
| 5        | SGD       | L1          | 1000   | Yes(50 Epochs)            | 0.03    | 0.006          | 0.4880   | 0.5180   | 0.4880 | 0.5802    |

> 🏆 **Best NN Model:** Instance 5 (SGD + L1 + Dropout + LR 0.006)

---

## ⚙️ Traditional ML Model Results

From the **Improved ML Comparison Notebook**:

| Model              | Accuracy | F1 Score | ROC AUC |
|--------------------|----------|----------|---------|
| RandomForest       | 1.0000   | 1.0000   | 1.0000  |
| GradientBoosting   | 1.0000   | 1.0000   | 1.0000  |
| XGBoost            | 0.9968   | 0.9961   | 0.9993  |
| LogisticRegression | 0.7738   | 0.8021   | 0.9627  |
| SVC                | 0.7282   | 0.7684   | 0.9748  |

---

## 📌 Key Insights

- 🔥 Despite aggressive optimization (dropout, learning rate tuning, L1/L2, early stopping), **neural networks underperformed traditional models** on this task.
- ⚠️ Likely causes include:
  - Class imbalance (heavily skewed toward "Funding Eligible")
  - Neural networks being more data-hungry and sensitive to noise
- ✅ **XGBoost**, **RandomForest**, and **GradientBoosting** outperformed all others, with nearly perfect accuracy and F1.

---

## ✅ Final Prediction Test (Comparison)

### Neural Network:
- ✅ Prediction: Funding Eligible  
- ❌ Ground truth: Mentorship Needed → **Incorrect**

### RandomForest:
- ✅ Prediction: Funding Eligible  
- ✅ Ground truth: Funding Eligible → **Correct**

---

## 🧱 Project Structure

```
📁 Intro2ML_Summative/
├── big_startup_secsees_dataset.csv # Original dataset
├── cleaned_big_startup_secsees_dataset.csv # Processed dataset with additional funding_class column
├── handled_big_startup_secsees_dataset.csv # Processed dataset with all needed features and handled missing values
├── 📄 README.md
├── 📓 Summative_Intro_to_ml_[David Niyonshuti]_assignment.ipynb # Contains all neural network models
├── 📓 Improved_Model_Comparison_Notebook.ipynb # Contains all traditional ML models
├── 📁 Data Visualizations/
│   ├── 📄 Fundingclass_distribution.png
│   ├── 📄 distribution by funding_rounds.png
│   ├── 📄 distribution by status.pdf
│   └── ...
├── 📁 Models/
│   ├── model_1.pkl
│   ├── model_2.pkl
│   ├── model_3.pkl
│   ├── model_4.pkl
│   ├── model_5.pkl
│   └── best_model_RandomForest.pkl
```

---

## ▶️ How to Run

```bash
git clone https://github.com/NiyonshutiDavid/Intro2ML_Summative.git
cd Intro2ML_Summative
```

1. Launch Jupyter or VSCode and open the notebook.
2. Run all cells in sequence.
3. To load best model:
```python
import joblib
model = joblib.load("best_model_RandomForest.pkl")
prediction = model.predict(X_test)
```

---

## 🎥 Video Walkthrough

🎬 [Click here to watch](https://youtu.be/nIOJiL6qH-k)

---
