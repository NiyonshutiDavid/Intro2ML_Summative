# Uruti AgriTech: ML-Driven Startup Evaluation

## Overview

Rwanda’s agriculture sector employs over 61% of the workforce and contributes 24% to GDP, yet youth-led agri-startups struggle to access funding and mentorship due to manual, biased evaluations.  
Uruti, a Rwandan AgriTech platform, integrates a machine learning classification model to categorize applicants as “funding eligible,” “mentorship suitable,” or “rejection” based on experience, innovation, and business model clarity.  
This project aims to enhance transparency, streamline support allocation, and empower Rwanda’s youth, supporting national innovation and inclusive growth.  
The dataset consists of applicant features and outcomes, enabling fair, data-driven decision-making.

---

## Experimental Setup

We trained and evaluated several models (Neural Networks and a traditional ML algorithm) using different hyperparameters:

| Model    | Accuracy | Precision | Recall  | F1 Score | ROC AUC |
|----------|----------|-----------|---------|----------|---------|
| model_1  | 0.5413   | 0.5496    | 0.5413  | 0.5453   | 0.5121  |
| model_3  | 0.4893   | 0.5300    | 0.4893  | 0.4967   | 0.4865  |
| model_2  | 0.4327   | 0.5472    | 0.4327  | 0.4680   | 0.4822  |
| model_4  | 0.4067   | 0.5340    | 0.4067  | 0.4228   | 0.4988  |
| model_5  | 0.2873   | 0.5426    | 0.2873  | 0.2509   | 0.4959  |

---

## Findings

- **Best Combination:**  
  The best performing model was **model_1**, achieving an F1 Score of 0.5453, ROC AUC of 0.5121, Accuracy of 0.5413, Precision of 0.5496, and Recall of 0.5413.

- **ML Algorithm vs Neural Network:**  
  Based on the results, **model_1** outperformed the other models in all key metrics. (Please specify in your notebook which model corresponds to a neural network and which to a traditional ML algorithm, e.g., logistic regression.)  
  The best results were achieved with the hyperparameters used in model_1. For the ML algorithm, tuning hyperparameters such as regularization and optimizer had a significant impact on performance.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/NiyonshutiDavid/Intro2ML_Summative.git
   cd NiyonshutiDavid/Intro2ML_Summative

Open Summative_Intro_to_ml_[David Niyonshuti]_assignment_FIXED.ipynb in Jupyter or VS Code.

Run all cells in order. Each model implementation is modularized in its own cell.

To load the best saved model:
```bash
# Example for loading a pickle model
import joblib
model = joblib.load('saved_models/model_1.pkl')
```
Directory Structure
Intro2ML_SUMMATIVE/
├── Summative_Intro_to_ml_[David Niyonshuti]_assignment_FIXED.ipynb
├── Data Visualization/ Contains all visualized data and plots
│---model_1.pkl
└── README.md

## Video Demo

[Insert link to video demo]
