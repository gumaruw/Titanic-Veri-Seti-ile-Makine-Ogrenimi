# Titanic Survival Prediction

This project predicts passenger survival on the Titanic. Three classification algorithms are applied and compared: **K-Nearest Neighbors (KNN)**, **Multi Layer Perceptron (MLP)**, and **Naive Bayes (NB)**.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Preprocessing](#preprocessing)
- [Models & Evaluation](#models--evaluation)
- [Results](#results)
- [Usage](#usage)

---

## Overview
The goal is to predict passenger survival (`Survived`) using features like age, sex, class, and fare. Models are evaluated using accuracy, precision, recall, and F1-score.

---

## Dataset
- The dataset is available on Kaggle: [Titanic Dataset](https://www.kaggle.com/datasets/heptapod/titanic).  
- It includes features such as `PassengerId`, `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Cabin`, and `Embarked`

---

## Features
Key features used in modeling:
- Numerical: `Age`, `SibSp`, `Parch`, `Fare`
- Categorical: `Pclass`, `Sex`, `Embarked`
- Target: `Survived`

---

## Models & Evaluation
Algorithms used:
- **KNN (K-Nearest Neighbors):** k=3, 7, 11  
- **MLP (Multi Layer Perceptron):** 1-3 hidden layers (32 neurons each)  
- **Naive Bayes:** GaussianNB with default parameters  

---

## Results
| Metric       | Best Model        | Value   |
|--------------|-----------------|--------|
| Accuracy     | KNN (k=11)       | 0.882  |
| Precision    | KNN (k=11)       | 0.875  |
| Recall       | MLP (32,32)      | 0.712  |
| F1-score     | KNN (k=11)       | 0.760  |

Overall best performance: **KNN (k=11)**.

---

## Usage

```bash
python titanic_classification.py
