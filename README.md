# Credit Card Fraud Detection

Hi! This project is about detecting fraudulent credit card transactions using Data science. I used the Kaggle credit card fraud dataset to train a model that can predict whether a transaction is genuine or fraud.

## What’s inside?

- Data preprocessing and handling imbalance with SMOTE 
- Saving the trained model and scaler for later use  
- An interactive script to predict fraud on new transactions by entering feature values

## Dataset

- Contains transactions with 30 features (V1 to V28, Time, Amount)  
- Target variable `Class` (0 = genuine, 1 = fraud)

##Libraries I used:
pandas
numpy
scikit-learn
imblearn (for SMOTE)
joblib

## How to run
1. Install the required libraries:
   pip install pandas, numpy, scikit-learn, imblearn, joblib.

2. Train the model by running:
   python train_model.py

3. Predict on new transactions by running:
   python predict_transaction.py


