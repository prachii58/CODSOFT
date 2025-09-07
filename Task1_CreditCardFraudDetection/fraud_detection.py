# CREDIT CARD FRAUD DETECTION PROJECT
# Author: Prachi Patil
# Description: Detect fraudulent transactions using Logistic Regression


# 1. IMPORT LIBRARIES

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 2. LOAD DATA

print("📂 Loading data...")
data = pd.read_csv("creditcard.csv")
print("✅ Data loaded successfully!")
print(data.head())

# 3. SPLIT FEATURES & TARGET

X = data.drop("Class", axis=1)  # Features
y = data["Class"]               # Target (Fraud or Not)

# 4. BALANCE THE DATA (SMOTE)

print("\n✂ Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. SPLIT INTO TRAIN & TEST

print("\n⚖ Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. TRAIN MODEL

print("\n🤖 Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. EVALUATE MODEL

y_pred = model.predict(X_test)


print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred))

import joblib

# 8. Save model
joblib.dump(model, "fraud_detection_model.pkl")
print("✅ Model saved as fraud_detection_model.pkl")






