import numpy as np
import joblib

# 1. Load the model and scaler you saved in Step 1
model = joblib.load("fraud_model.pkl")    # Your trained model file
scaler = joblib.load("scaler.pkl")        # Your scaler to scale features

# 2. Prepare the new transaction data you want to test
# This should be a 2D numpy array with 1 row and 30 columns (features)
new_transaction = np.array([[  
    0, -1.3598071336738, -0.0727811733098497, 2.53634673796914,
    1.37815522427443, -0.338320769942518, 0.462387777762292,
    0.239598554061257, 0.0986979012610507, 0.363786969611213,
    0.0907941719789316, -0.551599533260813, -0.617800855762348,
    -0.991389847235408, -0.311169353699879, 1.46817697209427,
    -0.470400525259478, 0.207971241929242, 0.0257905801985591,
    0.403992960255733, 0.251412098239705, -0.018306777944153,
    0.277837575558899, -0.110473910188767, 0.0669280749146731,
    0.128539358273528, -0.189114843888824, 0.133558376740387,
    -0.0210530534538215, 149.62
]])

# 3. Scale the new transaction data using the scaler (IMPORTANT!)
new_transaction_scaled = scaler.transform(new_transaction)

# 4. Use the trained model to predict the class of the new transaction
prediction = model.predict(new_transaction_scaled)

# 5.  Print the prediction result to the user

if prediction[0] == 1:
    print("⚠ Fraudulent transaction detected!")
else:
    print("✅ Genuine transaction.")


    
    
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# List of feature names in the order they appear in the dataset
features = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]

print("Enter the transaction details:")



# Collect inputs from user
user_input = []
for feature in features:
    while True:
        try:
            val = float(input(f"{feature}: "))
            user_input.append(val)
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

# Convert to numpy array and reshape
input_array = np.array(user_input).reshape(1, -1)

# Scale input
input_scaled = scaler.transform(input_array)

# Predict
prediction = model.predict(input_scaled)

if prediction[0] == 1:
    print("\n⚠ Fraudulent transaction detected!")
else:
    print("\n✅ Genuine transaction.")

