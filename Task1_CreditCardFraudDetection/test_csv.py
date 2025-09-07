import pandas as pd

try:
    df = pd.read_csv('creditcard.csv')
    print("✅ File loaded successfully!")
    print(df.head())
except FileNotFoundError:
    print("❌ File not found. Make sure creditcard.csv is in the same folder.")

