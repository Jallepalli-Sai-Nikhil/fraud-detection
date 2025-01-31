import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Suppress warnings
import warnings as wr
wr.filterwarnings(action="ignore")

# Load the data
account = pd.read_csv("fraud detect/Data/Customer Profiles/account_activity.csv")
customer = pd.read_csv("fraud detect/Data/Customer Profiles/customer_data.csv")
fraud = pd.read_csv("fraud detect/Data/Fraudulent Patterns/fraud_indicators.csv")
suspision = pd.read_csv("fraud detect/Data/Fraudulent Patterns/suspicious_activity.csv")
merchant = pd.read_csv("fraud detect/Data/Merchant Information/merchant_data.csv")
tran_cat = pd.read_csv("fraud detect/Data/Merchant Information/transaction_category_labels.csv")
amount = pd.read_csv("fraud detect/Data/Transaction Amounts/amount_data.csv")
anamoly = pd.read_csv("fraud detect/Data/Transaction Amounts/anomaly_scores.csv")
tran_data = pd.read_csv("fraud detect/Data/Transaction Data/transaction_metadata.csv")
tran_rec = pd.read_csv("fraud detect/Data/Transaction Data/transaction_records.csv")

# Merging data
costumer_data = pd.merge(customer, account, on='CustomerID')
costumer_data = pd.merge(costumer_data, suspision, on='CustomerID')

transaction_data1 = pd.merge(fraud, tran_cat, on="TransactionID")
transaction_data2 = pd.merge(amount, anamoly, on="TransactionID")
transaction_data3 = pd.merge(tran_data, tran_rec, on="TransactionID")
transaction_data = pd.merge(transaction_data1, transaction_data2, on="TransactionID")
transaction_data = pd.merge(transaction_data, transaction_data3, on="TransactionID")

# Final merge of customer and transaction data
data = pd.merge(transaction_data, costumer_data, on="CustomerID")

# Feature Engineering
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Hour'] = data['Timestamp'].dt.hour
data['LastLogin'] = pd.to_datetime(data['LastLogin'])
data['gap'] = (data['Timestamp'] - data['LastLogin']).dt.days.abs()

# Encoding the 'Category' column
label_encoder = LabelEncoder()
data['Category'] = label_encoder.fit_transform(data['Category'])

# List of all columns in the dataset
all_columns = list(data.columns)
print(f"All Columns in the dataset: {all_columns}")

# Dropping irrelevant columns for model training
columns_to_be_dropped = ['TransactionID', 'MerchantID', 'CustomerID', 'Name', 'Age', 'Address', 'Timestamp', 'LastLogin']
data1 = data.drop(columns_to_be_dropped, axis=1)

# List of columns used in the model (after dropping irrelevant columns)
model_columns = list(data1.columns)
print(f"Columns used for model training: {model_columns}")

# Ensure that 'Category' is not in the features for training (it is the target variable)
X = data1.drop(['FraudIndicator'], axis=1)
Y = data1['FraudIndicator']

# Check that the columns in X are consistent during train and test
print(f"Feature columns in X: {list(X.columns)}")

# Splitting the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Ensure that the test data has the same columns as the training data
X_test = X_test[X_train.columns]

# Decision Tree Classifier Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, Y_train)

# Handling class imbalance using SMOTE
smote = SMOTE()
X_res, Y_res = smote.fit_resample(X_train, Y_train)

# Re-train the model with balanced data
model.fit(X_res, Y_res)

# Streamlit Application
st.title("Fraud Detection")

# Show DataFrame
st.subheader("Merged Data Preview")
st.dataframe(data.head())

# Input for prediction (must match the feature set used in the model)
st.subheader("Enter Transaction Details")
amount_input = st.number_input("Transaction Amount", min_value=0.0, key="amount_input")
anomaly_score_input = st.number_input("Anomaly Score", min_value=0.0, key="anomaly_score_input")
suspicious_flag_input = st.selectbox("Suspicious Flag", ["Yes", "No"], key="suspicious_flag_input")
category_input = st.selectbox("Category", options=data['Category'].unique(), key="category_input")

# Add missing fields for prediction
account_balance_input = st.number_input("Account Balance", min_value=0.0, key="account_balance_input")
transaction_amount_input = st.number_input("Transaction Amount", min_value=0.0, key="transaction_amount_input")

# Process the input (ensure it matches the training data columns)
new_data = pd.DataFrame({
    'Amount': [amount_input],
    'AnomalyScore': [anomaly_score_input],
    'SuspiciousFlag': [suspicious_flag_input],
    'Category': [category_input],
    'AccountBalance': [account_balance_input],
    'TransactionAmount': [transaction_amount_input],
    'Hour': [12],  # Dummy data for Hour
    'gap': [5],    # Dummy data for gap (days)
})

# Handle unseen labels
if category_input not in label_encoder.classes_:
    st.write(f"Category '{category_input}' is not recognized. Assigning default category.")
    category_input = label_encoder.classes_[0]  # Default to first class (or any default you prefer)

# Encoding the 'Category' column with known categories
new_data['Category'] = label_encoder.transform([category_input])

# Encode 'SuspiciousFlag' as binary (1 for 'Yes' and 0 for 'No')
new_data['SuspiciousFlag'] = new_data['SuspiciousFlag'].apply(lambda x: 1 if x == 'Yes' else 0)

# Drop 'FraudIndicator' column if it exists in the new data, as it is not part of the feature set
new_data = new_data.drop(columns=['FraudIndicator'], errors='ignore')

# Drop any missing or extra columns from the new data (using method 2)
new_data = new_data.drop(columns=[col for col in new_data.columns if col not in model_columns], axis=1)

# Ensure the features match the training data
missing_cols = set(model_columns) - set(new_data.columns)
if missing_cols:
    st.write(f"Error: Missing columns for prediction: {missing_cols}")
else:
    # Prediction
    prediction = model.predict(new_data)

    # Show prediction result
    if prediction == 1:
        st.write("Prediction: Fraudulent Transaction")
    else:
        st.write("Prediction: Not Fraudulent")

# Measures to Avoid Fraud
st.subheader("Measures to Avoid Credit Fraud")
st.write("""
1. **Use Strong Authentication:** Implement multi-factor authentication (MFA).
2. **Monitor Transactions:** Regularly monitor transactions for suspicious activities.
3. **Analyze Customer Behavior:** Use machine learning algorithms to predict unusual behavior.
4. **Keep Software Updated:** Ensure your payment systems and security software are up-to-date.
5. **User Awareness:** Educate users on phishing and fraud tactics.
""")
