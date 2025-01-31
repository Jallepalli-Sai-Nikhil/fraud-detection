import numpy as np
import pandas as pd
import warnings as wr
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import streamlit as st
from imblearn.over_sampling import SMOTE

# Suppress warnings
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

# Streamlit Header
st.title("Fraud Detection Streamlit App")
st.write("This app analyzes customer and transaction data to detect fraudulent activity.")

# Display Dataframe
st.subheader("Data Preview")
st.write(data.head())

# Identify numerical and categorical columns
numerical_features = data.select_dtypes(include=['number']).columns.tolist()
categorical_features = data.select_dtypes(include=['object']).columns.tolist()
st.write(f"Numerical Features: {numerical_features}")
st.write(f"Categorical Features: {categorical_features}")

# Plot count plot for categorical features
st.subheader("Categorical Feature Count Plots")
for column in categorical_features:
    top_10_values = data[column].value_counts().head(10)
    plt.figure(figsize=(10, 5))
    sns.countplot(x=column, data=data, order=top_10_values.index)
    plt.title(f'Count Plot for {column}')
    plt.xticks(rotation=90)
    st.pyplot(plt)

# Create box plots for numerical columns
st.subheader("Box Plots for Numerical Columns")
num_cols = len(numerical_features)
num_rows = (num_cols // 2) + (num_cols % 2)
fig, axes = plt.subplots(num_rows, 2, figsize=(12, 6*num_rows))
fig.suptitle("Box Plots for Numerical Columns")
for i, column in enumerate(numerical_features):
    row = i // 2
    col = i % 2
    sns.boxplot(x=data[column], ax=axes[row, col])
    axes[row, col].set_title(column)
if num_cols % 2 != 0:
    fig.delaxes(axes[num_rows-1, 1])
plt.tight_layout()
plt.subplots_adjust(top=0.95)
st.pyplot(plt)

# Plot for SuspiciousFlag
st.subheader("Count Plot for Suspicious Flag")
plt.figure(figsize=(8, 6))
sns.countplot(x='SuspiciousFlag', data=data, palette='Set2')
plt.title('Count Plot for Suspicious Flag')
plt.xlabel('Suspicious Flag')
plt.ylabel('Count')
plt.xticks(rotation=45)
st.pyplot(plt)

# Correlation matrix for numerical columns
numeric_data = data.select_dtypes(include=['number'])
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap for Numeric Columns')
st.pyplot(plt)

# Dropping irrelevant columns
columns_to_be_dropped = ['TransactionID', 'MerchantID', 'CustomerID', 'Name', 'Age', 'Address']
data1 = data.drop(columns_to_be_dropped, axis=1)

# Feature Engineering: Creating 'Hour' of transaction and 'gap' between transaction date and last login
if pd.api.types.is_datetime64_any_dtype(data1['Timestamp']):
    st.write("The 'Timestamp' column is already in datetime format.")
else:
    data1['Timestamp'] = pd.to_datetime(data1['Timestamp'])

data1['Timestamp1'] = data1['Timestamp']
data1['Hour'] = data1['Timestamp1'].dt.hour
data1['LastLogin'] = pd.to_datetime(data1['LastLogin'])
data1['gap'] = (data1['Timestamp1'] - data1['LastLogin']).dt.days.abs()

# Encoding the 'Category' column
label_encoder = LabelEncoder()
data1['Category'] = label_encoder.fit_transform(data1['Category'])

# Splitting the data into features and target
X = data1.drop(['FraudIndicator', 'Timestamp', 'Timestamp1', 'LastLogin'], axis=1)
Y = data1['FraudIndicator']

# Splitting the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(Y_test, y_pred)
precision = precision_score(Y_test, y_pred)
recall = recall_score(Y_test, y_pred)
f1 = f1_score(Y_test, y_pred)
conf_matrix = confusion_matrix(Y_test, y_pred)

# Display model evaluation
st.subheader("Model Evaluation")
st.write(f"Accuracy: {accuracy}")
st.write(f"Precision: {precision}")
st.write(f"Recall: {recall}")
st.write(f"F1 Score: {f1}")
st.write("Confusion Matrix:")
st.write(conf_matrix)

# Handling class imbalance using SMOTE
smote = SMOTE()
X_res, Y_res = smote.fit_resample(X_train, Y_train)

# Re-train the model with balanced data
model.fit(X_res, Y_res)

# Predict on test data again
y_pred_res = model.predict(X_test)

# Evaluate the re-trained model
accuracy_res = accuracy_score(Y_test, y_pred_res)
st.write(f"Accuracy after balancing with SMOTE: {accuracy_res}")

# Prediction Section
st.subheader("Fraud Prediction")
# Select input values for prediction
hour = st.slider('Hour of transaction', min_value=0, max_value=23)
gap = st.slider('Gap (days since last login)', min_value=0, max_value=1000)
category = st.selectbox('Category', data1['Category'].unique())

# Making a prediction
new_data = pd.DataFrame([[hour, gap, category]], columns=['Hour', 'gap', 'Category'])
prediction = model.predict(new_data)

if prediction[0] == 1:
    st.write("Fraudulent Activity Detected!")
else:
    st.write("No Fraudulent Activity Detected.")

# Measures to avoid credit fraud
st.subheader("Measures to Avoid Credit Fraud")
st.write("""
1. **Regular Monitoring**: Continuously monitor transactions and user behavior for any suspicious activity.
2. **Multi-Factor Authentication**: Implement multi-factor authentication for user login to enhance security.
3. **Use Fraud Detection Models**: Train and deploy machine learning models that can predict and prevent fraudulent activities.
4. **Transaction Limits**: Set transaction limits for high-risk transactions.
5. **User Education**: Educate customers on recognizing phishing attacks and secure payment practices.
6. **Real-time Alerts**: Implement real-time alerts for unusual transaction patterns.
""")
