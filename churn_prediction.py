# Customer Churn Prediction
# Author: Manasa Ungarala

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# === Load dataset ===
df = pd.read_csv("telco_customer_churn.csv")  # Replace with actual path

# === Basic cleaning ===
df.dropna(inplace=True)
df = df[df['TotalCharges'] != ' ']  # Remove invalid entries
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

# === Encode categorical columns ===
for col in df.select_dtypes(include='object'):
    if col != 'customerID':
        df[col] = LabelEncoder().fit_transform(df[col])

# === Split features and target ===
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Standardize features ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Train Random Forest ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Confusion Matrix:
", confusion_matrix(y_test, y_pred))
print("📝 Classification Report:
", classification_report(y_test, y_pred))

# === Feature importance ===
importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
