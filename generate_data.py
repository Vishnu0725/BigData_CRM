import numpy as np
import pandas as pd
import os

# Reproducibility
np.random.seed(42)

# Number of customers
N = 1500

# Generate customer data
customer_id = [f"CUST{1000+i}" for i in range(N)]

age = np.random.randint(18, 70, N)
tenure = np.random.randint(1, 72, N)
monthly_charges = np.random.randint(200, 2000, N)
complaints = np.random.randint(0, 6, N)
support_calls = np.random.randint(0, 10, N)

contract_type = np.random.choice(
    ["Monthly", "Yearly", "Two-Year"],
    N,
    p=[0.55, 0.30, 0.15]
)

payment_method = np.random.choice(
    ["UPI", "Credit Card", "Debit Card", "Cash"],
    N
)

internet_usage_gb = np.random.randint(1, 200, N)

# Create churn probability based on customer behaviour
churn_probability = (
    0.15
    + (complaints * 0.08)
    + (support_calls * 0.035)
    - (tenure * 0.003)
    + (monthly_charges / 10000)
)

# Contract type affects churn
churn_probability += np.where(contract_type == "Monthly", 0.15, 0)
churn_probability += np.where(contract_type == "Yearly", -0.05, 0)
churn_probability += np.where(contract_type == "Two-Year", -0.12, 0)

# Keep probability within valid range
churn_probability = np.clip(churn_probability, 0.02, 0.95)

# Generate churn outcome
churn = np.random.binomial(1, churn_probability)

# Create DataFrame
df = pd.DataFrame({
    "CustomerID": customer_id,
    "Age": age,
    "Tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "Complaints": complaints,
    "SupportCalls": support_calls,
    "ContractType": contract_type,
    "PaymentMethod": payment_method,
    "InternetUsageGB": internet_usage_gb,
    "Churn": churn
})

# Save dataset
os.makedirs("data", exist_ok=True)

df.to_csv("data/customers.csv", index=False)

print("Dataset generated successfully!")
print(f"Total customers: {len(df)}")
print("\nDataset preview:")
print(df.head())

print("\nChurn distribution:")
print(df["Churn"].value_counts())

print("\nDataset saved to data/customers.csv")