# ============================================
# Milestone 2: Exploratory Data Analysis
# & Feature Engineering (Vi-SaaS)
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------
# 1. Load Dataset
# -------------------------------

dataset_path = os.path.join("..", "Milestone_1_Dataset", "visa_dataset.csv")
if not os.path.exists(dataset_path):
    print(f"Warning: Dataset not found at {dataset_path}. Using fallback 'visa_dataset.csv'.")
    dataset_path = "visa_dataset.csv"

df = pd.read_csv(dataset_path)

print(f"Dataset loaded successfully from: {dataset_path}")
print("Shape:", df.shape)

# -------------------------------
# 2. Basic EDA: Processing Time Distribution
# -------------------------------

plt.figure(figsize=(10, 6))
sns.histplot(df["processing_days"], bins=30, kde=True, color='teal')
plt.title("Distribution of Visa Processing Time (Vi-SaaS)")
plt.xlabel("Processing Days")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# -------------------------------
# 3. Processing Time vs Visa Type
# -------------------------------

plt.figure(figsize=(12, 6))
sns.boxplot(x="visa_type", y="processing_days", data=df, palette="Set2")
plt.title("Processing Time by Visa Type")
plt.xlabel("Visa Type")
plt.ylabel("Processing Days")
plt.xticks(rotation=45)
plt.show()

# -------------------------------
# 4. Processing Time vs Destination Country (Top 10)
# -------------------------------

top_destinations = df['destination_country'].value_counts().nlargest(10).index
df_top_dest = df[df['destination_country'].isin(top_destinations)]

plt.figure(figsize=(14, 7))
sns.boxplot(x="destination_country", y="processing_days", data=df_top_dest, palette="viridis")
plt.title("Processing Time by Top 10 Destination Countries")
plt.xticks(rotation=45)
plt.xlabel("Destination Country")
plt.ylabel("Processing Days")
plt.show()

# -------------------------------
# 5. Approval Status Distribution
# -------------------------------

plt.figure(figsize=(8, 6))
df['approval_status'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], startangle=90)
plt.title("Visa Approval Status Distribution")
plt.ylabel("")
plt.show()

# -------------------------------
# 6. Correlation Analysis (Relevant Features)
# -------------------------------

# Factors impacting processing time
relevant_numeric_cols = [
    'age', 'annual_income', 'bank_balance', 'travel_history_count', 
    'previous_rejections', 'processing_days'
]
numerical_df = df[relevant_numeric_cols]

plt.figure(figsize=(10, 8))
sns.heatmap(numerical_df.corr(), annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Correlation Matrix of Key Numeric Features")
plt.show()

# -------------------------------
# 7. Simple Feature Insights
# -------------------------------

print("\nInsights from Feature Analysis:")
avg_process_by_visa = df.groupby('visa_type')['processing_days'].mean().sort_values()
print("\nAverage Processing Days by Visa Type:")
print(avg_process_by_visa)

# -------------------------------
# 8. Final Dataset Check
# -------------------------------

print("\nFinal EDA dataset check completed.")
print("Dataset contains columns:", df.columns.tolist())

print("\nMilestone 2 Completed Successfully")
