# ============================================
# Milestone 2: Exploratory Data Analysis
# & Feature Engineering
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load Dataset
# -------------------------------

df = pd.read_csv("ai_visa_prediction_synthetic_dataset.csv")

print("Dataset loaded successfully")
print("Shape:", df.shape)

# -------------------------------
# 2. Basic EDA
# -------------------------------

# Distribution of processing time
plt.figure()
sns.histplot(df["processing_days"], bins=30, kde=True)
plt.title("Distribution of Visa Processing Time")
plt.xlabel("Processing Days")
plt.ylabel("Frequency")
plt.show()

# -------------------------------
# 3. Processing Time vs Visa Type
# -------------------------------

plt.figure()
sns.boxplot(x="visa_type", y="processing_days", data=df)
plt.title("Processing Time by Visa Type")
plt.xlabel("Visa Type")
plt.ylabel("Processing Days")
plt.show()

# -------------------------------
# 4. Processing Time vs Destination Country
# -------------------------------

plt.figure(figsize=(10, 5))
sns.boxplot(x="destination_country", y="processing_days", data=df)
plt.title("Processing Time by Destination Country")
plt.xticks(rotation=45)
plt.xlabel("Destination Country")
plt.ylabel("Processing Days")
plt.show()

# -------------------------------
# 5. Seasonal Analysis
# -------------------------------

plt.figure()
sns.boxplot(x="application_month", y="processing_days", data=df)
plt.title("Processing Time by Application Month")
plt.xlabel("Application Month")
plt.ylabel("Processing Days")
plt.show()

# -------------------------------
# 6. Correlation Analysis
# -------------------------------

numerical_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(8, 6))
sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# -------------------------------
# 7. Feature Engineering
# -------------------------------

# Seasonal Index (Peak vs Non-Peak)
df["seasonal_index"] = df["application_month"].apply(
    lambda x: 1 if x in [5, 6, 7, 8] else 0
)

# Country-wise average processing time
country_avg = df.groupby("destination_country")["processing_days"].mean()
df["country_avg_processing_days"] = df["destination_country"].map(country_avg)

# Visa-type average processing time
visa_avg = df.groupby("visa_type")["processing_days"].mean()
df["visa_type_avg_processing_days"] = df["visa_type"].map(visa_avg)

print("Feature engineering completed")

# -------------------------------
# 8. Final Dataset Check
# -------------------------------

print("Final dataset shape:", df.shape)
print("New features added:")
print([
    "seasonal_index",
    "country_avg_processing_days",
    "visa_type_avg_processing_days"
])

print("\nMilestone 2 Completed Successfully")
