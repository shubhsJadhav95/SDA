# Required modules: pandas, numpy
# Install with: pip install pandas numpy

import pandas as pd
import numpy as np

np.random.seed(1)

# -------------------------------
# 1. Create dataset (50 values)
# -------------------------------
age = np.random.randint(20, 70, 50)
glucose = np.random.randint(70, 180, 50)

df = pd.DataFrame({
    "Age": age,
    "Glucose": glucose
})

# -------------------------------
# 2. Add NOISE (extreme values)
# -------------------------------
df.loc[2, "Glucose"] = 350   # high noise
df.loc[8, "Glucose"] = 10    # low noise

# -------------------------------
# 3. Add MISSING values
# -------------------------------
df.loc[5, "Glucose"] = np.nan
df.loc[10, "Age"] = np.nan

# -------------------------------
# 4. Add DUPLICATES
# -------------------------------
df = pd.concat([df, df.iloc[0:5]], ignore_index=True)

# -------------------------------
# 5. Save Noisy Dataset
# -------------------------------
df.to_csv("age_glucose_noisy.csv", index=False)
print("Noisy dataset saved")

# -------------------------------
# 6. CLEANING
# -------------------------------

# Remove duplicates
df_clean = df.drop_duplicates()

# Fill missing values (mean)
df_clean["Age"].fillna(df_clean["Age"].mean(), inplace=True)
df_clean["Glucose"].fillna(df_clean["Glucose"].mean(), inplace=True)

# Remove noise using IQR
Q1 = df_clean["Glucose"].quantile(0.25)
Q3 = df_clean["Glucose"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df_clean = df_clean[
    (df_clean["Glucose"] >= lower) &
    (df_clean["Glucose"] <= upper)
]

# -------------------------------
# 7. Save Cleaned Dataset
# -------------------------------
df_clean.to_csv("age_glucose_cleaned.csv", index=False)
print("Cleaned dataset saved")

# -------------------------------
# 8. SHOW RESULTS
# -------------------------------
print("\nOriginal size:", len(df))
print("After cleaning:", len(df_clean))

print("\nSample Data:\n", df.head())