import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Dataset (cleaned - no missing values to avoid errors)
data = {
    'Year': [2005, 2010, 2015, 2020, 2018, 2012, 2016],
    'Engine Size': [1.5, 2.0, 1.8, 2.2, 1.7, 1.6, 2.0],
    'Mileage': [50000, 40000, 30000, 20000, 35000, 45000, 38000],
    'Price': [300000, 500000, 700000, 900000, 650000, 550000, 750000]
}

df = pd.DataFrame(data)

print("\nDATASET:")
print(df)

# Basic statistics
print("\nSTATISTICS:")
print(df.describe())

# Correlation
corr = stats.pearsonr(df['Year'], df['Price'])[0]
print("\nCorrelation (Year vs Price):", round(corr, 2))

# T-test
older = df[df['Year'] <= 2012]['Price']
newer = df[df['Year'] > 2012]['Price']

t_stat, p_val = stats.ttest_ind(older, newer)
print("\nT-test:")
print("T-stat:", round(t_stat, 2))
print("P-value:", round(p_val, 4))

# Linear Regression
slope, intercept, r, p, std_err = stats.linregress(df['Year'], df['Price'])

print("\nRegression Equation:")
print("Price =", round(intercept, 2), "+", round(slope, 2), "* Year")

# Plot (IMPORTANT FIX)
plt.figure(figsize=(6,4))
plt.scatter(df['Year'], df['Price'])

plt.plot(
    df['Year'],
    intercept + slope * df['Year'],
    color='red'
)

plt.title("Year vs Price")
plt.xlabel("Year")
plt.ylabel("Price")
plt.grid()
plt.show()