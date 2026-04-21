import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import andrews_curves

# Create sample dataset
np.random.seed(1)
df = pd.DataFrame({
    'A': np.random.normal(50,10,100),
    'B': np.random.normal(30,5,100),
    'C': np.random.normal(20,7,100),
    'Category': np.random.choice(['X','Y'], 100)
})

# -------------------------------
# 1. Quartile Plot (Box Plot)
# -------------------------------
plt.figure()
df[['A','B','C']].plot(kind='box', title="Box Plot (Quartiles)")
plt.show()

# -------------------------------
# 2. Scatter Plot
# -------------------------------
plt.figure()
plt.scatter(df['A'], df['B'])
plt.title("Scatter Plot")
plt.xlabel("A")
plt.ylabel("B")
plt.show()

# -------------------------------
# 3. Bubble Chart
# -------------------------------
plt.figure()
plt.scatter(df['A'], df['B'], s=df['C']*5)
plt.title("Bubble Chart")
plt.xlabel("A")
plt.ylabel("B")
plt.show()

# -------------------------------
# 4. Density Plot
# -------------------------------
plt.figure()
df['A'].plot(kind='density', title="Density Plot")
plt.show()

# -------------------------------
# 5. Andrews Curves
# -------------------------------
plt.figure()
andrews_curves(df, 'Category')
plt.title("Andrews Curves")
plt.show()