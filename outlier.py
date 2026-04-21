# Required modules: pandas, numpy, scikit-learn
# Install with: pip install pandas numpy scikit-learn

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor

# Create dataset
np.random.seed(0)  # for consistent output
df = pd.DataFrame(np.random.normal(50, 10, (200, 2)), columns=["F1", "F2"])

# -------- Distance-based Outlier Detection --------
nn = NearestNeighbors(n_neighbors=5)
nn.fit(df)

distances, _ = nn.kneighbors(df)
mean_dist = distances.mean(axis=1)

threshold = mean_dist.mean() + 2 * mean_dist.std()
df["Dist_Out"] = mean_dist > threshold

# -------- LOF (Local Outlier Factor) --------
lof = LocalOutlierFactor(n_neighbors=20)
df["LOF_Out"] = lof.fit_predict(df) == -1

# -------- Results --------
print("\nDistance-based Outliers:")
print(df[df["Dist_Out"]])

print("\nLOF-based Outliers:")
print(df[df["LOF_Out"]])