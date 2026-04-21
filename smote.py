import os
import warnings
import pandas as pd
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from collections import Counter

warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# FIXED dataset
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,   # 🔥 FIX
    n_classes=2,
    weights=[0.9, 0.1],
    random_state=1
)

print("Before SMOTE:", {int(k): v for k, v in Counter(y).items()})

# Apply SMOTE
X_res, y_res = SMOTE(random_state=1).fit_resample(X, y)

print("After SMOTE:", {int(k): v for k, v in Counter(y_res).items()})

# Save to CSV
df = pd.DataFrame(X_res, columns=["Feature1", "Feature2"])
df["Target"] = y_res
df.to_csv("smote_dataset.csv", index=False)

print("Dataset saved successfully")