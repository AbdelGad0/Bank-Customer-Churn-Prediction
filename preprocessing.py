import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

BASE = r"E:\MyProjects\Bank Customer Churn Prediction"
DATA_PATH = os.path.join(BASE, "Churn_Modelling_cleaned.csv")
OUT_DIR = os.path.join(BASE, "data_splits")
os.makedirs(OUT_DIR, exist_ok=True)

CAT_COLS = ["Geography", "Gender"]
TARGET = "Exited"
RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
)

encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
encoder.fit(X_train[CAT_COLS])

def encode(df_in):
    encoded = pd.DataFrame(
        encoder.transform(df_in[CAT_COLS]),
        columns=encoder.get_feature_names_out(CAT_COLS),
        index=df_in.index,
    )
    numeric = df_in.drop(columns=CAT_COLS)
    return pd.concat([numeric, encoded], axis=1)

X_train_e = encode(X_train)
X_val_e = encode(X_val)
X_test_e = encode(X_test)

print("Class balance (target=1):")
for name, yy in [("train", y_train), ("val", y_val), ("test", y_test)]:
    print(f"  {name:6s}: shape={len(yy):5d}  churn={yy.mean()*100:.2f}%")

print("\nEncoded feature columns:")
print(list(X_train_e.columns))

print("\nTotals:", len(X_train), len(X_val), len(X_test), "=", len(X_train) + len(X_val) + len(X_test))

for name_f, x, yy in [
    ("X_train", X_train_e, y_train),
    ("X_val", X_val_e, y_val),
    ("X_test", X_test_e, y_test),
]:
    x.to_csv(os.path.join(OUT_DIR, f"{name_f}.csv"), index=False)
    yy.to_csv(os.path.join(OUT_DIR, f"y_{name_f[2:]}.csv"), index=False)

print("\nFiles saved to:", OUT_DIR)
print(sorted(os.listdir(OUT_DIR)))