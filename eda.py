import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = r"E:\MyProjects\Bank Customer Churn Prediction\Churn_Modelling.csv"
OUT_DIR = r"E:\MyProjects\Bank Customer Churn Prediction\eda_plots"
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.dpi"] = 110

df = pd.read_csv(DATA_PATH)

def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), bbox_inches="tight")
    plt.close(fig)

print("=" * 60)
print("1) DATA OVERVIEW")
print("=" * 60)
print("Shape:", df.shape)
print("\nColumns and dtypes:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())

print("\n" + "=" * 60)
print("2) MISSING VALUES")
print("=" * 60)
missing = df.isna().sum()
print(pd.DataFrame({"missing": missing, "pct": (missing / len(df) * 100).round(2)}))

print("\n" + "=" * 60)
print("3) DUPLICATES")
print("=" * 60)
print("Exact duplicate rows:", df.duplicated().sum())
for col in ["CustomerId", "RowNumber"]:
    print(f"Unique values in {col}:", df[col].nunique(), "(nrows:", len(df), ")")

print("\n" + "=" * 60)
print("4) TARGET VARIABLE - Exited")
print("=" * 60)
print(df["Exited"].value_counts())
print("\nChurn rate: {:.2f}%".format(df["Exited"].mean() * 100))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
counts = df["Exited"].value_counts().sort_index()
axes[0].bar(["No (0)", "Yes (1)"], counts.values, color=["#2e8b57", "#d62728"])
axes[0].set_title("Churn Counts")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 20, str(v), ha="center")
axes[1].pie(counts.values, labels=["Stayed", "Churned"], autopct="%1.1f%%",
            colors=["#2e8b57", "#d62728"], startangle=90)
axes[1].set_title("Churn Proportion")
fig.suptitle("Target Distribution", fontweight="bold")
save(fig, "1_target.png")

print("\n" + "=" * 60)
print("5) NUMERIC FEATURES - DESCRIPTIVE STATS")
print("=" * 60)
numeric_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                "EstimatedSalary"]
print(df[numeric_cols].describe().round(2).to_string())

print("\n" + "=" * 60)
print("6) CATEGORICAL FEATURES")
print("=" * 60)
for col in ["Geography", "Gender", "HasCrCard", "IsActiveMember"]:
    print(f"\n{col}:")
    print(df[col].value_counts().to_string())

print("\n" + "=" * 60)
print("7) CHURN RATE BY CATEGORY")
print("=" * 60)
for col in ["Geography", "Gender", "HasCrCard", "IsActiveMember", "NumOfProducts"]:
    rate = df.groupby(col)["Exited"].agg(["mean", "count"]).round(4)
    rate["mean"] = (rate["mean"] * 100).round(2)
    rate = rate.rename(columns={"mean": "churn_pct", "count": "n"})
    print(f"\n{col}:")
    print(rate.to_string())

print("\n" + "=" * 60)
print("8) NUMERIC DISTRIBUTION BY CHURN")
print("=" * 60)
for col in ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]:
    print(f"\n{col}:")
    print(df.groupby("Exited")[col].describe().round(2).to_string())

fig, axes = plt.subplots(3, 2, figsize=(13, 11))
for ax, col in zip(axes.ravel(), numeric_cols):
    for label, color in [(0, "#2e8b57"), (1, "#d62728")]:
        subset = df[df["Exited"] == label][col]
        sns.histplot(subset, kde=True, ax=ax, color=color,
                     label=("Stayed" if label == 0 else "Churned"), stat="density", alpha=0.5)
    ax.set_title(f"{col} by Churn")
    ax.set_xlabel("")
    ax.legend()
fig.tight_layout()
save(fig, "2_numeric_by_churn.png")

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, col in zip(axes.ravel(), numeric_cols):
    sns.boxplot(x="Exited", y=col, data=df, ax=ax,
                palette=["#2e8b57", "#d62728"])
    ax.set_title(col)
fig.tight_layout()
save(fig, "3_boxplots_by_churn.png")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
cat_cols = ["Geography", "Gender", "HasCrCard"]
for ax, col in zip(axes, cat_cols):
    ctab = pd.crosstab(df[col], df["Exited"], normalize="index") * 100
    ax2 = ctab.plot(kind="bar", stacked=True, color=["#2e8b57", "#d62728"], ax=ax, legend=False)
    ax.set_title(col)
    ax.set_ylabel("Proportion (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    for i, idx in enumerate(ctab.index):
        ax2.text(i, 100, f"n={len(df[df[col]==idx])}", ha="center", va="bottom", fontsize=8)
    for i in range(len(ctab)):
        for j, lab in enumerate(["Stayed", "Churned"]):
            v = 100 - ctab.iloc[i, 1] if j == 0 else ctab.iloc[i, 1]
            ax2.text(i, v - 4, f"{v:.1f}%", ha="center", fontsize=8, color="white")
axes[-1].legend(["Stayed", "Churned"], loc="upper right")
fig.suptitle("Churn Rate by Categorical Feature", fontweight="bold")
fig.tight_layout()
save(fig, "4_categorical_churn.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
isactive = pd.crosstab(df["IsActiveMember"], df["Exited"], normalize="index") * 100
isactive.rename(index={0: "Inactive", 1: "Active"}).plot(
    kind="bar", stacked=True, color=["#2e8b57", "#d62728"], ax=axes[0])
axes[0].set_title("IsActiveMember vs Churn")
axes[0].set_ylabel("Proportion (%)"); axes[0].legend(["Stayed", "Churned"])
np_by_prod = pd.crosstab(df["NumOfProducts"], df["Exited"], normalize="index") * 100
np_by_prod.plot(kind="bar", stacked=True, color=["#2e8b57", "#d62728"], ax=axes[1])
axes[1].set_title("NumOfProducts vs Churn")
axes[1].set_ylabel("Proportion (%)"); axes[1].legend(["Stayed", "Churned"])
fig.tight_layout()
save(fig, "5_active_nproducts.png")

print("\n" + "=" * 60)
print("9) CORRELATION WITH TARGET")
print("=" * 60)
corr = df[numeric_cols + ["Exited"]].corr()["Exited"].drop("Exited").sort_values(
    key=abs, ascending=False)
print(corr.round(3).to_string())

print("\nFull correlation matrix:")
print(df[numeric_cols + ["Exited"]].corr().round(2).to_string())

fig, ax = plt.subplots(figsize=(8, 7))
mask = np.triu(np.ones_like(df[numeric_cols + ["Exited"]].corr(), dtype=bool))
sns.heatmap(df[numeric_cols + ["Exited"]].corr(), mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", vmin=-1, vmax=1, center=0, cbar_kws={"label": "Pearson r"})
ax.set_title("Correlation Heatmap of Numeric Features")
save(fig, "6_correlation.png")

print("\n" + "=" * 60)
print("10) HIGH-RISK SEGMENTS (example slices)")
print("=" * 60)
slices = {
    "Age > 50": df["Age"] > 50,
    "Age > 50 & Germany": (df["Age"] > 50) & (df["Geography"] == "Germany"),
    "Inactive members": df["IsActiveMember"] == 0,
    "Inactive & Age > 50": (df["IsActiveMember"] == 0) & (df["Age"] > 50),
    "4 products": df["NumOfProducts"] == 4,
}
for name, mask in slices.items():
    rate = df[mask]["Exited"].mean() * 100 if mask.sum() else 0
    print(f"{name:25s}: n={mask.sum():5d}  churn={rate:5.2f}%")

print("\nEDA plots saved to:", OUT_DIR)