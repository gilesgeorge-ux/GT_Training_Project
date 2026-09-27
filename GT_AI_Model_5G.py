#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# In[2]:


df = pd.read_csv('5g_network_data.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.dtypes


# In[6]:


df.isna().sum().sum()


# In[7]:


int(df.duplicated().sum())


# In[8]:


df.value_counts().to_string()


# In[9]:


FEATURES = ["Signal Strength (dBm)", "Download Speed (Mbps)", "Upload Speed (Mbps)", "Latency (ms)", "Jitter (ms)"]
df[FEATURES].describe().round(2).to_string()


# In[10]:


"Limited", "Standard", "Strong"]


# In[ ]:


def performance_score(frame: pd.DataFrame) -> pd.Series:
    c = lambda values: values.clip(0, 1)
    return (
        0.30 * c(frame["Download Speed (Mbps)"] / 1000)
        + 0.15 * c(frame["Upload Speed (Mbps)"] / 150)
        + 0.25 * c(1 - frame["Latency (ms)"] / 25)
        + 0.15 * c(1 - frame["Jitter (ms)"] / 5)
        + 0.15 * c((frame["Signal Strength (dBm)"] + 120) / 70)
    )

df["Performance Score"] = performance_score(df)
df["Performance Tier"] = pd.cut(
    df["Performance Score"],
    bins=[-np.inf, 0.45, 0.60, np.inf],
    labels=["Limited", "Standard", "Strong"],
    right=False,
).astype(str)
print(df["Performance Tier"].value_counts().reindex(["Limited", "Standard", "Strong"]).to_string())
df.min()
df.max()


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(12, 4))
order = ["Limited", "Standard", "Strong"]
sns.countplot(data=df, x="Performance Tier", order=order, ax=axes[0], color="#178c86")
axes[0].set(title="Derived performance tiers", xlabel="Tier", ylabel="Rows")
sns.histplot(df["Performance Score"], bins=35, ax=axes[1], color="#178c86")
for boundary in (0.45, 0.60):
    axes[1].axvline(boundary, color="#e39c55", linestyle="--", linewidth=2)
axes[1].set(title="Illustrative performance score", xlabel="Score")
plt.tight_layout()
plt.show()


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
sns.heatmap(df[FEATURES].corr(), annot=True, fmt=".2f", cmap="BrBG", center=0, ax=axes[0])
axes[0].set_title("Input correlations")
sample = df.sample(n=3000, random_state=42)
sns.scatterplot(data=sample, x="Download Speed (Mbps)", y="Latency (ms)",
                hue="Performance Tier", hue_order=["Limited", "Standard", "Strong"],
                alpha=0.55, s=22, ax=axes[1])
axes[1].set_title("Speed and latency by derived tier")
plt.tight_layout()
plt.show()


# In[ ]:


X = df[FEATURES].copy()
y = df["Performance Tier"].copy()
assert X.notna().all().all() and y.notna().all(), "Missing model values require a documented imputation policy."
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"Training rows: {len(X_train):,}; test rows: {len(X_test):,}")
print("Feature order:", list(scaler.feature_names_in_))


# In[ ]:


baseline = DummyClassifier(strategy="most_frequent").fit(X_train_scaled, y_train)
model = DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
baseline_pred = baseline.predict(X_test_scaled)
baseline_accuracy = accuracy_score(y_test, baseline_pred)
test_accuracy = accuracy_score(y_test, y_pred)
test_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
print(f"Majority baseline accuracy: {baseline_accuracy:.4f}")
print(f"Decision tree test accuracy: {test_accuracy:.4f}")
print(f"Decision tree balanced accuracy: {test_balanced_accuracy:.4f}")


# In[ ]:


#Held-out classification report:
print(classification_report(y_test, y_pred, digits=4, zero_division=0))


# In[ ]:


labels = ["Limited", "Standard", "Strong"]
cm = confusion_matrix(y_test, y_pred, labels=labels)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
    cmap="BuGn", values_format="d", colorbar=False
)
plt.title("Held-out confusion matrix")
plt.tight_layout()
plt.show()
print(pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_})
      .sort_values("Importance", ascending=False).round(4).to_string(index=False))


# In[ ]:


import os
from pathlib import Path

# Define OUTPUT_DIR - you can change this path to your desired output directory
OUTPUT_DIR = Path("output")  # or use Path("/path/to/your/output/directory")

# Create the directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

joblib.dump(model, OUTPUT_DIR / "model.pkl")
joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")
metrics = {
    "target": "Performance Tier (derived from five input measurements)",
    "target_type": "deterministic illustrative rule, not independently measured ground truth",
    "features": FEATURES,
    "score_weights": {"download": 0.30, "upload": 0.15, "latency": 0.25, "jitter": 0.15, "signal": 0.15},
    "thresholds": {"limited_below": 0.45, "strong_at_or_above": 0.60},
    "split": {"train_rows": len(X_train), "test_rows": len(X_test), "test_fraction": 0.2, "stratified": True, "random_state": 42},
    "model": "DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=42)",
    "baseline_accuracy": round(float(baseline_accuracy), 6),
    "test_accuracy": round(float(test_accuracy), 6),
    "balanced_accuracy": round(float(test_balanced_accuracy), 6),
    "class_counts": {k: int(v) for k, v in y.value_counts().items()},
    "confusion_labels": labels,
    "confusion_matrix": cm.tolist(),
    "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    "feature_importances": {name: float(importance) for name, importance in zip(FEATURES, model.feature_importances_)},
}
(OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
loaded_model = joblib.load(OUTPUT_DIR / "model.pkl")
loaded_scaler = joblib.load(OUTPUT_DIR / "scaler.pkl")
assert np.array_equal(y_pred[:10], loaded_model.predict(loaded_scaler.transform(X_test.iloc[:10])))

