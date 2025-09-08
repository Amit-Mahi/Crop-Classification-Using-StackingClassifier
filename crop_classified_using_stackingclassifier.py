import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("Crop_recommendation.csv")

# Drop rainfall column
df = df.drop(columns=["rainfall"])

# Split features and target
X = df.drop(columns=["label"])
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Base learners (Level-0 models)
# -----------------------------
base_learners = [
    ('svm', SVC(probability=True)),
    ('nb', GaussianNB()),
    ('knn', KNeighborsClassifier()),
    ('cart', DecisionTreeClassifier())
]

# -----------------------------
# Meta learner (Level-1 model)
# -----------------------------
meta_learner = LogisticRegression(max_iter=500)

# -----------------------------
# Stacking Classifier
# -----------------------------
stack_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    passthrough=False,
    n_jobs=-1
)

# Train model
stack_clf.fit(X_train, y_train)

# Predictions
y_pred = stack_clf.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
acc = accuracy_score(y_test, y_pred) * 100
print(f"✅ Test Accuracy: {acc:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Heatmap visualization
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
             xticklabels=np.unique(y),
             yticklabels=np.unique(y))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Crop Classification")
plt.show()

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
