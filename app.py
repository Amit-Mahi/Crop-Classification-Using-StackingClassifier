from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Load and Train Model at Startup
# -----------------------------
df = pd.read_csv("Crop_recommendation.csv")

# Drop rainfall column
df = df.drop(columns=["rainfall"])

# Encode labels
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

# Split features and target
X = df.drop(columns=["label", "label_encoded"])
y = df["label_encoded"]

# Train-test split (not strictly needed for Flask, but for scaling consistency)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Base learners
base_learners = [
    ('svm', SVC(probability=True)),
    ('nb', GaussianNB()),
    ('knn', KNeighborsClassifier()),
    ('cart', DecisionTreeClassifier())
]

# Meta learner
meta_learner = LogisticRegression(max_iter=500)

# Stacking classifier
model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    passthrough=False,
    n_jobs=-1
)

# Train the model
model.fit(X_train, y_train)


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")  # Create index.html in templates/


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array([[float(data['N']), float(data['P']), float(data['K']),
                              float(data['temperature']), float(data['humidity']),
                              float(data['ph'])]])
        features_scaled = scaler.transform(features)
        pred_encoded = model.predict(features_scaled)[0]
        crop = label_encoder.inverse_transform([pred_encoded])[0]
        return jsonify({"crop": crop})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
