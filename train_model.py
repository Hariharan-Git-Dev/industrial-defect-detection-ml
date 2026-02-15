import os
import numpy as np
import joblib
from extract_features import extract_features
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Paths
good_path = "data_balanced/good"
defect_path = "data_balanced/defective"

X = []
y = []

print("Extracting features from GOOD images...")
for img in os.listdir(good_path):
    img_path = os.path.join(good_path, img)
    features = extract_features(img_path)
    X.append(features)
    y.append(0)  # 0 = good

print("Extracting features from DEFECTIVE images...")
for img in os.listdir(defect_path):
    img_path = os.path.join(defect_path, img)
    features = extract_features(img_path)
    X.append(features)
    y.append(1)  # 1 = defective

X = np.array(X)
y = np.array(y)

print("Total samples:", len(X))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save model
joblib.dump(model, "model.pkl")
print("\nModel saved as model.pkl")
