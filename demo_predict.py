import cv2
import joblib
import numpy as np
from extract_features import extract_features

# Load model
model = joblib.load("model.pkl")

# If you used scaler, load it too:
# scaler = joblib.load("scaler.pkl")

# Image to test
image_path = "test.jpg"   # change this to any image path

# Extract features
features = extract_features(image_path)

# Convert to array
features = np.array(features).reshape(1, -1)

# If using scaler:
# features = scaler.transform(features)

# Predict
prediction = model.predict(features)[0]

if prediction == 1:
    print("Prediction: DEFECTIVE")
else:
    print("Prediction: GOOD")

# Optional: show image
img = cv2.imread(image_path)
cv2.putText(img,
            "DEFECTIVE" if prediction == 1 else "GOOD",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255) if prediction == 1 else (0, 255, 0),
            2)

cv2.imshow("Prediction", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
