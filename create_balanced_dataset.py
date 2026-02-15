import os
import random
import shutil

# Paths
good_path = "data/good"
defect_path = "data/defective"

balanced_path = "data_balanced"
balanced_good = os.path.join(balanced_path, "good")
balanced_defect = os.path.join(balanced_path, "defective")

os.makedirs(balanced_good, exist_ok=True)
os.makedirs(balanced_defect, exist_ok=True)

# Get image lists
good_images = os.listdir(good_path)
defect_images = os.listdir(defect_path)

print(f"Good images available: {len(good_images)}")
print(f"Defective images available: {len(defect_images)}")

# Randomly sample good images equal to defective count
sampled_good = random.sample(good_images, len(defect_images))

# Copy defective images
for img in defect_images:
    shutil.copy(os.path.join(defect_path, img),
                os.path.join(balanced_defect, img))

# Copy sampled good images
for img in sampled_good:
    shutil.copy(os.path.join(good_path, img),
                os.path.join(balanced_good, img))

print("Balanced dataset created successfully.")
