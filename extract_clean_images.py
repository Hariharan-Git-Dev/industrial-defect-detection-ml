import pandas as pd
import os
import shutil

# Paths
csv_path = "train.csv"
image_folder = "train_images"
output_folder = "data/good"

os.makedirs(output_folder, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)

# Get defective image names
defective_images = set(df["ImageId"].unique())

# Get all image names in folder
all_images = set(os.listdir(image_folder))

# Clean images = images NOT listed in CSV
clean_images = all_images - defective_images

print(f"Total images in folder: {len(all_images)}")
print(f"Total defective images (from CSV): {len(defective_images)}")
print(f"Total clean images found: {len(clean_images)}")

# Copy clean images
for img in clean_images:
    src = os.path.join(image_folder, img)
    dst = os.path.join(output_folder, img)
    if os.path.exists(src):
        shutil.copy(src, dst)

print("Copied clean images successfully.")
