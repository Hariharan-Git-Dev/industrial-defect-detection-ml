import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog


def extract_features(image_path):

    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize to fixed size
    img = cv2.resize(img, (256, 256))

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # -------------------------
    # 1. Edge Density
    # -------------------------
    edges = cv2.Canny(blurred, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size

    # -------------------------
    # 2 & 3. Contours
    # -------------------------
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_count = len(contours)

    areas = [cv2.contourArea(c) for c in contours]
    avg_area = np.mean(areas) if areas else 0

    # -------------------------
    # 4. Intensity Variance
    # -------------------------
    intensity_var = np.var(blurred)

    # -------------------------
    # 5. Texture (Improved LBP)
    # -------------------------
    radius = 3
    n_points = 8 * radius

    lbp = local_binary_pattern(blurred, n_points, radius, method='uniform')
    n_bins = n_points + 2

    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins)
    )

    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    # -------------------------
    # 6. HOG Features
    # -------------------------
    hog_features = hog(
        blurred,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=False,
        feature_vector=True
    )

    # Reduce HOG dimensionality
    hog_features = hog_features[:100]

    # -------------------------
    # Combine all features
    # -------------------------
    features = [edge_density, contour_count, avg_area, intensity_var]
    features.extend(lbp_hist.tolist())
    features.extend(hog_features.tolist())

    return features
