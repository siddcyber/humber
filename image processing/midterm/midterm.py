import cv2
import numpy as np
import os
import random
import pandas as pd

# Scale conversion: 1 mm = 3.7795275591 pixels
PIXELS_PER_MM = 3.7795275591


# Function to preprocess images with noise reduction
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Median Filtering (better for salt-and-pepper noise)
    median_filtered = cv2.medianBlur(blurred, 5)

    # Morphological Opening (removes small noise patches)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(median_filtered, cv2.MORPH_OPEN, kernel)

    # Adaptive Thresholding for better binarization
    adaptive_thresh = cv2.adaptiveThreshold(cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)

    return adaptive_thresh


# Function to add noise to an image
def add_noise(image, noise_level=0.05):
    noisy_image = image.copy()
    num_noise_pixels = int(noise_level * image.size)

    # Add salt (white) noise
    for _ in range(num_noise_pixels // 2):
        x, y = random.randint(0, image.shape[1] - 1), random.randint(0, image.shape[0] - 1)
        noisy_image[y, x] = 255

    # Add pepper (black) noise
    for _ in range(num_noise_pixels // 2):
        x, y = random.randint(0, image.shape[1] - 1), random.randint(0, image.shape[0] - 1)
        noisy_image[y, x] = 0

    return noisy_image


# Function to detect contours and extract shape features
def detect_and_measure_shapes(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []

    for contour in contours:
        area_px = cv2.contourArea(contour)
        perimeter_px = cv2.arcLength(contour, True)

        # Convert to real-world units
        area_mm2 = area_px / (PIXELS_PER_MM ** 2)
        perimeter_mm = perimeter_px / PIXELS_PER_MM

        # Smooth contours to reduce noise distortions
        epsilon = 0.02 * perimeter_px
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)

        # Classify shape
        if vertices == 3:
            shape_type = 'Triangle'
        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            shape_type = 'Square' if 0.9 <= aspect_ratio <= 1.1 else 'Rectangle'
        elif vertices == 5:
            shape_type = 'Pentagon'
        elif vertices == 6:
            shape_type = 'Hexagon'
        else:
            shape_type = 'Circle' if is_circular(contour, area_px, perimeter_px) else 'Polygon'

        shapes.append({
            "Shape": shape_type,
            "Area (px²)": area_px,
            "Perimeter (px)": perimeter_px,
            "Area (mm²)": round(area_mm2, 2),
            "Perimeter (mm)": round(perimeter_mm, 2)
        })

    return shapes


# Function to determine if a shape is a circle
def is_circular(contour, area, perimeter):
    if perimeter == 0:
        return False
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return 0.75 <= circularity <= 1.25  # Approximate range for circular shapes


# Function to process images in the **main dataset folder only** (no subfolders)
def process_dataset(dataset_path):
    results = []

    for filename in os.listdir(dataset_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(dataset_path, filename)
            print(f"Processing Image: {image_path}")

            # Preprocess image
            image = preprocess_image(image_path)
            shapes = detect_and_measure_shapes(image)

            for shape in shapes:
                results.append({
                    "filename": filename,
                    "shape": shape["Shape"],
                    "area_px": round(shape["Area (px²)"], 2),
                    "perimeter_px": round(shape["Perimeter (px)"], 2),
                    "area_mm2": round(shape["Area (mm²)"], 2),
                    "perimeter_mm": round(shape["Perimeter (mm)"], 2)
                })

    return results


# Example usage6
dataset_path = "dataset"  # Replace with the actual dataset path
shape_results = process_dataset(dataset_path)

# Save results to a JSON file
import json

with open("shape_measurements.json", "w") as f:
    json.dump(shape_results, f, indent=4)

print("Processing complete! Results saved to shape_measurements.json")
