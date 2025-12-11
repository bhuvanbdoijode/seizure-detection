import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("data.csv")

# Convert 5-class labels into 2 classes
# 🔥 MODIFY THIS MAPPING AFTER YOU TELL ME YOUR VALUES
df["y"] = df["y"].apply(lambda x: 1 if x == 1 else 0)


# Remove unwanted columns
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

X = df.drop(columns=["y"]).values
y = df["y"].values

# Base directory for heatmaps
BASE_DIR = "heatmap_images"
os.makedirs(BASE_DIR, exist_ok=True)

# Unique label folders
for label in np.unique(y):
    os.makedirs(f"{BASE_DIR}/{label}", exist_ok=True)

# Heatmap image size
IMG_SIZE = 14  # 14x14

for i in range(len(X)):
    vec = X[i]

    padded = np.pad(vec, (0, 196 - len(vec)), mode='constant')
    mat = padded.reshape(IMG_SIZE, IMG_SIZE)

    label = y[i]
    filename = f"{BASE_DIR}/{label}/{i}.png"   # <--- THIS MUST COME BEFORE savefig

    plt.figure(figsize=(2,2))
    plt.imshow(mat, cmap="viridis")
    plt.axis("off")
    plt.savefig(filename, dpi=60, bbox_inches='tight', pad_inches=0)
    plt.close('all')

    if i % 100 == 0:
        print(f"Generated {i}/{len(X)} images...")




print("DONE! Organized heatmap images saved to: heatmap_images/")
