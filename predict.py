import tensorflow as tf
import numpy as np
import cv2

# Your model input size
IMG_SIZE = 128

# Load model once at startup (important for Render)
model = tf.keras.models.load_model("seizure_heatmap_model.keras")

# Class labels
class_names = ["Non-Seizure", "Seizure"]

def predict_image(img_path):
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Could not read image from path: " + img_path)

    # Preprocess
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)
    class_id = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))

    return class_names[class_id], confidence
