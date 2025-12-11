from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from predict import predict_image

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_api():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # run prediction
    label, confidence = predict_image(filepath)

    return jsonify({
        "prediction": label,
        "confidence": float(confidence)
    })

# Render/Gunicorn will call app:app so no need for app.run()
