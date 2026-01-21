import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "model/haaland_lock.pth"
SECRET_FILE = "secret/hidden_file.pdf"
IMG_SIZE = 160
THRESHOLD = 0.85   # increase for stricter lock

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = models.mobilenet_v2(pretrained=False)
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, 1),
    nn.Sigmoid()
)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/verify", methods=["POST"])
def verify():
    file = request.files.get("image")
    if not file:
        return jsonify({"status":"error"})

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        score = model(img).item()

    if score > THRESHOLD:
        return jsonify({
            "status":"success",
            "confidence": round(score*100,2)
        })
    else:
        return jsonify({
            "status":"denied",
            "confidence": round(score*100,2)
        })

@app.route("/download")
def download():
    return send_file(SECRET_FILE, as_attachment=True)

if __name__ == "__main__":
    app.run()
