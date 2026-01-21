import os
import gc
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file

# -------------------------
# SYSTEM OPTIMIZATION
# -------------------------
torch.set_num_threads(1)
DEVICE = "cpu"

# -------------------------
# APP CONFIG
# -------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "model/haaland_lock.pth"
SECRET_FILE = "secret/hidden_file.pdf"
IMG_SIZE = 160
THRESHOLD = 0.40   # higher = stricter

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------
# LOAD MODEL ONCE
# -------------------------
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, 1),
    nn.Sigmoid()
)

state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()

# -------------------------
# TRANSFORMS
# -------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# -------------------------
# ROUTES
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/verify", methods=["POST"])
def verify():
    if "image" not in request.files:
        return jsonify({"status": "error"})

    file = request.files["image"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        raw_score = model(img).item()

    # IMPORTANT:
    # model learned: 1 = "others", 0 = "haaland"
    haaland_score = 1 - raw_score

    # cleanup
    del img
    gc.collect()

    if haaland_score >= THRESHOLD:
        return jsonify({
            "status": "success",
            "confidence": round(haaland_score * 100, 2)
        })
    else:
        return jsonify({
            "status": "denied",
            "confidence": round(haaland_score * 100, 2)
        })

@app.route("/download")
def download():
    return send_file(SECRET_FILE, as_attachment=True)

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
