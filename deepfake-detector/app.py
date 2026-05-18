from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs('uploads', exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# ── Load CNN model ─────────────────────────────────────
cnn_model = None
if os.path.exists('deepfake_model_v3.pth'):
    from predict import load_model as load_cnn
    cnn_model = load_cnn()
    print("✓ CNN model loaded")
else:
    print("⚠️  deepfake_model_v3.pth not found")

# ── Load EfficientNet model ────────────────────────────
eff_model = None
if os.path.exists('deepfake_efficientnet.pth'):
    from predict_efficientnet import load_efficientnet
    eff_model = load_efficientnet()
    print("✓ EfficientNet model loaded")
else:
    print("⚠️  deepfake_efficientnet.pth not found")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Pages ──────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/detect-efficientnet')
def detect_efficientnet():
    return render_template('detect_efficientnet.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html')


# ── CNN prediction ─────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if cnn_model is None:
        return jsonify({'error': 'CNN model not loaded. Add deepfake_model_v3.pth to project folder.'}), 503
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use JPG, PNG, or WEBP.'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    from predict import predict_image
    label, confidence = predict_image(filepath, cnn_model)
    os.remove(filepath)
    return jsonify({'result': label, 'confidence': confidence})


# ── EfficientNet prediction ────────────────────────────
@app.route('/predict-efficientnet', methods=['POST'])
def predict_eff():
    if eff_model is None:
        return jsonify({'error': 'EfficientNet model not loaded. Add deepfake_efficientnet.pth to project folder.'}), 503
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use JPG, PNG, or WEBP.'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    from predict_efficientnet import predict_image_efficientnet
    label, confidence = predict_image_efficientnet(filepath, eff_model)
    os.remove(filepath)
    return jsonify({'result': label, 'confidence': confidence})


if __name__ == '__main__':
    app.run(debug=True)