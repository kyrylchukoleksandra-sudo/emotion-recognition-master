from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import pillow_heif
import io
import base64

app = Flask(__name__)

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
model = load_model('fer2013_cnn_model_advanced_best_1.h5')

def preprocess_image(file_stream):
    file_bytes = file_stream.read()
    img = None

    # 1. Try HEIC
    try:
        heif_file = pillow_heif.read_heif(io.BytesIO(file_bytes))
        img_pil = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw"
        )
        img = img_pil
    except Exception:
        pass

    # 2. Try Pillow (JPG, PNG, etc.)
    if img is None:
        try:
            img_pil = Image.open(io.BytesIO(file_bytes))
            img = img_pil
        except Exception:
            pass

    # 3. Try OpenCV
    if img is None:
        file_bytes_np = np.frombuffer(file_bytes, np.uint8)
        img_cv = cv2.imdecode(file_bytes_np, cv2.IMREAD_UNCHANGED)
        if img_cv is not None:
            if len(img_cv.shape) == 3:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_cv)
        else:
            raise ValueError("Cannot read the image. Please upload a valid image file.")

    # --- FACE DETECTION AND CROPPING ---
    # Convert PIL image to OpenCV format
    img_cv = np.array(img)
    if img_cv.ndim == 2:
        img_gray = img_cv
    else:
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    # Load Haar cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Take the largest face
        x, y, w, h = max(faces, key=lambda rect: rect[2]*rect[3])
        face_img = img.crop((x, y, x + w, y + h))
        img = face_img
    # Якщо обличчя не знайдено — використовуємо все зображення

    # 4. Convert to grayscale, resize, normalize
    img = img.convert('L')
    img = img.resize((64, 64))
    img_np = np.array(img).astype('float32') / 255.0
    img_np = np.expand_dims(img_np, axis=-1)
    img_np = np.expand_dims(img_np, axis=0)

    # 5. Convert processed image to base64 for display
    img_display = Image.fromarray((img_np[0, :, :, 0] * 255).astype(np.uint8))
    buffered = io.BytesIO()
    img_display.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return img_np, img_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    emotions = []
    errors = []
    img_base64_list = []
    if request.method == 'POST':
        files = request.files.getlist('image')
        camera_image_data = request.form.get('camera_image')
        if camera_image_data:
            # Обробка фото з камери (dataURL)
            try:
                header, encoded = camera_image_data.split(",", 1)
                file_bytes = base64.b64decode(encoded)
                file = io.BytesIO(file_bytes)
                files = [file]
            except Exception:
                errors.append("Invalid camera image data.")
        for file in files:
            try:
                img, img_base64 = preprocess_image(file)
                pred = model.predict(img)
                predicted_class = np.argmax(pred, axis=1)[0]
                emotion = emotion_labels[predicted_class]
                emotions.append(emotion)
                img_base64_list.append(img_base64)
            except Exception as e:
                errors.append(str(e))
    return render_template('index.html', emotions=emotions, errors=errors, img_base64_list=img_base64_list)

if __name__ == '__main__':
    app.run(debug=True)