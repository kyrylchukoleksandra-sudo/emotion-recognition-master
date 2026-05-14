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

    # 4. Convert to grayscale, resize, normalize
    img = img.convert('L')
    img = img.resize((64, 64))
    img_np = np.array(img).astype('float32') / 255.0
    img_np = np.expand_dims(img_np, axis=-1)
    img_np = np.expand_dims(img_np, axis=0)

    # 5. Convert processed image to base64 for display
    # Convert back to PIL for encoding
    img_display = Image.fromarray((img_np[0, :, :, 0] * 255).astype(np.uint8))
    buffered = io.BytesIO()
    img_display.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return img_np, img_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    emotion = None
    error = None
    img_base64 = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            try:
                img, img_base64 = preprocess_image(file)
                pred = model.predict(img)
                predicted_class = np.argmax(pred, axis=1)[0]
                emotion = emotion_labels[predicted_class]
                print("Predicted emotion:", emotion)
            except Exception as e:
                error = str(e)
                print("Error:", error)
    return render_template('index.html', emotion=emotion, error=error, img_base64=img_base64)

if __name__ == '__main__':
    app.run(debug=True)