from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Якщо немає імпорту, просто додайте:
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

model = load_model('fer2013_cnn_model_advanced_best_1.h5')

# Завантаження зображення
img = cv2.imread('./fer2013_images/privateTest/angry/34887.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (64, 64))
img = img.astype('float32') / 255.0  # нормалізація
img = np.expand_dims(img, axis=-1)   # додаємо канал
img = np.expand_dims(img, axis=0)    # додаємо batch dimension

pred = model.predict(img)
predicted_class = np.argmax(pred, axis=1)[0]
emotion = emotion_labels[predicted_class]
print(f"Predicted emotion: {emotion}")