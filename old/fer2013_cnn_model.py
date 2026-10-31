import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# --- Data Generator ---
class FER2013DataGenerator(Sequence):
    def __init__(self, root_dir, emotion_labels, batch_size=64, shuffle=True):
        self.root_dir = root_dir
        self.emotion_labels = emotion_labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.samples = []
        for idx, emotion in enumerate(emotion_labels):
            folder = os.path.join(root_dir, emotion)
            for fname in os.listdir(folder):
                self.samples.append((os.path.join(folder, fname), idx))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, idx):
        batch_samples = self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y = [], []
        for path, label in batch_samples:
            arr = np.load(path)
            X.append(arr)
            y.append(label)
        X = np.array(X)[..., np.newaxis]  # Add channel dimension
        y = np.array(y)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.samples)

# --- Labels ---
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# --- Generators ---
train_gen = FER2013DataGenerator('fer2013_processed/Training', emotion_labels, batch_size=64, shuffle=True)
val_gen = FER2013DataGenerator('fer2013_processed/PublicTest', emotion_labels, batch_size=64, shuffle=False)
test_gen = FER2013DataGenerator('fer2013_processed/PrivateTest', emotion_labels, batch_size=64, shuffle=False)

# --- Model ---
model = models.Sequential([
    layers.Input(shape=(64, 64, 1)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Training ---
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15
)

# --- Plot Training History and Save ---
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')
plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

# --- Evaluation ---
test_loss, test_acc = model.evaluate(test_gen)
with open('test_results.txt', 'w') as f:
    f.write(f"Test loss: {test_loss:.4f}\n")
    f.write(f"Test accuracy: {test_acc:.4f}\n")

# --- Confusion Matrix and Save ---
X_test, y_test = [], []
for i in range(len(test_gen)):
    X_batch, y_batch = test_gen[i]
    X_test.append(X_batch)
    y_test.append(y_batch)
X_test = np.concatenate(X_test, axis=0)
y_test = np.concatenate(y_test, axis=0)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# --- Classification Report and Save ---
report = classification_report(y_test, y_pred_classes, target_names=emotion_labels)
with open('classification_report.txt', 'w') as f:
    f.write(report)

# --- Save Model ---
model.save('fer2013_cnn_model.h5')

print("All results saved: training_history.png, confusion_matrix.png, test_results.txt, classification_report.txt, fer2013_cnn_model.h5")