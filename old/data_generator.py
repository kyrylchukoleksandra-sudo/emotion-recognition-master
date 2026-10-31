import numpy as np
import os
from tensorflow.keras.utils import Sequence

class FER2013DataGenerator(Sequence):
    def __init__(self, root_dir, emotion_labels, batch_size=32, shuffle=True):
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

# Usage example:
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

train_gen = FER2013DataGenerator(
    root_dir='fer2013_processed/Training',
    emotion_labels=emotion_labels,
    batch_size=64,
    shuffle=True
)

val_gen = FER2013DataGenerator(
    root_dir='fer2013_processed/PublicTest',
    emotion_labels=emotion_labels,
    batch_size=64,
    shuffle=False
)

test_gen = FER2013DataGenerator(
    root_dir='fer2013_processed/PrivateTest',
    emotion_labels=emotion_labels,
    batch_size=64,
    shuffle=False
)

# Example: get a batch
X_batch, y_batch = train_gen[0]
print(X_batch.shape)  # (batch_size, 64, 64, 1)
print(y_batch.shape)  # (batch_size,)