import pandas as pd
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

# Load CSV
df = pd.read_csv('fer2013/fer2013.csv')

# Emotion labels (may vary depending on the dataset)
emotion_labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

# Create folder structure
for split in ['Training', 'PublicTest', 'PrivateTest']:
    for emotion in emotion_labels.values():
        os.makedirs(f'fer2013_images/{split}/{emotion}', exist_ok=True)

# Save images
for idx, row in tqdm(df.iterrows(), total=len(df)):
    pixels = np.array([int(p) for p in row['pixels'].split()]).reshape(48, 48)
    img = Image.fromarray(pixels.astype(np.uint8))
    emotion = emotion_labels[row['emotion']]
    split = row['Usage']  # Training, PublicTest, PrivateTest
    img.save(f'fer2013_images/{split}/{emotion}/{idx}.png')

print("Done! All images are split into folders by split and emotion class.")