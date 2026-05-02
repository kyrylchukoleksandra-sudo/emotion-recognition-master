import os
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# Parameters
input_root = 'fer2013_images'  # Folder with extracted images
output_root = 'fer2013_processed'  # Where to save processed images
target_size = (64, 64)  # New image size

# Augmentation (only for Training)
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Loop through all subfolders (Training, PublicTest, PrivateTest)
for split in ['Training', 'PublicTest', 'PrivateTest']:
    for emotion in os.listdir(os.path.join(input_root, split)):
        input_dir = os.path.join(input_root, split, emotion)
        output_dir = os.path.join(output_root, split, emotion)
        os.makedirs(output_dir, exist_ok=True)
        files = os.listdir(input_dir)
        for fname in tqdm(files, desc=f"{split}/{emotion}"):
            img_path = os.path.join(input_dir, fname)
            img = Image.open(img_path).convert('L')  # Grayscale
            img = img.resize(target_size)
            img_np = np.array(img)
            # Denoising
            img_np = cv2.medianBlur(img_np, 3)
            # Normalization
            img_np = img_np / 255.0
            # Save normalized image (as float32 .npy)
            np.save(os.path.join(output_dir, fname.replace('.png', '.npy')), img_np)
            # For Training, apply augmentation (e.g., 2 additional variants)
            if split == 'Training':
                img_aug = img_np.reshape((1, *target_size, 1))
                aug_iter = datagen.flow(img_aug, batch_size=1)
                for i in range(2):
                    aug_img = next(aug_iter)[0].reshape(target_size)
                    np.save(os.path.join(output_dir, fname.replace('.png', f'_aug{i}.npy')), aug_img)

print("Batch preprocessing complete! All processed images are saved as .npy files.")