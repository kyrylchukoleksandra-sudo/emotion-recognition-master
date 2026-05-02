import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import keras_tuner as kt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# --- Data Generator ---
class FER2013DataGenerator(tf.keras.utils.Sequence):
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

# --- Labels ---
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# --- Hyperparameterized Model ---
def build_model(hp):
    l2_reg = tf.keras.regularizers.l2(hp.Float('l2', 1e-5, 1e-2, sampling='log'))
    learning_rate = hp.Choice('learning_rate', [1e-3, 5e-4, 1e-4])
    dropout_rate = hp.Choice('dropout', [0.3, 0.5, 0.7])
    dense_units = hp.Choice('dense_units', [128, 256, 512])

    def se_block(input_tensor, ratio=8):
        filters = input_tensor.shape[-1]
        se = layers.GlobalAveragePooling2D()(input_tensor)
        se = layers.Dense(filters // ratio, activation='relu', kernel_regularizer=l2_reg)(se)
        se = layers.Dense(filters, activation='sigmoid', kernel_regularizer=l2_reg)(se)
        se = layers.Reshape((1, 1, filters))(se)
        return layers.multiply([input_tensor, se])

    def resnet_block(x, filters, kernel_size=3, stride=1):
        shortcut = x
        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same',
                          activation='relu', kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, kernel_size, strides=1, padding='same',
                          activation='relu', kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization()(x)
        x = se_block(x)
        if shortcut.shape[-1] != filters or stride != 1:
            shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same',
                                     kernel_regularizer=l2_reg)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        x = layers.add([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    inputs = layers.Input(shape=(64, 64, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = resnet_block(x, 64)
    x = layers.MaxPooling2D((2, 2))(x)

    x = resnet_block(x, 128)
    x = layers.MaxPooling2D((2, 2))(x)

    x = resnet_block(x, 256)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation='relu', kernel_regularizer=l2_reg)(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(7, activation='softmax', kernel_regularizer=l2_reg)(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- Tuner ---
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=20,
    factor=3,
    directory='kt_dir',
    project_name='fer2013_tuning'
)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# --- Головна функція для пошуку з оптимізацією batch_size ---
def run_tuner_with_batchsize():
    batch_sizes = [16, 32, 64, 128]
    best_score = -np.inf
    best_trial = None
    best_hp = None

    for batch_size in batch_sizes:
        print(f"\n--- Tuning with batch_size={batch_size} ---")
        train_gen = FER2013DataGenerator('fer2013_processed/Training', emotion_labels, batch_size=batch_size, shuffle=True)
        val_gen = FER2013DataGenerator('fer2013_processed/PublicTest', emotion_labels, batch_size=batch_size, shuffle=False)

        tuner.search(
            train_gen,
            validation_data=val_gen,
            epochs=20,
            callbacks=[stop_early]
        )

        trial_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        val_acc = tuner.oracle.get_best_trials(1)[0].metrics.get_last_value('val_accuracy')
        print(f"Best val_accuracy for batch_size={batch_size}: {val_acc}")

        if val_acc > best_score:
            best_score = val_acc
            best_trial = batch_size
            best_hp = trial_hp

    print(f"\nBest batch_size: {best_trial}")
    print(f"Best hyperparameters: learning_rate={best_hp.get('learning_rate')}, l2={best_hp.get('l2')}, dropout={best_hp.get('dropout')}, dense_units={best_hp.get('dense_units')}")

    # --- Побудова та навчання фінальної моделі ---
    train_gen = FER2013DataGenerator('fer2013_processed/Training', emotion_labels, batch_size=best_trial, shuffle=True)
    val_gen = FER2013DataGenerator('fer2013_processed/PublicTest', emotion_labels, batch_size=best_trial, shuffle=False)
    test_gen = FER2013DataGenerator('fer2013_processed/PrivateTest', emotion_labels, batch_size=best_trial, shuffle=False)

    model = tuner.hypermodel.build(best_hp)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=40,
        callbacks=[stop_early]
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
    plt.savefig('training_history_advanced.png')
    plt.close()

    # --- Evaluation ---
    test_loss, test_acc = model.evaluate(test_gen)
    with open('test_results_advanced.txt', 'w') as f:
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
    plt.title('Confusion Matrix (Advanced Model)')
    plt.savefig('confusion_matrix_advanced.png')
    plt.close()

    # --- Classification Report and Save ---
    report = classification_report(y_test, y_pred_classes, target_names=emotion_labels)
    with open('classification_report_advanced.txt', 'w') as f:
        f.write(report)

    # --- Save Model ---
    model.save('fer2013_cnn_model_advanced.h5')

    print("All results saved: training_history_advanced.png, confusion_matrix_advanced.png, test_results_advanced.txt, classification_report_advanced.txt, fer2013_cnn_model_advanced.h5")


if __name__ == "__main__":
    run_tuner_with_batchsize()