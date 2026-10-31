import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import keras_tuner as kt

# --- List of emotion labels for classification ---
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# --- Hyperparameterized model builder function for Keras Tuner ---
def build_model(hp):
    # L2 regularization strength (logarithmic sampling between 1e-5 and 1e-2)
    l2_reg = tf.keras.regularizers.l2(hp.Float('l2', 1e-5, 1e-2, sampling='log'))
    # Learning rate for the optimizer (choice from predefined values)
    learning_rate = hp.Choice('learning_rate', [1e-3, 5e-4, 1e-4])
    # Dropout rate for regularization (choice from predefined values)
    dropout_rate = hp.Choice('dropout', [0.3, 0.5, 0.7])
    # Number of units in the dense layer (choice from predefined values)
    dense_units = hp.Choice('dense_units', [128, 256, 512])

    # Squeeze-and-Excitation block for channel-wise attention
    def se_block(input_tensor, ratio=8):
        filters = input_tensor.shape[-1]
        se = layers.GlobalAveragePooling2D()(input_tensor)
        se = layers.Dense(filters // ratio, activation='relu', kernel_regularizer=l2_reg)(se)
        se = layers.Dense(filters, activation='sigmoid', kernel_regularizer=l2_reg)(se)
        se = layers.Reshape((1, 1, filters))(se)
        return layers.multiply([input_tensor, se])

    # Residual block with Squeeze-and-Excitation and optional projection shortcut
    def resnet_block(x, filters, kernel_size=3, stride=1):
        shortcut = x
        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same',
                          activation='relu', kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, kernel_size, strides=1, padding='same',
                          activation='relu', kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization()(x)
        x = se_block(x)
        # If input and output shapes differ, project shortcut to match
        if shortcut.shape[-1] != filters or stride != 1:
            shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same',
                                     kernel_regularizer=l2_reg)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        x = layers.add([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    # Input layer for 64x64 grayscale images
    inputs = layers.Input(shape=(64, 64, 1))
    # Initial convolutional layer
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # First residual block
    x = resnet_block(x, 64)
    x = layers.MaxPooling2D((2, 2))(x)

    # Second residual block
    x = resnet_block(x, 128)
    x = layers.MaxPooling2D((2, 2))(x)

    # Third residual block
    x = resnet_block(x, 256)
    # Global average pooling to reduce spatial dimensions
    x = layers.GlobalAveragePooling2D()(x)
    # Dense layer with tunable number of units
    x = layers.Dense(dense_units, activation='relu', kernel_regularizer=l2_reg)(x)
    # Dropout for regularization
    x = layers.Dropout(dropout_rate)(x)
    # Output layer for 7 emotion classes
    outputs = layers.Dense(7, activation='softmax', kernel_regularizer=l2_reg)(x)

    # Build and compile the model
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- Keras Tuner Hyperband setup for hyperparameter search ---
tuner = kt.Hyperband(
    build_model,                # Model-building function
    objective='val_accuracy',   # Metric to optimize
    max_epochs=20,              # Maximum number of epochs per trial
    factor=3,                   # Reduction factor for Hyperband
    directory='kt_dir',         # Directory to save tuning results
    project_name='fer2013_tuning' # Project name for organization
)