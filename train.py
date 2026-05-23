import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
IMG_SIZE = (150, 150)       # Resize all images to this
BATCH_SIZE = 32             # How many images to process at once
EPOCHS = 15                 # How many times to train over full dataset
TRAIN_DIR = 'model/dataset/train'
VAL_DIR   = 'model/dataset/val'
# Training data: apply augmentation to make model more robust
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize pixel values from 0-255 to 0-1
    rotation_range=20,        # Randomly rotate images
    zoom_range=0.2,           # Randomly zoom
    horizontal_flip=True      # Randomly flip horizontally
)

# Validation data: only normalize, NO augmentation
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'       # binary because it's 2 classes
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2,2),

    # Block 2
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    # Block 3
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    # Flatten and classify
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),          # Prevents overfitting
    layers.Dense(1, activation='sigmoid')  # Output: 0=cat, 1=dog
])

model.summary()   # Print a table of all layers
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',   # Standard loss for binary classification
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)
model.save('model/cats_vs_dogs.h5')
print("Model saved!")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.show()