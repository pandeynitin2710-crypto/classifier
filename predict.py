import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load saved model
model = tf.keras.models.load_model('model/cats_vs_dogs.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Load & resize
    img_array = image.img_to_array(img)                     # Convert to array
    img_array = img_array / 255.0                           # Normalize
    img_array = np.expand_dims(img_array, axis=0)           # Add batch dimension

    prediction = model.predict(img_array)[0][0]

    if prediction >= 0.5:
        print(f"It's a DOG! (confidence: {prediction:.2f})")
    else:
        print(f"It's a CAT! (confidence: {1 - prediction:.2f})")

# Test it
predict_image(r'C:\Users\Nitin\Downloads\download.jpg')