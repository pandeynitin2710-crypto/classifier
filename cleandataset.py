import os
from PIL import Image

DATASET_DIR = 'model/dataset'

removed = 0

for folder in ['train', 'val']:
    for category in ['Cat', 'Dog']:
        path = os.path.join(DATASET_DIR, folder, category)
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            try:
                img = Image.open(filepath)
                img.verify()       # checks if image is valid
            except Exception:
                print(f"Removing bad file: {filepath}")
                os.remove(filepath)
                removed += 1

print(f"Done! Removed {removed} bad files.")