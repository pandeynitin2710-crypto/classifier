🐶🐱 Dogs vs Cats Image Classifier
A Convolutional Neural Network (CNN) built with TensorFlow and Keras that classifies images as either a dog or a cat. This is a beginner-friendly deep learning project trained on the popular Kaggle Dogs vs Cats dataset.
---
📁 Project Structure
```
Project_Classifier/
├── model/
│   ├── dataset/
│   │   ├── train/
│   │   │   ├── Dog/
│   │   │   └── Cat/
│   │   └── val/
│   │       ├── Dog/
│   │       └── Cat/
│   ├── train.py
│   └── predict.py
├── split_dataset.py
├── clean_dataset.py
├── requirements.txt
└── .gitignore
```
---
📊 Dataset
Source: Kaggle Dogs vs Cats Dataset
Total Images: ~25,000 (12,500 dogs, 12,500 cats)
Split: 80% training (20,000 images), 20% validation (5,000 images)
Image Size: Resized to 150x150 pixels during training
> The dataset is not included in this repository due to its large size. Download it from Kaggle and run `split_dataset.py` to prepare it.
---
🧠 Model Architecture
The CNN consists of three convolutional blocks followed by fully connected layers:
Layer	Output Shape	Parameters
Conv2D (32 filters, 3x3)	148x148x32	896
MaxPooling2D (2x2)	74x74x32	0
Conv2D (64 filters, 3x3)	72x72x64	18,496
MaxPooling2D (2x2)	36x36x64	0
Conv2D (128 filters, 3x3)	34x34x128	73,856
MaxPooling2D (2x2)	17x17x128	0
Flatten	36,992	0
Dense (256 units, ReLU)	256	9,470,208
Dropout (0.5)	256	0
Dense (1 unit, Sigmoid)	1	257
Total Parameters: 9,563,713
---
⚙️ Setup and Installation
Prerequisites
Python 3.11
pip
1. Clone the repository
```bash
git clone https://github.com/your-username/Project_Classifier.git
cd Project_Classifier
```
2. Create and activate a virtual environment
```bash
py -3.11 -m venv venv311
venv311\Scripts\activate        # Windows
source venv311/bin/activate     # Mac/Linux
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
---
🗂️ Preparing the Dataset
Step 1 — Download the dataset
Download the Dogs vs Cats dataset from Kaggle and place the zip file inside `model/dataset/`.
Step 2 — Extract the zip
Add this at the top of `split_dataset.py` and run it once:
```python
import zipfile
with zipfile.ZipFile('model/dataset/your-zip-name.zip', 'r') as zip_ref:
    zip_ref.extractall('model/dataset/raw')
```
Step 3 — Clean corrupted images
```bash
python clean_dataset.py
```
Step 4 — Split into train and val sets
```bash
python split_dataset.py
```
This splits the dataset into 80% training and 20% validation automatically.
---
🏋️ Training
```bash
python model/train.py
```
Training Configuration
Parameter	Value
Image Size	150x150
Batch Size	32
Epochs	15
Optimizer	Adam
Loss Function	Binary Crossentropy
Activation (output)	Sigmoid
Data Augmentation applied during training
Random rotation (up to 20°)
Random zoom (up to 20%)
Random horizontal flip
The trained model is saved as `model/cats_vs_dogs.h5` after training completes.
---
🔍 Prediction
To classify a new image, edit the path in `predict.py` and run:
```bash
python model/predict.py
```
Example output:
```
It's a DOG! (confidence: 0.98)
```
A confidence above 0.5 means Dog, below 0.5 means Cat.
---
🛠️ Tech Stack
Python 3.11
TensorFlow 2.13 — deep learning framework
Keras — high level neural network API
NumPy — numerical operations
Matplotlib — plotting training graphs
Pillow — image loading and processing
Scikit-learn — evaluation utilities
---
📈 Results
The model was trained for 15 epochs on ~20,000 images and achieves solid binary classification performance between dogs and cats.
Training and validation accuracy/loss curves are plotted automatically after training using Matplotlib.
---
🚀 Future Improvements
Add transfer learning using pretrained models like VGG16 or ResNet50 for higher accuracy
Build a simple web interface using Flask or Streamlit for drag and drop predictions
Enable GPU acceleration with CUDA for faster training
Experiment with more data augmentation techniques
Add Grad-CAM visualizations to see what the model focuses on
---
👤 Author
Nitin
GitHub: Nitin Pandey
---
