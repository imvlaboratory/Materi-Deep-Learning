## **1. Prepare Data**
Tahap ini mencakup pemuatan dataset, pembersihan dan preprocessing dataset, serta pembagian dataset ke dalam subset untuk pelatihan, validasi, dan pengujian. Beberapa langkah penting pada bagian ini adalah menangani data yang hilang, melakukan scaling dan normalisasi dataset, dan memastikan data sudah siap dalam format yang sesuai untuk arsitektur deep learning yang akan digunakan.
### 1.1 Import Library yang Digunakan
Langkah pertama yang perlu kita lakukan adalah mengimpor library yang akan kita gunakan. Misalnya, untuk membuat grafik, kita membutuhkan ‘matplotlib’, dan karena kita bekerja dengan matriks dan array, kita memerlukan Numpy. Untuk dataset ini, kita menggunakan Keras dan TensorFlow untuk pengembangan model deep learning. Jadi, mari kita mulai!

```python
# Basic
import os
import numpy as np
import pandas as pd
import urllib.request
import zipfile
import shutil
import random

# visuals
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay

# Tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,MaxPooling2D,Dropout,Flatten,BatchNormalization,Conv2D, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

import warnings
warnings.filterwarnings('ignore')
```
### 1.2 Menyiapkan Dataset
Pada tahap ini kita menyiapkan dataset Microsoft Cats vs Dogs agar siap digunakan dalam model deep learning. Proses dimulai dengan mengunduh dataset dari URL sumber, kemudian mengekstraknya ke folder lokal. Setelah file ZIP diekstrak, folder khusus untuk data pelatihan dan pengujian dibuat, yang nantinya akan menampung gambar-gambar anjing dan kucing yang sudah terpisah. Dataset ini kemudian dibagi berdasarkan rasio yang telah ditentukan, yaitu 80% untuk pelatihan dan 20% untuk pengujian. Setiap gambar dari kategori “Cat” dan “Dog” dipindahkan secara acak ke folder pelatihan atau pengujian sesuai rasio tersebut. Dengan menjalankan kode ini, dataset akan tersusun rapi dalam folder pelatihan dan pengujian, siap untuk digunakan dalam melatih model klasifikasi.
```python
# Download dan ekstrak dataset
data_url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip'
urllib.request.urlretrieve(data_url, 'D:/IMV/Test Image Classification/kagglecatsanddogs_5340.zip')

# Ekstrak dataset
local_file = 'D:/IMV/Test Image Classification/kagglecatsanddogs_5340.zip'
with zipfile.ZipFile(local_file, 'r') as zip_ref:
    zip_ref.extractall('D:/IMV/Test Image Classification/data/')

# Inisialisasi folder untuk test dan train
base_dir = 'D:/IMV/Test Image Classification/data/PetImages'
train_dir = 'D:/IMV/Test Image Classification/data/dogs-vs-cats/train'
test_dir = 'D:/IMV/Test Image Classification/data/dogs-vs-cats/test'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Split dataset
split_ratio = 0.8  # 80% training, 20% testing

for category in ['Cat', 'Dog']:
    category_path = os.path.join(base_dir, category)
    images = os.listdir(category_path)
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)

    train_images = images[:split_index]
    test_images = images[split_index:]

    for img in train_images:
        src = os.path.join(category_path, img)
        dst = os.path.join(train_dir, f"{category.lower()}.{img}")
        if os.path.isfile(src):
            shutil.move(src, dst)

    for img in test_images:
        src = os.path.join(category_path, img)
        dst = os.path.join(test_dir, f"{category.lower()}.{img}")
        if os.path.isfile(src):
            shutil.move(src, dst)

print("Dataset split into train and test folders successfully.")
```
### 1.3 Mengorganisir Gambar dan Label ke dalam DataFrame
Pada tahap ini kita menyiapkan data gambar pelatihan dengan mengorganisir nama file dan label kategori (kucing atau anjing) ke dalam sebuah DataFrame. Nama file gambar diambil dari direktori, lalu label diekstrak berdasarkan nama file tersebut. DataFrame yang dihasilkan memudahkan pemrosesan lebih lanjut dengan menyimpan informasi file dan labelnya dalam format yang terstruktur. Kode ini mempermudah penggunaan dataset dalam model klasifikasi.
```python
# Inisialisasi folder
image_dir = "D:/IMV/Test Image Classification/data/dogs-vs-cats/train"

# List filenames in the directory
filenames = os.listdir(image_dir)

# Extract labels from filenames
labels = [x.split(".")[0] for x in filenames]

# Create a DataFrame with filenames and labels
data = pd.DataFrame({"filename": filenames, "label": labels})

# Display the first few rows of the DataFrame
data.head()
```
### 1.4 Pembersihan Dataset
Pada tahap kali ini kita melakukan pembersihan data untuk menghindari gambar yang duplikat atau gambar yang tidak sesuai format.

```python
# Function to check if images are valid
def validate_images(df):
    valid_rows = []
    for idx, row in df.iterrows():
        file_path = row['file_path']
        try:
            if file_path.endswith('.db'):
                print(f"Non-image file detected and removed: {file_path}")
                os.remove(file_path)
                continue
            with Image.open(file_path) as img:
                img.verify()  
            valid_rows.append(row) 
        except (IOError, SyntaxError, Image.DecompressionBombError, OSError):
            print(f"Invalid or truncated image file detected and removed: {file_path}")
            os.remove(file_path)  
    return pd.DataFrame(valid_rows) 

# Apply the function to validate and filter out invalid or non-image files
X_train = validate_images(X_train)
X_val = validate_images(X_val)
X_test = validate_images(X_test)

# Re-check the dataset size after cleaning
print("Cleaned dataset sizes:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)
```
### 1.5 Visualisasi Dataset
Pada tahap ini kita melakukan visualisasi dataset untuk melihat 
```python
def visualize_data(data_dir, categories=['cat', 'dog'], num_images=5):
    plt.figure(figsize=(12, 6))

    for i, category in enumerate(categories):
        images = [img for img in os.listdir(data_dir) if img.startswith(category)]
        selected_images = random.sample(images, num_images)
        for j, img_name in enumerate(selected_images):
            img_path = os.path.join(data_dir, img_name)
            img = load_img(img_path, target_size=(150, 150)) 
            img_array = img_to_array(img) / 255.0 

            # Plot image
            plt.subplot(len(categories), num_images, i * num_images + j + 1)
            plt.imshow(img_array)
            plt.axis('off')
            if j == num_images // 2:
                plt.title(category.capitalize())

    plt.tight_layout()
    plt.show()

# Visualize 5 images from each category in the training dataset
train_dir = 'D:/IMV/Test Image Classification/data/dogs-vs-cats/train'
visualize_data(train_dir, num_images=5)
```

### 1.6 Image Augmentation
```python
train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range = 15,
                                    horizontal_flip = True,
                                    zoom_range = 0.2,
                                    shear_range = 0.1,
                                    fill_mode = 'reflect',
                                    width_shift_range = 0.1,
                                    height_shift_range = 0.1)

test_datagen = ImageDataGenerator(rescale=1./255)
```
```python
train_generator = train_datagen.flow_from_dataframe(
    X_train,
    directory=None,
    x_col='file_path',
    y_col='label',
    batch_size=bat_size,
    target_size=(image_size, image_size),
    class_mode='binary'
)

val_generator = test_datagen.flow_from_dataframe(
    X_val,
    directory=None,
    x_col='file_path',
    y_col='label',
    batch_size=bat_size,
    target_size=(image_size, image_size),
    class_mode='binary',
    shuffle=False
)

test_generator = test_datagen.flow_from_dataframe(
    X_test,
    directory=None, 
    x_col='file_path',
    y_col='label',
    batch_size=bat_size,
    target_size=(image_size, image_size),
    class_mode='binary',
    shuffle=False
)
```
