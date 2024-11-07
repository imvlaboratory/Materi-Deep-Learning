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
Tahap ini bertujuan untuk menyiapkan dataset Microsoft Cats vs Dogs agar siap digunakan dalam pelatihan model deep learning. Proses dimulai dengan mengunduh dataset dari URL sumber dan mengekstraknya ke folder lokal. Setelah file ZIP diekstrak, langkah selanjutnya adalah mengatur direktori data untuk pelatihan dan pengujian. 
```python
# Langkah 1: Mengunduh dan mengekstrak dataset
data_url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip'
urllib.request.urlretrieve(data_url, 'D:/IMV/Test Image Classification/kagglecatsanddogs_5340.zip')

# Mengekstrak dataset
local_file = 'D:/IMV/Test Image Classification/kagglecatsanddogs_5340.zip'
with zipfile.ZipFile(local_file, 'r') as zip_ref:
    zip_ref.extractall('D:/IMV/Test Image Classification/data/')

# Mendefinisikan direktori dasar tempat gambar berada
base_dir = 'D:/IMV/Test Image Classification/data/PetImages'
```
### 1.3 Mengorganisir Gambar dan Label ke dalam DataFrame
Tahap ini bertujuan untuk mengorganisir direktori file dan label dari dataset Microsoft Cats vs Dogs ke dalam format DataFrame. Dengan DataFrame ini, pengelolaan dataset menjadi lebih mudah karena data terstruktur dalam format tabel, memungkinkan akses langsung ke direktori file dan label kategori untuk tiap gambar. Struktur tabel ini juga mempermudah tahap preprocessing, analisis, dan pembagian dataset ke dalam subset pelatihan, validasi, dan pengujian.
```python
# Langkah 2: Mengorganisir Gambar dan Label ke dalam DataFrame
file_paths = []
labels = []

for category in ['Cat', 'Dog']:
    category_path = os.path.join(base_dir, category)
    for img_file in os.listdir(category_path):
        if img_file.endswith('.jpg') or img_file.endswith('.png'): # Menyaring file gambar yang valid
            labels.append(category)
            file_paths.append(os.path.join(category_path, img_file))

# Membuat DataFrame untuk mengelola pembagian dan analisis data
data = pd.DataFrame({'file_path': file_paths, 'label': labels})
```
### 1.4 Membagi Dataset ke dalam Subset Pelatihan, Validasi, dan Pengujian
Tahap ini bertujuan untuk membagi dataset gambar ke dalam tiga subset yang akan digunakan untuk pelatihan, validasi, dan pengujian model. Pembagian dataset ini penting untuk memastikan bahwa model dilatih, divalidasi, dan diuji pada data yang berbeda sehingga hasil evaluasi lebih akurat dan model tidak mengalami overfitting. 
```python
# Langkah 3: Membagi data menjadi train (80%), test (10%), dan validation (10%)
X_train, X_temp = train_test_split(data, test_size=0.2, stratify=data['label'], random_state=42)
X_test, X_val = train_test_split(X_temp, test_size=0.5, stratify=X_temp['label'], random_state=42)

# Menampilkan ukuran setiap subset untuk konfirmasi
print('Ukuran data train:', X_train.shape)
print('Ukuran data test:', X_test.shape)
print('Ukuran data validation:', X_val.shape)
```
### 1.5 Visualisasi Data (Optional)
Tahap ini merupakan langkah opsional yang bertujuan untuk menampilkan beberapa contoh gambar dari setiap kategori (kucing dan anjing) dalam subset dataset pelatihan. Visualisasi ini berguna untuk memastikan bahwa data sudah terorganisir dengan benar dan untuk memeriksa kualitas gambar yang akan digunakan dalam pelatihan model.

```python
# Langkah 4: Visualisasi gambar dari setiap kategori
def visualize_data(data_dir, categories=['cat', 'dog'], num_images=5):
    plt.figure(figsize=(12, 6))  # Mengatur ukuran gambar

    # Loop melalui setiap kategori
    for i, category in enumerate(categories):
        # Menyaring gambar berdasarkan kategori
        images = [img for img in os.listdir(data_dir) if img.startswith(category)]
        selected_images = random.sample(images, num_images)  # Memilih beberapa gambar secara acak
        for j, img_name in enumerate(selected_images):
            img_path = os.path.join(data_dir, img_name)
            img = load_img(img_path, target_size=(150, 150))  # Memuat gambar dan mengubah ukurannya
            img_array = img_to_array(img) / 255.0  # Mengonversi gambar ke array dan menormalisasi

            # Menampilkan gambar
            plt.subplot(len(categories), num_images, i * num_images + j + 1)
            plt.imshow(img_array)
            plt.axis('off')  # Menghilangkan sumbu
            if j == num_images // 2:
                plt.title(category.capitalize())  # Menambahkan judul kategori di tengah gambar

    plt.tight_layout()  # Mengatur tata letak agar gambar tidak saling tumpang tindih
    plt.show()

# Visualisasikan 5 gambar dari setiap kategori dalam dataset pelatihan
train_dir = 'D:/IMV/Test Image Classification/data/dogs-vs-cats/train'
visualize_data(train_dir, num_images=5)
```

### 1.6 Visualisasi Distribusi Kelas di Set Pelatihan, Validasi, dan Pengujian (Opsional)
Tahap ini merupakan langkah opsional yang bertujuan untuk memverifikasi keseimbangan jumlah gambar antara kelas "Cat" dan "Dog" di setiap subset dataset (pelatihan, validasi, dan pengujian). Visualisasi distribusi kelas dilakukan untuk memastikan bahwa proporsi data dalam setiap subset sudah seimbang dan sesuai dengan pembagian yang diinginkan. Keseimbangan ini membantu model dalam melakukan klasifikasi yang adil antara kedua kelas, tanpa bias yang disebabkan oleh ketidakseimbangan data.
```python
# Langkah 4: Visualisasi distribusi kelas dari dataset
labels = ['Cat', 'Dog']

# Mendapatkan jumlah unik untuk setiap subset
label1, count1 = np.unique(X_train.label, return_counts=True)
label2, count2 = np.unique(X_val.label, return_counts=True)
label3, count3 = np.unique(X_test.label, return_counts=True)

# Membuat DataFrame untuk setiap subset
uni1 = pd.DataFrame(data=count1, index=labels, columns=['Count'])
uni2 = pd.DataFrame(data=count2, index=labels, columns=['Count'])
uni3 = pd.DataFrame(data=count3, index=labels, columns=['Count'])

# Mengatur ukuran dan gaya visualisasi
plt.figure(figsize=(18, 6), dpi=200)
sns.set_style('whitegrid')

# Fungsi untuk menambahkan label jumlah di atas batang grafik
def add_labels(data, ax):
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 5),
                    textcoords='offset points',
                    fontsize=10, color='black')

# Membuat plot distribusi kelas pada data pelatihan
plt.subplot(1, 3, 1)
ax1 = sns.barplot(data=uni1.reset_index(), x='index', y='Count', hue='index', palette='Blues', dodge=False, legend=False, width=0.4)
ax1.set_title('Distribusi Kelas pada Data Pelatihan', fontsize=15)
ax1.set_xlabel('Label', fontsize=12)
ax1.set_ylabel('Jumlah', fontsize=12)
add_labels(uni1, ax1)

# Membuat plot distribusi kelas pada data validasi
plt.subplot(1, 3, 2)
ax2 = sns.barplot(data=uni2.reset_index(), x='index', y='Count', hue='index', palette='Greens', dodge=False, legend=False, width=0.4)
ax2.set_title('Distribusi Kelas pada Data Validasi', fontsize=15)
ax2.set_xlabel('Label', fontsize=12)
ax2.set_ylabel('Jumlah', fontsize=12)
add_labels(uni2, ax2)

# Membuat plot distribusi kelas pada data pengujian
plt.subplot(1, 3, 3)
ax3 = sns.barplot(data=uni3.reset_index(), x='index', y='Count', hue='index', palette='Oranges', dodge=False, legend=False, width=0.4)
ax3.set_title('Distribusi Kelas pada Data Pengujian', fontsize=15)
ax3.set_xlabel('Label', fontsize=12)
ax3.set_ylabel('Jumlah', fontsize=12)
add_labels(uni3, ax3)

# Menampilkan tata letak yang rapi
plt.tight_layout()
plt.show()
```

### 1.7 Image Augmentation (Opsional)
Tahap ini bertujuan untuk menerapkan image augmentation pada data pelatihan, sehingga meningkatkan variasi data yang akan diberikan ke model. Image augmentation membantu model untuk mengenali objek dalam berbagai kondisi, sehingga dapat memperbaiki kinerja dan mengurangi overfitting pada data pelatihan.
- Parameter untuk Mendukung Augmentasi Gambar
```python
# parameters
image_size = 128
image_channel = 3
bat_size = 32
```
- Augmentasi pada Data Pelatihan
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
- Membuat Generator Data dari DataFrame
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