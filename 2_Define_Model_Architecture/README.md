## **2. Define Model Architecture**
Pada tahap ini, kita mendefinisikan arsitektur model Convolutional Neural Network (CNN) menggunakan Keras dengan tipe model Sequential. Model ini bertujuan untuk melakukan klasifikasi gambar biner (kucing atau anjing). Arsitektur ini terdiri dari beberapa lapisan konvolusi, lapisan batch normalization, pooling, dan lapisan fully connected.
```python
model = Sequential()

# Input Layer
model.add(Input(shape=(image_size, image_size, image_channel)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Block 1
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Block 2
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Block 3
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Fully Connected layers
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Output layer for binary classification
model.add(Dense(1, activation='sigmoid'))

model.summary()
```
### 2.1 Penjelasan Architecture
1. Input Layer
- Layer input menerima gambar dengan ukuran image_size x image_size dan tiga kanal warna (RGB).
- Lapisan pertama adalah Conv2D dengan 32 filter dan ukuran kernel (3, 3), yang diaktifkan dengan fungsi ReLU.
- Dilanjutkan dengan BatchNormalization, MaxPooling2D untuk pengurangan dimensi, dan Dropout(0.2) untuk mengurangi overfitting.
2. Blok Konvolusi 1
- Blok pertama memiliki Conv2D dengan 64 filter dan kernel (3, 3), disertai BatchNormalization, MaxPooling2D, dan Dropout(0.2).
3. Blok Konvolusi 2
- Blok kedua memiliki Conv2D dengan 128 filter dan kernel (3, 3), disertai BatchNormalization, MaxPooling2D, dan Dropout(0.2).
4. Blok Konvolusi 3
- Blok ketiga memiliki Conv2D dengan 256 filter dan kernel (3, 3), disertai BatchNormalization, MaxPooling2D, dan Dropout(0.2).
5. Lapisan Fully Connected
- Lapisan Flatten digunakan untuk mengubah data hasil konvolusi menjadi vektor 1 dimensi.
- Diikuti oleh lapisan Dense dengan 512 neuron dan fungsi aktivasi ReLU, BatchNormalization, serta Dropout(0.2).
6. Lapisan Output
- Lapisan terakhir adalah lapisan Dense dengan satu neuron dan fungsi aktivasi sigmoid untuk klasifikasi biner (kucing atau anjing).
### 2.2 Gambar Arsitektur Model
Untuk representasi arsitektur model, gambar berikut menunjukkan diagram arsitektur CNN dari lapisan input hingga lapisan output.


Pada diagram ini, setiap blok konvolusi diikuti oleh lapisan batch normalization, pooling, dan dropout. Lapisan fully connected dan output melengkapi arsitektur, membuat model siap untuk klasifikasi biner antara kategori "Cat" dan "Dog."

