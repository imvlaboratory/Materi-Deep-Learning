## 🧠 **Convolutional Neural Network**

### 📝 **Definisi Convolutional Neural Network**

Convolutional Neural Network (CNN/ConvNet) merupakan bagian dari *deep neural network*, yaitu jenis jaringan saraf tiruan yang banyak digunakan dalam pengenalan dan pemrosesan gambar. Algoritma ini memiliki dua bagian utama:
- **Ekstraksi Fitur** 📸 dilakukan oleh **Convolutional Backbone**
- **Klasifikasi** 🎯 dilakukan oleh **Classifier Head** yang berisi Artificial Neural Network.

<div align="center">
    <img src="../contents/Fully-Connected-ANN.png" alt="Fully Connected ANN" width="720" align="center">
</div>

---

### 🛠️ **Arsitektur Convolutional Neural Network**

#### **1. Convolution Layer**

Convolutional layer melakukan konvolusi terhadap citra input dengan *filter* untuk menghasilkan `feature map` 🌌. `Feature map` menyimpan informasi penting dari gambar yang jauh lebih esensial dibandingkan gambar mentah.

<div align="center">
    <img src="../contents/Conv-Backbone.gif" alt="Conv Backbone" width="720" align="center">
</div>

Proses konvolusi dilakukan dengan **sliding filter** dari kiri atas hingga kanan bawah matriks citra. 

<div align="center">
    <img src="../contents/Conv.gif" alt="Convolution" width="720" align="center">
</div>

Convolutional layer akan menemukan *kernel* atau *weight* yang optimal. Semakin banyak *kernel*, semakin kaya fitur yang diekstrak 🧩. Penumpukan layer konvolusi memungkinkan informasi menjadi lebih bermakna dan berlapis.

<div align="center">
    <img src="../contents/conv stride 1 pad 0.gif" alt="convolution" width="360" style="vertical-align:left">
    <img src="../contents/conv stride 1 pad 1.gif" alt="convolution" width="360" style="vertical-align:left">
</div>
<div align="center">
    <img src="../contents/conv stride 2 pad 0.gif" alt="convolution" width="360" style="vertical-align:left">
    <img src="../contents/conv stride 2 pad 1.gif" alt="convolution" width="360" style="vertical-align:left">
</div>

```python
tf.keras.layers.Conv2D(
    filters,          # Dimensi ruang output → jumlah filter output
    kernel_size,      # Ukuran spasial filter (lebar/tinggi)
    strides=(1, 1),   # Besar pergeseran filter dalam konvolusi
    padding='valid',  # Jumlah padding → 'valid' tidak ada padding
    activation=None,  # Fungsi aktivasi yang digunakan
    input_shape=(height, width, color_channels)  # Input gambar
)
```

- filters → dimensi ruang output → jumlah filter output dalam konvolusi
- kernel_size → ukuran spasial dari filter (lebar/tinggi)
- stride → besar pergeseran filter dalam konvolusi
- padding → jumlah penambahan nol pada gambar
    - valid → tidak ada padding
    - same → padding nol merata kiri/kanan/atas/bawah
- activation → fungsi aktivasi untuk digunakan
- input_shape → input gambar

#### **2. Batch Normalization**

⚙️ Batch Normalization berfungsi untuk mengurangi *covariance shift* dengan menormalkan distribusi setiap nilai input yang berubah selama pelatihan 🔄. Hal ini membuat pelatihan lebih stabil dan cepat.

<div align="center"> 
    <img src="../contents/Batchnorm.png" alt="Batchnorm" width="360" align="center"> 
</div>

```python
tf.keras.layers.BatchNormalization()
```

#### **3. Pooling Layer**

Pooling layer digunakan untuk mengurangi dimensi citra fitur (downsampling) sambil mempertahankan informasi penting 🗜️.

<div align="center"> <img src="../contents/Pooling-Process.png" alt="Pooling Process" width="720" align="center"> </div> <div align="center"> <img src="../contents/Maxpool-Stride1-Pad1.gif" alt="convolution" width="360" style="vertical-align:left"> <img src="../contents/Maxpool-Stride2-Pad1.gif" alt="convolution" width="360" style="vertical-align:left"> </div>

```python
tf.keras.layers.MaxPool2D(
    pool_size=(2, 2),
    strides=None,
    padding='same',
)
```

```python
tf.keras.layers.AveragePooling2D(
    pool_size=(2, 2),
    strides=None,
    padding='same',
)
```

- pool_size → ukuran pool
- strides → besar pergeseran
- padding → jumlah penambahan nol pada gambar
    - valid → tidak ada padding
    - same → padding nol merata kiri/kanan/atas/bawah

### **🎨 Visualisasi Proses Konvolusi dan Pooling pada Gambar**

Contoh berikut menunjukkan bagaimana gambar anjing diproses setelah melewati Convolutional Backbone yang terdiri dari beberapa convolution layer dan max pooling layer. Terlihat pada gambar bahwa fitur-fitur penting pada wajah anjing mulai muncul setelah beberapa lapisan konvolusi dan pooling. Proses ini membantu model mengenali dan memprediksi gambar dengan lebih baik 📈.

<div align="center"> <img src="../contents/Convolution-Visualization.png" alt="Convolution Visualization" width="720" align="center"> </div> <br>