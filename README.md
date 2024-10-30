<h1 align="center"> Deep Learning </h1>

<p align="center">
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" style="vertical-align:middle">
    <img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white" style="vertical-align:middle">
    <img src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white" style="vertical-align:middle">
    <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" style="vertical-align:middle">
    <img src="https://img.shields.io/badge/Matplotlib-black.svg?style=for-the-badge&logo=Matplotlib&logoColor=white" style="vertical-align:middle">

</p>

----

# Pengantar Deep Learning

<p align="center">
    <img src="contents/what-is-deep-learning.png" width="720" align="center">
</p>

**Deep Learning** adalah salah satu cabang dari **Artificial Intelligence (AI)** yang meniru cara kerja otak manusia dalam memproses data dan membuat keputusan. Menggunakan struktur yang dikenal sebagai **neural networks**, deep learning memungkinkan komputer untuk belajar dari sejumlah besar data dengan cara yang menyerupai pola pembelajaran manusia.

## Pentingnya Deep Learning dalam AI

Deep learning telah menjadi pilar penting dalam pengembangan teknologi AI karena kemampuannya untuk:

- **Mengolah Data Kompleks**: Deep learning unggul dalam memahami pola dalam data yang besar dan kompleks, seperti gambar, suara, dan teks.
- **Mengotomatisasi Tugas Kompleks**: Dengan model deep learning, AI dapat menjalankan tugas yang sebelumnya sulit untuk diotomatisasi, seperti memahami bahasa alami atau mengenali objek dalam gambar.
- **Memberikan Akurasi Tinggi**: Dalam berbagai tugas, seperti klasifikasi gambar dan prediksi, deep learning sering kali mencapai tingkat akurasi yang lebih tinggi dibandingkan metode machine learning tradisional.

## Aplikasi Utama Deep Learning

Deep learning telah membuka jalan bagi berbagai inovasi dan aplikasi di dunia nyata, seperti:

- **Computer Vision**: Penggunaan deep learning dalam pengenalan objek, deteksi wajah, analisis video, dan diagnosis medis melalui citra medis.
- **Natural Language Processing (NLP)**: Teknologi ini digunakan untuk analisis teks, chatbot, penerjemahan otomatis, dan asisten virtual.
- **Speech Recognition**: Mengubah suara menjadi teks, seperti yang digunakan dalam perangkat pintar atau sistem call center otomatis.
- **Rekomendasi Produk**: Digunakan di e-commerce dan platform hiburan (seperti Netflix dan Spotify) untuk memberikan rekomendasi produk atau konten yang relevan kepada pengguna.
- **Pengendalian Kendaraan Otonom**: Deep learning digunakan dalam pemrosesan gambar dan data sensor di mobil otonom untuk membantu mereka mengidentifikasi jalan, pejalan kaki, dan rambu lalu lintas.

Dengan perkembangan teknologi dan meningkatnya volume data, deep learning terus menjadi alat yang semakin kuat dan relevan dalam kehidupan sehari-hari dan berbagai industri.

## Artificial Neural Network (ANN)

<p align="center">
    <img src="contents/Neuron-Model.png" alt="Neural Network" width="720" align="center">
</p>

Dalam Machine Learning, Neural Network yang juga disebut Jaringan Saraf Tiruan (JST) atau Artificial Neural Network (ANN) adalah model algoritma yang terinspirasi oleh struktur dan fungsi jaringan saraf biologis dalam otak hewan (termasuk manusia).

### Arsitektur Artificial Neural Network (ANN)

<p align="center">
    <img src="contents/Activation-Function.gif" alt="Activation Function" width="720" align="center">
</p>

Sebuah ANN terdiri dari unit-unit atau node-node yang saling terhubung yang disebut neuron buatan atau artificial neuron, yang secara teori meniru neuron-neuron di otak. Node-node ini dihubungkan oleh tepi-tepi (edges), yang meniru sinapsis di otak. Setiap neuron buatan menerima sinyal dari neuron-neuron yang terhubung, kemudian memprosesnya dan mengirimkan sinyal ke neuron-neuron lain yang terhubung. "Sinyal" tersebut adalah bilangan riil, dan keluaran dari setiap neuron dihitung oleh suatu fungsi non-linear dari jumlah inputnya, yang disebut fungsi aktivasi (activation function). Kekuatan sinyal di setiap koneksi ditentukan oleh sebuah bobot (weight), yang disesuaikan selama proses pembelajaran.

<p align="center">
    <img src="contents/Neural-Network.gif" alt="Neural Network" width="720" align="center">
</p>

Biasanya, neuron-neuron dikelompokkan ke dalam lapisan-lapisan. Lapisan-lapisan yang berbeda mungkin melakukan transformasi yang berbeda pada input mereka. Sinyal-sinyal bergerak dari lapisan pertama (lapisan input) ke lapisan terakhir (lapisan output), dan mungkin melewati beberapa lapisan antara (lapisan tersembunyi) yang disebut hidden layer. Suatu jaringan disebut deep neural network jika memiliki setidaknya dua hidden layer.

#### **Input Layer**

Lapisan input dari ANN terdiri dari neuron-neuron input yang membawa data awal ke dalam sistem untuk diproses lebih lanjut oleh lapisan-lapisan artificial neuron berikutnya. Lapisan input adalah awal dari alur kerja untuk ANN. Jumlah neuron input sama dengan jumlah fitur pada data input.

```python
tf.keras.Input(shape=[height, width, color_channels])
```

```python
tf.keras.layers.InputLayer(input_shape=(height, width, color_channels))
```

```python
tf.keras.layers.Conv2D(input_shape=(height, width, color_channels))
```

```python
tf.keras.layers.Flatten(input_shape=(height, width, color_channels))
```

```python
tf.keras.layers.Dense(input_shape=(height, width, color_channels))
```

- input_shape/shape → dimensi ruang input

#### **Hidden Layer**

Dalam Artificial Neural Network (ANN), `hidden layer` adalah lapisan neuron buatan yang dapat ditambahkan atau diterapkan dalam rancangan. Lapisan ini bukanlah lapisan input maupun lapisan output, lapisan ini berada di antara keduanya. Contoh dari ANN yang menggunakan hidden layer adalah `feedforward neural network`.

Hidden layer mengubah input dari lapisan input ke lapisan output. Dilakukan dengan cara menerapkan `weight` pada input dan melewatkannya melalui `activation function`, yang menghitung output berdasarkan input dan bobot tersebut. Proses ini memungkinkan ANN untuk mempelajari hubungan non-linear antara data input dan output. Bobot pada input dapat ditetapkan secara acak, dan juga dapat disesuaikan serta dikalibrasi melalui metode yang disebut `backpropagation`. Jumlah lapisan tersembunyi sebaiknya disesuaikan dengan kompleksitas permasalahan yang dihadapi.

- Semakin banyak jumlah lapisan (layer), semakin banyak waktu komputasi yang dibutuhkan.
- Semakin banyak jumlah node (neuron), semakin memungkinkan ANN untuk mempelajari pola yang lebih rumit.
- Untuk mencegah `overfitting`, jumlah node (neuron) sebaiknya ditambahkan secara bertahap.

    ```python
    tf.keras.layers.Dense(units, activation=None)
    ```

    - units → dimensi ruang output
    - activation → fungsi aktivasi untuk digunakan, misalnya ReLu

#### **Output Layer**

Lapisan output pada artificial neural network adalah lapisan terakhir yang menghasilkan prediksi atau hasil akhir dari model. Lapisan ini menerima input dari hidden layer sebelumnya, mengolahnya, dan mengeluarkan hasil yang digunakan untuk menentukan keputusan atau klasifikasi akhir. Jumlah neuron disesuaikan dengan permasalahan yang dihadapi.

- Untuk klasifikasi `binary dan regresi`, output layer terdiri dari `satu neuron`.
- Untuk klasifikasi `multiclass atau categorical`, output layer terdiri dari jumlah neuron yang sama dengan `jumlah class`.

    ```python
    tf.keras.layers.Dense(units, activation='sigmoid')
    ```

    - units → dimensi ruang output
        - binary → satu neuron.
        - categorical → jumlah neuron sesuai jumlah class
    - activation → fungsi aktivasi yang digunakan

        | activation | output/class mode | loss function                             |
        | ---------- | ----------------- | ----------------------------------------- |
        | sigmoid    | binary            | binary_crossentropy → 0/1                 |
        | softmax    | categorical       | categorical_crossentropy → [1 0] [0 1]    |
        | softmax    | categorical       | sparse_categorical_crossentropy → [0] [1] |

Artificial Neural Network digunakan untuk berbagai tugas, termasuk pemodelan prediktif, kontrol adaptif, dan pemecahan masalah dalam kecerdasan buatan. Artificial Neural Network dapat belajar dari pengalaman, dan dapat menarik kesimpulan dari kumpulan informasi yang kompleks.

<br>

## Jenis-Jenis Artificial Neural Network

### Feedforward Neural Networks (FNNs)

Let's start with the most basic type of neural network: the Feedforward Neural Network (FNN). As the name suggests, data in an FNN flows in one direction — from the input layer, through the hidden layers, and finally to the output layer. There are no loops or cycles; it's a straightforward process. This simplicity makes FNNs easy to understand and implement, which is why they're often the first type of network you'll encounter when learning about ANNs.

FNNs are commonly used for tasks like image classification, where the goal is to assign labels to images, and regression problems, where you're predicting a continuous value based on input data. For example, if you wanted to predict house prices based on features like square footage and number of bedrooms, an FNN could be a good fit.

### Convolutional Neural Networks (CNNs)

Next up, we have Convolutional Neural Networks (CNNs), which are specifically designed for processing grid-like data such as images. CNNs are like the vision specialists of the neural network world. They use convolutional layers to scan over an image, detecting patterns like edges, textures, and even complex shapes. This ability to capture spatial hierarchies makes CNNs incredibly powerful for tasks like image recognition, object detection, and even video analysis.

Imagine you're building an app that can recognize objects in photos. A CNN would be your go-to tool, as it can identify and classify different objects within an image with high accuracy. CNNs are the backbone of many modern computer vision applications, from facial recognition systems to self-driving cars.

### Recurrent Neural Networks (RNNs)

Recurrent Neural Networks (RNNs) are a bit different from FNNs and CNNs because they're designed to handle sequential data. This means they're great at processing data where the order of information matters, like time series data or sentences in a text. RNNs have loops in their architecture that allow information to be passed from one step of the sequence to the next, making them ideal for tasks that require memory of previous inputs.

For example, if you're developing a speech recognition system or a language translation tool, an RNN would be a strong choice. It can take into account the context of previous words or sounds to make more accurate predictions about what comes next. Popular RNN variants like Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) have further improved their ability to capture long-range dependencies in sequences.
