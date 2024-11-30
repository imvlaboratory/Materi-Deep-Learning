## **3. Compile Model**
Pada tahap ini, kita akan melakukan konfigurasi akhir pada model CNN yang telah kita bangun dengan mendefinisikan parameter optimasi dan metrik evaluasi. Proses ini mencakup penentuan optimizer, loss function, dan metrics untuk memonitor kinerja model selama pelatihan.
### 3.1 Menentukan Callback
Callback adalah fungsi yang dijalankan pada titik tertentu selama pelatihan model, yang memungkinkan kita melakukan penyesuaian otomatis berdasarkan kinerja model pada data validasi.
```python
learning_rate_reduction = ReduceLROnPlateau(monitor='val_binary_accuracy',
                                            patience=8,
                                            factor=0.2,
                                            min_lr=1e-6,
                                            verbose=1)

early_stopping = EarlyStopping(monitor='val_loss', 
                               patience=12, 
                               restore_best_weights=True, 
                               verbose=1)
```
- monitor → Metrik yang dipantau.
- patience → Jumlah epoch tanpa peningkatan sebelum learning rate dikurangi atau pelatihan dihentikan.
- factor → Learning rate akan dikurangi dari nilai sebelumnya ketika tidak ada peningkatan.
- min_lr → Nilai minimum learning rate untuk memastikan proses tidak menjadi terlalu lambat.
- verbose → Menampilkan informasi saat learning rate berubah
- restore_best_weights → Mengembalikan bobot model ke nilai terbaik saat pelatihan dihentikan, sehingga kinerja terbaik yang dicapai selama pelatihan dapat digunakan.
### 3.2 Mengompilasi Model
Setelah callback didefinisikan, langkah berikutnya adalah mengompilasi model dengan optimizer, loss function, dan metrik yang diperlukan:

```python
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=[BinaryAccuracy(), Precision(), Recall()])
```

- optimizer='adam': Optimizer Adam digunakan karena adaptif dan stabil dalam proses pelatihan.
- loss='binary_crossentropy': Loss function binary_crossentropy dipilih karena ini adalah masalah klasifikasi biner.
metrics=[BinaryAccuracy(), Precision(), Recall()]: Metrik evaluasi yang digunakan adalah:
- BinaryAccuracy: Mengukur akurasi model pada klasifikasi biner.
- Precision: Mengukur ketepatan model dalam memprediksi kelas positif.
- Recall: Mengukur sensitivitas model dalam mendeteksi kelas positif.

