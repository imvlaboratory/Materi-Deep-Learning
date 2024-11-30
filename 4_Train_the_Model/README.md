## **4. Train the Model**
Pada tahap ini, kita akan melatih model Convolutional Neural Network (CNN) yang telah dikompilasi pada data pelatihan menggunakan generator data yang telah disiapkan sebelumnya. Pelatihan ini bertujuan untuk memperbarui bobot pada model, sehingga model dapat belajar mengenali pola-pola dari gambar kucing dan anjing. Selama proses pelatihan, kita juga memantau metrik kinerja pada data validasi untuk menghindari overfitting.
```python
cat_dog = model.fit(train_generator,
                    validation_data = val_generator,
                    callbacks=[early_stopping,learning_rate_reduction],
                    epochs = 100,
                   )
```
- train_generator → Dataset untuk pelatihan.
- validation_data → Dataset untuk validasi.
- callback → [early_stopping, learning_rate_reduction]:
    - early_stopping → Menghentikan pelatihan lebih awal jika model tidak lagi menunjukkan perbaikan.
    - learning_rate_reduction → Mengurangi learning rate jika kinerja model stagnan.
- epochs → Menentukan jumlah epoch pelatihan, 100 kali iterasi.

