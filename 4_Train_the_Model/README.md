## **4. Train the Model**
Pada tahap ini, kita akan melatih model Convolutional Neural Network (CNN) yang telah dikompilasi pada data pelatihan menggunakan generator data yang telah disiapkan sebelumnya. Pelatihan ini bertujuan untuk memperbarui bobot pada model, sehingga model dapat belajar mengenali pola-pola dari gambar kucing dan anjing. Selama proses pelatihan, kita juga memantau metrik kinerja pada data validasi untuk menghindari overfitting.
```python
cat_dog = model.fit(train_generator,
                    validation_data = val_generator,
                    callbacks=[early_stopping,learning_rate_reduction],
                    epochs = 100,
                   )
```
