## **4. Evaluate the Model**
Setelah model selesai dilatih, tahap berikutnya adalah mengevaluasi kinerja model. Tahap evaluasi ini dilakukan dengan menganalisis performa model pada data pengujian (test set), memvisualisasikan metrik pelatihan, serta menampilkan hasil prediksi model dalam berbagai bentuk, seperti grafik loss dan accuracy, confusion matrix, dan hasil prediksi pada beberapa gambar.
### 4.1 Visualisasi Loss dan Accuracy
Pada bagian ini, kita membuat grafik untuk melihat perkembangan loss dan accuracy pada data pelatihan dan validasi selama epoch pelatihan.
```python
# Setelah pelatihan, konversi history pelatihan ke dalam DataFrame
error = pd.DataFrame(cat_dog.history)

# Mengatur ukuran dan gaya plot
plt.figure(figsize=(18, 5), dpi=200)
sns.set_style('darkgrid')

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(error['loss'], label='Training Loss', color='blue')
plt.plot(error['val_loss'], label='Validation Loss', color='orange')
plt.title('Training and Validation Loss', fontsize=15)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()

# Plot Akurasi Biner
plt.subplot(1, 2, 2)
plt.plot(error['binary_accuracy'], label='Training Accuracy', color='blue')
plt.plot(error['val_binary_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Training and Validation Accuracy', fontsize=15)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Akurasi', fontsize=12)
plt.legend()

# Menampilkan plot
plt.tight_layout()
plt.show()
```
### 4.2 Evaluasi Model pada Test Set
Kita menggunakan data pengujian (test set) untuk mengevaluasi model secara keseluruhan, melihat nilai loss, accuracy, precision, dan recall.
```python
# Evaluasi model pada test set
results = model.evaluate(test_generator)

# Print hasil evaluasi
print(f'Loss: {results[0]}, Binary Accuracy: {results[1]}, Precision: {results[2]}, Recall: {results[3]}')
```
- Metrik yang ditampilkan meliputi:
    - Loss: Menunjukkan kesalahan prediksi pada data pengujian.
    - Binary Accuracy: Mengukur seberapa sering model memprediksi kelas yang benar.
    - Precision: Mengukur ketepatan prediksi kelas positif (anjing).
    - Recall: Mengukur sensitivitas model dalam mendeteksi kelas positif.
### 4.3 Confusion Matrix
Confusion matrix membantu memvisualisasikan jumlah prediksi benar dan salah yang dilakukan oleh model pada data pengujian, menunjukkan berapa banyak gambar yang diklasifikasikan dengan benar atau salah di antara kedua kelas (Cat dan Dog).
```python
# Mendapatkan prediksi untuk test set
predictions = model.predict(test_generator)
binary_predictions = (predictions > 0.5).astype(int).flatten() 

# Dapatkan label sebenarnya dari test set
true_labels = test_generator.classes 

# Buat confusion matrix
cm = confusion_matrix(true_labels, binary_predictions)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(cm, display_labels=['Cat', 'Dog']).plot(cmap='Blues')
plt.title('Confusion Matrix for Test Set')
plt.show()
```
### 4.4  Visualisasi Prediksi Model pada Gambar Test Set
Pada bagian ini, kita akan menampilkan beberapa gambar dari test set dengan label prediksi dan label sebenarnya untuk memeriksa kinerja model secara visual.
```python
# Mengambil satu batch dari test_generator
images, labels = next(test_generator)

# Membuat prediksi untuk batch ini
predictions = model.predict(images)
binary_predictions = (predictions > 0.5).astype(int).flatten()

# Menampilkan gambar dan prediksinya
plt.figure(figsize=(12, 12))
for i in range(len(images)):
    plt.subplot(4, 4, i + 1) 
    plt.imshow(images[i])
    plt.axis('off')
    pred_label = 'Dog' if binary_predictions[i] == 1 else 'Cat'
    true_label = 'Dog' if labels[i] == 1 else 'Cat'
    plt.title(f'Pred: {pred_label}\nTrue: {true_label}', color="green" if pred_label == true_label else "red")
    if i == 15: 
        break

plt.tight_layout()
plt.show()
```
