# Mini Search Engine - Sistem Temu Kembali Informasi

Project ini merupakan implementasi sederhana sistem temu kembali informasi menggunakan Python. Sistem ini dapat membaca dokumen teks, melakukan preprocessing dengan stemming bahasa Indonesia, membangun indeks menggunakan TF-IDF, dan melakukan pencarian dokumen yang relevan berdasarkan input pengguna.

## Fitur

- Preprocessing teks: lowercase, stopword removal, stemming
- Indexing dengan TF-IDF
- Query pencarian dengan cosine similarity
- Bahasa Indonesia

## Cara Menjalankan

1. Letakkan dokumen teks di folder `documents/`
2. Jalankan perintah:

```
python main.py
```

## Library yang Digunakan

- scikit-learn
- Sastrawi
