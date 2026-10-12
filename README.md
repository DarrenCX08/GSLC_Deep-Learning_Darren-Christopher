# Tugas 05 – Transfer Learning untuk Klasifikasi Citra dengan Data Terbatas

**Nama:** Darren Christopher | **NIM:** 2802522304  
Mata kuliah: Deep Learning – Sesi 05 *Advanced CNNs and Transfer Learning*

## Deskripsi
Perbandingan tiga paradigma pelatihan ResNet-18 pada subset Oxford-IIIT Pet (10 kelas) dengan data terbatas, ditambah uji kontrol augmentasi dan arsitektur pembanding.

| Kode | Kondisi |
|---|---|
| B1 | Baseline from scratch |
| B2 | Feature extraction (backbone beku, hanya `fc` dilatih) |
| B3 | Fine-tuning `layer4` + `fc` dengan differential learning rate (1e-4 / 1e-3) |
| B4 | B3 tanpa augmentasi (uji kontrol augmentasi) |
| B5 | EfficientNet-B0 (bonus) |

## Dataset dan Protokol
- Oxford-IIIT Pet, 10 kelas (5 kucing, 5 anjing): Abyssinian, Beagle, Bengal, Bombay, Boxer, British Shorthair, English Setter, Shiba Inu, Siamese, Wheaten Terrier.
- 1.980 citra, split stratified 70/15/15: 1.386 train / 297 validasi / 297 test.
- Input 224×224, normalisasi ImageNet, batch 32, AdamW (weight decay 1e-2), CrossEntropyLoss.
- Early stopping pada validation loss (patience 4, maks. 20 epoch). Seed tetap 42 (Python, NumPy, PyTorch).
- Augmentasi (RandomResizedCrop, Horizontal Flip, Rotation 15°, Color Jitter) hanya pada train set.

## Ringkasan Hasil
| Kode | Trainable params | Epoch terbaik / total | Durasi (s) | Val acc | Test acc | Test Macro F1 |
|---|---|---|---|---|---|---|
| B1 | 11.181.642 | 20 / 20* | 292,1 | 0,5219 | 0,5623 | 0,5575 |
| B2 | 5.130 | 15 / 19 | 250,2 | 0,9596 | 0,9663 | 0,9661 |
| B3 | 8.398.858 | 4 / 8 | 105,1 | 0,9596 | 0,9731 | 0,9731 |
| B4 | 8.398.858 | 8 / 12 | 117,9 | 0,9764 | 0,9731 | 0,9731 |
| B5 | 1.142.202 | 11 / 15 | 199,6 | 0,9933 | 0,9899 | 0,9900 |

\*B1 berhenti di batas 20 epoch (belum memicu early stopping). Sumber angka: `hasil_eksperimen.csv`.

## Struktur Folder
```
Tugas 5_2802522304_Darren Christopher/
├── README.md
├── requirements.txt
├── Bagian A/
│   └── Bagian A GSLC Deep Learning.docx      # analisis konseptual
├── Bagian B/
│   └── Bagian B Deep Learning.ipynb          # notebook eksperimen (output tersimpan)
├── Bagian C/
│   └── Bagian C GSLC Deep Learning.docx      # laporan dan analisis
├── hasil_eksperimen.csv                      # ringkasan metrik per kondisi
├── kurva_training.json                       # log mentah per epoch (semua kondisi)
├── misklasifikasi_B3.csv                     # 8 sampel salah prediksi model B3
├── kurva_loss.png                            # train vs val loss B1-B3
├── failure_analysis_B3.png                   # 8 kesalahan model B3
├── failure_analysis.png                      # 3 kesalahan model B5 (terbaik)
├── Advanced CNNs and Transfer Learning.pptx  # materi kuliah (rujukan)
└── Tugas Take Home - Sesi 05 Advanced CNNs and Transfer Learning.pdf   # soal tugas
```

## Cara Menjalankan
1. Buka `Bagian B/Bagian B Deep Learning.ipynb` di Google Colab, pilih *Runtime > Change runtime type > T4 GPU*.
2. Jalankan *Runtime > Run all*. Dataset diunduh otomatis lewat `torchvision.datasets.OxfordIIITPet`.
3. Untuk lingkungan lokal: `pip install -r requirements.txt`, lalu jalankan notebook dengan Jupyter.

## Catatan dan Keterbatasan
- Hasil berbasis single seed (42); varians antar-run tidak diuji.
- Test set hanya 297 citra, sehingga selisih 1-2 citra antar kondisi tidak signifikan.
- B1 belum konvergen penuh pada batas 20 epoch.
- Penggunaan AI diungkapkan pada akhir laporan (Bagian C).

## Referensi
- He et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
- Tan & Le (2019). EfficientNet. ICML.
- Parkhi et al. (2012). Cats and Dogs. CVPR.
- Materi kuliah Sesi 05 – Advanced CNNs and Transfer Learning.
