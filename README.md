# 📈 S&P 500 — Derin Öğrenme ile Makro Olay Etkisi Analizi

> **S&P 500 sektör ETF'lerinin makro-ekonomik ve jeopolitik olaylar etrafındaki fiyat hareketlerini derin öğrenme modelleri ile analiz eden bir proje.**

---

## 📋 İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Proje Yapısı](#proje-yapısı)
- [Veri Setleri](#veri-setleri)
- [Model Mimarisi](#model-mimarisi)
- [Kullanılan Teknolojiler](#kullanılan-teknolojiler)
- [Deneysel Sonuçlar](#deneysel-sonuçlar)
- [Kurulum ve Çalıştırma](#kurulum-ve-çalıştırma)
- [Raporlar](#raporlar)
- [Lisans](#lisans)

---

## 🎯 Proje Hakkında

Bu proje, S&P 500 endeksi ile ilişkili hisse senetleri ve emtia (commodities) verilerini, belirli makroekonomik ve jeopolitik olaylar etrafında analiz ederek **T+5 gün sonra fiyatın yükselip yükselmeyeceğini** tahmin eden bir **İkili Sınıflandırma (Binary Classification)** modeli inşa etmektedir.

**Temel Amaç:**  
Makro olayların (seçimler, pandemiler, savaşlar vb.) sektör bazlı hisse senedi fiyatlarına etkisini derin öğrenme yöntemleri ile ölçmek ve gelecekteki fiyat yönünü tahmin etmektir.

### İncelenen Makro Olaylar

| Olay | Yıl |
|------|-----|
| ABD Seçimleri | 2016 |
| COVID-19 Pandemisi | 2020 |
| Rusya-Ukrayna Savaşı | 2022 |
| İsrail-İran Gerginliği | 2024 |
| ABD Seçimleri | 2024 |

### Analiz Edilen Hisseler / ETF'ler

`SPY`, `XLK`, `XLF`, `XLE`, `XLV`, `XLY`, `XLP`, `XLRE`, `ITA`, `GLD`, `CL=F`

---

## 📁 Proje Yapısı

```
DL_projesi/
├── sp500_feedforward_dense_model1.ipynb      # Ana model (büyük veri seti, ~257K satır)
├── sp500_feedforward_dense_ham_veri.ipynb     # Ham veri seti ile model (~17K satır)
├── sp500_sektor_analiz.ipynb                 # Sektör ETF analizi (istatistiksel çıkarım)
├── sp500_olay_calismasi_ham_veri.csv         # Ham veri seti (~17.5K satır)
├── sp500_deep_learning_massive_data.csv      # Büyük veri seti (~257K satır)
├── SP500_Denemeler_Raporu.pdf                # Denemeler raporu
├── SP500_DerinOgrenme_Sunum.pdf              # Sunum dosyası
│
├── denenen_modeller_ve_grafikleri/            # Denenen model versiyonları ve grafikleri
│   ├── sp500_feedforward_dense_model1_swis-verbose.ipynb
│   ├── sp500_feedforward_dense_model_v2.ipynb
│   ├── sp500_feedforward_dense_model_v3.ipynb
│   ├── 01_egitim_grafikleri.png
│   ├── egitim_grafikleri.png
│   ├── egitim_grafikleri_ham_veri.png
│   ├── egitim_grafikleri_v2.png
│   ├── egitim_grafikleri_v3.png
│   ├── loss_grafikleri.png
│   └── optimizer-ve-patience-batchsize-degisimi.png
│
├── Raporlar_ve_ornek/                        # Akademik raporlar ve referans çalışmalar
│   ├── SP500_Akademik_Karsilastirma_Raporu.pdf
│   ├── SP500_Dense_vs_LSTM_Rapor.pdf
│   ├── SP500_Karsilastirmali_Rapor.pdf
│   ├── SP500_Rapor_5Sayfa.pdf
│   └── ssrn_id300.pdf
│
└── README.md
```

---

## 📊 Veri Setleri

### 1. Ham Veri Seti (`sp500_olay_calismasi_ham_veri.csv`)

| Özellik | Değer |
|---------|-------|
| Satır sayısı | ~17.520 |
| Hisse/Enstrüman sayısı | 11 |
| Olay sayısı | 5 |
| Tarih aralığı | 2015-11-16 → 2025-01-29 |

**Özellikler (Features):**
- `Log_Getiri` — Logaritmik getiri
- `Volatilite_10g` — 10 günlük volatilite
- `Volatilite_30g` — 30 günlük volatilite
- `Log_Hacim` — Logaritmik hacim
- `Hacim_Degisimi` — Hacim değişimi
- `RSI_14` — 14 günlük RSI
- `SMA_Uzaklik_20` — 20 günlük SMA'dan uzaklık

### 2. Büyük Veri Seti (`sp500_deep_learning_massive_data.csv`)

| Özellik | Değer |
|---------|-------|
| Satır sayısı | ~257.603 |
| Sütun sayısı | 15 |

**Ek Özellikler:**
- `MACD_12_26_9` — MACD göstergesi
- `MACDh_12_26_9` — MACD histogram
- `MACDs_12_26_9` — MACD sinyal
- `BBL_20_2.0` — Bollinger Alt Bandı
- `BBU_20_2.0` — Bollinger Üst Bandı

---

## 🧠 Model Mimarisi

Projede **Feedforward (Dense) Yapay Sinir Ağı** kullanılmıştır.

### Mimari

```
Input (7 veya 9 özellik)
  → Dense (64, ReLU) → Dropout (0.5)
  → Dense (32, ReLU) → Dropout (0.5)
  → Dense (1, Sigmoid)
```

### Neden Bu Mimari?

- **Kompakt model:** Veri boyutu göz önüne alındığında derin/geniş mimariler overfitting riski taşır
- **Dropout (0.5):** Agresif regularizasyon ile ezberleme engellenir
- **ReLU:** Hızlı, basit ve standart aktivasyon fonksiyonu
- **Sigmoid:** 0-1 arası olasılık çıktısı (ikili sınıflandırma)

### Eğitim Parametreleri

| Parametre | Değer |
|-----------|-------|
| Kayıp Fonksiyonu | Binary Crossentropy |
| Optimizer | Adam (lr=0.001) |
| Metrik | Accuracy |
| Batch Size | 64 |
| Epochs | 50 (maksimum) |
| EarlyStopping | patience=10, restore_best_weights=True |
| Train/Test | %80 / %20 (shuffle=False) |

### Veri Sızıntısı Önleme (Data Leakage Prevention)

- Zaman serisi olduğu için `shuffle=False` kullanılmıştır
- `StandardScaler` **sadece eğitim setine** fit edilmiştir
- Her iki sete de aynı scaler ile `transform` uygulanmıştır

---

## 🛠 Kullanılan Teknolojiler

| Teknoloji | Sürüm / Açıklama |
|-----------|-------------------|
| **Python** | 3.9+ |
| **TensorFlow / Keras** | 2.10+ |
| **NumPy** | Sayısal hesaplamalar |
| **Pandas** | Veri işleme ve analiz |
| **Matplotlib** | Grafik ve görselleştirme |
| **Seaborn** | İstatistiksel görselleştirme |
| **Scikit-learn** | Ön işleme, model değerlendirme |
| **SciPy** | İstatistiksel testler |

---

## 📈 Deneysel Sonuçlar

### Ham Veri Seti (17K satır, 7 özellik)

| Metrik | Değer |
|--------|-------|
| Test Doğruluğu | ~%58.79 |
| Test Kaybı | ~0.6768 |
| Toplam Parametre | 2.625 |
| Eğitim Epoch | 27 (EarlyStopping: 17) |

### Büyük Veri Seti (257K satır, 9 özellik)

| Metrik | Değer |
|--------|-------|
| Test Doğruluğu | ~%55.84 |
| Test Kaybı | ~0.6866 |
| Toplam Parametre | 2.753 |
| Eğitim Epoch | 12 (EarlyStopping: 2) |

### Sektör Analizi

Sektör ETF'leri üzerinde yapılan analizde:
- **Anormal Getiri (AR)** hesabı yapılmıştır
- **Kümülatif Anormal Getiri (CAR)** analizi gerçekleştirilmiştir
- **İstatistiksel hipotez testi** (t-testi) ile olay öncesi ve sonrası getiriler karşılaştırılmıştır

---

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn tensorflow
```

### Çalıştırma

1. Repository'yi klonlayın:
```bash
git clone <repo-url>
cd DL_projesi
```

2. Jupyter Notebook'u başlatın:
```bash
jupyter notebook
```

3. İlgili notebook dosyasını açın:
   - `sp500_feedforward_dense_model1.ipynb` — Ana model (büyük veri seti)
   - `sp500_feedforward_dense_ham_veri.ipynb` — Ham veri seti modeli
   - `sp500_sektor_analiz.ipynb` — Sektör analizi

---

## 📑 Raporlar

Proje kapsamında hazırlanan raporlar `Raporlar_ve_ornek/` dizininde bulunmaktadır:

- **SP500_Akademik_Karsilastirma_Raporu.pdf** — Akademik karşılaştırma raporu
- **SP500_Dense_vs_LSTM_Rapor.pdf** — Dense vs LSTM karşılaştırma raporu
- **SP500_Karsilastirmali_Rapor.pdf** — Karşılaştırmalı rapor
- **SP500_Rapor_5Sayfa.pdf** — 5 sayfalık özet rapor

Ayrıca kök dizinde:
- **SP500_Denemeler_Raporu.pdf** — Tüm denemeleri kapsayan rapor
- **SP500_DerinOgrenme_Sunum.pdf** — Proje sunumu

---

## 📄 Lisans

Bu proje akademik amaçlarla geliştirilmiştir.

---
