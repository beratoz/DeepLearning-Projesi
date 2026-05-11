# S&P 500 Yön Tahmini: Feedforward Neural Network

> **T+5 gün sonra S&P 500 sektör ETF'lerinin yönünü (yükseliş / düşüş) tahmin eden bir ikili sınıflandırma modeli.**
> Gürültülü finansal zaman serilerinden öğrenilebilir sinyal çıkarmaya yönelik deneysel bir derin öğrenme çalışmasıdır.

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-API-red?logo=keras&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Tamamland%C4%B1-success)

---

## 📌 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Problem Tanımı](#-problem-tanımı)
- [Veri Seti ve Özellikler](#-veri-seti-ve-özellikler)
- [Model Mimarisi](#-model-mimarisi)
- [Sonuçlar](#-sonuçlar)
- [Deneylerin Karşılaştırmalı Özeti](#-deneylerin-karşılaştırmalı-özeti)
- [RNN/LSTM ile Karşılaştırma](#-rnn--lstm-ile-karşılaştırma)
- [Çıkarılan Dersler](#-çıkarılan-dersler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Proje Yapısı](#-proje-yapısı)
- [Lisans](#-lisans)

---

## 🎯 Proje Hakkında

Bu proje, **finansal zaman serisi tahmininin sınırlarını** ileri beslemeli (feedforward) bir sinir ağı üzerinden deneysel olarak araştırır. Çalışmanın amacı yüksek doğruluk peşinde koşmak değil; **gürültülü piyasa verisinde derin öğrenmenin nasıl davrandığını**, hiperparametre değişikliklerinin ne kadar etkili olduğunu ve veri kalitesinin model kapasitesinden daha belirleyici olup olmadığını anlamaktır.

**Öne Çıkan Bulgular:**

| Bulgu | Değer |
|---|---|
| 🏆 En İyi Test Accuracy | **%56.77** |
| 📊 Baseline Test Accuracy | %55.84 |
| 🔬 Toplam Deneme Sayısı | 6 farklı konfigürasyon |
| 📈 Test Edilen Veri Seti | 2 (~257.000 ve ~17.500 satır) |
| 🧠 En İyi Mimari | Dense(64) → Dense(32) → Sigmoid |

> 📚 Literatürde S&P 500 tipi endekslerin günlük/haftalık yön tahmininde **%54-58 aralığı normal** kabul edilir. %60 üstü değerler çoğunlukla veri sızıntısı (data leakage) işaretidir. Bu çalışmanın sonuçları literatürle tutarlıdır.

---

## ❓ Problem Tanımı

Finansal piyasalar, **düşük sinyal-gürültü oranına (low signal-to-noise)** sahip karmaşık sistemlerdir. Fiyat hareketleri; makro haberler, likidite şokları, sektör rotasyonu ve sayısız psikolojik etmenin süperpozisyonudur.

**Hedef Değişken:**

```
Hedef = 1   eğer  Fiyat(T+5) > Fiyat(T)     (Yükseliş)
Hedef = 0   aksi halde                       (Düşüş)
```

**Neden Zor?**
- Rastgele tahmin tabanı ~%50
- Volatilite rejim değişiklikleri genellemeyi sürekli tehdit eder
- Her 1 puanlık gerçek iyileşme ticari uygulamada anlamlıdır

---

## 📊 Veri Seti ve Özellikler

### Veri Seti
- **~257.000 satır** günlük fiyat verisi
- S&P 500 **sektör ETF'leri**
- Makro olay pencereleri etrafında derlenmiş

### Kullanılan 9 Teknik Gösterge

| Özellik | Türkçe Açılımı | Modele Ne Söyler? |
|---|---|---|
| `Log_Getiri` | Logaritmik getiri | Fiyatın bir önceki güne göre oransal değişimi |
| `Volatilite_10g` | 10 günlük oynaklık | Kısa vadeli dalgalanma şiddeti |
| `Volatilite_30g` | 30 günlük oynaklık | Orta vadeli dalgalanma şiddeti |
| `RSI_14` | Göreceli Güç Endeksi | Aşırı alım / aşırı satım sinyali (0–100) |
| `MACD_12_26_9` | MACD çizgisi | Trendin yönü ve gücü |
| `MACDh_12_26_9` | MACD histogram | Trend ivmesi |
| `MACDs_12_26_9` | MACD sinyal | Yön değişim sinyali |
| `BBL_20_2.0` | Bollinger alt bandı | İstatistiksel "aşırı düşüş" sınırı |
| `BBU_20_2.0` | Bollinger üst bandı | İstatistiksel "aşırı yükseliş" sınırı |

### Leakage-Safe Pipeline 🔒

```python
# Zaman serisi bütünlüğü için karıştırma YOK
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Ölçeklendirme sadece eğitim setine fit edilir
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)   # fit DEĞİL, transform!
```

> ⚠️ `shuffle=False` ve eğitim setinden öğrenilmiş scaler, geleceği bugüne taşımayı (data leakage) önleyerek gerçekçi metrikler üretir.

---

## 🧠 Model Mimarisi

```
Input (9 özellik)
   │
   ▼
Dense(64) ── ReLU
   │
Dropout(0.5)
   │
   ▼
Dense(32) ── ReLU
   │
Dropout(0.5)
   │
   ▼
Dense(1) ── Sigmoid  →  P(Yükseliş)
```

**Hiperparametreler:**

| Parametre | Değer |
|---|---|
| Optimizer | Adam (lr=0.001) |
| Loss | Binary Crossentropy |
| Batch Size | 64 |
| Max Epoch | 50 |
| Early Stopping | patience=10, `restore_best_weights=True` |
| Toplam Parametre | **2.753** |

**Neden Küçük Model?** Finansal veride gerçek sinyal seyrektir; büyük modeller gürültüyü ezberlemeye eğilimlidir (Deneme 5'te bu açıkça gözlemlendi — bkz. aşağıdaki tablo).

---

## 📈 Sonuçlar

### Baseline Model — 9 Özellikli Büyük Veri Seti

| Metrik | Değer |
|---|---|
| Test Accuracy | **%55.84** |
| Test Loss | 0.6866 |
| AUC | 0.514 |
| Eğitim Epoch | 12 (Early Stopping) |
| Train-Val Fark | 0.0032 (sağlıklı genelleme) |

### En İyi Model — 7 Özellikli Hacim Bilgili Veri Seti

| Metrik | Değer |
|---|---|
| Test Accuracy | **🏆 %56.77** |
| Test Loss | 0.6838 |
| Train-Val Fark | 0.0020 (denemeler arası en düşük) |

> 💡 **Paradoks:** 13 kat daha az veriyle daha iyi sonuç. Sebep: **Hacim bilgisi** (`Log_Hacim`, `Hacim_Degisimi`), fiyat türevli MACD/Bollinger'dan farklı yeni bilgi taşıyor.

---

## 🧪 Deneylerin Karşılaştırmalı Özeti

| # | Konfigürasyon | Test Loss | Test Acc | Durum |
|---|---|---|---|---|
| 1 | Adam, küçük model (baseline) | — | — | ✅ Baseline |
| 2 | Adam lr=0.0001, batch=128 | 0.6865 | %55.87 | ✅ Kabul edilebilir |
| 3 | AdamW varsayılan | 0.6868 | %55.93 | ✅ Kabul edilebilir |
| 4 | AdamW düşük LR + uzun eğitim (121 epoch) | 0.6867 | %55.77 | ✅ Kabul edilebilir |
| 5 | Büyük model (12.417 param) + BN + swish | 0.6911 | %55.45 | ❌ **Overfitting** |
| 6 | **Yeni veri seti (hacim bilgisi)** | **0.6838** | **%56.77** | 🏆 **EN İYİ** |

### Deneme 5'in Hikayesi (Başarısızlık Anatomisi)

Deneme 5'te parametre sayısını 2.753 → 12.417'ye (4.5 kat) çıkardık: BatchNormalization, swish aktivasyon, daha geniş katmanlar. Sonuç klasik overfitting — train loss düşerken val loss tırmandı ("makas açıldı"). Veri seti büyük olsa da (236k satır), **bilgi yoğunluğu** büyük modeli besleyecek kadar yüksek değildi.

---

## 🔬 RNN / LSTM ile Karşılaştırma

Bu projede gözlemlediğim sınırı anlamlandırmak için **Samuel Edet'in RNN/LSTM/GRU tabanlı akademik çalışmasıyla** karşılaştırma yaptım:

| Özellik | Bu Proje (Feedforward) | Edet (RNN/LSTM/GRU) |
|---|---|---|
| Hafıza | ❌ Yok | ✅ Var (hidden state) |
| Girdi yapısı | 2D `(Örneklem, Özellik)` | 3D `(Örneklem, Zaman, Özellik)` |
| Test Accuracy | **%55–57** | **%74–75** |
| Zaman serisi uygunluğu | Düşük | Yüksek |

**Çıkarım:** Feedforward mimari her satırı bağımsız bir örneklem olarak işler — geçmişin akışını göremez. Finansal zaman serisinde bilginin önemli kısmı **ardışık değişim dizilerinde** saklıdır. ~%20'lik başarı farkının asıl kaynağı budur.

> 🌦️ *Analoji:* Feedforward, yarınki havayı tahmin ederken sadece bugünün gazetesine bakar. LSTM, son bir haftanın arşivini zihninde tutar.

---

## 🎓 Çıkarılan Dersler

### 1️⃣ Küçük Model Her Zaman Daha Kötü Değildir
Finansal veride sinyal zayıftır; büyük modeller gürültüyü ezberler. 2.500–3.000 parametreli kompakt bir ağ, 12.000+ parametreli ağdan **daha iyi** test performansı verdi.

### 2️⃣ Hiperparametre Ayarlamanın Sınırı Vardır
Optimizer (Adam → AdamW), learning rate, batch size, epoch — hepsi değiştirildi. Sonuç hep %55.8 civarında kaldı. Hiperparametre ayarının kazandırabileceği maksimum yaklaşık **%1–2 accuracy**'dir.

### 3️⃣ Veri Kalitesi > Veri Miktarı
13 kat daha az satırlık veri seti, daha iyi sonuç verdi. **Özellik mühendisliği**, model mühendisliğinden değerli çıktı.

### 4️⃣ Grafik Okuması Şart
Sadece son sayılara bakmak yanıltıcıdır. Train/Val eğrilerinin birlikte hareket etmesi, makasın açılıp açılmaması, val_loss minimum noktası — hepsi karar verici.

### 5️⃣ Finansal Tahminin Gerçek Limitleri
Literatürde günlük yön tahmininde %54–58 normaldir. %60+ değerler çoğunlukla **veri sızıntısı** işaretidir. %56.77 sonucumuz literatürle tutarlı bir başarıdır.

---

## ⚙️ Kurulum

```bash
# Repoyu klonla
git clone https://github.com/beratoz/sp500-feedforward-prediction.git
cd sp500-feedforward-prediction

# Sanal ortam oluştur (önerilir)
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# Gerekli paketleri kur
pip install -r requirements.txt
```

### Gereksinimler

```
tensorflow==2.10.1
numpy==1.23.5
pandas==2.3.1
scikit-learn
matplotlib
```

---

## ▶️ Kullanım

Jupyter Notebook'u açıp hücreleri sırayla çalıştırın:

```bash
jupyter notebook sp500_feedforward_dense_model1.ipynb
```

Notebook akışı:

1. **Adım 1:** Veriyi okuma ve hedef değişken (T+5 yön) oluşturma
2. **Adım 2:** Özellik seçimi (9 teknik gösterge)
3. **Adım 3:** Train/Test split (`shuffle=False`)
4. **Adım 4:** StandardScaler (leakage-safe)
5. **Adım 5:** Model kurulumu ve eğitim
6. **Adım 6:** Loss/Accuracy grafikleri ve test değerlendirmesi

---

## 📁 Proje Yapısı

```
sp500-feedforward-prediction/
│
├── 📓 sp500_feedforward_dense_model1.ipynb   # Ana notebook
├── 📊 data/
│   └── sp500_deep_learning_massive_data.csv  # ~257k satırlık veri
├── 📑 docs/
│   ├── SP500_DerinOgrenme_Sunum.pdf          # Proje sunumu
│   ├── SP500_Denemeler_Raporu.pdf            # 6 denemenin detaylı raporu
│   └── ssrn_makalesi_karsilastirma.pdf       # RNN/LSTM ile karşılaştırma
├── 🖼️ images/
│   └── egitim_grafikleri.png                 # Loss & Accuracy grafikleri
├── 📄 requirements.txt
├── 📄 LICENSE
└── 📄 README.md
```

---

## 🔭 İleri Adımlar

- [ ] **LSTM / GRU** ile tekrarlayan sinir ağı denemesi (zamansal hafıza eklemek)
- [ ] **class_weight** ile sınıf dengesizliğinin yönetimi
- [ ] **Walk-forward validation** ile zaman bazlı çapraz doğrulama
- [ ] **Makro değişken** entegrasyonu (VIX, faiz, dolar endeksi)
- [ ] **Threshold optimization** (0.50 yerine ROC üzerinden optimal eşik)
- [ ] **Ensemble** yaklaşımları (Feedforward + LSTM oylama)

---

## 📚 Referanslar

- Edet, S. — *RNN-Based Stock Price Direction Prediction with S&P 500 Data*
- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

---

## 📜 Lisans

Bu proje **MIT Lisansı** altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakınız.

---

## 👤 İletişim

Projeyle ilgili soru, öneri veya iş birliği için issue açabilir ya da bana ulaşabilirsiniz.

> *"Finansal tahminde model değişiklikleri sınırlı kazanım verir. Asıl kazanım özellik mühendisliğinden ve veri kalitesinden gelir."*

⭐ Faydalı bulduysanız repoya yıldız vermeyi unutmayın!
