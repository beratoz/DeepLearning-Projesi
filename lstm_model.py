# ==============================================================================
# S&P 500 (^GSPC) Yon Tahmini — BiLSTM + Attention + Focal Loss (Faz 7)
# ==============================================================================
# Kalibrasyon: alpha=0.75 -> 0.60 (perma-bear fix) + 2. BiLSTM katmani
#
# Mimari: Input -> BiLSTM(64) -> Drop(0.3) -> BiLSTM(32) -> Drop(0.3)
#         -> Attention -> Dense(32,relu) -> Dense(1,sigmoid)
# Loss  : Custom Focal Loss (gamma=2.0, alpha=0.60)
# Eval  : Dinamik Threshold Tuning [0.35, 0.40, 0.45, 0.50, 0.55]
# Split : Kronolojik olay bolmesi (%80 Train / %20 Test)
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional, Layer
)
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

# Tekrarlanabilirlik
np.random.seed(42)
tf.random.set_seed(42)

# ==============================================================================
# SABITLER
# ==============================================================================

CSV_DOSYA_ADI = "sp500_deep_learning_massive_data.csv"

OZELLIK_SUTUNLARI = [
    'Log_Getiri', 'Volatilite_10g', 'Volatilite_30g',
    'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
    'BBL_20_2.0', 'BBU_20_2.0'
]

LOOKBACK_GUN = 14
HEDEF_UFUK_GUN = 10
EGITIM_ORANI = 0.80
ESIK_DEGERLERI = [0.35, 0.40, 0.45, 0.50, 0.55]


# ==============================================================================
# CUSTOM ATTENTION LAYER
# ==============================================================================
# Bu katman, BiLSTM'in cikti dizisindeki her zaman adimina (t=-14..t=0)
# bir "dikkat agirligi" (attention weight) atar.
#
# Matematiksel olarak:
#   e_t = tanh(W * h_t + b)           -> her zaman adimi icin skor
#   alpha_t = softmax(e_t)             -> skorlar normalize edilip agirlik olur
#   context = sum(alpha_t * h_t)       -> agirlikli toplam = baglam vektoru
#
# Ufak ama cok etkili: Model, enflasyon verisi aciklanan gune %80 dikkat
# edebilirken, sessiz gunleri gormezden gelebilir.
# ==============================================================================

class AttentionLayer(Layer):
    """
    Bahdanau-tipi (additive) Attention katmani.
    
    Girdi : 3D tensor (batch_size, time_steps, features)  — BiLSTM ciktisi
    Cikti : 2D tensor (batch_size, features)               — agirlikli baglam vektoru
    
    Ek cikti: Attention agirliklari (attention_weights) — sonradan
              gorsellestirilmek uzere saklanir.
    """
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # input_shape: (batch_size, time_steps, features)
        feature_dim = input_shape[-1]
        
        # Ogrenilecek agirliklar: W (features -> features), b (bias), u (skor vektoru)
        self.W = self.add_weight(
            name='attention_W',
            shape=(feature_dim, feature_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_b',
            shape=(feature_dim,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_u',
            shape=(feature_dim, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        """
        inputs: (batch_size, time_steps, features) — BiLSTM sequence ciktisi
        
        Adimlar:
        1. Her zaman adimi icin skor hesapla: e_t = tanh(h_t @ W + b)
        2. Skorlari tek boyuta indir: score = e_t @ u
        3. Softmax ile normalize et: alpha = softmax(score)
        4. Agirlikli toplam: context = sum(alpha * h_t)
        """
        # Step 1: e_t = tanh(h_t @ W + b)
        # inputs @ W -> (batch, time_steps, features)
        e = K.tanh(K.dot(inputs, self.W) + self.b)
        
        # Step 2: score = e_t @ u -> (batch, time_steps, 1)
        score = K.dot(e, self.u)
        
        # Step 3: alpha = softmax(score) -> (batch, time_steps, 1)
        alpha = K.softmax(score, axis=1)
        
        # Agirliklari sakla (gorsellestirilme icin)
        self.attention_weights = alpha
        
        # Step 4: context = sum(alpha * h_t) -> (batch, features)
        context = K.sum(alpha * inputs, axis=1)
        
        return context
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config


# ==============================================================================
# CUSTOM FOCAL LOSS (Faz 5'ten — DEGISTIRILMEDI)
# ==============================================================================

def focal_loss(gamma=2.0, alpha=0.60):
    """
    Focal Loss: Azinlik sinifini (Dusus) cezalandirir (alpha=0.60, kalibre edildi),
    kolay orneklerin (perma-bull) loss'unu suppress eder.
    """
    def focal_loss_fn(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        pos_loss = -y_true * (1 - alpha) * K.pow(1.0 - y_pred, gamma) * K.log(y_pred)
        neg_loss = -(1 - y_true) * alpha * K.pow(y_pred, gamma) * K.log(1.0 - y_pred)
        return K.mean(pos_loss + neg_loss)
    return focal_loss_fn


# ==============================================================================
# 1. VERI YUKLEME
# ==============================================================================

def veri_yukle_ve_filtrele(csv_yolu: str) -> pd.DataFrame:
    """CSV'den ^GSPC verisi yukler ve NaN temizler."""
    print("=" * 70)
    print("[BILGI] CSV okunuyor...")
    df = pd.read_csv(csv_yolu)
    print(f"[BILGI] Toplam: {len(df):,} satir")
    df_gspc = df[df["Hisse"] == "^GSPC"].copy()
    print(f"[BILGI] ^GSPC: {len(df_gspc):,} satir")
    onceki = len(df_gspc)
    df_gspc = df_gspc.dropna(subset=OZELLIK_SUTUNLARI)
    print(f"[BILGI] NaN temizleme: {onceki - len(df_gspc)} silindi -> {len(df_gspc):,}")
    print(f"[BILGI] Olay: {df_gspc['Olay_Ismi'].nunique()}")
    print("=" * 70)
    return df_gspc


# ==============================================================================
# 2. HEDEF DEGISKEN
# ==============================================================================

def hedef_degisken_olustur(df: pd.DataFrame, ufuk: int = HEDEF_UFUK_GUN) -> pd.DataFrame:
    """t+1..t+ufuk kumulatif Log_Getiri -> Label (0/1)."""
    print(f"\n[BILGI] Hedef: t+1..t+{ufuk} kumulatif getiri")
    label_dict = {}
    for olay, odf in df.groupby('Olay_Ismi'):
        mask = (odf['T0_Goreceli_Gun'] > 0) & (odf['T0_Goreceli_Gun'] <= ufuk)
        label_dict[olay] = 1 if odf.loc[mask, 'Log_Getiri'].sum() > 0 else 0
    df = df.copy()
    df['Label'] = df['Olay_Ismi'].map(label_dict)
    y1 = sum(1 for v in label_dict.values() if v == 1)
    y0 = sum(1 for v in label_dict.values() if v == 0)
    print(f"[BILGI] Yukselis={y1} ({y1/(y0+y1)*100:.1f}%), Dusus={y0} ({y0/(y0+y1)*100:.1f}%)")
    return df


# ==============================================================================
# 3. KRONOLOJIK BOLME + TENSOR
# ==============================================================================

def kronolojik_bolme_ve_tensor(df, lookback=LOOKBACK_GUN, egitim_orani=EGITIM_ORANI):
    """Kronolojik olay bolmesi ve tensor uretimi."""
    print(f"\n{'='*70}")
    print(f"[BILGI] Kronolojik bolme: {lookback+1} gun lookback")
    print(f"{'='*70}")
    
    olay_tarihleri = {}
    for olay, odf in df.groupby('Olay_Ismi'):
        t0 = odf[odf['T0_Goreceli_Gun'] == 0]
        olay_tarihleri[olay] = t0['Tarih'].iloc[0] if len(t0) > 0 else odf['Tarih'].iloc[0]
    
    sirali = sorted(olay_tarihleri.keys(), key=lambda x: olay_tarihleri[x])
    kesim = int(len(sirali) * egitim_orani)
    eg_o = sirali[:kesim]
    te_o = sirali[kesim:]
    print(f"[BILGI] Egitim: {len(eg_o)} | Test: {len(te_o)}")
    
    def tensorler(olay_listesi):
        X, Y, basarili = [], [], []
        beklenen = lookback + 1
        for olay in olay_listesi:
            odf = df[df['Olay_Ismi'] == olay].sort_values('T0_Goreceli_Gun')
            p = odf[(odf['T0_Goreceli_Gun'] >= -lookback) & (odf['T0_Goreceli_Gun'] <= 0)]
            if len(p) < beklenen: continue
            label = odf['Label'].iloc[0]
            if pd.isna(label): continue
            x = p.tail(beklenen)[OZELLIK_SUTUNLARI].values
            if np.isnan(x).any(): continue
            X.append(x); Y.append(int(label)); basarili.append(olay)
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), basarili
    
    X_eg, Y_eg, b_eg = tensorler(eg_o)
    X_te, Y_te, b_te = tensorler(te_o)
    print(f"[BILGI] X_egitim: {X_eg.shape} | Y: 1={int(Y_eg.sum())}, 0={int(len(Y_eg)-Y_eg.sum())}")
    print(f"[BILGI] X_test  : {X_te.shape} | Y: 1={int(Y_te.sum())}, 0={int(len(Y_te)-Y_te.sum())}")
    return X_eg, X_te, Y_eg, Y_te, b_eg, b_te


# ==============================================================================
# 4. OLCEKLENDIRME
# ==============================================================================

def veriyi_olceklendir(X_eg, X_te):
    """StandardScaler (sadece egitim fit)."""
    print("\n[BILGI] Olceklendirme...")
    n, z, f = X_eg.shape
    s = StandardScaler()
    X_eg_s = s.fit_transform(X_eg.reshape(-1, f)).reshape(n, z, f)
    X_te_s = s.transform(X_te.reshape(-1, f)).reshape(X_te.shape[0], z, f)
    return X_eg_s, X_te_s


# ==============================================================================
# 5. BiLSTM + ATTENTION + FOCAL LOSS MODEL
# ==============================================================================

def model_olustur(girdi_sekli: tuple):
    """
    BiLSTM x2 + Custom Attention + Focal Loss modeli.
    
    Mimari (Faz 7 — kalibre edildi):
        Input(15, 9)
        -> Bidirectional(LSTM(64, return_sequences=True))
        -> Dropout(0.3)
        -> Bidirectional(LSTM(32, return_sequences=True))  [YENİ]
        -> Dropout(0.3)                                    [YENİ]
        -> AttentionLayer
        -> Dense(32, relu)
        -> Dense(1, sigmoid)
    """
    print(f"\n{'='*70}")
    print("[BILGI] BiLSTM x2 + ATTENTION + FOCAL LOSS (Faz 7 Kalibre)")
    print(f"[BILGI] Girdi: {girdi_sekli}")
    print(f"{'='*70}")

    # Functional API
    girdi = Input(shape=girdi_sekli, name="Girdi")
    
    # BiLSTM Katman 1 (64 birim) — return_sequences=True
    x = Bidirectional(
        LSTM(64, return_sequences=True),
        name="BiLSTM_64"
    )(girdi)
    x = Dropout(0.3, name="Dropout_1")(x)
    
    # BiLSTM Katman 2 (32 birim) — return_sequences=True (Attention icin)
    x = Bidirectional(
        LSTM(32, return_sequences=True),
        name="BiLSTM_32"
    )(x)
    x = Dropout(0.3, name="Dropout_2")(x)
    
    # CUSTOM ATTENTION — dizideki en kritik zaman adimini bulur
    attention_layer = AttentionLayer(name="Attention")
    x = attention_layer(x)
    
    # Dense katmanlar
    x = Dense(32, activation='relu', name="Dense_32")(x)
    cikti = Dense(1, activation='sigmoid', name="Cikis")(x)
    
    # Model
    model = Model(inputs=girdi, outputs=cikti, name="SP500_BiLSTM_Attention_v2")
    
    # Focal Loss ile derle (alpha=0.60, kalibre edildi)
    model.compile(
        optimizer="adam",
        loss=focal_loss(gamma=2.0, alpha=0.60),
        metrics=["accuracy"]
    )

    print("[BILGI] DERLENDI")
    print("[BILGI] Loss: Focal Loss (gamma=2.0, alpha=0.60) — kalibre edildi")
    print("[BILGI] Mimari: BiLSTM(64)->Drop->BiLSTM(32)->Drop->ATTENTION->Dense(32)->Sigmoid")
    print("\n--- Model Ozeti ---")
    model.summary()
    
    # Attention agirliklarini cikaracak alt model
    attention_model = Model(
        inputs=girdi,
        outputs=attention_layer.output
    )
    
    return model, attention_layer, attention_model


# ==============================================================================
# 6. EGITIM
# ==============================================================================

def modeli_egit(model, X_eg, Y_eg, epoch_sayisi=100, parti_boyutu=16, sabir=15):
    """Focal Loss ile egitim (class_weight YOK)."""
    print(f"\n{'='*70}")
    print(f"[BILGI] Egitim (Focal Loss, Attention aktif)")
    print(f"[BILGI] Epoch: {epoch_sayisi} | Batch: {parti_boyutu} | Sabir: {sabir}")
    print(f"{'='*70}")

    gecmis = model.fit(
        X_eg, Y_eg,
        epochs=epoch_sayisi,
        batch_size=parti_boyutu,
        validation_split=0.2,
        callbacks=[EarlyStopping(
            monitor="val_loss", patience=sabir,
            restore_best_weights=True, verbose=1
        )],
        verbose=1
    )
    print(f"\n[BILGI] Egitim bitti. Epoch: {len(gecmis.history['loss'])}")
    return gecmis


# ==============================================================================
# 7. DINAMIK ESIK DEGERLENDIRME (Faz 5'ten — DEGISTIRILMEDI)
# ==============================================================================

def dinamik_esik_degerlendirme(Y_prob, Y_gercek, esik_degerleri=ESIK_DEGERLERI):
    """Her esik icin confusion matrix + sinif 0 Recall/F1 raporlar."""
    print(f"\n{'#'*70}")
    print(f"#  DINAMIK ESIK TUNING: {esik_degerleri}")
    print(f"{'#'*70}")
    
    sinif_isimleri = ["Dusus (0)", "Yukselis (1)"]
    sonuclar = {}
    
    for esik in esik_degerleri:
        Y_t = (Y_prob > esik).astype(int).flatten()
        acc = accuracy_score(Y_gercek, Y_t)
        km = confusion_matrix(Y_gercek, Y_t, labels=[0, 1])
        rapor = classification_report(Y_gercek, Y_t, target_names=sinif_isimleri,
                                       output_dict=True, zero_division=0)
        
        r0, f0, p0 = rapor['Dusus (0)']['recall'], rapor['Dusus (0)']['f1-score'], rapor['Dusus (0)']['precision']
        r1, f1 = rapor['Yukselis (1)']['recall'], rapor['Yukselis (1)']['f1-score']
        
        sonuclar[esik] = {'accuracy': acc, 'cm': km, 'tahminler': Y_t,
                          'recall_0': r0, 'f1_0': f0, 'precision_0': p0,
                          'recall_1': r1, 'f1_1': f1}
        
        print(f"\n{'='*70}")
        print(f"  ESIK = {esik} | Accuracy: {acc:.4f}")
        print(f"{'='*70}")
        print(f"                     Tahmin:0   Tahmin:1")
        print(f"    Gercek:0 (Dusus)    {km[0,0]:>5}      {km[0,1]:>5}")
        print(f"    Gercek:1 (Yukselis) {km[1,0]:>5}      {km[1,1]:>5}")
        print(f"  Dusus (0)  -> P:{p0:.4f} R:{r0:.4f} F1:{f0:.4f}")
        print(f"  Yukselis(1)-> R:{r1:.4f} F1:{f1:.4f}")
        
        if r0 == 0:
            print(f"  *** PERMA-BULL! ***")
        elif r0 >= 0.3:
            print(f"  >> Dusus yakalaniyor! <<")
    
    en_iyi = max(sonuclar.items(), key=lambda x: x[1]['f1_0'])
    print(f"\n{'#'*70}")
    print(f"#  ONERILEN ESIK: {en_iyi[0]} (Dusus F1: {en_iyi[1]['f1_0']:.4f})")
    print(f"{'#'*70}")
    
    return sonuclar


# ==============================================================================
# 8. ATTENTION AGIRLIKLARI GORSELLESTIRME
# ==============================================================================

def attention_agirliklarini_cizdir(
    model,
    attention_layer: AttentionLayer,
    X_test: np.ndarray,
    test_olaylari: list,
    lookback: int = LOOKBACK_GUN,
    kayit_dizini: str = "."
) -> None:
    """
    Attention agirliklarini gorsellestirir.
    Her zaman adimina (t=-14..t=0) modelin ne kadar dikkat ettigini gosterir.
    
    2 grafik uretir:
    1. Ortalama attention agirliklari (tum test seti uzerinden)
    2. Ilk 6 olayin bireysel attention profilleri
    """
    print(f"\n[BILGI] Attention agirliklari hesaplaniyor...")
    
    # Attention katmanindan agirliklari cikar
    # Forward pass yaparak attention_weights'i hesaplat
    attention_weight_model = Model(
        inputs=model.input,
        outputs=attention_layer.output
    )
    
    # Tum test seti icin attention skorlarini hesapla
    # Bunun icin attention_weights attribute'unu kullanacagiz
    # Attention'in ciktisi context vector, bize agirliklar lazim
    # Agirliklari almak icin ara katman ciktisi gerek
    
    # Yontem: Her ornek icin forward pass yapip attention_weights'i al
    # Attention katmaninin weights'ini kullanarak manual hesaplama
    W = attention_layer.get_weights()[0]  # (features, features)
    b = attention_layer.get_weights()[1]  # (features,)
    u = attention_layer.get_weights()[2]  # (features, 1)
    
    # BiLSTM ciktisini al (Attention oncesi — 2. Dropout'un ciktisi)
    bilstm_model = Model(
        inputs=model.input,
        outputs=model.get_layer("Dropout_2").output
    )
    bilstm_out = bilstm_model.predict(X_test, verbose=0)  # (batch, time, features)
    
    # Attention agirliklarini hesapla
    e = np.tanh(np.dot(bilstm_out, W) + b)  # (batch, time, features)
    score = np.dot(e, u).squeeze(-1)          # (batch, time)
    
    # Softmax
    exp_score = np.exp(score - np.max(score, axis=1, keepdims=True))
    alpha = exp_score / np.sum(exp_score, axis=1, keepdims=True)  # (batch, time)
    
    zaman_etiketleri = [f"t={i}" for i in range(-lookback, 1)]
    
    # --- GRAFIK 1: Ortalama Attention Agirliklari ---
    ort_alpha = alpha.mean(axis=0)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    renkler = plt.cm.YlOrRd(ort_alpha / ort_alpha.max())
    bars = ax.bar(zaman_etiketleri, ort_alpha, color=renkler, edgecolor='gray', linewidth=0.5)
    
    # En yuksek agirliktaki gunu isaretle
    max_idx = np.argmax(ort_alpha)
    bars[max_idx].set_edgecolor('red')
    bars[max_idx].set_linewidth(2.5)
    
    ax.set_title("Ortalama Attention Agirliklari (Tum Test Seti)\n"
                 "Model hangi gunlere daha cok dikkat ediyor?",
                 fontsize=13, pad=15)
    ax.set_xlabel("Goreceli Gun (t=0 = Olay Gunu)", fontsize=11)
    ax.set_ylabel("Attention Agirligi", fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # En onemli gunu annote et
    ax.annotate(f'En onemli:\n{zaman_etiketleri[max_idx]}',
                xy=(max_idx, ort_alpha[max_idx]),
                xytext=(max_idx, ort_alpha[max_idx] + 0.01),
                fontsize=10, ha='center', fontweight='bold', color='red')
    
    plt.tight_layout()
    yol1 = os.path.join(kayit_dizini, "attention_ortalama.png")
    plt.savefig(yol1, dpi=150)
    print(f"[BILGI] Ortalama attention grafigi: {yol1}")
    plt.close()
    
    # --- GRAFIK 2: Bireysel Olay Attention Profilleri (ilk 6) ---
    n_goster = min(6, len(test_olaylari))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(n_goster):
        renkler_i = plt.cm.YlOrRd(alpha[i] / alpha[i].max())
        axes[i].bar(range(len(zaman_etiketleri)), alpha[i],
                    color=renkler_i, edgecolor='gray', linewidth=0.3)
        axes[i].set_title(test_olaylari[i][:35], fontsize=9, fontweight='bold')
        axes[i].set_xticks(range(0, len(zaman_etiketleri), 3))
        axes[i].set_xticklabels([zaman_etiketleri[j] for j in range(0, len(zaman_etiketleri), 3)],
                                fontsize=7)
        axes[i].grid(True, alpha=0.2)
    
    plt.suptitle("Bireysel Olay Attention Profilleri", fontsize=14, fontweight='bold')
    plt.tight_layout()
    yol2 = os.path.join(kayit_dizini, "attention_bireysel.png")
    plt.savefig(yol2, dpi=150, bbox_inches='tight')
    print(f"[BILGI] Bireysel attention grafigi: {yol2}")
    plt.close()
    
    # Konsola ozet
    print(f"\n[BILGI] Attention Ozet (ortalama agirliklar):")
    for i, (gun, ag) in enumerate(zip(zaman_etiketleri, ort_alpha)):
        bar = "█" * int(ag * 200)
        print(f"  {gun:>5}: {ag:.4f} {bar}")


# ==============================================================================
# 9. DETAYLI DEGERLENDIRME
# ==============================================================================

def modeli_degerlendir(model, X_test, Y_test, test_olaylari=None, kayit_dizini="."):
    """Threshold tuning + en iyi esikle olay bazinda detay."""
    Y_prob = model.predict(X_test, verbose=0)
    sonuclar = dinamik_esik_degerlendirme(Y_prob, Y_test.astype(int))
    
    en_iyi_esik = max(sonuclar.items(), key=lambda x: x[1]['f1_0'])[0]
    Y_tahmin = sonuclar[en_iyi_esik]['tahminler']
    Y_gercek = Y_test.astype(int)
    km = sonuclar[en_iyi_esik]['cm']
    
    if test_olaylari and len(test_olaylari) == len(Y_gercek):
        print(f"\n{'='*70}")
        print(f"OLAY BAZINDA TAHMIN (Esik={en_iyi_esik})")
        print(f"{'='*70}")
        print(f"{'Olay':<50} {'Gercek':>7} {'Tahmin':>7} {'Prob':>7} {'Sonuc':>7}")
        print("-" * 80)
        dogru = 0
        for i, olay in enumerate(test_olaylari):
            g, t, p = int(Y_gercek[i]), int(Y_tahmin[i]), float(Y_prob[i][0])
            s = "DOGRU" if g == t else "YANLIS"
            if g == t: dogru += 1
            print(f"{olay:<50} {g:>7} {t:>7} {p:>7.4f} {s:>7}")
        print(f"\n[OZET] Dogru: {dogru}/{len(test_olaylari)} "
              f"({dogru/len(test_olaylari)*100:.1f}%) | Esik: {en_iyi_esik}")

    sinif_isimleri = ["Dusus (0)", "Yukselis (1)"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(km, annot=True, fmt="d", cmap="Blues",
                xticklabels=sinif_isimleri, yticklabels=sinif_isimleri,
                linewidths=0.5, annot_kws={"size": 16})
    plt.title(f"Confusion Matrix\n[BiLSTM+Attention+FocalLoss | Esik={en_iyi_esik}]",
              fontsize=14, pad=15)
    plt.xlabel("Tahmin"); plt.ylabel("Gercek")
    plt.tight_layout()
    yol = os.path.join(kayit_dizini, "karisiklik_matrisi.png")
    plt.savefig(yol, dpi=150)
    print(f"\n[BILGI] Confusion matrix: {yol}")
    plt.close()
    
    return Y_prob


def egitim_gecmisini_cizdir(gecmis, kayit_dizini="."):
    """Egitim grafikleri."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(gecmis.history["loss"], label="Egitim", lw=2, color="#2196F3")
    ax[0].plot(gecmis.history["val_loss"], label="Dogrulama", lw=2, color="#FF5722", ls="--")
    ax[0].set_title("Focal Loss"); ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss"); ax[0].legend(); ax[0].grid(True, alpha=0.3)
    ax[1].plot(gecmis.history["accuracy"], label="Egitim", lw=2, color="#4CAF50")
    ax[1].plot(gecmis.history["val_accuracy"], label="Dogrulama", lw=2, color="#9C27B0", ls="--")
    ax[1].set_title("Dogruluk"); ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy"); ax[1].legend(); ax[1].grid(True, alpha=0.3)
    plt.suptitle("BiLSTM + Attention Egitim Gecmisi", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    yol = os.path.join(kayit_dizini, "egitim_gecmisi.png")
    plt.savefig(yol, dpi=150, bbox_inches="tight")
    print(f"[BILGI] Egitim grafigi: {yol}")
    plt.close()


# ==============================================================================
# 10. FINANSAL BACKTEST MOTORU
# ==============================================================================

def olay_getirilerini_hesapla(
    df: pd.DataFrame,
    test_olaylari: list,
    ufuk: int = HEDEF_UFUK_GUN
) -> dict:
    """
    Her test olayi icin t+1..t+ufuk arasindaki kumulatif Log_Getiri'yi hesaplar.
    
    Dondurur:
        {olay_ismi: kumulatif_log_getiri} sozlugu
    """
    getiriler = {}
    for olay in test_olaylari:
        odf = df[df['Olay_Ismi'] == olay]
        mask = (odf['T0_Goreceli_Gun'] > 0) & (odf['T0_Goreceli_Gun'] <= ufuk)
        getiriler[olay] = odf.loc[mask, 'Log_Getiri'].sum()
    return getiriler


def finansal_backtest(
    Y_prob: np.ndarray,
    Y_gercek: np.ndarray,
    test_olaylari: list,
    olay_getirileri: dict,
    esik: float,
    kayit_dizini: str = "."
) -> None:
    """
    Finansal Backtest Motoru.
    
    Buy & Hold: Tum olaylarda pozisyon al -> kumulatif getiri
    Model Long/Short: Sinyal=1 -> Long (getiriyi ekle),
                      Sinyal=0 -> Short (getiriyi tersine cevir)
    
    Parametreler:
        Y_prob: Model sigmoid olasuliklari
        Y_gercek: Gercek etiketler
        test_olaylari: Kronolojik olay listesi
        olay_getirileri: {olay: kumulatif_log_getiri} sozlugu
        esik: Sinyal esigi (orn: 0.50)
        kayit_dizini: Grafik kayit dizini
    """
    print(f"\n{'#'*70}")
    print(f"#  FINANSAL BACKTEST MOTORU")
    print(f"#  Esik: {esik} | Strateji: Long/Short | Olaylar: {len(test_olaylari)}")
    print(f"{'#'*70}")
    
    Y_sinyal = (Y_prob > esik).astype(int).flatten()
    
    # Her olayin getirisi
    bh_getirileri = []   # Buy & Hold
    model_getirileri = [] # Model stratejisi
    olay_detay = []       # Detay tablosu
    
    for i, olay in enumerate(test_olaylari):
        olay_getiri = olay_getirileri.get(olay, 0.0)
        sinyal = int(Y_sinyal[i])
        gercek = int(Y_gercek[i])
        prob = float(Y_prob[i][0])
        
        # Buy & Hold: Her zaman Long
        bh_getirileri.append(olay_getiri)
        
        # Model Stratejisi: Long/Short
        if sinyal == 1:
            # LONG: Yukselis beklentisi -> getiriyi aynen al
            model_getiri = olay_getiri
            islem = "LONG"
        else:
            # SHORT: Dusus beklentisi -> getiriyi tersine cevir
            model_getiri = -olay_getiri
            islem = "SHORT"
        
        model_getirileri.append(model_getiri)
        
        # Trade basarili mi?
        basarili = (sinyal == 1 and olay_getiri > 0) or (sinyal == 0 and olay_getiri <= 0)
        
        olay_detay.append({
            'olay': olay, 'sinyal': sinyal, 'islem': islem,
            'gercek': gercek, 'prob': prob,
            'olay_getiri': olay_getiri, 'model_getiri': model_getiri,
            'basarili': basarili
        })
    
    # Kumulatif getiriler
    bh_kum = np.cumsum(bh_getirileri)
    model_kum = np.cumsum(model_getirileri)
    
    # ========================================================================
    # RISK METRIKLERI HESAPLAMA
    # ========================================================================
    bh_arr = np.array(bh_getirileri)
    model_arr = np.array(model_getirileri)
    
    # --- Sharpe Ratio (Yilliklandirilmis) ---
    # Sharpe = (Ort_Getiri / Std_Getiri) * sqrt(252)
    # Risk-free rate = 0 kabul edildi
    def sharpe_hesapla(getiriler):
        if len(getiriler) < 2 or np.std(getiriler) == 0:
            return 0.0
        return (np.mean(getiriler) / np.std(getiriler)) * np.sqrt(252)
    
    bh_sharpe = sharpe_hesapla(bh_arr)
    model_sharpe = sharpe_hesapla(model_arr)
    
    # --- Maksimum Cokus (Max Drawdown) ---
    # MDD = min( (kumul_t - max(kumul_0..t)) / max(kumul_0..t) )
    def mdd_hesapla(kumulatif_seri):
        # Kumulatif log getiriyi fiyat serisine cevir
        fiyat = np.exp(kumulatif_seri)  # P(t) = e^(cumsum(log_returns))
        zirve = np.maximum.accumulate(fiyat)
        drawdown = (fiyat - zirve) / zirve
        return np.min(drawdown) * 100  # Yuzde olarak
    
    bh_mdd = mdd_hesapla(bh_kum)
    model_mdd = mdd_hesapla(model_kum)
    # ========================================================================
    # PROFESYONEL FINANSAL RAPOR
    # ========================================================================
    toplam_trade = len(test_olaylari)
    long_sayisi = sum(1 for d in olay_detay if d['islem'] == 'LONG')
    short_sayisi = sum(1 for d in olay_detay if d['islem'] == 'SHORT')
    basarili_sayisi = sum(1 for d in olay_detay if d['basarili'])
    win_rate = basarili_sayisi / toplam_trade * 100 if toplam_trade > 0 else 0
    bh_win = sum(1 for d in olay_detay if d['olay_getiri'] > 0)
    bh_win_rate = bh_win / toplam_trade * 100 if toplam_trade > 0 else 0
    
    bh_toplam = bh_kum[-1] * 100
    model_toplam = model_kum[-1] * 100
    alfa = model_toplam - bh_toplam
    
    print(f"\n{'='*70}")
    print(f"  PROFESYONEL FINANSAL PERFORMANS RAPORU")
    print(f"{'='*70}")
    print(f"")
    print(f"  +{'─'*39}+{'─'*17}+{'─'*17}+")
    print(f"  | {'Metrik':<37} | {'S&P 500 B&H':>15} | {'LSTM Model':>15} |")
    print(f"  +{'─'*39}+{'─'*17}+{'─'*17}+")
    print(f"  | {'Kumulatif Getiri':<37} | {bh_toplam:>+14.2f}% | {model_toplam:>+14.2f}% |")
    print(f"  | {'Sharpe Orani (Yillik, Rf=0)':<37} | {bh_sharpe:>15.3f} | {model_sharpe:>15.3f} |")
    print(f"  | {'Maksimum Cokus (Max Drawdown)':<37} | {bh_mdd:>14.2f}% | {model_mdd:>14.2f}% |")
    print(f"  | {'Win Rate':<37} | {bh_win_rate:>14.1f}% | {win_rate:>14.1f}% |")
    print(f"  +{'─'*39}+{'─'*17}+{'─'*17}+")
    print(f"  | {'Alfa (Model - Piyasa)':<37} | {alfa:>+32.2f}% |")
    print(f"  +{'─'*39}+{'─'*35}+")
    print(f"")
    print(f"  +{'─'*39}+{'─'*17}+")
    print(f"  | {'Trade Istatistikleri':<37} | {'Deger':>15} |")
    print(f"  +{'─'*39}+{'─'*17}+")
    print(f"  | {'Toplam Trade':<37} | {toplam_trade:>15} |")
    print(f"  | {'  Long (Yukselis Sinyali)':<37} | {long_sayisi:>15} |")
    print(f"  | {'  Short (Dusus Sinyali)':<37} | {short_sayisi:>15} |")
    print(f"  | {'Basarili Trade':<37} | {basarili_sayisi:>15} |")
    print(f"  | {'Basarisiz Trade':<37} | {toplam_trade - basarili_sayisi:>15} |")
    print(f"  +{'─'*39}+{'─'*17}+")
    
    # Sonuc yorumu
    print(f"")
    if alfa > 0:
        print(f"  >> MODEL PIYASAYI YENDI! Alfa: +{alfa:.2f}% <<")
    else:
        print(f"  ** Model piyasanin altinda kaldi. Alfa: {alfa:.2f}% **")
    
    if model_sharpe > bh_sharpe:
        print(f"  >> Model daha iyi riske gore getiri sunuyor (Sharpe: {model_sharpe:.3f} vs {bh_sharpe:.3f}) <<")
    
    if abs(model_mdd) < abs(bh_mdd):
        print(f"  >> Model daha dusuk maksimum cokus yasadi (MDD: {model_mdd:.2f}% vs {bh_mdd:.2f}%) <<")
    
    # ========================================================================
    # OLAY BAZINDA TRADE DETAYI
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"  TRADE DETAYI (Kronolojik Sira)")
    print(f"{'='*70}")
    print(f"  {'Olay':<40} {'Islem':>6} {'Prob':>6} {'Getiri':>8} {'Model':>8} {'Sonuc':>7}")
    print(f"  {'-'*77}")
    
    for d in olay_detay:
        sonuc = "WIN" if d['basarili'] else "LOSS"
        print(f"  {d['olay']:<40} {d['islem']:>6} {d['prob']:>6.3f} "
              f"{d['olay_getiri']*100:>+7.2f}% {d['model_getiri']*100:>+7.2f}% {sonuc:>7}")
    
    # ========================================================================
    # GORSELLESTIRME: backtest_sonucu.png
    # ========================================================================
    fig, ax = plt.subplots(figsize=(16, 7))
    
    x = range(len(test_olaylari))
    
    # Buy & Hold cizgisi
    ax.plot(x, bh_kum * 100, label="S&P 500 Buy & Hold",
            color="#2196F3", linewidth=2.5, alpha=0.8)
    ax.fill_between(x, 0, bh_kum * 100, alpha=0.1, color="#2196F3")
    
    # Model stratejisi cizgisi
    ax.plot(x, model_kum * 100, label="LSTM Model Stratejisi",
            color="#FF5722", linewidth=2.5, alpha=0.8)
    ax.fill_between(x, 0, model_kum * 100, alpha=0.1, color="#FF5722")
    
    # Sifir cizgisi
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Etiketler
    ax.set_title(f"Finansal Backtest: Buy & Hold vs LSTM Model\n"
                 f"[BiLSTM+Attention+FocalLoss | Esik={esik} | {toplam_trade} Trade]",
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Test Olayi (Kronolojik Sira)", fontsize=12)
    ax.set_ylabel("Kumulatif Getiri (%)", fontsize=12)
    
    # X ekseni etiketleri (her 5 olayda bir)
    step = max(1, len(test_olaylari) // 10)
    tick_pos = list(range(0, len(test_olaylari), step))
    tick_lab = [test_olaylari[i][:20] for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=45, ha='right', fontsize=7)
    
    # Son noktalari annote et
    ax.annotate(f"B&H: {bh_toplam:+.2f}%",
                xy=(len(test_olaylari)-1, bh_kum[-1]*100),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold', color='#2196F3',
                arrowprops=dict(arrowstyle='->', color='#2196F3'))
    ax.annotate(f"Model: {model_toplam:+.2f}%",
                xy=(len(test_olaylari)-1, model_kum[-1]*100),
                xytext=(10, -15), textcoords='offset points',
                fontsize=10, fontweight='bold', color='#FF5722',
                arrowprops=dict(arrowstyle='->', color='#FF5722'))
    
    # Win Rate + Risk metrikleri kutusu
    textstr = (f"Model Sharpe: {model_sharpe:.3f}\n"
               f"B&H Sharpe: {bh_sharpe:.3f}\n"
               f"Model MDD: {model_mdd:.1f}%\n"
               f"B&H MDD: {bh_mdd:.1f}%\n"
               f"Win Rate: {win_rate:.1f}%\n"
               f"Alfa: {alfa:+.2f}%")
    props = dict(boxstyle='round,pad=0.6', facecolor='lightyellow',
                 edgecolor='gray', alpha=0.9)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace', bbox=props)
    
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    yol = os.path.join(kayit_dizini, "backtest_sonucu.png")
    plt.savefig(yol, dpi=150, bbox_inches='tight')
    print(f"\n[BILGI] Backtest grafigi: {yol}")
    plt.close()


# ==============================================================================
# 11. ANA CALISTIRMA
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  S&P 500 YON TAHMINI — BiLSTM x2 + ATTENTION + FOCAL LOSS (Faz 7)")
    print("#  alpha=0.60 (kalibre) | 2x BiLSTM | Attention | Threshold Tuning")
    print("#" * 70 + "\n")

    PROJE = os.path.dirname(os.path.abspath(__file__))
    CSV = os.path.join(PROJE, CSV_DOSYA_ADI)

    # 1. Veri yukle
    df = veri_yukle_ve_filtrele(CSV)

    # 2. Hedef degisken
    df = hedef_degisken_olustur(df, ufuk=HEDEF_UFUK_GUN)

    # 3. Kronolojik bolme
    X_eg, X_te, Y_eg, Y_te, eg_o, te_o = kronolojik_bolme_ve_tensor(df)

    if len(X_eg) < 10 or len(X_te) < 3:
        print(f"[HATA] Yetersiz! Eg:{len(X_eg)}, Te:{len(X_te)}")
        exit(1)

    # 4. Olceklendirme
    X_eg, X_te = veriyi_olceklendir(X_eg, X_te)

    # 5. Model (BiLSTM + Attention + Focal Loss)
    model, attention_layer, _ = model_olustur((X_eg.shape[1], X_eg.shape[2]))

    # 6. Egitim
    gecmis = modeli_egit(model, X_eg, Y_eg, epoch_sayisi=100, parti_boyutu=16, sabir=15)

    # 7. Egitim grafikleri
    egitim_gecmisini_cizdir(gecmis, kayit_dizini=PROJE)

    # 8. Detayli degerlendirme + Threshold Tuning
    Y_prob = modeli_degerlendir(model, X_te, Y_te, test_olaylari=te_o, kayit_dizini=PROJE)

    # 9. ATTENTION GORSELLESTIRME
    attention_agirliklarini_cizdir(
        model=model,
        attention_layer=attention_layer,
        X_test=X_te,
        test_olaylari=te_o,
        lookback=LOOKBACK_GUN,
        kayit_dizini=PROJE
    )

    # 10. FINANSAL BACKTEST
    # En iyi esigi threshold tuning'den sec
    sonuclar = dinamik_esik_degerlendirme(Y_prob, Y_te.astype(int))
    en_iyi_esik = max(sonuclar.items(), key=lambda x: x[1]['f1_0'])[0]
    
    # Olay bazli getirileri hesapla
    olay_getirileri = olay_getirilerini_hesapla(df, te_o, ufuk=HEDEF_UFUK_GUN)
    
    # Backtest calistir
    finansal_backtest(
        Y_prob=Y_prob,
        Y_gercek=Y_te,
        test_olaylari=te_o,
        olay_getirileri=olay_getirileri,
        esik=en_iyi_esik,
        kayit_dizini=PROJE
    )

    print("\n" + "#" * 70)
    print("#  ISLEM TAMAMLANDI")
    print("#" * 70 + "\n")
