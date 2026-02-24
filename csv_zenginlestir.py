"""
Mevcut sp500_deep_learning_raw_data.csv dosyasini (9 sutun) okuyarak,
teknik gostergeleri (RSI, MACD, Bollinger, Volume_Change_5d) hesaplayip
zenginlestirilmis 16 sutunlu yeni CSV ureten yardimci betik.

Bu betik yfinance'e bagli DEGILDIR — mevcut Duzeltilmis_Kapanis verisini kullanir.

Kullanim:
    conda run -n tf_projesi python csv_zenginlestir.py
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# TEKNIK GOSTERGE FONKSIYONLARI
# ==============================================================================

def hesapla_rsi(seri: pd.Series, periyot: int = 14) -> pd.Series:
    """RSI (Wilder's EMA yontemi)"""
    delta = seri.diff()
    kazanc = delta.clip(lower=0)
    kayip = (-delta).clip(lower=0)
    ort_kazanc = kazanc.ewm(alpha=1/periyot, min_periods=periyot, adjust=False).mean()
    ort_kayip = kayip.ewm(alpha=1/periyot, min_periods=periyot, adjust=False).mean()
    rs = ort_kazanc / ort_kayip
    return 100 - (100 / (1 + rs))


def hesapla_macd(seri: pd.Series, hizli=12, yavas=26, sinyal=9) -> pd.DataFrame:
    """MACD, Histogram ve Sinyal cizgisi"""
    ema_h = seri.ewm(span=hizli, adjust=False).mean()
    ema_y = seri.ewm(span=yavas, adjust=False).mean()
    macd = ema_h - ema_y
    sinyal_c = macd.ewm(span=sinyal, adjust=False).mean()
    return pd.DataFrame({
        'MACD_12_26_9': macd,
        'MACDh_12_26_9': macd - sinyal_c,
        'MACDs_12_26_9': sinyal_c
    })


def hesapla_bollinger(seri: pd.Series, periyot=20, std_c=2.0) -> pd.DataFrame:
    """Bollinger Alt ve Ust Bantlari"""
    sma = seri.rolling(window=periyot).mean()
    std = seri.rolling(window=periyot).std()
    return pd.DataFrame({
        'BBL_20_2.0': sma - (std_c * std),
        'BBU_20_2.0': sma + (std_c * std)
    })


# ==============================================================================
# ANA ISLEM
# ==============================================================================

if __name__ == "__main__":
    PROJE_DIZINI = os.path.dirname(os.path.abspath(__file__))
    CSV_GIRDI = os.path.join(PROJE_DIZINI, "sp500_deep_learning_raw_data.csv")
    CSV_CIKTI = os.path.join(PROJE_DIZINI, "sp500_deep_learning_raw_data.csv")  # Ayni dosyanin ustune
    
    logger.info(f"CSV okunuyor: {CSV_GIRDI}")
    df = pd.read_csv(CSV_GIRDI)
    logger.info(f"Okunan satir: {len(df)}, Sutunlar: {list(df.columns)}")
    
    # Yeni sutunlarin zaten olup olmadigini kontrol et
    if 'RSI_14' in df.columns:
        logger.info("CSV zaten zenginlestirilmis! Islem yapilmadi.")
        exit(0)
    
    sonuc_listesi = []
    
    # Her (Olay_Ismi, Hisse) grubu icin teknik gostergeleri hesapla
    gruplar = df.groupby(['Olay_Ismi', 'Hisse'])
    toplam = len(gruplar)
    logger.info(f"Toplam {toplam} grup islenecek...")
    
    for i, ((olay, hisse), grup_df) in enumerate(gruplar):
        grup_df = grup_df.sort_values('T0_Goreceli_Gun').copy()
        kapanis = grup_df['Duzeltilmis_Kapanis'].reset_index(drop=True)
        
        # RSI
        grup_df['RSI_14'] = hesapla_rsi(kapanis).values
        
        # MACD
        macd = hesapla_macd(kapanis)
        grup_df['MACD_12_26_9'] = macd['MACD_12_26_9'].values
        grup_df['MACDh_12_26_9'] = macd['MACDh_12_26_9'].values
        grup_df['MACDs_12_26_9'] = macd['MACDs_12_26_9'].values
        
        # Bollinger
        bb = hesapla_bollinger(kapanis)
        grup_df['BBL_20_2.0'] = bb['BBL_20_2.0'].values
        grup_df['BBU_20_2.0'] = bb['BBU_20_2.0'].values
        
        # Volume_Change_5d — Hacim verisi CSV'de yok, NaN olarak birak
        grup_df['Volume_Change_5d'] = np.nan
        
        # Pencereyi t=-30 ile t=+30 arasina filtrele
        grup_df = grup_df[
            (grup_df['T0_Goreceli_Gun'] >= -30) &
            (grup_df['T0_Goreceli_Gun'] <= 30)
        ].copy()
        
        sonuc_listesi.append(grup_df)
        
        if (i + 1) % 100 == 0:
            logger.info(f"  {i+1}/{toplam} grup islendi...")
    
    # Birlestir
    son_df = pd.concat(sonuc_listesi, ignore_index=True)
    
    # Sutun siralamasini ayarla
    sutun_sirasi = [
        'Tarih', 'Olay_Ismi', 'Hisse', 'Grup', 'T0_Goreceli_Gun',
        'Duzeltilmis_Kapanis', 'Log_Getiri', 'Volatilite_10g', 'Volatilite_30g',
        'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'BBL_20_2.0', 'BBU_20_2.0', 'Volume_Change_5d'
    ]
    son_df = son_df[sutun_sirasi]
    
    # NaN raporu
    nan_ozet = son_df.isnull().sum()
    nan_olan = nan_ozet[nan_ozet > 0]
    if len(nan_olan) > 0:
        logger.warning(f"NaN bulunan sutunlar:\n{nan_olan}")
    
    # Kaydet
    son_df.to_csv(CSV_CIKTI, index=False)
    logger.info(f"Zenginlestirilmis CSV kaydedildi: {CSV_CIKTI}")
    logger.info(f"Toplam satir: {len(son_df)}, Sutun sayisi: {len(son_df.columns)}")
    logger.info(f"Sutunlar: {list(son_df.columns)}")
    
    # Kontrol: t=-30 ile t=+30 arasinda kac benzersiz olay var
    logger.info(f"Benzersiz olay sayisi: {son_df['Olay_Ismi'].nunique()}")
    logger.info(f"Benzersiz hisse sayisi: {son_df['Hisse'].nunique()}")
