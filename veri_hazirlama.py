# ==============================================================================
# DEVASA VERI URETIM SCRIPTI (Faz 3)
# ==============================================================================
# Bu script, LSTM modeli icin ~260K+ satirlik zengin veri seti uretir.
#
# 3 Adim:
#   1. Otomatik olay uretimi (TUFE + FOMC + mevcut sok olaylari)
#   2. yfinance ile fiyat indirme + teknik gosterge hesaplama
#   3. Tampon bolge mantigi ile t=-30/+30 filtreleme
#
# Cikti: sp500_deep_learning_massive_data.csv
# ==============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import logging
import os
import time
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Loglama yapilandirmasi
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# TEKNIK GOSTERGE FONKSIYONLARI (saf pandas/numpy)
# ==============================================================================

def hesapla_rsi(seri: pd.Series, periyot: int = 14) -> pd.Series:
    """RSI (Relative Strength Index) - Wilder's EMA yontemiyle."""
    delta = seri.diff()
    kazanc = delta.clip(lower=0)
    kayip = (-delta).clip(lower=0)
    ort_kazanc = kazanc.ewm(alpha=1/periyot, min_periods=periyot, adjust=False).mean()
    ort_kayip = kayip.ewm(alpha=1/periyot, min_periods=periyot, adjust=False).mean()
    rs = ort_kazanc / ort_kayip
    return 100 - (100 / (1 + rs))


def hesapla_macd(seri: pd.Series, hizli: int = 12, yavas: int = 26, sinyal: int = 9) -> pd.DataFrame:
    """MACD cizgisi, histogram ve sinyal cizgisi."""
    ema_h = seri.ewm(span=hizli, adjust=False).mean()
    ema_y = seri.ewm(span=yavas, adjust=False).mean()
    macd = ema_h - ema_y
    sinyal_c = macd.ewm(span=sinyal, adjust=False).mean()
    return pd.DataFrame({
        'MACD_12_26_9': macd,
        'MACDh_12_26_9': macd - sinyal_c,
        'MACDs_12_26_9': sinyal_c
    })


def hesapla_bollinger(seri: pd.Series, periyot: int = 20, std_c: float = 2.0) -> pd.DataFrame:
    """Bollinger Alt ve Ust Bantlari."""
    sma = seri.rolling(window=periyot).mean()
    std = seri.rolling(window=periyot).std()
    return pd.DataFrame({
        'BBL_20_2.0': sma - (std_c * std),
        'BBU_20_2.0': sma + (std_c * std)
    })


# ==============================================================================
# ADIM 1: OTOMATIK OLAY URETIM FONKSIYONLARI
# ==============================================================================

def tufe_olaylari_uret(baslangic_yil: int = 2000, bitis_yil: int = 2024) -> Dict[str, str]:
    """
    Her ayin 13. gununu yaklasik TUFE (CPI) aciklama gunu kabul ederek
    olay sozlugune ekler.
    
    Format: "YYYY_MM_TUFE" -> "YYYY-MM-13"
    
    Parametreler:
        baslangic_yil: Baslangic yili (dahil)
        bitis_yil: Bitis yili (dahil)
    
    Dondurur:
        Olay ismi -> tarih sozlugu
    """
    olaylar = {}
    for yil in range(baslangic_yil, bitis_yil + 1):
        for ay in range(1, 13):
            olay_ismi = f"{yil}_{ay:02d}_TUFE"
            tarih = f"{yil}-{ay:02d}-13"
            olaylar[olay_ismi] = tarih
    
    logger.info(f"TUFE olaylari uretildi: {len(olaylar)} adet ({baslangic_yil}-{bitis_yil})")
    return olaylar


def fomc_olaylari_uret(baslangic: str = "2000-01-01", bitis: str = "2024-12-31") -> Dict[str, str]:
    """
    FRED veritabanindan DFEDTAR (Fed Funds Target Rate) verisini ceker.
    Faizin degistigi (farkin sifirdan farkli oldugu) tarihleri bulur.
    
    Format: "YYYY_MM_FOMC" -> "YYYY-MM-DD"
    
    Parametreler:
        baslangic: Baslangic tarihi
        bitis: Bitis tarihi
    
    Dondurur:
        Olay ismi -> tarih sozlugu
    """
    olaylar = {}
    
    try:
        logger.info("FRED veritabanindan DFEDTAR verisi cekiliyor...")
        df = web.DataReader('DFEDTAR', 'fred', baslangic, bitis)
        
        # Faiz degisim tarihlerini bul
        degisim = df['DFEDTAR'].diff()
        degisen_tarihler = degisim[degisim != 0].dropna()
        
        for tarih in degisen_tarihler.index:
            yil = tarih.year
            ay = tarih.month
            gun = tarih.day
            olay_ismi = f"{yil}_{ay:02d}_FOMC"
            
            # Ayni ay icinde birden fazla degisim varsa gun numarasi ekle
            if olay_ismi in olaylar:
                olay_ismi = f"{yil}_{ay:02d}_{gun:02d}_FOMC"
            
            olaylar[olay_ismi] = tarih.strftime("%Y-%m-%d")
        
        logger.info(f"FOMC olaylari uretildi: {len(olaylar)} adet (DFEDTAR degisim tarihleri)")
        
    except Exception as e:
        logger.error(f"FRED verisi cekilirken hata: {e}")
        logger.warning("FOMC olaylari uretilemedi, bos sozluk donduruluyor.")
    
    return olaylar


def mevcut_sok_olaylari() -> Dict[str, str]:
    """
    Onceden tanimlanmis 49 spesifik makroekonomik/jeopolitik sok olayini dondurur.
    Bu olaylar sabit kodlanmistir ve degistirilmemelidir.
    """
    return {
        "2008_Bear_Stearns_Satisi": "2008-03-17",
        "2008_Lehman_Iflasi": "2008-09-15",
        "2008_TARP_Kurtarma_Reddi_Cokusu": "2008-09-29",
        "2009_Piyasa_Dibi_Rallisi": "2009-03-09",
        "2010_Yunanistan_Kurtarma": "2010-05-02",
        "2010_Flash_Crash": "2010-05-06",
        "2011_Fukushima_Felaketi": "2011-03-11",
        "2011_ABD_Not_Indirimi_Soku": "2011-08-08",
        "2012_Avrupa_Borc_Krizi_Draghi": "2012-07-26",
        "2013_Taper_Tantrum": "2013-05-22",
        "2014_Kirim_Ilhaki": "2014-02-27",
        "2014_Ebola_Korkusu_Cokusu": "2014-10-15",
        "2015_Isvicre_Frangi_Soku": "2015-01-15",
        "2015_Cin_Yuan_Devaluasyonu": "2015-08-11",
        "2015_Fed_Ilk_Faiz_Artisi": "2015-12-16",
        "2016_Brexit_Referandumu": "2016-06-24",
        "2016_ABD_Secimleri_Trump": "2016-11-08",
        "2018_Volmageddon_VIX_Soku": "2018-02-05",
        "2018_Cin_ABD_Ticaret_Savasi": "2018-03-22",
        "2018_Ekim_Teknoloji_Satis_Dalgasi": "2018-10-10",
        "2018_Fed_Faiz_Artisi_Hatasi": "2018-12-19",
        "2019_Getiri_Egrisi_Tersinmesi": "2019-08-14",
        "2019_Aramco_Saldirisi": "2019-09-16",
        "2020_Kasim_Suleymani_Suikasti": "2020-01-03",
        "2020_COVID19_Pandemi_Ilani": "2020-03-11",
        "2020_Fed_Acil_Sinirsiz_QE_Mudahalesi": "2020-03-23",
        "2020_Petrol_Fiyat_Cokusu": "2020-04-20",
        "2020_ABD_Secimleri_Biden": "2020-11-03",
        "2020_Pfizer_Asi_Rallisi": "2020-11-09",
        "2021_Kongre_Baskini": "2021-01-06",
        "2021_Meme_Stock_Cilginligi_GME": "2021-01-27",
        "2021_Archegos_Fonu_Cokusu": "2021-03-26",
        "2021_Evergrande_Krizi": "2021-09-20",
        "2021_Enflasyon_Gecici_Degil_Aciklamasi": "2021-11-30",
        "2022_Rusya_Ukrayna_Savasi": "2022-02-24",
        "2022_LME_Nikel_Soku": "2022-03-08",
        "2022_Fed_Agresif_Faiz_Artisi": "2022-06-15",
        "2022_ABD_Tufe_Enflasyon_Soku": "2022-09-13",
        "2022_Ingiltere_Tahvil_Krizi": "2022-09-28",
        "2022_FTX_Kripto_Iflasi": "2022-11-08",
        "2023_SVB_Banka_Iflasi": "2023-03-10",
        "2023_Credit_Suisse_Krizi": "2023-03-19",
        "2023_Fitch_ABD_Not_Indirimi": "2023-08-01",
        "2023_Israil_Hamas_Catismasi": "2023-10-09",
        "2023_ABD_Tahvil_Krizi_Yuzde5": "2023-10-19",
        "2024_Israil_Iran_Gerilimi": "2024-04-15",
        "2024_Japonya_Carry_Trade_Soku": "2024-08-05",
        "2024_Fed_50_Baz_Puan_Indirimi": "2024-09-18",
        "2024_ABD_Secimleri": "2024-11-05"
    }


def tum_olaylari_birlestir() -> Dict[str, str]:
    """
    Tum olay kaynaklarini birlestirip tek bir sozluk dondurur.
    Oncelik: Sok Olaylari > FOMC > TUFE (ayni tarih varsa sok olayi kalir)
    """
    # Once dusuk onceliklileri ekle, sonra yuksek oncelikliler ustune yazsin
    birlesik = {}
    
    # 1. TUFE olaylari (en dusuk oncelik)
    tufe = tufe_olaylari_uret(2000, 2024)
    birlesik.update(tufe)
    
    # 2. FOMC olaylari
    fomc = fomc_olaylari_uret("2000-01-01", "2024-12-31")
    birlesik.update(fomc)
    
    # 3. Sok olaylari (en yuksek oncelik — ayni isim varsa ustune yazar)
    sok = mevcut_sok_olaylari()
    birlesik.update(sok)
    
    logger.info(f"Toplam birlesmis olay sayisi: {len(birlesik)}")
    logger.info(f"  - TUFE: {len(tufe)}, FOMC: {len(fomc)}, Sok: {len(sok)}")
    
    return birlesik


# ==============================================================================
# ADIM 2: YFINANCE VERI INDIRME (TICKER BUG WORKAROUND)
# ==============================================================================

def ticker_verisi_indir(ticker: str, baslangic: str = "1999-01-01") -> Optional[pd.DataFrame]:
    """
    Tek bir ticker icin yfinance'den veri indirir.
    
    yfinance 1.1.0'daki ^^GSPC bug'i nedeniyle ticker'lar tek tek indirilir.
    Hata durumunda None dondurur.
    
    Parametreler:
        ticker: Hisse/endeks sembol (orn: "^GSPC", "XLF")
        baslangic: Veri baslangic tarihi
    
    Dondurur:
        Adj Close sutunu iceren DataFrame veya None
    """
    for deneme in range(3):  # 3 deneme hakki
        try:
            df = yf.download(ticker, start=baslangic, progress=False)
            
            if df is None or df.empty:
                logger.warning(f"  {ticker}: Bos veri dondu (deneme {deneme+1}/3)")
                time.sleep(2)
                continue
            
            # Sutun secimi: Adj Close > Close
            if isinstance(df.columns, pd.MultiIndex):
                # MultiIndex durumu (bu yfinance versiyonunda her zaman boyle)
                if 'Adj Close' in df.columns.get_level_values(0):
                    fiyat = df['Adj Close'].copy()
                else:
                    fiyat = df['Close'].copy()
                
                # MultiIndex'ten ticker ismini cikar
                if isinstance(fiyat, pd.DataFrame):
                    fiyat = fiyat.iloc[:, 0]  # Tek sutun al
            else:
                col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                fiyat = df[col].copy()
            
            # Series -> DataFrame cevirimi
            result = pd.DataFrame({'Fiyat': fiyat.values}, index=fiyat.index)
            result = result.ffill().bfill()
            
            if len(result) > 0:
                return result
            
        except Exception as e:
            logger.warning(f"  {ticker}: Indirme hatasi (deneme {deneme+1}/3): {e}")
            time.sleep(2)
    
    logger.error(f"  {ticker}: 3 denemede de basarisiz, atlaniyor.")
    return None


def tum_ticker_verilerini_indir(ticker_gruplari: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Tum ticker'lari tek tek indirip sozlukte dondurur.
    
    Parametreler:
        ticker_gruplari: {ticker: grup} sozlugu
    
    Dondurur:
        {ticker: DataFrame} sozlugu
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"yfinance uzerinden {len(ticker_gruplari)} ticker indiriliyor (tek tek)...")
    logger.info(f"{'='*70}")
    
    veri_cache = {}
    
    for ticker in tqdm(ticker_gruplari.keys(), desc="Ticker indirme", unit="ticker"):
        df = ticker_verisi_indir(ticker)
        if df is not None:
            veri_cache[ticker] = df
            logger.info(f"  {ticker}: {len(df)} satir indirildi "
                       f"({df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')})")
        else:
            logger.error(f"  {ticker}: BASARISIZ - veri alinamadi!")
    
    logger.info(f"\nBasarili ticker sayisi: {len(veri_cache)}/{len(ticker_gruplari)}")
    return veri_cache


# ==============================================================================
# ADIM 3: OLAY ISLEMLERI VE GOSTERGE HESAPLAMA
# ==============================================================================

def olaylari_isle(
    veri_cache: Dict[str, pd.DataFrame],
    ticker_gruplari: Dict[str, str],
    olaylar: Dict[str, str],
    tampon_eksi: int = 100,
    tampon_arti: int = 60,
    nihai_eksi: int = 30,
    nihai_arti: int = 30
) -> List[pd.DataFrame]:
    """
    Tum olaylari isler: tampon bolgeli pencere kesimi + teknik gosterge hesaplama.
    
    Pipeline (her olay-ticker cifti icin):
        1. Genis pencere (t=-tampon_eksi .. t=+tampon_arti) kesilir
        2. Teknik gostergeler hesaplanir (RSI, MACD, Bollinger)
        3. Nihai kesim (t=-nihai_eksi .. t=+nihai_arti) yapilir
    
    Parametreler:
        veri_cache: {ticker: DataFrame} indirilen veriler
        ticker_gruplari: {ticker: grup} sozlugu
        olaylar: {olay_ismi: tarih} sozlugu
        tampon_eksi/arti: Genis pencere sinirlari
        nihai_eksi/arti: Son kesim sinirlari
    
    Dondurur:
        DataFrame listesi (concat edilecek)
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Olay penceresi isleme basliyor...")
    logger.info(f"  Tampon bolge : t=-{tampon_eksi} / t=+{tampon_arti}")
    logger.info(f"  Nihai kesim  : t=-{nihai_eksi} / t=+{nihai_arti}")
    logger.info(f"  Toplam olay  : {len(olaylar)}")
    logger.info(f"  Toplam ticker: {len(veri_cache)}")
    logger.info(f"{'='*70}")
    
    sonuc_listesi = []
    islenen = 0
    atlanan = 0
    hata_sayisi = 0
    
    for olay_ismi, olay_tarihi_str in tqdm(olaylar.items(), desc="Olay isleme", unit="olay"):
        try:
            olay_tarihi = pd.to_datetime(olay_tarihi_str)
        except Exception:
            atlanan += 1
            continue
        
        for ticker, grup in ticker_gruplari.items():
            if ticker not in veri_cache:
                continue
            
            try:
                ham = veri_cache[ticker]
                
                # Olayin veri setindeki konumunu bul
                if olay_tarihi not in ham.index:
                    idx = ham.index.searchsorted(olay_tarihi)
                    if idx >= len(ham.index) or idx == 0:
                        continue  # Sessizce atla
                else:
                    idx = ham.index.get_loc(olay_tarihi)
                
                # Genis pencere indeksleri
                baslangic_idx = idx - tampon_eksi
                bitis_idx = idx + tampon_arti
                
                if baslangic_idx < 0 or bitis_idx >= len(ham.index):
                    continue  # Yetersiz veri, sessizce atla
                
                # ============================================================
                # ADIM 1: Genis pencereden veri kes
                # ============================================================
                pencere = ham.iloc[baslangic_idx : bitis_idx + 1].copy()
                
                df_t = pd.DataFrame()
                df_t['Tarih'] = pencere.index.strftime('%Y-%m-%d')
                
                kapanis = pencere['Fiyat'].values
                df_t['Duzeltilmis_Kapanis'] = kapanis
                
                # Logaritmik Getiri
                kapanis_seri = pd.Series(kapanis, dtype=np.float64)
                df_t['Log_Getiri'] = np.log(kapanis_seri / kapanis_seri.shift(1)).values
                
                # Volatilite
                log_g = pd.Series(df_t['Log_Getiri'].values, dtype=np.float64)
                df_t['Volatilite_10g'] = log_g.rolling(window=10).std().values
                df_t['Volatilite_30g'] = log_g.rolling(window=30).std().values
                
                # ============================================================
                # ADIM 2: Teknik gostergeler (genis pencere uzerinde)
                # ============================================================
                df_t['RSI_14'] = hesapla_rsi(kapanis_seri, periyot=14).values
                
                macd_df = hesapla_macd(kapanis_seri, hizli=12, yavas=26, sinyal=9)
                df_t['MACD_12_26_9'] = macd_df['MACD_12_26_9'].values
                df_t['MACDh_12_26_9'] = macd_df['MACDh_12_26_9'].values
                df_t['MACDs_12_26_9'] = macd_df['MACDs_12_26_9'].values
                
                bb_df = hesapla_bollinger(kapanis_seri, periyot=20, std_c=2.0)
                df_t['BBL_20_2.0'] = bb_df['BBL_20_2.0'].values
                df_t['BBU_20_2.0'] = bb_df['BBU_20_2.0'].values
                
                # Goreceli gun
                olay_sifir = tampon_eksi
                df_t['T0_Goreceli_Gun'] = np.arange(len(df_t)) - olay_sifir
                
                # ============================================================
                # ADIM 3: Nihai kesim (t=-30 .. t=+30)
                # ============================================================
                df_f = df_t[
                    (df_t['T0_Goreceli_Gun'] >= -nihai_eksi) &
                    (df_t['T0_Goreceli_Gun'] <= nihai_arti)
                ].copy()
                
                # Etiketler
                df_f['Olay_Ismi'] = olay_ismi
                df_f['Hisse'] = ticker
                df_f['Grup'] = grup
                
                # Sutun sirasi
                sutun_sirasi = [
                    'Tarih', 'Olay_Ismi', 'Hisse', 'Grup', 'T0_Goreceli_Gun',
                    'Duzeltilmis_Kapanis', 'Log_Getiri', 'Volatilite_10g', 'Volatilite_30g',
                    'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
                    'BBL_20_2.0', 'BBU_20_2.0'
                ]
                df_f = df_f[sutun_sirasi]
                
                sonuc_listesi.append(df_f)
                islenen += 1
                
            except Exception as e:
                hata_sayisi += 1
                if hata_sayisi <= 10:  # Ilk 10 hatayi logla
                    logger.debug(f"  Hata: {olay_ismi} / {ticker}: {e}")
                continue
    
    logger.info(f"\nOlay isleme tamamlandi:")
    logger.info(f"  Basarili olay-ticker cifti: {islenen}")
    logger.info(f"  Atlanan (veri yetersiz)    : {atlanan}")
    logger.info(f"  Hata                       : {hata_sayisi}")
    logger.info(f"  Toplam veri parcasi        : {len(sonuc_listesi)}")
    
    return sonuc_listesi


def csv_olarak_kaydet(
    sonuc_listesi: List[pd.DataFrame],
    dosya_adi: str = "sp500_deep_learning_massive_data.csv"
) -> None:
    """Tum islenmis parcalari birlestirip CSV olarak kaydeder."""
    if not sonuc_listesi:
        logger.error("Kaydedilecek veri bulunamadi!")
        return
    
    logger.info("\nVeriler birlestiriliyor...")
    son_veri = pd.concat(sonuc_listesi, ignore_index=True)
    
    # NaN raporu
    nan_ozet = son_veri.isnull().sum()
    nan_olan = nan_ozet[nan_ozet > 0]
    if len(nan_olan) > 0:
        logger.warning(f"NaN bulunan sutunlar:\n{nan_olan}")
    
    son_veri.to_csv(dosya_adi, index=False)
    logger.info(f"\n{'='*70}")
    logger.info(f"SONUC RAPORU")
    logger.info(f"{'='*70}")
    logger.info(f"Dosya         : {dosya_adi}")
    logger.info(f"Toplam satir  : {len(son_veri):,}")
    logger.info(f"Sutun sayisi  : {len(son_veri.columns)}")
    logger.info(f"Sutunlar      : {list(son_veri.columns)}")
    logger.info(f"Benzersiz olay: {son_veri['Olay_Ismi'].nunique()}")
    logger.info(f"Benzersiz hisse: {son_veri['Hisse'].nunique()}")
    logger.info(f"Tarih araligi : {son_veri['Tarih'].min()} -> {son_veri['Tarih'].max()}")
    logger.info(f"{'='*70}")


# ==============================================================================
# ANA CALISTIRMA BLOGU
# ==============================================================================

if __name__ == "__main__":
    
    print("\n" + "#" * 70)
    print("#  DEVASA VERI URETIM SCRIPTI (Faz 3)")
    print("#  TUFE + FOMC + Sok Olaylari -> ~260K+ Satir")
    print("#" * 70 + "\n")
    
    # 1. Ticker ve Gruplar (12 ticker)
    TICKER_GRUPLARI = {
        "^GSPC": "Gosterge_Endeks",
        "^VIX": "Gosterge_Endeks",
        "XLF": "Odak_Sektor",
        "XLK": "Odak_Sektor",
        "XLE": "Odak_Sektor",
        "XLV": "Odak_Sektor",
        "ITA": "Odak_Sektor",
        "XLP": "Kontrol_Grubu",
        "XLRE": "Kontrol_Grubu",
        "XLY": "Kontrol_Grubu",
        "GLD": "Emtia_Makro",
        "CL=F": "Emtia_Makro"
    }
    
    try:
        # =====================================================================
        # ADIM 1: Tum olaylari birlestir
        # =====================================================================
        OLAYLAR = tum_olaylari_birlestir()
        
        # =====================================================================
        # ADIM 2: Tum ticker verilerini indir (tek tek, bug workaround)
        # =====================================================================
        veri_cache = tum_ticker_verilerini_indir(TICKER_GRUPLARI)
        
        if not veri_cache:
            logger.critical("Hicbir ticker'dan veri indirilemedi! Cikiliyor.")
            exit(1)
        
        # =====================================================================
        # ADIM 3: Olaylari isle (tampon bolge -> gosterge -> nihai kesim)
        # =====================================================================
        sonuc = olaylari_isle(
            veri_cache=veri_cache,
            ticker_gruplari=TICKER_GRUPLARI,
            olaylar=OLAYLAR,
            tampon_eksi=100,
            tampon_arti=60,
            nihai_eksi=30,
            nihai_arti=30
        )
        
        # =====================================================================
        # ADIM 4: CSV kaydi
        # =====================================================================
        csv_yolu = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "sp500_deep_learning_massive_data.csv"
        )
        csv_olarak_kaydet(sonuc, dosya_adi=csv_yolu)
        
    except Exception as e:
        logger.critical(f"Kritik hata: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "#" * 70)
    print("#  ISLEM TAMAMLANDI")
    print("#" * 70 + "\n")
