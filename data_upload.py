import pandas as pd
from sqlalchemy import create_engine, text
import pymysql
import sys

# --- AYARLAR ---
# Buraya kendi endpoint ve şifreni tekrar girmen gerekecek
db_host = 'database-1.c7ypwp71pvda.us-east-1.rds.amazonaws.com'
db_user = 'admin'
db_pass = 'kerem2003'
yeni_db_adi = 'nasa_db'  # Yeni oluşturacağımız veritabanı adı

# --- ADIM 1: VERİYİ OKU ---
print("1. Adım: NASA verisi okunuyor...")
cols = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + \
       ['s_{}'.format(i) for i in range(1, 22)]

try:
    df = pd.read_csv('train_FD001.txt', sep='\s+', header=None, names=cols)
    print(f"✅ Veri okundu! Satır sayısı: {df.shape[0]}")
except FileNotFoundError:
    sys.exit()

# --- ADIM 2: VERİTABANINI OLUŞTUR (YENİ KISIM) ---
print(f"\n2. Adım: '{yeni_db_adi}' adında veritabanı oluşturuluyor...")

try:
    # Önce genel sunucuya bağlanıyoruz (Veritabanı ismi vermeden)
    conn = pymysql.connect(host=db_host, user=db_user, password=db_pass)
    cursor = conn.cursor()

    # "nasa_db" yoksa yarat diyoruz
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {yeni_db_adi}")
    conn.close()
    print(f"✅ Veritabanı ({yeni_db_adi}) hazır!")

except Exception as e:
    print("❌ Veritabanı oluşturma hatası:", e)
    sys.exit()

# --- ADIM 3: VERİYİ YÜKLE ---
print(f"\n3. Adım: Veriler '{yeni_db_adi}' içine yükleniyor...")

# Şimdi yeni yarattığımız veritabanına bağlanıyoruz
baglanti_adresi = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:3306/{yeni_db_adi}"

try:
    engine = create_engine(baglanti_adresi)
    df.to_sql(name='sensor_data', con=engine, if_exists='replace', index=False, chunksize=1000)
    print(" Tüm veriler başarıyla yüklendi.")

except Exception as e:
    print("❌ Yükleme hatası:", e)