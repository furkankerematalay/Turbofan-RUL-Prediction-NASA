import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
import pymysql
import time

# --- 1. AYARLAR ---
print("🔌 AWS Bağlantısı hazırlanıyor...")
db_host = 'database-1.c7ypwp71pvda.us-east-1.rds.amazonaws.com'
db_user = 'admin'
db_pass = 'kerem2003'        # Şifren
db_name = 'nasa_db'

# --- 2. GÜÇLENDİRİLMİŞ BAĞLANTI (MÜHENDİSLİK DOKUNUŞU) ---
# Burada AWS'ye "Beni hemen atma, bekle" diyoruz.
connect_args = {
    'connect_timeout': 300,  # 5 Dakika bekleme süresi (Normalde 10-60 saniyedir)
    'read_timeout': 300,
    'write_timeout': 300
}

conn_str = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:3306/{db_name}"
engine = create_engine(conn_str, connect_args=connect_args)

# --- 3. VERİYİ AWS'DEN ÇEK (ISRARLA) ---
print("☁️  Veriler AWS Bulutundan indiriliyor...")
query = "SELECT * FROM sensor_data"

try:
    # Chunksize'ı biraz düşürdük ki paketler daha kolay geçsin
    chunk_list = []
    # Her 1000 satırda bir ekrana nokta koyar
    for chunk in pd.read_sql(query, engine, chunksize=1000):
        chunk_list.append(chunk)
        print(f"📦 Paket indi... ({len(chunk)} satır)")

    df = pd.concat(chunk_list)
    print(f"✅ BAŞARDIK! AWS'den {df.shape[0]} satır veri çekildi.")

except Exception as e:
    print("Hata detayı:", e)
    exit()

#  (RUL HESABI) ---
max_cycles = df.groupby('unit_nr')['time_cycles'].max().reset_index()
max_cycles.columns = ['unit_nr', 'max']
df = df.merge(max_cycles, on='unit_nr', how='left')
df['RUL'] = df['max'] - df['time_cycles']
df.drop('max', axis=1, inplace=True)

# --- 5. MODEL EĞİTİMİ ---
print("🧠 Model AWS verisiyle eğitiliyor...")
features = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
X = df[features]
y = df['RUL']

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# --- 6. SONUÇLARI KAYDET (HEM CSV, HEM AWS) ---
print("💾 Sonuçlar işleniyor...")
predictions = rf_model.predict(X)
results_df = df[['unit_nr', 'time_cycles', 'RUL']].copy()
results_df['Predicted_RUL'] = predictions

# A) Bilgisayara Kaydet (Tableau için garanti olsun)
results_df.to_csv("final_sonuclar.csv", index=False)
print("✅ Bilgisayara CSV yedeği alındı.")

# B) AWS'ye Geri Yükle (Projenin Şanı İçin!)
print("☁️  Tahminler AWS'ye geri yükleniyor...")
try:
    results_df.to_sql('predictions', engine, if_exists='replace', index=False, chunksize=500)
    print("🎉 MÜKEMMEL! Döngü tamamlandı: AWS -> Python -> AWS")
except Exception as e:
    print("yüklenemedi :", e)

