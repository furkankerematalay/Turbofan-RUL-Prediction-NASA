import pandas as pd
from sqlalchemy import create_engine
import pymysql

# --- AYARLAR ---
db_host = 'database-1.c7ypwp71pvda.us-east-1.rds.amazonaws.com'
db_user = 'admin'
db_pass = 'kerem2003'        # Şifren
db_name = 'nasa_db'

# --- BAĞLANTI ---
print(" Buluta bağlanılıyor...")
conn_str = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:3306/{db_name}"
engine = create_engine(conn_str)

# --- SORGULA VE GÖSTER ---
print(" Tahminler okunuyor...\n")

# SQL ile "predictions" tablosundan ilk 10 satırı çekiyoruz
query = "SELECT * FROM predictions LIMIT 10"
df = pd.read_sql(query, engine)

# Sonuçları ekrana bas
print(df)
print("Sütunlar: unit_nr (Motor No), time_cycles (Zaman), RUL (Gerçek Ömür), Predicted_RUL (Yapay Zeka Tahmini)")