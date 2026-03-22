import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

import joblib

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

np.random.seed(1234)

index_names = ['unit_number', 'time_cycles']
operational_settings = ['setting_1', 'setting_2', 'setting_3']

sensor_names = [
    'T2','T24','T30','T50','P2','P15','P30',
    'Nf','Nc','epr','Ps30','phi','NRf','NRc',
    'BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32'
]

col_names = index_names + operational_settings + sensor_names
print(f"Total columns: {len(col_names)}")


base_path = "."

fd001_train = pd.read_csv(
    f"{base_path}/train_FD001.txt",
    sep=r"\s+", header=None, names=col_names, engine="python"
)

fd001_test = pd.read_csv(
    f"{base_path}/test_FD001.txt",
    sep=r"\s+", header=None, names=col_names, engine="python"
)

rul = pd.read_csv(
    f"{base_path}/RUL_FD001.txt",
    sep=r"\s+", header=None, names=['RUL'], engine="python"
)

print("Train Shape:", fd001_train.shape)
print("Test Shape :", fd001_test.shape)
print("RUL Shape  :", rul.shape)

print("Missing values (train):", fd001_train.isnull().sum().sum())
print("Missing values (test) :", fd001_test.isnull().sum().sum())

fd001_train.info()
fd001_train.describe().T


constant_features = {
    col for col in fd001_train.columns
    if fd001_train[col].nunique() <=1  ## tüm satırlar aynı mı (constant var mı)
}

print("Dead Sensors: ")
print(constant_features)

## Dead sensor vs Active sensor


dead_sensor = 'P2'
active_sensor ='T50'

fig, ax = plt.subplots(1, 2, figsize=(16, 5))

sns.lineplot(data=fd001_train[fd001_train['unit_number'] == 1],
             x='time_cycles',y=dead_sensor,ax=ax[0],color ='red'
)

ax[0].set_title(f"Dead sensor : {dead_sensor}")

sns.lineplot(data=fd001_train[fd001_train['unit_number']==1],
             x='time_cycles',y=active_sensor,ax=ax[1],color ='green')

ax[1].set_title(f"active sensor : {active_sensor}")

# plt.tight_layout()
# plt.show()


fd001_train.drop(columns=constant_features,inplace=True)
fd001_test.drop(columns=constant_features,inplace=True)

print("New train shape",fd001_train.shape)

## formula RUL(t) = Tmax- t  The dataset provides the total life, but we need to predict the RUL

def calculate_rul(data):
    max_cycle = (
        data.groupby('unit_number')['time_cycles']
        .max()
        .reset_index()
        .rename(columns={'time_cycles':'max_cycle'})
    )
    data = data.merge(max_cycle, on='unit_number')
    data['RUL'] = data['max_cycle'] - data['time_cycles']
    data.drop(columns=['max_cycle'], inplace=True)
    return data

fd001_train = calculate_rul(fd001_train)
print(fd001_train[['unit_number','time_cycles','RUL']].head())








#  Neden 120 Eşiğini Seçiyoruz?

sample_engine = fd001_train[fd001_train['unit_number'] == 1]

plt.figure(figsize=(10, 5))
# Y ekseni sıcaklık anlık sıcaklığı gösterir sıcaklığın artmasının en önemli sebebi motordaki aşınma  aşınmayla beraber RUL negative lineer correlation gösteriyor
sns.scatterplot(x=sample_engine['RUL'], y=sample_engine['T50'], color='blue', alpha=0.6)

# 120 eşiğine kırmızı çizgi
plt.axvline(x=120, color='red', linestyle='--', label='120')

plt.title('Sensör T50 Verisindeki Eğimin Başladığı Nokta')
plt.xlabel('RUL')
plt.ylabel('Sensör Değeri')
plt.legend()
plt.gca().invert_xaxis()
plt.show()



## x ekseni motoların toplam ömrünü (cycles) y ekseni o motordan kaç tane bulunduğunu söylüyor min 128 olduğunu görüyoruz
engine_life = fd001_train.groupby('unit_number')['time_cycles'].max()

print("Mean lifespan:", engine_life.mean())
print("Min lifespan :", engine_life.min())
print("Max lifespan :", engine_life.max())


sns.histplot(engine_life, kde=True,binwidth=10)
plt.title("Engine Lifespan Distribution")
plt.show()







## rul 300-400 gibi sayılarda çok az bir eğimle azalıyor makineler de bu eğim 120den sonra
RUL_THRESHOLD = 120
fd001_train['RUL'] = fd001_train['RUL'].clip(upper=RUL_THRESHOLD)
print(fd001_train['RUL'].describe())




corr = fd001_train.corr()
rul_corr = corr[['RUL']].sort_values(by='RUL', ascending=False)

plt.figure(figsize=(10, 8))
sns.heatmap(rul_corr, annot=True, cmap='RdYlGn', fmt=".2f")
plt.title("Sensor Correlation with RUL")
plt.show()


## scale edip scale ettiğimiz veri ile train ediyoruz
features_to_scale = [
    col for col in fd001_train.columns
    if col not in ['unit_number', 'time_cycles', 'RUL']
]
scaler = MinMaxScaler()
fd001_train[features_to_scale] = scaler.fit_transform(fd001_train[features_to_scale])
fd001_test[features_to_scale] = scaler.transform(fd001_test[features_to_scale])


##yeni scla ettiklerimizi x train ile veriyoruz makinaya
##y_train normal veirler aradaki ilişkiye yani y ye bakarak çöz diyoruz
## x_tet ile ilişkini test et diyoruz
#y_true ile kıyasla diyoruz



X_train = fd001_train[features_to_scale]
y_train = fd001_train['RUL']

X_test = (
    fd001_test
    .groupby('unit_number')
    .last()
    .reset_index()[features_to_scale]
)

y_true = rul['RUL']

result_summary = []
# benchmarking ve deneme yanılma

def evaluete_model(model,name):
    model.fit(X_train ,y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    r2 = r2_score(y_true,y_pred)
    result_summary.append({
        "Model" : name,
        "RMSE"  : rmse,
        "R2"    :   r2
    })

    plt.plot(y_true.values, '--',label = 'Actual')
    plt.plot(y_pred, label ='Predicted')
    plt.title(name)
    plt.legend()
    plt.show()

    print(name,"RMSE:",rmse,"R2:", r2)

evaluete_model(LinearRegression(), "Linear Regression")
evaluete_model(SVR(C=10,epsilon=0.1),"SVR")
evaluete_model(RandomForestRegressor(n_estimators=100,max_depth=12,n_jobs=-1,random_state=42),"Random Forest")
evaluete_model(XGBRegressor(n_estimators=100,max_depth=6,learning_rate=0.1,random_state=42),"XGBoost")


## lowes RMSE  bizim criteriamız

best_result_dict = min(result_summary, key=lambda x: x['RMSE'])
best_model = best_result_dict['Model']

if best_model == "XGBoost":
    best_model_obj = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
elif best_model == "Random Forest":
    best_model_obj = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
elif best_model == "SVR":
    best_model_obj = SVR(C=10, epsilon=0.1)
else:
    best_model_obj = LinearRegression()

best_model_obj.fit(X_train, y_train)

#       #Real-time Simulation (The "Digital Twin")


def simulate_engine_life(unit_id, model=best_model_obj, W1=30, window_std=5):

    engine = fd001_test[fd001_test['unit_number'] == unit_id]
    X_engine = engine[features_to_scale]

    # Predict RUL
    preds = model.predict(X_engine).flatten()

    # Compute rolling standard deviation as a simple uncertainty measure /?
    preds_std = pd.Series(preds).rolling(window_std, min_periods=1).std()

    plt.figure(figsize=(12, 6))
    plt.plot(engine['time_cycles'], preds, label='Predicted RUL', color='blue')

    plt.fill_between(
        engine['time_cycles'],
        preds - preds_std,
        preds + preds_std,
        color='blue', alpha=0.2, label='Prediction uncertainty'
    )


    plt.axhline(W1, color='red', linestyle='--', label=f'Warning threshold (RUL={W1})')

    #  "actual failure"
    final_rul = rul.iloc[unit_id - 1]['RUL']
    last_cycle = engine['time_cycles'].max()
    plt.axvline(last_cycle, color='black', linestyle='--', label='Actual failure')

    plt.xlabel("Time Cycles")
    plt.ylabel("Remaining Useful Life (RUL)")
    plt.title(f"Engine #{unit_id} – Digital Twin RUL Simulation")
    plt.legend()
    plt.show()

# Example
simulate_engine_life(24)
simulate_engine_life(31)




