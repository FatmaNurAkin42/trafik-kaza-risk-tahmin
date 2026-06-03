# %% [markdown]
# # Trafikte Kaza Riski Tahmini
# **Veri Seti:** US_Accidents_March23.csv
#
# ## Proje Aşamaları
# 1. Veri Temizleme ve Özellik Mühendisliği (EDA)
# 2. Kümeleme (Clustering) ile Risk Bölgeleri
# 3. Sınıflandırma (Classification) ile Risk Tahmini
# 4. Görselleştirme ve Analiz

# %% [markdown]
# ---
# ## Kütüphanelerin Yüklenmesi

# %%
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script mode
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', 50)

# Çıktı klasörü oluştur
OUTPUT_DIR = 'output_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Kütüphaneler yüklendi.")

# %% [markdown]
# ---
# ## AŞAMA 1: Veri Yükleme, Temizleme ve Özellik Mühendisliği (EDA)

# %% [markdown]
# ### 1.1 Veri Yükleme
# Veri seti çok büyük (~7.7M satır, 3 GB) olduğu için performans açısından
# 300.000 satırlık rastgele bir örneklem kullanıyoruz.

# %%
# Veri setini yükle (bellek optimizasyonu ile)
DATA_PATH = "US_Accidents_March23.csv/US_Accidents_March23.csv"

# Kullanılacak sütunlar
USE_COLS = [
    'Severity', 'Start_Time', 'End_Time', 'Start_Lat', 'Start_Lng',
    'Distance(mi)', 'City', 'State', 'Zipcode',
    'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
    'Pressure(in)', 'Visibility(mi)', 'Wind_Direction',
    'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition',
    'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
    'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
    'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop',
    'Sunrise_Sunset'
]

print("Veri yükleniyor...")
df_full = pd.read_csv(DATA_PATH, usecols=USE_COLS)
print(f"Toplam kayıt: {len(df_full):,}")

# Rastgele 300K örneklem al
SAMPLE_SIZE = 300_000
df = df_full.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
del df_full  # Belleği serbest bırak

print(f"Örneklem boyutu: {len(df):,}")
print(f"Sütun sayısı: {df.shape[1]}")

# %%
df.head()

# %%
df.info()

# %% [markdown]
# ### 1.2 Eksik Veri Analizi

# %%
# Eksik veri oranları
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Eksik Sayısı': missing,
    'Eksik Oranı (%)': missing_pct
}).sort_values('Eksik Oranı (%)', ascending=False)

print("=== EKSİK VERİ ANALİZİ ===")
print(missing_df[missing_df['Eksik Sayısı'] > 0])

# %%
# Eksik veri görselleştirmesi
fig, ax = plt.subplots(figsize=(12, 6))
cols_with_missing = missing_df[missing_df['Eksik Sayısı'] > 0].index
missing_pcts = missing_df.loc[cols_with_missing, 'Eksik Oranı (%)']
bars = ax.barh(cols_with_missing, missing_pcts, color='coral')
ax.set_xlabel('Eksik Veri Oranı (%)')
ax.set_title('Sütunlardaki Eksik Veri Oranları')
for bar, pct in zip(bars, missing_pcts):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{pct:.1f}%', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_eksik_veri.png', dpi=150, bbox_inches='tight')
plt.close()
print('Grafik kaydedildi: 01_eksik_veri.png')

# %% [markdown]
# ### 1.3 Eksik Verileri Doldurma

# %%
# Sayısal sütunlardaki eksikleri medyan ile doldur
numeric_cols = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
                'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
                'Precipitation(in)']

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Kategorik sütunlardaki eksikleri mod ile doldur
categorical_cols = ['Wind_Direction', 'Weather_Condition', 'Sunrise_Sunset']
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Zipcode eksiklerini doldur
df['Zipcode'] = df['Zipcode'].fillna('Unknown')
df['City'] = df['City'].fillna('Unknown')

print("Eksik veriler dolduruldu.")
print(f"Kalan eksik veri: {df.isnull().sum().sum()}")

# %% [markdown]
# ### 1.4 Zaman Özellik Mühendisliği
# `Start_Time` sütunundan yeni özellikler türetiyoruz:
# - **Saat (Hour)**: 0-23
# - **Pik Saat (Is_Peak_Hour)**: Sabah 7-9, Akşam 16-19
# - **Gece (Is_Night)**: Gece saatleri (21-6)
# - **Hafta Sonu (Is_Weekend)**: Cumartesi & Pazar
# - **Ay (Month)**, **Mevsim (Season)**

# %%
# Tarih/saat dönüşümü
df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='ISO8601')
df['End_Time'] = pd.to_datetime(df['End_Time'], format='ISO8601')

# Kaza süresi (dakika)
df['Duration_min'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60
df['Duration_min'] = df['Duration_min'].clip(lower=0, upper=1440)  # 0-24 saat arası

# Zaman özellikleri
df['Hour'] = df['Start_Time'].dt.hour
df['Day_of_Week'] = df['Start_Time'].dt.dayofweek  # 0=Pazartesi
df['Month'] = df['Start_Time'].dt.month
df['Year'] = df['Start_Time'].dt.year

# Pik saat (sabah 7-9, akşam 16-19)
df['Is_Peak_Hour'] = df['Hour'].apply(lambda h: 1 if (7 <= h <= 9) or (16 <= h <= 19) else 0)

# Gece (21:00 - 06:00)
df['Is_Night'] = df['Hour'].apply(lambda h: 1 if h >= 21 or h <= 6 else 0)

# Hafta sonu
df['Is_Weekend'] = df['Day_of_Week'].apply(lambda d: 1 if d >= 5 else 0)

# Mevsim
def get_season(month):
    if month in [12, 1, 2]: return 'Kış'
    elif month in [3, 4, 5]: return 'İlkbahar'
    elif month in [6, 7, 8]: return 'Yaz'
    else: return 'Sonbahar'

df['Season'] = df['Month'].apply(get_season)

print("Zaman özellikleri oluşturuldu:")
print(df[['Hour', 'Is_Peak_Hour', 'Is_Night', 'Is_Weekend', 'Season']].head(10))

# %% [markdown]
# ### 1.5 Keşifsel Veri Analizi (EDA)

# %%
# Severity (Kaza Şiddeti) dağılımı
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Severity bar plot
severity_counts = df['Severity'].value_counts().sort_index()
colors = ['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']
axes[0].bar(severity_counts.index, severity_counts.values, color=colors)
axes[0].set_xlabel('Kaza Şiddeti (Severity)')
axes[0].set_ylabel('Kaza Sayısı')
axes[0].set_title('Kaza Şiddeti Dağılımı')
axes[0].set_xticks([1, 2, 3, 4])
for i, v in enumerate(severity_counts.values):
    axes[0].text(severity_counts.index[i], v + 500, f'{v:,}', ha='center', fontsize=9)

# Severity pie chart
axes[1].pie(severity_counts.values, labels=[f'Severity {i}' for i in severity_counts.index],
            autopct='%1.1f%%', colors=colors, startangle=90)
axes[1].set_title('Kaza Şiddeti Yüzdeleri')

plt.suptitle('Kaza Şiddeti Analizi', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_kaza_siddeti.png', dpi=150, bbox_inches='tight')
plt.close()
print('Grafik kaydedildi: 02_kaza_siddeti.png')

# %%
# Saatlik kaza dağılımı
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Saatlik dağılım
hourly = df.groupby('Hour').size()
axes[0].plot(hourly.index, hourly.values, 'o-', color='#e74c3c', linewidth=2, markersize=6)
axes[0].fill_between(hourly.index, hourly.values, alpha=0.2, color='#e74c3c')
axes[0].set_xlabel('Saat')
axes[0].set_ylabel('Kaza Sayısı')
axes[0].set_title('Günün Saatlerine Göre Kaza Sayısı')
axes[0].set_xticks(range(0, 24))
axes[0].axvspan(7, 9, alpha=0.1, color='orange', label='Sabah Pik')
axes[0].axvspan(16, 19, alpha=0.1, color='red', label='Akşam Pik')
axes[0].legend()

# Haftalık dağılım
days = ['Pzt', 'Sal', 'Çar', 'Per', 'Cum', 'Cmt', 'Paz']
daily = df.groupby('Day_of_Week').size()
bars = axes[1].bar(daily.index, daily.values, color=['#3498db']*5 + ['#e74c3c']*2)
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(days)
axes[1].set_xlabel('Gün')
axes[1].set_ylabel('Kaza Sayısı')
axes[1].set_title('Haftanın Günlerine Göre Kaza Sayısı')

plt.suptitle('Zaman Bazlı Kaza Analizi', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_zaman_analizi.png', dpi=150, bbox_inches='tight')
plt.close()
print('Grafik kaydedildi: 03_zaman_analizi.png')

# %%
# Hava durumu ve sıcaklık analizi
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# En yaygın 15 hava durumu - kaza sayısı
weather_top = df['Weather_Condition'].value_counts().head(15)
axes[0].barh(weather_top.index[::-1], weather_top.values[::-1], color='steelblue')
axes[0].set_xlabel('Kaza Sayısı')
axes[0].set_title('Hava Durumuna Göre Kaza Sayısı (Top 15)')

# Sıcaklık vs kaza şiddeti
df.boxplot(column='Temperature(F)', by='Severity', ax=axes[1],
           patch_artist=True,
           boxprops=dict(facecolor='lightblue'))
axes[1].set_xlabel('Kaza Şiddeti (Severity)')
axes[1].set_ylabel('Sıcaklık (°F)')
axes[1].set_title('Sıcaklık vs Kaza Şiddeti')
plt.suptitle('')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_hava_sicaklik.png', dpi=150, bbox_inches='tight')
plt.close()
print('Grafik kaydedildi: 04_hava_sicaklik.png')

# %%
# Mevsimsel analiz ve Gece/Gündüz karşılaştırması
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Mevsimsel dağılım
season_order = ['İlkbahar', 'Yaz', 'Sonbahar', 'Kış']
season_counts = df['Season'].value_counts().reindex(season_order)
season_colors = ['#2ecc71', '#f1c40f', '#e67e22', '#3498db']
axes[0].bar(season_counts.index, season_counts.values, color=season_colors)
axes[0].set_xlabel('Mevsim')
axes[0].set_ylabel('Kaza Sayısı')
axes[0].set_title('Mevsimlere Göre Kaza Dağılımı')

# Gece/Gündüz
sunrise_sev = df.groupby(['Sunrise_Sunset', 'Severity']).size().unstack(fill_value=0)
sunrise_sev.plot(kind='bar', ax=axes[1], colormap='RdYlGn_r')
axes[1].set_xlabel('Gündüz / Gece')
axes[1].set_ylabel('Kaza Sayısı')
axes[1].set_title('Gece/Gündüz ve Kaza Şiddeti')
axes[1].legend(title='Severity')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_mevsim_gece.png', dpi=150, bbox_inches='tight')
plt.close()
print('Grafik kaydedildi: 05_mevsim_gece.png')

# %%
# Korelasyon matrisi
fig, ax = plt.subplots(figsize=(12, 8))
corr_cols = ['Severity', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
             'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)',
             'Hour', 'Is_Peak_Hour', 'Is_Night', 'Is_Weekend', 'Duration_min']
corr_matrix = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title('Özellikler Arası Korelasyon Matrisi', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_korelasyon.png', dpi=150, bbox_inches='tight')
plt.close()
print('Grafik kaydedildi: 06_korelasyon.png')

# %% [markdown]
# ---
# ## AŞAMA 2: Kümeleme (Clustering) ile Risk Bölgeleri
# Kaza lokasyonlarını (Enlem/Boylam) kullanarak **K-Means** ile coğrafi risk kümeleri oluşturuyoruz.

# %%
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

# Coğrafi veriyi hazırla
geo_data = df[['Start_Lat', 'Start_Lng']].dropna().copy()

# MiniBatchKMeans - büyük veri için hızlı
print("K-Means kümeleme başlıyor...")

# Optimal k için Elbow yöntemi (küçük örneklem üzerinde)
geo_sample = geo_data.sample(n=min(50000, len(geo_data)), random_state=42)
scaler = StandardScaler()
geo_scaled = scaler.fit_transform(geo_sample)

inertias = []
K_range = range(3, 12)
for k in K_range:
    kmeans_temp = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=5000)
    kmeans_temp.fit(geo_scaled)
    inertias.append(kmeans_temp.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(K_range, inertias, 'bx-', linewidth=2, markersize=8)
ax.set_xlabel('Küme Sayısı (k)')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Yöntemi - Optimal Küme Sayısı')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_elbow.png', dpi=150, bbox_inches='tight')
plt.close()
print('Grafik kaydedildi: 07_elbow.png')

# %%
# K=6 ile kümeleme yap
OPTIMAL_K = 6
print(f"K={OPTIMAL_K} ile kümeleme yapılıyor...")

geo_all_scaled = scaler.transform(geo_data)
kmeans = MiniBatchKMeans(n_clusters=OPTIMAL_K, random_state=42, batch_size=5000)
geo_data['Cluster'] = kmeans.fit_predict(geo_all_scaled)

# Küme merkezlerini gerçek koordinatlara dönüştür
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Kümelere ait özet istatistikler
df_with_cluster = df.copy()
df_with_cluster['Cluster'] = geo_data['Cluster'].values

print("\n=== KÜMELERİN ÖZETİ ===")
cluster_summary = df_with_cluster.groupby('Cluster').agg(
    Kaza_Sayisi=('Severity', 'count'),
    Ort_Siddet=('Severity', 'mean'),
    Ort_Lat=('Start_Lat', 'mean'),
    Ort_Lng=('Start_Lng', 'mean')
).round(2)
cluster_summary['Risk_Seviyesi'] = pd.cut(
    cluster_summary['Ort_Siddet'],
    bins=[0, 2.0, 2.5, 3.0, 5.0],
    labels=['Düşük', 'Orta', 'Yüksek', 'Çok Yüksek']
)
print(cluster_summary)

# %%
# Kümelerin coğrafi dağılımı
fig, ax = plt.subplots(figsize=(14, 8))
scatter = ax.scatter(
    geo_data['Start_Lng'], geo_data['Start_Lat'],
    c=geo_data['Cluster'], cmap='tab10', alpha=0.1, s=1
)
# Küme merkezleri
for i, center in enumerate(cluster_centers):
    ax.scatter(center[1], center[0], c='red', marker='X', s=200, linewidths=2,
               edgecolors='black', zorder=5)
    ax.annotate(f'Küme {i}', (center[1], center[0]),
                fontsize=10, fontweight='bold', color='red',
                textcoords="offset points", xytext=(10, 10))

ax.set_xlabel('Boylam (Longitude)')
ax.set_ylabel('Enlem (Latitude)')
ax.set_title('K-Means Kümeleme: Kaza Yoğunluk Bölgeleri', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Küme No')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_cluster_harita.png', dpi=150, bbox_inches='tight')
plt.close()
print('Grafik kaydedildi: 08_cluster_harita.png')

# %% [markdown]
# ---
# ## AŞAMA 3: Sınıflandırma (Classification) ile Risk Tahmini
# **Hedef:** Belirli koşullar altında kazanın şiddetini tahmin etmek.
# "Bu hava durumu ve bu saatte kaza riski YÜKSEK mi?"
#
# - Binary sınıflandırma: Severity <= 2 → **Düşük Risk**, Severity >= 3 → **Yüksek Risk**
# - Modeller: Random Forest + XGBoost
# - Dengesiz veri: SMOTE ile dengeleme

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Binary hedef değişken oluştur
df_with_cluster['Risk_Label'] = (df_with_cluster['Severity'] >= 3).astype(int)  # 1=Yüksek Risk

print(f"Risk dağılımı:")
print(df_with_cluster['Risk_Label'].value_counts())
print(f"\nDengesizlik oranı: {df_with_cluster['Risk_Label'].value_counts()[0] / df_with_cluster['Risk_Label'].value_counts()[1]:.1f}:1")

# %%
# Özellik matrisi hazırlama
feature_cols = [
    'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
    'Wind_Speed(mph)', 'Precipitation(in)', 'Distance(mi)',
    'Hour', 'Day_of_Week', 'Month',
    'Is_Peak_Hour', 'Is_Night', 'Is_Weekend',
    'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
    'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
    'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop',
    'Duration_min', 'Cluster'
]

X = df_with_cluster[feature_cols].copy()
y = df_with_cluster['Risk_Label'].copy()

# Bool sütunları int'e çevir
bool_cols = X.select_dtypes(include='bool').columns
X[bool_cols] = X[bool_cols].astype(int)

# Eksik kontrol
X = X.fillna(0)

print(f"Özellik matrisi: {X.shape}")
print(f"Hedef dağılımı:\n{y.value_counts()}")

# %%
# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Eğitim seti: {X_train.shape[0]:,} kayıt")
print(f"Test seti: {X_test.shape[0]:,} kayıt")

# SMOTE ile dengeleme (eğitim seti üzerinde)
print("\nSMOTE uygulanıyor...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"SMOTE sonrası eğitim seti: {X_train_sm.shape[0]:,}")
print(f"Dengeli dağılım:\n{pd.Series(y_train_sm).value_counts()}")

# %% [markdown]
# ### 3.1 Random Forest

# %%
print("Random Forest eğitiliyor...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_sm, y_train_sm)
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

print("\n=== RANDOM FOREST SONUÇLARI ===")
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, rf_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred, target_names=['Düşük Risk', 'Yüksek Risk']))

# %% [markdown]
# ### 3.2 XGBoost

# %%
print("XGBoost eğitiliyor...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
xgb_model.fit(X_train_sm, y_train_sm)
xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

print("\n=== XGBOOST SONUÇLARI ===")
print(f"Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, xgb_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, xgb_pred, target_names=['Düşük Risk', 'Yüksek Risk']))

# %% [markdown]
# ### 3.3 Model Karşılaştırma ve Değerlendirme

# %%
# ROC Eğrisi karşılaştırma
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# ROC Curves
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_proba)

axes[0].plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC={roc_auc_score(y_test, rf_proba):.3f})',
             linewidth=2)
axes[0].plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC={roc_auc_score(y_test, xgb_proba):.3f})',
             linewidth=2)
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Eğrisi Karşılaştırma')
axes[0].legend(loc='lower right')

# Confusion Matrix - Random Forest
cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Düşük', 'Yüksek'], yticklabels=['Düşük', 'Yüksek'])
axes[1].set_xlabel('Tahmin')
axes[1].set_ylabel('Gerçek')
axes[1].set_title('Random Forest - Confusion Matrix')

# Confusion Matrix - XGBoost
cm_xgb = confusion_matrix(y_test, xgb_pred)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges', ax=axes[2],
            xticklabels=['Düşük', 'Yüksek'], yticklabels=['Düşük', 'Yüksek'])
axes[2].set_xlabel('Tahmin')
axes[2].set_ylabel('Gerçek')
axes[2].set_title('XGBoost - Confusion Matrix')

plt.suptitle('Model Performans Karşılaştırması', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/09_model_performans.png', dpi=150, bbox_inches='tight')
plt.close()
print('Grafik kaydedildi: 09_model_performans.png')

# %%
# Feature Importance (Özellik Önem Derecesi)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Random Forest Feature Importance
rf_importance = pd.Series(rf_model.feature_importances_, index=feature_cols)
rf_importance.sort_values(ascending=True).tail(15).plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].set_title('Random Forest - Özellik Önem Sırası (Top 15)')
axes[0].set_xlabel('Önem Derecesi')

# XGBoost Feature Importance
xgb_importance = pd.Series(xgb_model.feature_importances_, index=feature_cols)
xgb_importance.sort_values(ascending=True).tail(15).plot(kind='barh', ax=axes[1], color='darkorange')
axes[1].set_title('XGBoost - Özellik Önem Sırası (Top 15)')
axes[1].set_xlabel('Önem Derecesi')

plt.suptitle('Kaza Riskini Belirleyen En Etkili Faktörler', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/10_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print('Grafik kaydedildi: 10_feature_importance.png')

# %% [markdown]
# ---
# ## AŞAMA 4: Görselleştirme ve Analiz

# %% [markdown]
# ### 4.1 Interaktif Risk Isı Haritası (Folium)

# %%
import folium
from folium.plugins import HeatMap

# ABD merkez koordinatları
us_center = [39.8283, -98.5795]
risk_map = folium.Map(location=us_center, zoom_start=4, tiles='CartoDB dark_matter')

# Yüksek riskli kazaları filtrele (Severity >= 3)
high_risk = df_with_cluster[df_with_cluster['Severity'] >= 3].sample(
    n=min(30000, len(df_with_cluster[df_with_cluster['Severity'] >= 3])),
    random_state=42
)

# Isı haritası verisi
heat_data = high_risk[['Start_Lat', 'Start_Lng']].values.tolist()

HeatMap(
    heat_data,
    radius=8,
    blur=10,
    max_zoom=13,
    gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1: 'red'}
).add_to(risk_map)

# Küme merkezlerini marker olarak ekle
for i, center in enumerate(cluster_centers):
    risk_level = cluster_summary.loc[i, 'Risk_Seviyesi']
    kaza_say = cluster_summary.loc[i, 'Kaza_Sayisi']
    folium.Marker(
        location=[center[0], center[1]],
        popup=f"<b>Küme {i}</b><br>Risk: {risk_level}<br>Kaza: {kaza_say:,}",
        icon=folium.Icon(color='red', icon='warning-sign', prefix='glyphicon')
    ).add_to(risk_map)

# Haritayı kaydet
risk_map.save('kaza_risk_haritasi.html')
print("Interaktif risk haritası 'kaza_risk_haritasi.html' olarak kaydedildi.")

# %% [markdown]
# ### 4.2 Plotly ile İnteraktif Görselleştirmeler

# %%
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Saatlik risk analizi - interaktif
hourly_risk = df_with_cluster.groupby(['Hour', 'Risk_Label']).size().unstack(fill_value=0)
hourly_risk.columns = ['Düşük Risk', 'Yüksek Risk']
hourly_risk['Yüksek Risk Oranı (%)'] = (
    hourly_risk['Yüksek Risk'] / (hourly_risk['Düşük Risk'] + hourly_risk['Yüksek Risk']) * 100
).round(1)

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Bar(x=hourly_risk.index, y=hourly_risk['Düşük Risk'],
           name='Düşük Risk', marker_color='#2ecc71', opacity=0.7),
    secondary_y=False
)
fig.add_trace(
    go.Bar(x=hourly_risk.index, y=hourly_risk['Yüksek Risk'],
           name='Yüksek Risk', marker_color='#e74c3c', opacity=0.7),
    secondary_y=False
)
fig.add_trace(
    go.Scatter(x=hourly_risk.index, y=hourly_risk['Yüksek Risk Oranı (%)'],
               name='Yüksek Risk Oranı (%)', line=dict(color='#8e44ad', width=3),
               mode='lines+markers'),
    secondary_y=True
)

fig.update_layout(
    title='Saatlere Göre Kaza Risk Analizi',
    xaxis_title='Saat',
    barmode='stack',
    template='plotly_white',
    height=500
)
fig.update_yaxes(title_text="Kaza Sayısı", secondary_y=False)
fig.update_yaxes(title_text="Yüksek Risk Oranı (%)", secondary_y=True)

fig.write_html('saatlik_risk_analizi.html')
print('Saatlik risk analizi kaydedildi: saatlik_risk_analizi.html')

# %%
# Hava durumu vs Risk - Plotly
weather_risk = df_with_cluster.groupby('Weather_Condition').agg(
    Toplam_Kaza=('Severity', 'count'),
    Ort_Siddet=('Severity', 'mean'),
    Yuksek_Risk_Orani=('Risk_Label', 'mean')
).sort_values('Toplam_Kaza', ascending=False).head(20)

weather_risk['Yuksek_Risk_Orani'] = (weather_risk['Yuksek_Risk_Orani'] * 100).round(1)

fig = px.scatter(
    weather_risk.reset_index(),
    x='Toplam_Kaza',
    y='Yuksek_Risk_Orani',
    size='Ort_Siddet',
    color='Ort_Siddet',
    text='Weather_Condition',
    color_continuous_scale='RdYlGn_r',
    title='Hava Durumu - Kaza Sayısı ve Risk Oranı İlişkisi (Top 20)',
    labels={
        'Toplam_Kaza': 'Toplam Kaza Sayısı',
        'Yuksek_Risk_Orani': 'Yüksek Risk Oranı (%)',
        'Ort_Siddet': 'Ort. Şiddet'
    },
    height=600
)
fig.update_traces(textposition='top center', textfont_size=8)
fig.update_layout(template='plotly_white')
fig.write_html('hava_durumu_risk.html')
print('Hava durumu risk analizi kaydedildi: hava_durumu_risk.html')

# %% [markdown]
# ### 4.3 Zaman Serisi Analizleri

# %%
# Aylık kaza trendi
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Aylık kaza sayısı - yıl bazında
monthly_yearly = df_with_cluster.groupby(['Year', 'Month']).size().unstack(level=0, fill_value=0)
monthly_yearly.plot(ax=axes[0, 0], marker='o', linewidth=2)
axes[0, 0].set_xlabel('Ay')
axes[0, 0].set_ylabel('Kaza Sayısı')
axes[0, 0].set_title('Aylık Kaza Trendi (Yıl Bazında)')
axes[0, 0].set_xticks(range(1, 13))
axes[0, 0].legend(title='Yıl', fontsize=8)

# Saatlik şiddet ortalaması
hourly_severity = df_with_cluster.groupby('Hour')['Severity'].mean()
axes[0, 1].plot(hourly_severity.index, hourly_severity.values, 'r-o', linewidth=2)
axes[0, 1].fill_between(hourly_severity.index, hourly_severity.values,
                         alpha=0.2, color='red')
axes[0, 1].set_xlabel('Saat')
axes[0, 1].set_ylabel('Ortalama Kaza Şiddeti')
axes[0, 1].set_title('Saatlere Göre Ortalama Kaza Şiddeti')
axes[0, 1].set_xticks(range(0, 24))
axes[0, 1].axhline(y=hourly_severity.mean(), color='gray', linestyle='--', alpha=0.5)

# Eyalet bazında kaza sayısı (Top 15)
state_counts = df_with_cluster['State'].value_counts().head(15)
axes[1, 0].barh(state_counts.index[::-1], state_counts.values[::-1], color='teal')
axes[1, 0].set_xlabel('Kaza Sayısı')
axes[1, 0].set_title('Eyalet Bazında Kaza Sayısı (Top 15)')

# Sıcaklık ve Görüş mesafesi - risk ilişkisi
high_risk_data = df_with_cluster[df_with_cluster['Risk_Label'] == 1]
low_risk_data = df_with_cluster[df_with_cluster['Risk_Label'] == 0]

axes[1, 1].scatter(low_risk_data['Temperature(F)'].sample(5000, random_state=42),
                    low_risk_data['Visibility(mi)'].sample(5000, random_state=42),
                    alpha=0.2, s=5, c='green', label='Düşük Risk')
axes[1, 1].scatter(high_risk_data['Temperature(F)'].sample(min(5000, len(high_risk_data)), random_state=42),
                    high_risk_data['Visibility(mi)'].sample(min(5000, len(high_risk_data)), random_state=42),
                    alpha=0.3, s=5, c='red', label='Yüksek Risk')
axes[1, 1].set_xlabel('Sıcaklık (°F)')
axes[1, 1].set_ylabel('Görüş Mesafesi (mi)')
axes[1, 1].set_title('Sıcaklık vs Görüş Mesafesi - Risk Dağılımı')
axes[1, 1].legend(markerscale=5)

plt.suptitle('Kapsamlı Kaza Analiz Panosu', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/11_kapsamli_analiz.png', dpi=150, bbox_inches='tight')
plt.close()
print('Grafik kaydedildi: 11_kapsamli_analiz.png')

# %% [markdown]
# ### 4.4 Özet ve Sonuçlar

# %%
# Proje özet istatistikleri
print("=" * 60)
print("     TRAFİKTE KAZA RİSKİ TAHMİNİ - PROJE ÖZETİ")
print("=" * 60)

print(f"\n📊 Veri Seti: {SAMPLE_SIZE:,} kayıt analiz edildi")
print(f"\n🔍 AŞAMA 1 - EDA:")
print(f"   • {len(feature_cols)} özellik mühendisliği ile üretildi")
print(f"   • Eksik veriler medyan/mod ile dolduruldu")

print(f"\n📍 AŞAMA 2 - Kümeleme:")
print(f"   • K-Means ile {OPTIMAL_K} coğrafi risk kümesi belirlendi")
for i in range(OPTIMAL_K):
    print(f"   • Küme {i}: {cluster_summary.loc[i, 'Kaza_Sayisi']:,} kaza, "
          f"Risk: {cluster_summary.loc[i, 'Risk_Seviyesi']}")

print(f"\n🎯 AŞAMA 3 - Sınıflandırma:")
rf_acc = accuracy_score(y_test, rf_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)
rf_auc = roc_auc_score(y_test, rf_proba)
xgb_auc = roc_auc_score(y_test, xgb_proba)
print(f"   • Random Forest - Accuracy: {rf_acc:.4f}, AUC: {rf_auc:.4f}")
print(f"   • XGBoost      - Accuracy: {xgb_acc:.4f}, AUC: {xgb_auc:.4f}")
best_model = "XGBoost" if xgb_auc > rf_auc else "Random Forest"
print(f"   • En iyi model: {best_model}")

print(f"\n📈 AŞAMA 4 - Görselleştirme:")
print(f"   • Interaktif risk haritası: kaza_risk_haritasi.html")
print(f"   • Saatlik risk analizi: saatlik_risk_analizi.html")

# En önemli risk faktörleri
best_imp = xgb_importance if best_model == "XGBoost" else rf_importance
top_factors = best_imp.sort_values(ascending=False).head(5)
print(f"\n⚠️  En Kritik Risk Faktörleri ({best_model}):")
for factor, imp in top_factors.items():
    print(f"   • {factor}: {imp:.4f}")

print("\n" + "=" * 60)
