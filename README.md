# TRAFİKTE KAZA RİSKİ TAHMİNİ - PROJE ANALİZ RAPORU

---

## 1. Projenin Çözdüğü Problem

Bu proje, **trafik kazalarının şiddetini ve riskini tahmin etme** problemini çözmektedir. Temel amaçlar:

- **Risk Tahmini:** Belirli hava koşulları, saat, konum ve yol özellikleri altında bir kazanın **yüksek riskli (Severity ≥ 3)** mi yoksa **düşük riskli (Severity ≤ 2)** mi olacağını makine öğrenmesi ile tahmin etmek.
- **Coğrafi Risk Bölgeleri:** K-Means kümeleme algoritması ile ABD genelinde **kaza yoğunluk bölgelerini** belirlemek ve coğrafi risk haritası oluşturmak.
- **Risk Faktörleri Analizi:** Kazaların şiddetini etkileyen en kritik faktörleri (trafik sinyali, kavşak, hava durumu, saat vb.) tespit etmek.
- **Zamansal Analiz:** Saatlik, günlük, aylık ve mevsimsel kaza eğilimlerini ortaya koymak.

**Pratikte karşılığı:** Trafik yönetim merkezleri, sigorta şirketleri ve şehir planlama birimleri bu tür analizleri kullanarak riskli bölgelerde önlem alabilir, trafik sinyalizasyonunu optimize edebilir ve kaynakları etkin dağıtabilir.

---

## 2. Kullanılan Veri Seti

| Özellik | Detay |
|---------|-------|
| **Veri Seti** | US_Accidents_March23.csv |
| **Kaynak** | ABD geneli trafik kaza kayıtları (Şubat 2016 - Mart 2023) |
| **Toplam Kayıt** | 7.728.394 satır |
| **Sütun Sayısı** | 46 sütun |
| **Dosya Boyutu** | ~3.06 GB |
| **Kullanılan Örneklem** | 300.000 kayıt (rastgele örnekleme, performans için) |

### Kullanılan Temel Sütunlar (22 adet):

| Kategori | Sütunlar |
|----------|----------|
| **Hedef Değişken** | `Severity` (1-4 arası şiddet seviyesi) |
| **Konum** | `Start_Lat`, `Start_Lng`, `State`, `City` |
| **Zaman** | `Start_Time`, `End_Time`, `Sunrise_Sunset` |
| **Hava Durumu** | `Temperature(F)`, `Humidity(%)`, `Pressure(in)`, `Visibility(mi)`, `Wind_Speed(mph)`, `Precipitation(in)`, `Weather_Condition` |
| **Yol Koşulları** | `Distance(mi)`, `Amenity`, `Bump`, `Crossing`, `Give_Way`, `Junction`, `No_Exit`, `Railway`, `Roundabout`, `Station`, `Stop`, `Traffic_Calming`, `Traffic_Signal`, `Turning_Loop` |

### Üretilen Mühendislik Özellikleri (10 adet):

| Özellik | Açıklama |
|---------|----------|
| `Hour` | Kazanın gerçekleştiği saat (0-23) |
| `Day_of_Week` | Haftanın günü (0=Pazartesi, 6=Pazar) |
| `Month`, `Year` | Ay ve yıl bilgisi |
| `Is_Peak_Hour` | Trafiğin yoğun olduğu saatler (07-09, 16-19) |
| `Is_Night` | Gece saatleri (21:00-06:00) |
| `Is_Weekend` | Hafta sonu mu? |
| `Season` | Mevsim (1-4) |
| `Duration_min` | Kazanın süresi (dakika) |
| `Risk_Label` | Binary hedef: Severity ≥ 3 → Yüksek Risk (1) |

---

## 3. Kod Tarafı: Kullanılan Kütüphaneler ve Algoritmalar

### 3.1 Programlama Dili ve Ortam

| Bileşen | Detay |
|---------|-------|
| **Dil** | Python 3.13 |
| **Çalışma Ortamı** | VS Code Interactive Python (`.py` dosyası, `# %%` hücre yapısı) |
| **Grafik Backend** | Matplotlib Agg (non-interactive, dosyaya kayıt) |

### 3.2 Kütüphane Listesi

| Kütüphane | Versiyon | Kullanım Amacı |
|-----------|----------|----------------|
| **pandas** | - | Veri manipülasyonu, CSV okuma, gruplama, eksik veri işleme |
| **numpy** | - | Sayısal hesaplamalar, üst/alt üçgen maske oluşturma |
| **scikit-learn** | - | K-Means kümeleme, Random Forest sınıflandırma, model değerlendirme (ROC, Confusion Matrix), train/test split, StandardScaler |
| **xgboost** | 3.2.0 | Gradient Boosting tabanlı sınıflandırma modeli |
| **imbalanced-learn** | 0.14.1 | SMOTE (Synthetic Minority Over-sampling Technique) ile veri dengeleme |
| **matplotlib** | - | Statik grafikler (11 adet PNG) |
| **seaborn** | - | İstatistiksel görselleştirme (heatmap, countplot, boxplot) |
| **folium** | - | Leaflet tabanlı interaktif harita ve ısı haritası |
| **plotly** | - | İnteraktif web tabanlı grafikler |

> **Not:** Projede **SimPy** kullanılmamıştır. SimPy, kesikli olay simülasyonu (discrete event simulation) kütüphanesidir ve trafik akış simülasyonu için uygundur. Bu projede bunun yerine gerçek kaza verileri üzerinde makine öğrenmesi uygulanmıştır.

### 3.3 Kullanılan Algoritmalar

| Algoritma | Aşama | Açıklama |
|-----------|-------|----------|
| **MiniBatch K-Means** | Kümeleme | Büyük veri için optimize edilmiş K-Means. 50.000 örneklem üzerinde Elbow yöntemi ile optimal K=6 belirlenmiş. Coğrafi risk kümeleri oluşturur. |
| **Random Forest** | Sınıflandırma | 200 ağaç (n_estimators=200), max_depth=15. Ensemble tabanlı sınıflandırıcı. Sonuçlar: ~%74.5 accuracy, ~0.80 AUC. |
| **XGBoost** | Sınıflandırma | 200 ağaç, max_depth=8, learning_rate=0.1. Gradient Boosting. **En iyi model.** Sonuçlar: ~%81.7 accuracy, ~0.84 AUC. |
| **SMOTE** | Veri Dengeleme | Dengesiz sınıf dağılımını düzeltmek için azınlık sınıfı (Yüksek Risk) yapay örneklerle çoğaltılmış. ~386K dengeli eğitim seti oluşturulmuş. |
| **Elbow Method** | Hiperparametre | K-Means'de optimal küme sayısını belirlemek için Inertia değerlerinin analiz edildiği yöntem. |
| **StandardScaler** | Ön İşleme | Coğrafi koordinatların (Lat/Lng) normalize edilmesi. |

### 3.4 Proje Aşamaları (Pipeline)

```
[CSV Veri Seti] → [300K Örnekleme] → [Eksik Veri Analizi & Doldurma]
       ↓
[Özellik Mühendisliği (10 yeni özellik)]
       ↓
[AŞAMA 1: EDA - 6 Görselleştirme]
       ↓
[AŞAMA 2: K-Means Kümeleme → 6 Coğrafi Risk Kümesi]
       ↓
[AŞAMA 3: SMOTE → Random Forest + XGBoost → Model Karşılaştırma]
       ↓
[AŞAMA 4: Folium Harita + Plotly Grafikler + Kapsamlı Analiz]
```

---

## 4. Arayüz Teknolojisi

### Mevcut Durum

Bu proje şu an **doğrudan arayüz (GUI/Web UI) içermemektedir**. Proje bir **batch analiz pipeline'ı** olarak tasarlanmıştır:

| Çıktı Türü | Teknoloji | Açıklama |
|-------------|-----------|----------|
| **Statik Grafikler** | Matplotlib/Seaborn | PNG dosyaları olarak `output_plots/` klasörüne kaydedilir. Tarayıcıda veya resim görüntüleyicide açılabilir. |
| **İnteraktif Harita** | Folium (Leaflet.js) | `kaza_risk_haritasi.html` - Tarayıcıda açılan interaktif ısı haritası |
| **İnteraktif Grafikler** | Plotly | `saatlik_risk_analizi.html`, `hava_durumu_risk.html` - Tarayıcıda açılan etkileşimli grafikler |
| **Konsol Çıktısı** | Python print | Model sonuçları ve istatistikler terminale yazdırılır |

> **Streamlit, Flask, Django, React, Vue.js** gibi web arayüz çerçeveleri şu an kullanılmamıştır.
>
> **Gelecek geliştirme önerisi:** Projeye **Streamlit** ile interaktif bir web arayüzü eklenebilir. Streamlit, Python tabanlı olduğu için mevcut kodla kolayca entegre edilebilir ve kullanıcıların parametreleri (saat, hava durumu, konum) değiştirerek canlı risk tahmini almasını sağlayabilir.

---

## 5. Görselleştirme Yapıları

### 5.1 Isı Haritası (Heat Map) ✅ MEVCUT

| Tür | Teknoloji | Dosya | Açıklama |
|-----|-----------|-------|----------|
| **Coğrafi Isı Haritası** | Folium HeatMap (Leaflet.js) | `kaza_risk_haritasi.html` | ABD haritası üzerinde yüksek riskli kazaların (Severity ≥ 3) yoğunluk dağılımı. 30.000 yüksek riskli kaza noktası. Mavi→Yeşil→Sarı→Turuncu→Kırmızı renk gradyanı. Zoom yapılabilir, küme merkezleri marker olarak işaretli. **CartoDB dark_matter** harita stili. |
| **Korelasyon Isı Haritası** | Seaborn heatmap | `output_plots/06_korelasyon.png` | Özellikler arası korelasyon matrisi. RdBu_r renk skalası, üst üçgen maskelenmiş. |
| **Confusion Matrix** | Seaborn heatmap | `output_plots/09_model_performans.png` | RF ve XGBoost için tahmin/gerçek karşılaştırma ısı haritaları. |

### 5.2 GUI (Grafik Kullanıcı Arayüzü) ❌ MEVCUT DEĞİL

Projede masaüstü veya web tabanlı bir GUI bulunmamaktadır. Tüm görselleştirmeler dosya çıktısı olarak üretilir ve tarayıcıda HTML dosyaları veya resim dosyaları olarak görüntülenir.

### 5.3 SUMO (Simulation of Urban Mobility) ❌ MEVCUT DEĞİL

Projede **SUMO trafik simülasyonu** kullanılmamıştır. SUMO, canlı trafik akışı simülasyonu için mikroskopik bir simülatördür. Bu proje bunun yerine:
- Gerçek kaza verileri üzerinde istatistiksel analiz ve makine öğrenmesi uygulamıştır.
- Coğrafi kümeleme ile risk bölgeleri belirlenmiştir.
- Interaktif haritalar ile sonuçlar görselleştirilmiştir.

### 5.4 Tüm Görselleştirme Çıktıları

#### Statik Grafikler (PNG - `output_plots/` klasörü)

| # | Dosya | İçerik | Grafik Türü |
|---|-------|--------|-------------|
| 01 | `01_eksik_veri.png` | Eksik veri yüzdeleri | Yatay çubuk grafik |
| 02 | `02_siddet_dagilimi.png` | Kaza şiddeti dağılımı (1-4) | Bar plot + Pie chart |
| 03 | `03_saatlik_gunluk.png` | Saatlik ve günlük kaza dağılımı | Çizgi grafik + Bar plot |
| 04 | `04_hava_sicaklik.png` | Hava durumu ve sıcaklık etkisi | Bar plot + Box plot |
| 05 | `05_mevsim_gece.png` | Mevsimsel ve gece/gündüz analizi | Grouped bar + Stacked bar |
| 06 | `06_korelasyon.png` | Korelasyon matrisi | Heatmap |
| 07 | `07_elbow.png` | Optimal küme sayısı (Elbow) | Çizgi grafik |
| 08 | `08_cluster_harita.png` | K-Means küme dağılımı | Scatter plot (harita) |
| 09 | `09_model_performans.png` | ROC eğrisi + Confusion Matrix | ROC curve + Heatmap |
| 10 | `10_feature_importance.png` | Özellik önem sırası (RF + XGB) | Yatay çubuk grafik |
| 11 | `11_kapsamli_analiz.png` | Kapsamlı analiz panosu | 4'lü subplot (trend, şiddet, eyalet, scatter) |

#### İnteraktif Çıktılar (HTML)

| Dosya | Teknoloji | Canlı İzlenebilirlik |
|-------|-----------|---------------------|
| `kaza_risk_haritasi.html` | Folium/Leaflet.js | ✅ Zoom, pan, marker tıklama, ısı haritası katmanı |
| `saatlik_risk_analizi.html` | Plotly | ✅ Hover bilgisi, zoom, seçim, indirme |
| `hava_durumu_risk.html` | Plotly | ✅ Hover bilgisi, zoom, bubble boyutu etkileşimi |

### 5.5 Canlı İzlenebilirlik Özeti

| Özellik | Durum |
|---------|-------|
| Statik grafikler (PNG) | ✅ 11 adet |
| İnteraktif ısı haritası (Folium) | ✅ Mevcut - tarayıcıda açılabilir |
| İnteraktif grafikler (Plotly) | ✅ 2 adet - tarayıcıda açılabilir |
| Canlı web arayüzü (Streamlit/React) | ❌ Mevcut değil |
| Trafik simülasyonu (SUMO) | ❌ Mevcut değil |
| Masaüstü GUI (Tkinter/PyQt) | ❌ Mevcut değil |
| API/Backend (Flask/FastAPI) | ❌ Mevcut değil |

---

## 6. Model Performans Özeti

| Metrik | Random Forest | XGBoost |
|--------|--------------|---------|
| **Accuracy** | ~%74.5 | ~%81.7 |
| **ROC-AUC** | ~0.80 | ~0.84 |
| **En İyi Model** | | ✅ XGBoost |

### En Kritik Risk Faktörleri (XGBoost):

1. **Traffic_Signal** (~0.194) - Trafik sinyali varlığı
2. **Crossing** (~0.141) - Yaya geçidi
3. **Stop** (~0.128) - Dur işareti
4. **Duration_min** - Kaza süresi
5. **Hour** - Saat bilgisi

---

## 7. Proje Dizin Yapısı

```
fatmanurbp/
├── trafik_kaza_riski_tahmini.py      # Ana proje kodu (~800 satır)
├── US_Accidents_March23.csv/
│   └── US_Accidents_March23.csv      # Veri seti (3.06 GB)
├── output_plots/                      # Statik grafikler
│   ├── 01_eksik_veri.png
│   ├── 02_siddet_dagilimi.png
│   ├── 03_saatlik_gunluk.png
│   ├── 04_hava_sicaklik.png
│   ├── 05_mevsim_gece.png
│   ├── 06_korelasyon.png
│   ├── 07_elbow.png
│   ├── 08_cluster_harita.png
│   ├── 09_model_performans.png
│   ├── 10_feature_importance.png
│   └── 11_kapsamli_analiz.png
├── kaza_risk_haritasi.html            # Folium interaktif harita
├── saatlik_risk_analizi.html          # Plotly interaktif grafik
├── hava_durumu_risk.html              # Plotly interaktif grafik
└── PROJE_RAPORU.md                    # Bu rapor
```

---

*Rapor Tarihi: 2025*
*Python 3.13 | scikit-learn | XGBoost | Folium | Plotly*
