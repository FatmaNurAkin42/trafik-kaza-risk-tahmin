# Trafik Kaza Riski Tahmini Dashboard

Bu proje, ABD trafik kazası verileri üzerinden kaza riskini analiz eden, model eğiten ve Streamlit tabanlı bir dashboard ile risk tahmini yapılmasını sağlayan bir Python projesidir. Temel amaç; hava koşulları, zaman, konum ve yol özelliklerine göre bir durumun yüksek riskli olup olmadığını tahmin etmek ve geçmiş kaza verilerini görsel olarak incelemektir.

> Not: Dashboard canlı trafik veya anlık hava durumu verisi kullanmaz. Tahminler, geçmiş kaza verileriyle eğitilen modelin olasılık hesaplamalarına dayanır.

## Proje Özeti

- `Severity >= 3` kayıtları yüksek risk, daha düşük şiddetteki kayıtlar düşük risk olarak sınıflandırılır.
- Model eğitimi için hava durumu, zaman, konum ve yol özelliklerinden türetilen özellikler kullanılır.
- Coğrafi risk bölgeleri için koordinatlar üzerinden K-Means/MiniBatchKMeans kümeleme yapılır.
- Sınıf dengesizliğini azaltmak için SMOTE kullanılır.
- Dashboard; risk tahmini, veri analizi, harita, model performansı, senaryo karşılaştırma ve veri önizleme bölümlerinden oluşur.

## Kullanılan Teknolojiler

- Python
- pandas, numpy
- scikit-learn
- imbalanced-learn / SMOTE
- joblib
- matplotlib, seaborn
- plotly
- folium
- streamlit

## Proje Yapısı

```text
fatmanurbp/
├── dashboard.py                        # Streamlit dashboard arayüzü
├── train_model.py                      # Dashboard için model eğitim scripti
├── trafik_kaza_riski_tahmini.py          # Detaylı analiz, görselleştirme ve modelleme pipeline'ı
├── requirements.txt                     # Python bağımlılıkları
├── PROJE_RAPORU.md                      # Ayrıntılı proje analiz raporu
├── data/
│   └── US_Accidents_March23.csv          # Ana veri seti
├── models/
│   └── risk_model.pkl                   # Eğitilmiş model paketi
├── output_plots/
│   ├── 01_eksik_veri.png
│   ├── 02_kaza_siddeti.png
│   ├── 03_zaman_analizi.png
│   ├── 04_hava_sicaklik.png
│   ├── 05_mevsim_gece.png
│   ├── 06_korelasyon.png
│   ├── 07_elbow.png
│   ├── 08_cluster_harita.png
│   ├── 09_model_performans.png
│   ├── 10_feature_importance.png
│   └── 11_kapsamli_analiz.png
├── kaza_risk_haritasi.html              # Folium interaktif risk haritası
├── saatlik_risk_analizi.html            # Plotly saatlik risk analizi
└── hava_durumu_risk.html                # Plotly hava durumu/risk analizi
Dashboard Bölümleri
1. Risk Tahmini
Kullanıcının girdiği koşullara göre model yüksek risk olasılığı hesaplar. Bu bölümde sıcaklık, nem, basınç, görüş mesafesi, rüzgar hızı, yağış, saat, gün, ay, konum ve yol özellikleri gibi parametreler kullanılır.

Dashboard şu çıktıları gösterir:

Risk sınıfı: düşük risk veya yüksek risk

Yüksek risk olasılığı

Tahmin edilen coğrafi küme

Tahmini etkileyebilecek başlıca koşullar

Modelin kullandığı nihai girdi tablosu

2. Veri Analizi
data/US_Accidents_March23.csv dosyasından ilk 100.000 kayıt okunur ve genel veri özeti sunulur.

Bu bölümde:

İncelenen kayıt sayısı

Ortalama kaza şiddeti

Yüksek risk oranı

Eyalet sayısı

Kaza şiddeti dağılımı

Saatlere göre kaza sayısı

En fazla kaza olan eyaletler

Hava durumuna göre kaza sayısı

Kayıtlı analiz görselleri

gösterilir.

3. Harita
kaza_risk_haritasi.html dosyası dashboard içinde gömülü olarak açılır. Bu dosya, Folium/Leaflet tabanlı interaktif bir risk haritasıdır.

Harita, yüksek riskli kaza noktalarının yoğunluğunu ve küme merkezlerini incelemek için kullanılır.

4. Model Performansı
Model performansı ve özellik önemi görselleri bu sekmede gösterilir.

Kullanılan dosyalar:

output_plots/09_model_performans.png

output_plots/10_feature_importance.png

Bu bölüm, modelin geçmiş veri üzerindeki başarı seviyesini ve karar verirken hangi özelliklere daha fazla ağırlık verdiğini anlamak için kullanılır.

5. Senaryo Karşılaştırma
İki farklı koşul seti yan yana girilir ve model her iki senaryo için risk olasılığı hesaplar.

Bu bölüm şu sorular için kullanışlıdır:

Hangi hava koşulu daha riskli görünüyor?

Gece ve gündüz koşulları arasında risk farkı var mı?

Kavşak, trafik ışığı veya yaya geçidi gibi yol özellikleri riski nasıl etkiliyor?

6. Veri Önizleme
Veri setinin ilk 1000 kaydı ve sütun bazlı eksik veri oranları gösterilir. Bu bölüm, dashboard'un kullandığı veriyi hızlı kontrol etmek için eklenmiştir.

Kod Akışı
train_model.py
Dashboard tarafında kullanılan modeli üretir.

Genel akış:

data/US_Accidents_March23.csv dosyasını okur.

300.000 kayıtlık örneklem alır.

Eksik sayıal verileri medyan ile doldurur.

Eksik kategorik verileri mod ile doldurur.

Zaman özellikleri üretir: Hour, Day_of_Week, Month, Is_Peak_Hour, Is_Night, Is_Weekend.

Kaza süresini Duration_min olarak hesaplar.

Koordinatları ölçekler ve MiniBatchKMeans ile Cluster özelliğini üretir.

Severity >= 3 için Risk_Label hedef değişkenini oluşturur.

SMOTE ile eğitim verisini dengeler.

RandomForestClassifier modelini eğitir.

Modeli, özellik listesini, coğrafi scaler'ı ve KMeans nesnesini models/risk_model.pkl dosyasına kaydeder.

Kaydedilen model paketi şu anahtarları içerir:

model

feature_cols

geo_scaler

kmeans

dashboard.py
Streamlit arayüzünü oluşturur ve models/risk_model.pkl model paketini kullanarak tahmin yapar.

Ana sorumlulukları:

Modeli cache ile yüklemek

Örnek veri setini dashboard için okumak

Kullanıcıdan tahmin parametrelerini almak

Tahmin girdisini modelin beklediği kolon sıralamasına getirmek

Risk olasılığını ve risk sınıfını hesaplamak

Analiz grafikleri, harita ve veri önizleme ekranlarını göstermek

trafik_kaza_riski_tahmini.py
Notebook/hücre yapısında yazılmış ana analiz pipeline'ıdır. EDA, kümeleme, model karşılaştırma ve görselleştirme çıktılarını üretir.

Ürettiği başlıca çıktılar:

output_plots/ altındaki PNG analiz grafikleri

kaza_risk_haritasi.html

saatlik_risk_analizi.html

hava_durumu_risk.html

Bu dosyada Random Forest ve XGBoost karşılaştırması da bulunur. Dashboard'daki model eğitiminden farklı olarak daha geniş bir analiz ve raporlama amacı taşır.

Dikkat: Bu dosyada veri yolu US_Accidents_March23.csv/US_Accidents_March23.csv olarak tanımlı görünüyor. Mevcut klasör yapısında veri dosyası data/US_Accidents_March23.csv altındadır. Ana analiz scriptini çalıştırmadan önce DATA_PATH değerinin mevcut veri yoluyla uyumlu olduğunu kontrol edin.

Kurulum
Python sanal ortamı kullanılması önerilir.

Bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Linux/macOS için aktivasyon komutu:

Bash
source .venv/bin/activate
Çalıştırma
1. Modeli eğitme
Model dosyası yoksa veya yeniden eğitmek istenirse:

Bash
python train_model.py
Bu komut models/risk_model.pkl dosyasını oluşturur.

2. Dashboard'u başlatma
Bash
streamlit run dashboard.py
Streamlit çalıştıktan sonra terminalde verilen lokal adres tarayıcıda açılarak dashboard kullanılabilir.

3. Analiz çıktılarını yeniden üretme
Statik grafikler ve HTML çıktıları yeniden üretilmek istenirse:

Bash
python trafik_kaza_riski_tahmini.py
Çalıştırmadan önce bu dosyadaki DATA_PATH değerinin data/US_Accidents_March23.csv ile uyumlu olduğunu kontrol edin.

Veri Seti
Proje, US_Accidents_March23.csv veri setini kullanır. Mevcut dosya data/ klasörü altındadır ve büyük boyutlu olduğu için repoya taşınırken veya paylaşılırken dikkat edilmelidir.

Dashboard tarafında veri önizleme ve analiz için ilk 100.000 kayıt okunur. Model eğitiminde ise performans nedeniyle 300.000 kayıtlık rastgele örneklem kullanılır.

Model Özellikleri
Modelde kullanılan başlıca özellik grupları:

Hava durumu: sıcaklık, nem, basınç, görüş mesafesi, rüzgar hızı, yağış

Zaman: saat, gün, ay, pik saat, gece, hafta sonu

Konum: enlem, boylamdan üretilen coğrafi küme

Yol özellikleri: kavşak, trafik ışığı, yaya geçidi, dur tabelası, kasis, istasyon, demiryolu vb.

Kaza bilgisi: etki mesafesi ve süre

Hedef değişken:

Plaintext
Severity >= 3  -> Yüksek Risk
Severity <= 2  -> Düşük Risk
Üretilen Çıktılar
PNG grafikler
output_plots/ klasörü içinde EDA, korelasyon, kümeleme, model performansı ve özellik önemi görselleri bulunur.

HTML çıktılar
kaza_risk_haritasi.html: interaktif risk haritası

saatlik_risk_analizi.html: saat bazlı risk analizi

hava_durumu_risk.html: hava durumu ve risk ilişkisi

Bu dosyalar doğrudan tarayıcıda açılabilir veya dashboard içinde gösterilebilir.

Önemli Notlar
models/risk_model.pkl dosyası olmadan dashboard tahmin sekmesi çalışmaz. Eksikse önce train_model.py çalıştırılmalıdır.

Dashboard tahminleri geçmiş veri tabanlı model çıktısıdır; gerçek zamanlı trafik karar sistemi olarak kullanılmamalıdır.

Veri seti büyük olduğu için okuma ve model eğitimi zaman alabilir.

Ana analiz scripti ile dashboard eğitim scripti benzer özellikler kullansa da birebir aynı amaca hizmet etmez: trafik_kaza_riski_tahmini.py analiz ve raporlama, train_model.py ise dashboard model paketi üretme odaklıdır.
