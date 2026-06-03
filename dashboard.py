import os
import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

MODEL_PATH = "models/risk_model.pkl"
DATA_PATH = "data/US_Accidents_March23.csv"
MAP_PATH = "kaza_risk_haritasi.html"
PLOTS_DIR = "output_plots"

st.set_page_config(
    page_title="Trafik Kaza Riski Dashboard",
    layout="wide"
)

st.title("Trafik Kaza Riski Tahmini Dashboard")

st.info(
    "Bu dashboard geçmiş kaza verileriyle eğitilmiş bir modele dayanır. "
    "Sonuçlar canlı trafik veya anlık hava durumu verisi değil, geçmiş veri tabanlı risk tahminidir."
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_sample_data():
    use_cols = [
        "Severity", "Start_Time", "Start_Lat", "Start_Lng",
        "City", "State", "Temperature(F)", "Humidity(%)",
        "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)",
        "Precipitation(in)", "Weather_Condition", "Sunrise_Sunset"
    ]

    df = pd.read_csv(DATA_PATH, usecols=use_cols, nrows=100_000)
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df["Hour"] = df["Start_Time"].dt.hour
    df["Month"] = df["Start_Time"].dt.month
    df["Risk_Label"] = (df["Severity"] >= 3).astype(int)

    return df


def calculate_extra_features(hour, day_of_week):
    is_peak_hour = 1 if (7 <= hour <= 9) or (16 <= hour <= 19) else 0
    is_night = 1 if hour >= 21 or hour <= 6 else 0
    is_weekend = 1 if day_of_week >= 5 else 0

    return is_peak_hour, is_night, is_weekend


def create_input_form(prefix=""):
    temperature = st.slider(f"{prefix}Sıcaklık (F)", -20, 130, 70, key=f"{prefix}_temperature")
    humidity = st.slider(f"{prefix}Nem (%)", 0, 100, 60, key=f"{prefix}_humidity")
    pressure = st.slider(f"{prefix}Basınç (in)", 20.0, 35.0, 29.5, key=f"{prefix}_pressure")
    visibility = st.slider(f"{prefix}Görüş Mesafesi (mi)", 0.0, 10.0, 8.0, key=f"{prefix}_visibility")
    wind_speed = st.slider(f"{prefix}Rüzgar Hızı (mph)", 0.0, 80.0, 10.0, key=f"{prefix}_wind_speed")
    precipitation = st.slider(f"{prefix}Yağış (in)", 0.0, 5.0, 0.0, key=f"{prefix}_precipitation")
    distance = st.slider(f"{prefix}Kaza Etki Mesafesi (mi)", 0.0, 20.0, 1.0, key=f"{prefix}_distance")

    hour = st.slider(f"{prefix}Saat", 0, 23, 8, key=f"{prefix}_hour")

    day_of_week = st.selectbox(
        f"{prefix}Gün",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: [
            "Pazartesi", "Salı", "Çarşamba", "Perşembe",
            "Cuma", "Cumartesi", "Pazar"
        ][x],
        key=f"{prefix}_day"
    )

    month = st.slider(f"{prefix}Ay", 1, 12, 5, key=f"{prefix}_month")

    lat = st.number_input(f"{prefix}Enlem", value=39.8283, key=f"{prefix}_lat")
    lng = st.number_input(f"{prefix}Boylam", value=-98.5795, key=f"{prefix}_lng")

    st.write("Yol Özellikleri")

    col1, col2, col3 = st.columns(3)

    with col1:
        amenity = st.checkbox(f"{prefix}Yakında tesis var mı?", key=f"{prefix}_amenity")
        bump = st.checkbox(f"{prefix}Kasis var mı?", key=f"{prefix}_bump")
        crossing = st.checkbox(f"{prefix}Yaya geçidi var mı?", key=f"{prefix}_crossing")
        give_way = st.checkbox(f"{prefix}Yol ver tabelası var mı?", key=f"{prefix}_give_way")

    with col2:
        junction = st.checkbox(f"{prefix}Kavşak var mı?", key=f"{prefix}_junction")
        no_exit = st.checkbox(f"{prefix}Çıkmaz yol mu?", key=f"{prefix}_no_exit")
        railway = st.checkbox(f"{prefix}Demiryolu var mı?", key=f"{prefix}_railway")
        roundabout = st.checkbox(f"{prefix}Dönel kavşak var mı?", key=f"{prefix}_roundabout")

    with col3:
        station = st.checkbox(f"{prefix}İstasyon var mı?", key=f"{prefix}_station")
        stop = st.checkbox(f"{prefix}Dur tabelası var mı?", key=f"{prefix}_stop")
        traffic_calming = st.checkbox(f"{prefix}Trafik sakinleştirici var mı?", key=f"{prefix}_traffic_calming")
        traffic_signal = st.checkbox(f"{prefix}Trafik ışığı var mı?", key=f"{prefix}_traffic_signal")
        turning_loop = st.checkbox(f"{prefix}Turning loop var mı?", key=f"{prefix}_turning_loop")

    return {
        "temperature": temperature,
        "humidity": humidity,
        "pressure": pressure,
        "visibility": visibility,
        "wind_speed": wind_speed,
        "precipitation": precipitation,
        "distance": distance,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "lat": lat,
        "lng": lng,
        "amenity": amenity,
        "bump": bump,
        "crossing": crossing,
        "give_way": give_way,
        "junction": junction,
        "no_exit": no_exit,
        "railway": railway,
        "roundabout": roundabout,
        "station": station,
        "stop": stop,
        "traffic_calming": traffic_calming,
        "traffic_signal": traffic_signal,
        "turning_loop": turning_loop
    }


def make_prediction_input(values, bundle):
    feature_cols = bundle["feature_cols"]
    geo_scaler = bundle["geo_scaler"]
    kmeans = bundle["kmeans"]

    is_peak_hour, is_night, is_weekend = calculate_extra_features(
        values["hour"],
        values["day_of_week"]
    )

    geo_scaled = geo_scaler.transform([[values["lat"], values["lng"]]])
    cluster = int(kmeans.predict(geo_scaled)[0])

    input_data = pd.DataFrame([{
        "Temperature(F)": values["temperature"],
        "Humidity(%)": values["humidity"],
        "Pressure(in)": values["pressure"],
        "Visibility(mi)": values["visibility"],
        "Wind_Speed(mph)": values["wind_speed"],
        "Precipitation(in)": values["precipitation"],
        "Distance(mi)": values["distance"],
        "Hour": values["hour"],
        "Day_of_Week": values["day_of_week"],
        "Month": values["month"],
        "Is_Peak_Hour": is_peak_hour,
        "Is_Night": is_night,
        "Is_Weekend": is_weekend,
        "Amenity": int(values["amenity"]),
        "Bump": int(values["bump"]),
        "Crossing": int(values["crossing"]),
        "Give_Way": int(values["give_way"]),
        "Junction": int(values["junction"]),
        "No_Exit": int(values["no_exit"]),
        "Railway": int(values["railway"]),
        "Roundabout": int(values["roundabout"]),
        "Station": int(values["station"]),
        "Stop": int(values["stop"]),
        "Traffic_Calming": int(values["traffic_calming"]),
        "Traffic_Signal": int(values["traffic_signal"]),
        "Turning_Loop": int(values["turning_loop"]),
        "Duration_min": 30,
        "Cluster": cluster
    }])

    input_data = input_data[feature_cols]

    return input_data, cluster, is_peak_hour, is_night, is_weekend


def predict_risk(values, bundle):
    model = bundle["model"]
    input_data, cluster, is_peak_hour, is_night, is_weekend = make_prediction_input(values, bundle)

    risk_probability = model.predict_proba(input_data)[0][1]
    risk_prediction = model.predict(input_data)[0]

    return {
        "input_data": input_data,
        "cluster": cluster,
        "risk_probability": risk_probability,
        "risk_prediction": risk_prediction,
        "is_peak_hour": is_peak_hour,
        "is_night": is_night,
        "is_weekend": is_weekend
    }


def show_risk_comment(result):
    probability = result["risk_probability"]

    if probability >= 0.70:
        st.error("Bu koşullarda yüksek risk tahmini yapıldı.")
    elif probability >= 0.40:
        st.warning("Bu koşullarda orta seviyede risk tahmini yapıldı.")
    else:
        st.success("Bu koşullarda düşük risk tahmini yapıldı.")

    reasons = []

    row = result["input_data"].iloc[0]

    if row["Visibility(mi)"] <= 3:
        reasons.append("Görüş mesafesi düşük.")
    if row["Precipitation(in)"] > 0:
        reasons.append("Yağış bilgisi mevcut.")
    if row["Wind_Speed(mph)"] >= 25:
        reasons.append("Rüzgar hızı yüksek.")
    if result["is_peak_hour"] == 1:
        reasons.append("Pik saat aralığında.")
    if result["is_night"] == 1:
        reasons.append("Gece saatlerinde.")
    if row["Junction"] == 1:
        reasons.append("Kavşak bilgisi mevcut.")
    if row["Traffic_Signal"] == 1:
        reasons.append("Trafik ışığı bilgisi mevcut.")
    if row["Crossing"] == 1:
        reasons.append("Yaya geçidi bilgisi mevcut.")

    if reasons:
        st.write("Tahmini etkileyebilecek başlıca koşullar:")
        for reason in reasons:
            st.write(f"- {reason}")
    else:
        st.write("Girilen koşullarda belirgin bir risk artırıcı faktör öne çıkmıyor.")


if not os.path.exists(MODEL_PATH):
    st.error("Model dosyası bulunamadı. Önce train_model.py dosyasını çalıştırıp modeli oluşturmalısın.")
    st.stop()

bundle = load_model()

tab_prediction, tab_analysis, tab_map, tab_model, tab_compare, tab_data = st.tabs([
    "Risk Tahmini",
    "Veri Analizi",
    "Harita",
    "Model Performansı",
    "Senaryo Karşılaştırma",
    "Veri Önizleme"
])

with tab_prediction:
    st.subheader("Risk Tahmini")

    with st.sidebar:
        st.header("Tahmin Parametreleri")
        values = create_input_form("Ana")

    result = predict_risk(values, bundle)

    col1, col2, col3 = st.columns(3)

    with col1:
        label = "Yüksek Risk" if result["risk_prediction"] == 1 else "Düşük Risk"
        st.metric("Risk Tahmini", label)

    with col2:
        st.metric("Yüksek Risk Olasılığı", f"{result['risk_probability'] * 100:.2f}%")

    with col3:
        st.metric("Coğrafi Küme", result["cluster"])

    show_risk_comment(result)

    st.divider()
    st.subheader("Modelin Kullandığı Girdi Değerleri")
    st.dataframe(result["input_data"], use_container_width=True)


with tab_analysis:
    st.subheader("Veri Seti Analizi")

    if os.path.exists(DATA_PATH):
        df = load_sample_data()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("İncelenen Kayıt", f"{len(df):,}")

        with col2:
            st.metric("Ortalama Şiddet", f"{df['Severity'].mean():.2f}")

        with col3:
            high_risk_rate = df["Risk_Label"].mean() * 100
            st.metric("Yüksek Risk Oranı", f"{high_risk_rate:.2f}%")

        with col4:
            st.metric("Eyalet Sayısı", df["State"].nunique())

        selected_state = st.selectbox(
            "Eyalet seç",
            options=["Tümü"] + sorted(df["State"].dropna().unique().tolist())
        )

        filtered_df = df.copy()

        if selected_state != "Tümü":
            filtered_df = filtered_df[filtered_df["State"] == selected_state]

        st.write("Kaza Şiddeti Dağılımı")
        severity_counts = filtered_df["Severity"].value_counts().sort_index()
        st.bar_chart(severity_counts)

        st.write("Saatlere Göre Kaza Sayısı")
        hourly_counts = filtered_df.groupby("Hour").size()
        st.line_chart(hourly_counts)

        st.write("En Fazla Kaza Olan Eyaletler")
        state_counts = df["State"].value_counts().head(15)
        st.bar_chart(state_counts)

        st.write("Hava Durumuna Göre Kaza Sayısı")
        weather_counts = filtered_df["Weather_Condition"].value_counts().head(15)
        st.bar_chart(weather_counts)
    else:
        st.warning("Veri dosyası bulunamadı. data/US_Accidents_March23.csv yolunu kontrol et.")

    st.divider()
    st.subheader("Kayıtlı Görseller")

    image_files = [
        "02_kaza_siddeti.png",
        "03_zaman_analizi.png",
        "04_hava_sicaklik.png",
        "05_mevsim_gece.png",
        "06_korelasyon.png",
        "11_kapsamli_analiz.png"
    ]

    for image_file in image_files:
        image_path = os.path.join(PLOTS_DIR, image_file)

        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)


with tab_map:
    st.subheader("İnteraktif Kaza Risk Haritası")

    if os.path.exists(MAP_PATH):
        with open(MAP_PATH, "r", encoding="utf-8") as file:
            map_html = file.read()

        components.html(map_html, height=650)
    else:
        st.warning("Harita dosyası bulunamadı. Önce ana analiz dosyanı çalıştırıp kaza_risk_haritasi.html üretmelisin.")


with tab_model:
    st.subheader("Model Performansı")

    col1, col2 = st.columns(2)

    with col1:
        performance_path = os.path.join(PLOTS_DIR, "09_model_performans.png")

        if os.path.exists(performance_path):
            st.image(performance_path, use_container_width=True)
        else:
            st.warning("Model performans grafiği bulunamadı.")

    with col2:
        importance_path = os.path.join(PLOTS_DIR, "10_feature_importance.png")

        if os.path.exists(importance_path):
            st.image(importance_path, use_container_width=True)
        else:
            st.warning("Özellik önem grafiği bulunamadı.")

    st.info(
        "Model performansı eğitim/test ayrımı üzerinden hesaplanır. "
        "Bu değerler modelin geçmiş verideki başarı seviyesini gösterir."
    )


with tab_compare:
    st.subheader("Senaryo Karşılaştırma")

    col_a, col_b = st.columns(2)

    with col_a:
        st.write("Senaryo A")
        scenario_a = create_input_form("A")
        result_a = predict_risk(scenario_a, bundle)

    with col_b:
        st.write("Senaryo B")
        scenario_b = create_input_form("B")
        result_b = predict_risk(scenario_b, bundle)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        label_a = "Yüksek Risk" if result_a["risk_prediction"] == 1 else "Düşük Risk"
        st.metric("Senaryo A", label_a, f"{result_a['risk_probability'] * 100:.2f}%")

    with col2:
        label_b = "Yüksek Risk" if result_b["risk_prediction"] == 1 else "Düşük Risk"
        st.metric("Senaryo B", label_b, f"{result_b['risk_probability'] * 100:.2f}%")

    difference = abs(result_a["risk_probability"] - result_b["risk_probability"]) * 100

    if result_a["risk_probability"] > result_b["risk_probability"]:
        st.warning(f"Senaryo A, Senaryo B'ye göre yaklaşık %{difference:.2f} daha riskli görünüyor.")
    elif result_b["risk_probability"] > result_a["risk_probability"]:
        st.warning(f"Senaryo B, Senaryo A'ya göre yaklaşık %{difference:.2f} daha riskli görünüyor.")
    else:
        st.success("İki senaryonun risk oranı aynı görünüyor.")


with tab_data:
    st.subheader("Veri Önizleme")

    if os.path.exists(DATA_PATH):
        df = load_sample_data()

        st.write("İlk 1000 kayıt")
        st.dataframe(df.head(1000), use_container_width=True)

        st.write("Eksik Veri Oranları")
        missing = df.isnull().mean().sort_values(ascending=False) * 100
        st.dataframe(
            missing.reset_index().rename(
                columns={"index": "Sütun", 0: "Eksik Oran (%)"}
            ),
            use_container_width=True
        )
    else:
        st.warning("Veri dosyası bulunamadı.")
