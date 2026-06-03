import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, roc_auc_score

DATA_PATH = "data/US_Accidents_March23.csv"
MODEL_PATH = "models/risk_model.pkl"

USE_COLS = [
    "Severity", "Start_Time", "End_Time", "Start_Lat", "Start_Lng",
    "Distance(mi)", "City", "State", "Zipcode",
    "Temperature(F)", "Wind_Chill(F)", "Humidity(%)",
    "Pressure(in)", "Visibility(mi)", "Wind_Direction",
    "Wind_Speed(mph)", "Precipitation(in)", "Weather_Condition",
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction",
    "No_Exit", "Railway", "Roundabout", "Station", "Stop",
    "Traffic_Calming", "Traffic_Signal", "Turning_Loop",
    "Sunrise_Sunset"
]

FEATURE_COLS = [
    "Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)",
    "Wind_Speed(mph)", "Precipitation(in)", "Distance(mi)",
    "Hour", "Day_of_Week", "Month",
    "Is_Peak_Hour", "Is_Night", "Is_Weekend",
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction",
    "No_Exit", "Railway", "Roundabout", "Station", "Stop",
    "Traffic_Calming", "Traffic_Signal", "Turning_Loop",
    "Duration_min", "Cluster"
]


def prepare_data(sample_size=300_000):
    df_full = pd.read_csv(DATA_PATH, usecols=USE_COLS)
    df = df_full.sample(n=min(sample_size, len(df_full)), random_state=42).reset_index(drop=True)

    numeric_cols = [
        "Temperature(F)", "Wind_Chill(F)", "Humidity(%)",
        "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)",
        "Precipitation(in)"
    ]

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    categorical_cols = ["Wind_Direction", "Weather_Condition", "Sunrise_Sunset"]

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    df["Zipcode"] = df["Zipcode"].fillna("Unknown")
    df["City"] = df["City"].fillna("Unknown")

    df["Start_Time"] = pd.to_datetime(df["Start_Time"], format="ISO8601")
    df["End_Time"] = pd.to_datetime(df["End_Time"], format="ISO8601")

    df["Duration_min"] = (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60
    df["Duration_min"] = df["Duration_min"].clip(lower=0, upper=1440)

    df["Hour"] = df["Start_Time"].dt.hour
    df["Day_of_Week"] = df["Start_Time"].dt.dayofweek
    df["Month"] = df["Start_Time"].dt.month

    df["Is_Peak_Hour"] = df["Hour"].apply(lambda h: 1 if (7 <= h <= 9) or (16 <= h <= 19) else 0)
    df["Is_Night"] = df["Hour"].apply(lambda h: 1 if h >= 21 or h <= 6 else 0)
    df["Is_Weekend"] = df["Day_of_Week"].apply(lambda d: 1 if d >= 5 else 0)

    geo_data = df[["Start_Lat", "Start_Lng"]].dropna().copy()

    scaler = StandardScaler()
    geo_scaled = scaler.fit_transform(geo_data)

    kmeans = MiniBatchKMeans(n_clusters=6, random_state=42, batch_size=5000)
    df["Cluster"] = kmeans.fit_predict(geo_scaled)

    df["Risk_Label"] = (df["Severity"] >= 3).astype(int)

    X = df[FEATURE_COLS].copy()
    y = df["Risk_Label"].copy()

    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    X = X.fillna(0)

    return df, X, y, scaler, kmeans


def train():
    os.makedirs("models", exist_ok=True)

    df, X, y, scaler, kmeans = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_sm, y_train_sm)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, pred))
    print("ROC-AUC:", roc_auc_score(y_test, proba))

    joblib.dump(
        {
            "model": model,
            "feature_cols": FEATURE_COLS,
            "geo_scaler": scaler,
            "kmeans": kmeans
        },
        MODEL_PATH
    )

    print(f"Model kaydedildi: {MODEL_PATH}")


if __name__ == "__main__":
    train()
