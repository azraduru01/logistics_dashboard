"""
model.py — ML model katmanı
Eğitim, tahmin ve feature importance işlemleri.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from typing import Any

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

from config import MODEL_FEATURES, MODEL_TARGET, RF_PARAMS


# ─────────────────────────────────────────────────────────
# EĞİTİM
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame) -> dict[str, Any] | None:
    """
    RandomForest pipeline'ını eğitir.

    Returns
    -------
    {
      'pipeline'   : sklearn Pipeline,
      'metrics'    : dict (accuracy, precision, recall, f1),
      'feature_cols': list[str],
      'cat_features': list[str],
      'num_features': list[str],
      'cm'         : confusion matrix ndarray,
      'importance_df': pd.DataFrame,
    }
    ya da None (eğitim mümkün değilse)
    """
    if MODEL_TARGET not in df.columns:
        return None

    feature_cols = [c for c in MODEL_FEATURES if c in df.columns]
    if not feature_cols:
        return None

    model_df = df[feature_cols + [MODEL_TARGET]].dropna(subset=[MODEL_TARGET]).copy()
    if model_df.empty or model_df[MODEL_TARGET].nunique() < 2:
        return None

    X = model_df[feature_cols]
    y = model_df[MODEL_TARGET].astype(int)

    # Explicitly identify numeric vs categorical by column name
    _numeric_names = {"distance_km", "package_weight_kg"}
    cat_features = [c for c in feature_cols if c not in _numeric_names]
    num_features = [c for c in feature_cols if c in _numeric_names]

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    preprocessor = ColumnTransformer([
        ("cat", cat_pipe, cat_features),
        ("num", num_pipe, num_features),
    ])

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(**RF_PARAMS)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Feature importance
    feat_names = pipeline.named_steps["pre"].get_feature_names_out()
    importances = pipeline.named_steps["clf"].feature_importances_
    importance_df = (
        pd.DataFrame({"feature": feat_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "pipeline":     pipeline,
        "feature_cols": feature_cols,
        "cat_features": cat_features,
        "num_features": num_features,
        "metrics": {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall":    recall_score(y_test, y_pred, zero_division=0),
            "f1":        f1_score(y_test, y_pred, zero_division=0),
        },
        "cm":            confusion_matrix(y_test, y_pred),
        "importance_df": importance_df,
    }


# ─────────────────────────────────────────────────────────
# TAHMİN
# ─────────────────────────────────────────────────────────
def predict(model_bundle: dict, input_dict: dict) -> dict[str, Any]:
    """
    Tekil bir teslimat için gecikme tahmini yapar.

    Returns
    -------
    {
      'label'       : 'Gecikecek' | 'Zamanında',
      'prediction'  : int (0 | 1),
      'probability' : float (0-100),
      'risk_level'  : 'high' | 'medium' | 'low',
    }
    """
    pipeline    = model_bundle["pipeline"]
    feature_cols = model_bundle["feature_cols"]

    row = {col: input_dict.get(col) for col in feature_cols}
    input_df = pd.DataFrame([row])

    pred = int(pipeline.predict(input_df)[0])
    prob = float(pipeline.predict_proba(input_df)[0][1]) * 100

    if prob >= 70:
        risk = "high"
    elif prob >= 40:
        risk = "medium"
    else:
        risk = "low"

    return {
        "label":       "Gecikecek" if pred == 1 else "Zamanında",
        "prediction":  pred,
        "probability": round(prob, 1),
        "risk_level":  risk,
    }


# ─────────────────────────────────────────────────────────
# AÇIKLAMA
# ─────────────────────────────────────────────────────────
def build_explanation(input_dict: dict, prob: float) -> list[str]:
    """Kural tabanlı tahmin açıklaması üretir."""
    notes = []

    dist = input_dict.get("distance_km", 0) or 0
    weight = input_dict.get("package_weight_kg", 0) or 0
    weather = str(input_dict.get("weather_condition", "")).lower()
    mode = str(input_dict.get("delivery_mode", "")).lower()
    region = str(input_dict.get("region", "")).lower()
    vehicle = str(input_dict.get("vehicle_type", "")).lower()

    if dist > 1500:
        notes.append(f"📍 Mesafe çok uzun ({dist:.0f} km) — gecikme riski artar.")
    elif dist > 800:
        notes.append(f"📍 Orta-uzun mesafe ({dist:.0f} km).")

    if weight > 1500:
        notes.append(f"⚖️ Yüksek ağırlık ({weight:.0f} kg) operasyon süresini uzatabilir.")

    if weather in ("rainy", "stormy", "snowy", "foggy"):
        notes.append(f"🌧️ '{weather.capitalize()}' hava koşulları teslimatı zorlaştırabilir.")

    if "same day" in mode or "express" in mode:
        notes.append(f"⚡ '{mode}' modu yüksek operasyonel baskı oluşturur.")

    if prob >= 70:
        notes.append("🔴 Risk yüksek — alternatif rota veya ek kapasite planlaması önerilir.")
    elif prob >= 40:
        notes.append("🟡 Orta risk — operasyon takibi artırılmalı.")
    else:
        notes.append("🟢 Risk düşük — standart operasyon yeterli.")

    return notes if notes else ["ℹ️ Belirgin bir risk faktörü tespit edilmedi."]
