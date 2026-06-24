"""
clustering.py — Gecikme riski segmentasyonu & zincir analizi
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ─────────────────────────────────────────────────────────
# 1. K-MEANS SEGMENTASYONu
# ─────────────────────────────────────────────────────────
WEATHER_RISK = {
    "clear": 0, "sunny": 0,
    "windy": 1, "cloudy": 1,
    "rainy": 2, "foggy": 2,
    "stormy": 3, "snowy": 3,
}

@st.cache_data(show_spinner=False)
def run_kmeans(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame | None:
    """
    mesafe_km + paket_ağırlık + hava_riski üzerinde KMeans çalıştırır.
    Küme etiketleri + risk seviyesi ile zenginleştirilmiş DataFrame döner.
    """
    needed = {"distance_km", "package_weight_kg", "weather_condition", "is_delayed"}
    if not needed.issubset(df.columns):
        return None

    work = df[list(needed)].copy().dropna()
    if len(work) < n_clusters * 10:
        return None

    # Hava durumunu sayısala çevir
    work["weather_risk"] = (
        work["weather_condition"]
        .str.strip().str.lower()
        .map(WEATHER_RISK)
        .fillna(1)
        .astype(float)
    )

    features = ["distance_km", "package_weight_kg", "weather_risk"]
    X = work[features].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    work["cluster"] = km.fit_predict(X_scaled)

    # Küme merkezlerini orijinal ölçeğe geri al
    centers = scaler.inverse_transform(km.cluster_centers_)
    center_df = pd.DataFrame(centers, columns=features)

    # Her kümenin gecikme oranına göre risk etiketi ver (düşük→yüksek)
    cluster_delay = (
        work.groupby("cluster")["is_delayed"].mean().reset_index()
    )
    cluster_delay = cluster_delay.sort_values("is_delayed").reset_index(drop=True)
    risk_labels = {
        row["cluster"]: label
        for row, label in zip(
            cluster_delay.to_dict("records"),
            ["🟢 Düşük Risk", "🟡 Orta Risk", "🔴 Yüksek Risk"][:n_clusters],
        )
    }
    work["risk_label"] = work["cluster"].map(risk_labels)

    # PCA (2D görselleştirme için)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    work["pca_x"] = coords[:, 0]
    work["pca_y"] = coords[:, 1]

    return work.reset_index(drop=True), center_df, risk_labels


def cluster_summary(clustered_df: pd.DataFrame) -> pd.DataFrame:
    """Küme bazında özet istatistikler."""
    summary = (
        clustered_df.groupby("risk_label")
        .agg(
            Kayıt=("distance_km", "count"),
            Ort_Mesafe=("distance_km", "mean"),
            Ort_Ağırlık=("package_weight_kg", "mean"),
            Ort_Hava_Riski=("weather_risk", "mean"),
            Gecikme_Oranı=("is_delayed", "mean"),
        )
        .reset_index()
        .rename(columns={"risk_label": "Risk Segmenti"})
    )
    summary["Gecikme_Oranı"] = (summary["Gecikme_Oranı"] * 100).round(1)
    summary["Ort_Mesafe"]    = summary["Ort_Mesafe"].round(1)
    summary["Ort_Ağırlık"]  = summary["Ort_Ağırlık"].round(1)
    summary["Ort_Hava_Riski"] = summary["Ort_Hava_Riski"].round(2)
    return summary


# ─────────────────────────────────────────────────────────
# 2. GECİKME ZİNCİRİ ANALİZİ
# ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def delay_chain_analysis(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    partner + region + weather_condition kombinasyonlarının gecikme oranını hesaplar.
    En riskli kombinasyonları döner.
    """
    needed = {"delivery_partner", "region", "weather_condition", "is_delayed"}
    if not needed.issubset(df.columns):
        return None

    work = df[list(needed)].dropna().copy()

    chain = (
        work.groupby(["delivery_partner", "region", "weather_condition"])
        .agg(
            toplam=("is_delayed", "count"),
            geciken=("is_delayed", "sum"),
        )
        .reset_index()
    )
    chain["gecikme_orani"] = (chain["geciken"] / chain["toplam"] * 100).round(1)

    # En az 5 gözlem içeren kombinasyonları al
    chain = chain[chain["toplam"] >= 5].copy()
    chain = chain.sort_values("gecikme_orani", ascending=False).reset_index(drop=True)

    # Risk etiketi
    def risk_tag(rate: float) -> str:
        if rate >= 70:
            return "🔴 Kritik"
        elif rate >= 40:
            return "🟡 Yüksek"
        else:
            return "🟢 Normal"

    chain["risk"] = chain["gecikme_orani"].apply(risk_tag)
    chain["kombinasyon"] = (
        chain["delivery_partner"].str.title() + " · "
        + chain["region"].str.title() + " · "
        + chain["weather_condition"].str.title()
    )

    return chain


def pivot_chain(chain_df: pd.DataFrame, dim1: str = "delivery_partner",
                dim2: str = "region") -> pd.DataFrame:
    """İki boyut üzerinde gecikme oranı pivot tablosu."""
    pivot = chain_df.pivot_table(
        values="gecikme_orani",
        index=dim1,
        columns=dim2,
        aggfunc="mean",
    ).round(1)
    return pivot
