"""
app.py — Logistics Intelligence Dashboard v2
Çalıştırmak için: streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from config import APP_TITLE, APP_ICON, APP_VERSION, PALETTE, CHART_COLORS
from clustering import run_kmeans, delay_chain_analysis, cluster_summary, pivot_chain
from data_loader import load_data, get_sql_kpis
from model import train_model, predict, build_explanation
from ui_components import (
    inject_css, kpi_card, fmt,
    panel_start, panel_end,
    apply_dark_theme,
    bar_chart, donut_chart, scatter_chart,
    heatmap_chart, histogram_chart, corr_heatmap,
    confusion_matrix_fig, feature_importance_fig,
)

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────
with st.spinner("Veri yükleniyor..."):
    try:
        data, data_source, mysql_warn = load_data()
    except RuntimeError as exc:
        st.error(f"**Veri yüklenemedi.**\n\n{exc}")
        st.stop()

if data.empty:
    st.warning("Veri seti boş görünüyor.")
    st.stop()

# ─────────────────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────────────────
with st.spinner("Model eğitiliyor..."):
    model_bundle = train_model(data)

# ─────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## {APP_ICON} Filtreler")
    st.caption("Seçimler tüm dashboard'u günceller.")

    filtered = data.copy()

    def apply_filter(label: str, col: str) -> None:
        global filtered
        if col not in data.columns:
            return
        options = ["Tümü"] + sorted(data[col].dropna().unique().tolist())
        key = f"f_{col}"
        if key not in st.session_state:
            st.session_state[key] = "Tümü"
        selected = st.selectbox(label, options, key=key)
        if selected != "Tümü":
            filtered = filtered[filtered[col] == selected]

    apply_filter("🗺️ Bölge", "region")
    apply_filter("🤝 Teslimat Partneri", "delivery_partner")
    apply_filter("🚗 Araç Tipi", "vehicle_type")
    apply_filter("📦 Paket Türü", "package_type")
    apply_filter("🌤️ Hava Durumu", "weather_condition")
    apply_filter("📋 Teslimat Modu", "delivery_mode")
    apply_filter("✅ Teslimat Durumu", "delivery_status")

    if "is_delayed" in data.columns:
        delay_opt = st.selectbox("⏱️ Gecikme Durumu", ["Tümü", "Geciken", "Zamanında"], key="f_delay")
        if delay_opt == "Geciken":
            filtered = filtered[filtered["is_delayed"] == 1]
        elif delay_opt == "Zamanında":
            filtered = filtered[filtered["is_delayed"] == 0]

    st.markdown("---")

    # Maliyet slider
    if "delivery_cost" in filtered.columns:
        valid = filtered["delivery_cost"].dropna()
        if not valid.empty and valid.min() < valid.max():
            mn, mx = float(valid.min()), float(valid.max())
            cost_range = st.slider("💰 Maliyet Aralığı", mn, mx, (mn, mx), key="f_cost")
            filtered = filtered[filtered["delivery_cost"].between(cost_range[0], cost_range[1])]

    # Puan slider
    if "delivery_rating" in filtered.columns:
        valid = filtered["delivery_rating"].dropna()
        if not valid.empty and valid.min() < valid.max():
            mn, mx = float(valid.min()), float(valid.max())
            rating_range = st.slider("⭐ Puan Aralığı", mn, mx, (mn, mx), key="f_rating")
            filtered = filtered[filtered["delivery_rating"].between(rating_range[0], rating_range[1])]

    st.markdown("---")

    col_r, col_c = st.columns(2)
    with col_r:
        if st.button("Sıfırla", use_container_width=True):
            keys = [k for k in st.session_state if k.startswith("f_")]
            for k in keys:
                del st.session_state[k]
            st.rerun()
    with col_c:
        if st.button("Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    st.markdown("---")
    st.download_button(
        "⬇️ CSV İndir",
        data=filtered.to_csv(index=False).encode("utf-8-sig"),
        file_name="logistics_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Kaynak badge
    is_csv = "CSV" in data_source
    badge_cls = "source-badge source-badge-warn" if is_csv else "source-badge"
    icon = "📄" if is_csv else "🗄️"
    st.markdown(
        f"<div class='{badge_cls}' style='margin-top:10px'>{icon} {data_source}</div>",
        unsafe_allow_html=True,
    )
    if mysql_warn:
        st.caption(f"⚠️ MySQL bağlanamadı, CSV kullanılıyor.")

# ─────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────
rec        = len(filtered)
delay_cnt  = int(filtered["is_delayed"].sum()) if "is_delayed" in filtered.columns else 0
delay_rate = delay_cnt / rec * 100 if rec > 0 else 0
avg_cost   = filtered["delivery_cost"].mean() if "delivery_cost" in filtered.columns else None
avg_rating = filtered["delivery_rating"].mean() if "delivery_rating" in filtered.columns else None
avg_dist   = filtered["distance_km"].mean() if "distance_km" in filtered.columns else None
avg_weight = filtered["package_weight_kg"].mean() if "package_weight_kg" in filtered.columns else None
max_cost   = filtered["delivery_cost"].max() if "delivery_cost" in filtered.columns else None
min_cost   = filtered["delivery_cost"].min() if "delivery_cost" in filtered.columns else None

best_partner = "—"
if {"delivery_partner", "delivery_rating"}.issubset(filtered.columns):
    gp = filtered.groupby("delivery_partner")["delivery_rating"].mean()
    if not gp.empty:
        best_partner = str(gp.idxmax())

worst_region = "—"
if {"region", "is_delayed"}.issubset(filtered.columns):
    gr = filtered.groupby("region").agg(t=("region", "count"), d=("is_delayed", "sum"))
    if not gr.empty:
        gr["rate"] = gr["d"] / gr["t"]
        worst_region = str(gr["rate"].idxmax())

most_vehicle = "—"
if "vehicle_type" in filtered.columns:
    vc = filtered["vehicle_type"].value_counts()
    if not vc.empty:
        most_vehicle = str(vc.index[0])

# ─────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-eyebrow">STAJ PROJESİ · {APP_VERSION}</div>
        <div class="hero-title">Logistics Intelligence Dashboard</div>
        <div class="hero-subtitle">
            SQL temizleme katmanı + Python ML pipeline + interaktif analitik
        </div>
        <div>
            <span class="hero-badge">📊 {len(data):,} Kayıt</span>
            <span class="hero-badge">🗺️ En Riskli: {worst_region.title()}</span>
            <span class="hero-badge">🏆 En İyi Partner: {best_partner.title()}</span>
            <span class="hero-badge">⏱️ Gecikme: %{delay_rate:.1f}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────
# KPI ROW 1
# ─────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    kpi_card("Toplam Kayıt", fmt(len(data)), f"Filtrelendi: {rec:,}", PALETTE["accent_blue"])
with c2:
    kpi_card("Geciken", fmt(delay_cnt), f"Oran: %{delay_rate:.1f}", PALETTE["accent_red"])
with c3:
    kpi_card("Ort. Maliyet", fmt(avg_cost, 2), "Teslimat başına", PALETTE["accent_cyan"])
with c4:
    kpi_card("Ort. Puan", fmt(avg_rating, 2), "Hizmet kalitesi", PALETTE["accent_amber"])
with c5:
    kpi_card("Ort. Mesafe", fmt(avg_dist, 1) + " km", "Filtreli ortalama", PALETTE["accent_green"])
with c6:
    kpi_card("Ort. Ağırlık", fmt(avg_weight, 1) + " kg", "Paket ağırlığı", "#8B5CF6")

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Maks. Maliyet", fmt(max_cost, 2), "En pahalı teslimat", PALETTE["accent_red"])
with c2:
    kpi_card("Min. Maliyet", fmt(min_cost, 2), "En ucuz teslimat", PALETTE["accent_green"])
with c3:
    kpi_card("En İyi Partner", best_partner.title(), "Ortalama puana göre", PALETTE["accent_amber"])
with c4:
    kpi_card("En Riskli Bölge", worst_region.title(), "Gecikme oranına göre", PALETTE["accent_red"])

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────
(
    tab_exec, tab_pred, tab_delay,
    tab_cost, tab_perf, tab_quality,
    tab_cluster, tab_chain,
    tab_insight, tab_data, tab_summary
) = st.tabs([
    "📊 Executive", "🤖 Tahmin", "⏱️ Gecikme",
    "💰 Maliyet", "📈 Performans", "🔍 Veri Kalitesi",
    "🔵 Risk Segmentasyonu", "🔗 Gecikme Zinciri",
    "💡 Insights", "📋 Veri", "📌 Proje",
])

# ═══════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════
with tab_exec:
    r1, r2 = st.columns(2)

    with r1:
        panel_start("Bölgeye Göre Sipariş Yoğunluğu", "Operasyonun en yoğun olduğu bölgeler")
        if "region" in filtered.columns:
            df_r = filtered.groupby("region").size().reset_index(name="count").sort_values("count", ascending=False)
            fig = bar_chart(df_r, "region", "count", PALETTE["accent_blue"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("region kolonu bulunamadı.")
        with st.expander("ℹ️ Ne anlatıyor?"):
            st.write("Hangi bölgede daha yoğun sipariş olduğunu gösterir. Yüksek yoğunluk gecikme ve maliyet baskısı yaratabilir.")
        panel_end()

    with r2:
        panel_start("Teslimat Durumu Dağılımı", "Operasyonel sonuçların genel görünümü")
        if "delivery_status" in filtered.columns:
            df_s = filtered["delivery_status"].value_counts().reset_index()
            df_s.columns = ["status", "count"]
            fig = donut_chart(df_s, "status", "count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("delivery_status kolonu bulunamadı.")
        with st.expander("ℹ️ Ne anlatıyor?"):
            st.write("Başarılı, gecikmiş veya iptale düşen teslimatların oranını gösterir.")
        panel_end()

    panel_start("Bölgesel Yönetici Özeti", "Sipariş, maliyet, puan ve gecikme tek tabloda")
    if {"region", "delivery_cost", "delivery_rating", "is_delayed"}.issubset(filtered.columns):
        summary = (
            filtered.groupby("region")
            .agg(
                Sipariş=("region", "count"),
                Maliyet=("delivery_cost", "mean"),
                Puan=("delivery_rating", "mean"),
                Geciken=("is_delayed", "sum"),
            )
            .reset_index()
        )
        summary["Gecikme %"] = (summary["Geciken"] / summary["Sipariş"] * 100).round(2)
        summary["Maliyet"] = summary["Maliyet"].round(2)
        summary["Puan"] = summary["Puan"].round(2)
        st.dataframe(
            summary.sort_values("Gecikme %", ascending=False),
            use_container_width=True, height=300,
        )
    else:
        st.info("Bu tablo için gerekli kolonlar eksik.")
    with st.expander("ℹ️ Yorumlama"):
        st.write("Yüksek maliyet + yüksek gecikme + düşük puan aynı bölgede birleşiyorsa o bölge operasyonel öncelik haline gelir.")
    panel_end()

    panel_start("Bölge × Araç Tipi Gecikme Heatmap", "Hangi araç hangi bölgede daha çok gecikiyor?")
    if {"region", "vehicle_type", "is_delayed"}.issubset(filtered.columns):
        heat_df = filtered.pivot_table("is_delayed", "region", "vehicle_type", aggfunc="mean")
        if not heat_df.empty:
            fig = heatmap_chart(heat_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Heatmap için yeterli veri yok.")
    else:
        st.info("Heatmap kolonları bulunamadı.")
    with st.expander("ℹ️ Ne anlatıyor?"):
        st.write("Gecikmenin araç tipi ve bölge kombinasyonuna göre nasıl değiştiğini gösterir. Koyu hücreler yüksek gecikme riskini işaret eder.")
    panel_end()

# ═══════════════════════════════════════════════════════════
# TAB 2 — PREDICTION
# ═══════════════════════════════════════════════════════════
with tab_pred:
    if model_bundle is None:
        st.warning("Model eğitilemedi. `is_delayed` veya özellik kolonları eksik.")
    else:
        # Model metrikleri
        panel_start("Model Performansı (Random Forest)", "Test seti üzerinde ölçülen metrikler")
        m = model_bundle["metrics"]
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            kpi_card("Accuracy", fmt(m["accuracy"] * 100, 1, "%"), "Genel doğruluk", PALETTE["accent_blue"])
        with mc2:
            kpi_card("Precision", fmt(m["precision"] * 100, 1, "%"), "Pozitif tahmin doğruluğu", PALETTE["accent_cyan"])
        with mc3:
            kpi_card("Recall", fmt(m["recall"] * 100, 1, "%"), "Gecikeni yakalama oranı", PALETTE["accent_amber"])
        with mc4:
            kpi_card("F1 Score", fmt(m["f1"] * 100, 1, "%"), "Dengeli performans", PALETTE["accent_green"])

        # Confusion matrix + Feature importance yan yana
        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown("<div class='panel-title' style='color:#F1F5F9;margin-bottom:8px'>Confusion Matrix</div>", unsafe_allow_html=True)
            fig = confusion_matrix_fig(model_bundle["cm"])
            st.plotly_chart(fig, use_container_width=True)
        with fc2:
            st.markdown("<div class='panel-title' style='color:#F1F5F9;margin-bottom:8px'>Top 15 Feature Importance</div>", unsafe_allow_html=True)
            fig = feature_importance_fig(model_bundle["importance_df"], top_n=15)
            st.plotly_chart(fig, use_container_width=True)
        panel_end()

        # Tahmin formu
        panel_start("Yeni Teslimat Tahmini", "Parametreleri girerek gecikme riskini hesapla")

        p1, p2, p3, p4 = st.columns(4)
        p5, p6, p7, p8 = st.columns(4)
        input_data: dict = {}

        def pick(col_obj, label, df_col, key):
            if df_col in data.columns:
                opts = sorted(data[df_col].dropna().unique().tolist())
                with col_obj:
                    input_data[df_col] = st.selectbox(label, opts, key=key)

        pick(p1, "Bölge",              "region",           "pi_region")
        pick(p2, "Teslimat Partneri",  "delivery_partner", "pi_partner")
        pick(p3, "Araç Tipi",          "vehicle_type",     "pi_vehicle")
        pick(p4, "Paket Türü",         "package_type",     "pi_package")
        pick(p5, "Teslimat Modu",      "delivery_mode",    "pi_mode")
        pick(p6, "Hava Durumu",        "weather_condition","pi_weather")

        if "distance_km" in data.columns:
            with p7:
                input_data["distance_km"] = st.number_input(
                    "Mesafe (km)", min_value=0.0,
                    value=float(data["distance_km"].median()), step=10.0, key="pi_dist"
                )
        if "package_weight_kg" in data.columns:
            with p8:
                input_data["package_weight_kg"] = st.number_input(
                    "Ağırlık (kg)", min_value=0.0,
                    value=float(data["package_weight_kg"].median()), step=10.0, key="pi_weight"
                )

        if st.button("🔮 Gecikme Riskini Hesapla", use_container_width=True):
            result = predict(model_bundle, input_data)
            prob = result["probability"]
            risk = result["risk_level"]

            res1, res2 = st.columns(2)
            with res1:
                kpi_card(
                    "Tahmin Sonucu",
                    result["label"],
                    f"Model güveni: %{prob:.1f}",
                    PALETTE["accent_red"] if result["prediction"] == 1 else PALETTE["accent_green"],
                )
            with res2:
                kpi_card(
                    "Gecikme Olasılığı",
                    f"%{prob:.1f}",
                    "Risk skoru",
                    PALETTE["accent_red"] if risk == "high" else (
                        PALETTE["accent_amber"] if risk == "medium" else PALETTE["accent_green"]
                    ),
                )

            risk_map = {
                "high":   ("risk-high",   "🔴 Yüksek Risk — Alternatif planlama önerilir."),
                "medium": ("risk-medium", "🟡 Orta Risk — Operasyon takibi artırılmalı."),
                "low":    ("risk-low",    "🟢 Düşük Risk — Standart operasyon yeterli."),
            }
            cls, msg = risk_map[risk]
            st.markdown(f"<div class='{cls}'>{msg}</div>", unsafe_allow_html=True)

            st.markdown("**Tahmin Açıklaması:**")
            notes = build_explanation(input_data, prob)
            for n in notes:
                st.markdown(f"- {n}")

        panel_end()

# ═══════════════════════════════════════════════════════════
# TAB 3 — GECİKME ANALİTİĞİ
# ═══════════════════════════════════════════════════════════
with tab_delay:
    d1, d2 = st.columns(2)

    with d1:
        panel_start("Bölgeye Göre Gecikme Oranı", "En riskli bölgeler")
        if {"region", "is_delayed"}.issubset(filtered.columns):
            df_r = (
                filtered.groupby("region")
                .agg(t=("region", "count"), d=("is_delayed", "sum"))
                .reset_index()
            )
            df_r["rate"] = df_r["d"] / df_r["t"] * 100
            fig = bar_chart(df_r.sort_values("rate", ascending=False), "region", "rate", PALETTE["accent_red"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Gerekli kolonlar bulunamadı.")
        panel_end()

    with d2:
        panel_start("Hava Durumuna Göre Gecikme", "Çevresel etkilerin operasyon üzerindeki etkisi")
        if {"weather_condition", "is_delayed"}.issubset(filtered.columns):
            df_w = (
                filtered.groupby("weather_condition")
                .agg(t=("weather_condition", "count"), d=("is_delayed", "sum"))
                .reset_index()
            )
            df_w["rate"] = df_w["d"] / df_w["t"] * 100
            fig = bar_chart(df_w.sort_values("rate", ascending=False), "weather_condition", "rate", PALETTE["accent_amber"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Gerekli kolonlar bulunamadı.")
        panel_end()

    panel_start("Partner Bazlı Gecikme Dağılımı", "Hangi partner daha sorunlu?")
    if {"delivery_partner", "is_delayed"}.issubset(filtered.columns):
        df_p = (
            filtered.groupby("delivery_partner")
            .agg(t=("delivery_partner", "count"), d=("is_delayed", "sum"))
            .reset_index()
        )
        df_p["rate"] = df_p["d"] / df_p["t"] * 100
        fig = bar_chart(df_p.sort_values("rate", ascending=False).head(15), "delivery_partner", "rate", PALETTE["accent_red"])
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("ℹ️ Yorumlama"):
        st.write("Yüksek gecikme oranlı partnerler için SLA gözden geçirme ya da kapasite artırımı önerilebilir.")
    panel_end()

    panel_start("Araç Tipi × Teslimat Modu Gecikme Heatmap", "Kombinasyon bazlı gecikme dağılımı")
    if {"vehicle_type", "delivery_mode", "is_delayed"}.issubset(filtered.columns):
        pivot = filtered.pivot_table("is_delayed", "vehicle_type", "delivery_mode", aggfunc="mean")
        if not pivot.empty:
            fig = heatmap_chart(pivot, height=350)
            st.plotly_chart(fig, use_container_width=True)
    panel_end()

# ═══════════════════════════════════════════════════════════
# TAB 4 — MALİYET ANALİTİĞİ
# ═══════════════════════════════════════════════════════════
with tab_cost:
    m1, m2 = st.columns(2)

    with m1:
        panel_start("Araç Tipine Göre Ortalama Maliyet", "Hangi araç daha pahalı?")
        if {"vehicle_type", "delivery_cost"}.issubset(filtered.columns):
            df_v = (
                filtered.groupby("vehicle_type")["delivery_cost"]
                .mean().reset_index().sort_values("delivery_cost", ascending=False)
            )
            fig = bar_chart(df_v, "vehicle_type", "delivery_cost", PALETTE["accent_cyan"])
            st.plotly_chart(fig, use_container_width=True)
        panel_end()

    with m2:
        panel_start("Bölgeye Göre Ortalama Maliyet", "Bölgesel maliyet baskısı")
        if {"region", "delivery_cost"}.issubset(filtered.columns):
            df_rc = (
                filtered.groupby("region")["delivery_cost"]
                .mean().reset_index().sort_values("delivery_cost", ascending=False)
            )
            fig = bar_chart(df_rc, "region", "delivery_cost", PALETTE["accent_blue"])
            st.plotly_chart(fig, use_container_width=True)
        panel_end()

    m3, m4 = st.columns(2)

    with m3:
        panel_start("Maliyet Dağılımı (Histogram)", "Maliyetler hangi aralıkta yoğunlaşıyor?")
        if "delivery_cost" in filtered.columns:
            fig = histogram_chart(filtered["delivery_cost"])
            st.plotly_chart(fig, use_container_width=True)
        panel_end()

    with m4:
        panel_start("Mesafe × Maliyet × Ağırlık", "Üç değişkeni aynı anda gör")
        if {"distance_km", "delivery_cost", "package_weight_kg", "is_delayed"}.issubset(filtered.columns):
            bdf = filtered.copy()
            bdf["Durum"] = bdf["is_delayed"].map({0: "Zamanında", 1: "Gecikmiş"})
            fig = scatter_chart(bdf, "distance_km", "delivery_cost", "package_weight_kg", "Durum",
                                ["region", "vehicle_type", "delivery_partner"])
            st.plotly_chart(fig, use_container_width=True)
        panel_end()

    # Paket tipi bazlı maliyet
    panel_start("Paket Türüne Göre Ortalama Maliyet", "Ürün kategorilerinin maliyet profili")
    if {"package_type", "delivery_cost"}.issubset(filtered.columns):
        df_pt = (
            filtered.groupby("package_type")["delivery_cost"]
            .mean().reset_index().sort_values("delivery_cost", ascending=False)
        )
        fig = bar_chart(df_pt, "package_type", "delivery_cost", "#8B5CF6")
        st.plotly_chart(fig, use_container_width=True)
    panel_end()

# ═══════════════════════════════════════════════════════════
# TAB 5 — PERFORMANS
# ═══════════════════════════════════════════════════════════
with tab_perf:
    p1, p2 = st.columns(2)

    with p1:
        panel_start("Partner Bazlı Ortalama Puan", "Kalite karşılaştırması")
        if {"delivery_partner", "delivery_rating"}.issubset(filtered.columns):
            df_pp = (
                filtered.groupby("delivery_partner")["delivery_rating"]
                .mean().reset_index().sort_values("delivery_rating", ascending=False).head(15)
            )
            fig = bar_chart(df_pp, "delivery_partner", "delivery_rating", PALETTE["accent_green"])
            st.plotly_chart(fig, use_container_width=True)
        panel_end()

    with p2:
        panel_start("Araç Kullanım Dağılımı", "Filo yapısı")
        if "vehicle_type" in filtered.columns:
            df_vt = filtered["vehicle_type"].value_counts().reset_index()
            df_vt.columns = ["vehicle_type", "count"]
            fig = bar_chart(df_vt, "vehicle_type", "count", PALETTE["accent_cyan"])
            st.plotly_chart(fig, use_container_width=True)
        panel_end()

    # Partner maliyet × puan scatter
    panel_start("Partner Bazlı Maliyet × Puan Matrisi", "Kalite ve maliyet birlikte")
    if {"delivery_partner", "delivery_cost", "delivery_rating"}.issubset(filtered.columns):
        comp = (
            filtered.groupby("delivery_partner")
            .agg(maliyet=("delivery_cost", "mean"), puan=("delivery_rating", "mean"),
                 toplam=("delivery_partner", "count"))
            .reset_index()
        )
        fig = px.scatter(
            comp, x="maliyet", y="puan", size="toplam", hover_name="delivery_partner",
            color_discrete_sequence=[PALETTE["accent_blue"]],
            labels={"maliyet": "Ortalama Maliyet", "puan": "Ortalama Puan"},
        )
        apply_dark_theme(fig, 400)
        st.plotly_chart(fig, use_container_width=True)
    panel_end()

    # Operasyonel kıyas tablosu
    panel_start("Operasyonel Kıyas Tablosu", "Partner × sipariş × puan × gecikme")
    if {"delivery_partner", "delivery_cost", "delivery_rating", "is_delayed"}.issubset(filtered.columns):
        ps = (
            filtered.groupby("delivery_partner")
            .agg(
                Sipariş=("delivery_partner", "count"),
                Maliyet=("delivery_cost", "mean"),
                Puan=("delivery_rating", "mean"),
                Geciken=("is_delayed", "sum"),
            )
            .reset_index()
        )
        ps["Gecikme %"] = (ps["Geciken"] / ps["Sipariş"] * 100).round(2)
        ps["Maliyet"] = ps["Maliyet"].round(2)
        ps["Puan"] = ps["Puan"].round(2)
        st.dataframe(
            ps.sort_values(["Puan", "Gecikme %"], ascending=[False, True]),
            use_container_width=True, height=340,
        )
    panel_end()

# ═══════════════════════════════════════════════════════════
# TAB 6 — VERİ KALİTESİ
# ═══════════════════════════════════════════════════════════
with tab_quality:
    panel_start("Data Quality Analysis", "Veri setinin temizliği ve güvenilirliği")

    q1, q2 = st.columns(2)
    with q1:
        st.subheader("Eksik Veri")
        miss = data.isnull().sum().reset_index()
        miss.columns = ["Kolon", "Eksik Veri"]
        miss["Oran (%)"] = (miss["Eksik Veri"] / len(data) * 100).round(2)
        st.dataframe(miss, use_container_width=True)

    with q2:
        st.subheader("Veri Tipleri")
        dtypes = pd.DataFrame(data.dtypes, columns=["Veri Tipi"])
        dtypes["Benzersiz"] = [data[c].nunique() for c in data.columns]
        st.dataframe(dtypes, use_container_width=True)

    st.subheader("Aykırı Değer — Delivery Cost")
    if "delivery_cost" in data.columns:
        fig = px.box(
            data, y="delivery_cost",
            color_discrete_sequence=[PALETTE["accent_blue"]],
        )
        apply_dark_theme(fig, 360)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sayısal Değişkenler Korelasyon Haritası")
    if data.select_dtypes(include="number").shape[1] >= 2:
        fig = corr_heatmap(data)
        st.plotly_chart(fig, use_container_width=True)

    panel_end()

# ═══════════════════════════════════════════════════════════
# TAB 7 — INSIGHTS & RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════
with tab_insight:
    panel_start("Kritik Bulgular", "Veriden çıkarılan operasyonel içgörüler")

    ia, ib, ic = st.columns(3)
    with ia:
        st.markdown(
            f"""<div style='background:{PALETTE["bg_surface"]};border:1px solid {PALETTE["border"]};
            border-radius:14px;padding:16px;'>
            <div style='color:{PALETTE["accent_red"]};font-size:11px;font-weight:700;
            text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px'>En Riskli Bölge</div>
            <div style='color:{PALETTE["text_primary"]};font-size:22px;font-weight:800'>{worst_region.title()}</div>
            <div style='color:{PALETTE["text_secondary"]};font-size:12px;margin-top:4px'>Gecikme oranı en yüksek</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with ib:
        st.markdown(
            f"""<div style='background:{PALETTE["bg_surface"]};border:1px solid {PALETTE["border"]};
            border-radius:14px;padding:16px;'>
            <div style='color:{PALETTE["accent_green"]};font-size:11px;font-weight:700;
            text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px'>En İyi Partner</div>
            <div style='color:{PALETTE["text_primary"]};font-size:22px;font-weight:800'>{best_partner.title()}</div>
            <div style='color:{PALETTE["text_secondary"]};font-size:12px;margin-top:4px'>En yüksek ortalama puan</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with ic:
        st.markdown(
            f"""<div style='background:{PALETTE["bg_surface"]};border:1px solid {PALETTE["border"]};
            border-radius:14px;padding:16px;'>
            <div style='color:{PALETTE["accent_amber"]};font-size:11px;font-weight:700;
            text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px'>Genel Gecikme Oranı</div>
            <div style='color:{PALETTE["text_primary"]};font-size:22px;font-weight:800'>%{delay_rate:.2f}</div>
            <div style='color:{PALETTE["text_secondary"]};font-size:12px;margin-top:4px'>Filtrelenmiş veri</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.subheader("Öneriler")
    st.markdown(
        f"""
        - 🗺️ **{worst_region.title()} bölgesinde** rota optimizasyonu ve kaynak artırımı uygulanmalı
        - 📉 Düşük performanslı partnerler için SLA gözden geçirilmeli
        - 🌧️ Olumsuz hava koşullarında dinamik planlama modeli devreye alınmalı
        - 🚗 Yüksek maliyetli araç tipleri için kullanım stratejisi optimize edilmeli
        - 🤖 ML modeli periyodik olarak yeni veriyle yeniden eğitilmeli
        """
    )
    st.subheader("İş Etkisi")
    st.markdown(
        """
        - Gecikme oranının %5 azaltılması müşteri memnuniyetini doğrudan artırır
        - Araç optimizasyonu ortalama teslimat maliyetini düşürebilir
        - Proaktif tahminleme operasyonel hazırlık süresini iyileştirir
        """
    )
    panel_end()

# ═══════════════════════════════════════════════════════════
# TAB 8 — HAM VERİ
# ═══════════════════════════════════════════════════════════
with tab_data:
    panel_start(
        "Ham Veri",
        f"Filtrelenmiş: {rec:,} kayıt · Toplam: {len(data):,} kayıt"
    )
    st.dataframe(filtered, use_container_width=True, height=520)
    with st.expander("📖 Kolon Açıklamaları"):
        st.markdown(
            """
            | Kolon | Açıklama |
            |-------|----------|
            | delivery_partner | Teslimatı gerçekleştiren iş ortağı |
            | package_type | Paket türü veya kategorisi |
            | vehicle_type | Kullanılan araç sınıfı |
            | delivery_mode | Teslimat modu (express, same-day, vb.) |
            | region | Hizmet verilen bölge |
            | weather_condition | Teslimat sırasındaki hava durumu |
            | distance_km | Teslimat mesafesi (km) |
            | package_weight_kg | Paket ağırlığı (kg) |
            | is_delayed | 1 = gecikmiş, 0 = zamanında |
            | delivery_status | Son teslimat sonucu |
            | delivery_rating | Kalite değerlendirme puanı |
            | delivery_cost | Teslimat maliyeti |
            """
        )
    panel_end()

# ═══════════════════════════════════════════════════════════
# TAB 9 — PROJE ÖZETİ
# ═══════════════════════════════════════════════════════════
with tab_summary:
    panel_start("Proje Özeti", "Teknik mimari ve akademik bağlam")

    st.subheader("Proje Amacı")
    st.write(
        "Bu proje, lojistik teslimat süreçlerinde gecikme, maliyet ve hizmet kalitesi "
        "arasındaki ilişkileri analiz etmek, operasyonel sorun alanlarını görünür hale "
        "getirmek ve karar vericilere veri temelli içgörü sunmak amacıyla geliştirilmiştir."
    )

    sa, sb = st.columns(2)

    with sa:
        st.subheader("Veri Seti")
        st.markdown(
            f"""
            - **Toplam Kayıt:** {len(data):,}
            - **Kolon Sayısı:** {len(data.columns)}
            - **Veri Kaynağı:** Kaggle — Delivery Logistics Dataset India
            - **Aktif Kaynak:** {data_source}
            """
        )

    with sb:
        st.subheader("Teknoloji Yığını")
        st.markdown(
            """
            | Katman | Teknoloji |
            |--------|-----------|
            | Veritabanı | MySQL / CSV |
            | Backend | Python 3.11+ |
            | Arayüz | Streamlit |
            | ML | Scikit-learn (RandomForest) |
            | Görselleştirme | Plotly Express |
            | Veri | Pandas |
            """
        )

    st.subheader("Mimari Akış")
    st.markdown(
        """
        ```
        Ham CSV / MySQL
              ↓
        data_loader.py  ← Fallback mekanizması (MySQL → CSV)
              ↓
        clean_data()    ← Tip dönüşümü, normalleştirme
              ↓
        model.py        ← RandomForest pipeline eğitimi
              ↓
        ui_components.py ← Dark theme, KPI kartları, Plotly şablonları
              ↓
        app.py          ← Streamlit dashboard (9 sekme)
        ```
        """
    )

    if model_bundle:
        st.subheader("Model Özeti")
        m = model_bundle["metrics"]
        st.markdown(
            f"""
            | Metrik | Değer |
            |--------|-------|
            | Accuracy | %{m['accuracy']*100:.2f} |
            | Precision | %{m['precision']*100:.2f} |
            | Recall | %{m['recall']*100:.2f} |
            | F1 Score | %{m['f1']*100:.2f} |
            | Eğitim Verisi | %80 |
            | Test Verisi | %20 |
            | Algoritma | RandomForestClassifier |
            """
        )

    panel_end()


# ═══════════════════════════════════════════════════════════
# TAB 10 — RİSK SEGMENTASYONu (K-MEANS)
# ═══════════════════════════════════════════════════════════
with tab_cluster:
    panel_start(
        "Risk Segmentasyonu — K-Means Clustering",
        "Mesafe · Paket ağırlığı · Hava durumu riskine göre teslimatlar üç segmente ayrılır"
    )

    n_clusters = st.slider("Küme sayısı", min_value=2, max_value=5, value=3, key="km_n")

    km_result = run_kmeans(filtered, n_clusters=n_clusters)

    if km_result is None:
        st.warning("Segmentasyon için yeterli veri veya kolon bulunamadı.")
    else:
        clustered, centers, risk_labels = km_result

        # ── KPI satırı ──
        summary_df = cluster_summary(clustered)
        cols = st.columns(len(summary_df))
        color_map = {
            "🟢 Düşük Risk": PALETTE["accent_green"],
            "🟡 Orta Risk":  PALETTE["accent_amber"],
            "🔴 Yüksek Risk": PALETTE["accent_red"],
        }
        for col, (_, row) in zip(cols, summary_df.iterrows()):
            accent = color_map.get(row["Risk Segmenti"], PALETTE["accent_blue"])
            with col:
                kpi_card(
                    row["Risk Segmenti"],
                    f"{row['Kayıt']:,} teslimat",
                    f"Gecikme: %{row['Gecikme_Oranı']} · Ort. mesafe: {row['Ort_Mesafe']} km",
                    accent,
                )

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        # ── PCA Scatter ──
        sc1, sc2 = st.columns([2, 1])

        with sc1:
            panel_start("2D Risk Haritası (PCA)", "Her nokta bir teslimat — renk risk segmentini gösterir")
            color_discrete = {
                "🟢 Düşük Risk": PALETTE["accent_green"],
                "🟡 Orta Risk":  PALETTE["accent_amber"],
                "🔴 Yüksek Risk": PALETTE["accent_red"],
            }
            fig = px.scatter(
                clustered, x="pca_x", y="pca_y",
                color="risk_label",
                color_discrete_map=color_discrete,
                opacity=0.65,
                labels={"pca_x": "PCA Boyut 1", "pca_y": "PCA Boyut 2", "risk_label": "Risk"},
            )
            fig.update_traces(marker_size=4)
            from ui_components import apply_dark_theme
            apply_dark_theme(fig, 420)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("ℹ️ PCA nedir?"):
                st.write(
                    "Principal Component Analysis (PCA), yüksek boyutlu veriyi 2 boyuta indirger. "
                    "Burada mesafe, ağırlık ve hava riski tek bir düzlemde gösterilir. "
                    "Kümeler ne kadar ayrışık görünüyorsa segmentasyon o kadar belirgin demektir."
                )
            panel_end()

        with sc2:
            panel_start("Segment Özeti", "Her kümenin ortalama karakteristikleri")
            st.dataframe(
                summary_df.set_index("Risk Segmenti"),
                use_container_width=True,
                height=280,
            )
            panel_end()

        # ── Gecikme oranı karşılaştırması ──
        panel_start("Segment Bazlı Gecikme Oranı", "Hangi segment daha fazla gecikiyor?")
        bar_df = summary_df[["Risk Segmenti", "Gecikme_Oranı"]].copy()
        bar_df.columns = ["Segment", "Gecikme %"]
        bar_colors = [color_map.get(s, PALETTE["accent_blue"]) for s in bar_df["Segment"]]

        fig2 = go.Figure(go.Bar(
            x=bar_df["Segment"],
            y=bar_df["Gecikme %"],
            marker_color=bar_colors,
            marker_line_width=0,
            text=bar_df["Gecikme %"].apply(lambda v: f"%{v}"),
            textposition="outside",
        ))
        apply_dark_theme(fig2, 320)
        st.plotly_chart(fig2, use_container_width=True)
        with st.expander("ℹ️ Yorumlama"):
            st.write(
                "Yüksek riskli segment; fırtınalı/yağışlı hava koşullarında gerçekleşen teslimatları "
                "içeriyor. Bu segmentteki siparişler için proaktif operasyonel önlem almak gecikme "
                "oranını anlamlı şekilde düşürebilir."
            )
        panel_end()

    panel_end()


# ═══════════════════════════════════════════════════════════
# TAB 11 — GECİKME ZİNCİRİ ANALİZİ
# ═══════════════════════════════════════════════════════════
with tab_chain:
    panel_start(
        "Gecikme Zinciri Analizi",
        "Partner · Bölge · Hava kombinasyonlarının gecikme oranı — en riskli üçlüler"
    )

    chain_df = delay_chain_analysis(filtered)

    if chain_df is None or chain_df.empty:
        st.warning("Gecikme zinciri için yeterli veri veya kolon bulunamadı.")
    else:
        # ── Üst KPI'lar ──
        kritik = (chain_df["risk"] == "🔴 Kritik").sum()
        yuksek = (chain_df["risk"] == "🟡 Yüksek").sum()
        normal = (chain_df["risk"] == "🟢 Normal").sum()
        en_riskli = chain_df.iloc[0]

        z1, z2, z3, z4 = st.columns(4)
        with z1:
            kpi_card("Kritik Kombinasyon", str(kritik), "≥%70 gecikme oranı", PALETTE["accent_red"])
        with z2:
            kpi_card("Yüksek Risk", str(yuksek), "%40–70 gecikme oranı", PALETTE["accent_amber"])
        with z3:
            kpi_card("Normal", str(normal), "<%40 gecikme oranı", PALETTE["accent_green"])
        with z4:
            kpi_card(
                "En Riskli Kombinasyon",
                f"%{en_riskli['gecikme_orani']}",
                en_riskli["kombinasyon"],
                PALETTE["accent_red"],
            )

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        # ── Top 20 bar chart ──
        panel_start("En Riskli 20 Kombinasyon", "Partner · Bölge · Hava üçlüsüne göre gecikme oranı")
        top20 = chain_df.head(20).copy()
        risk_colors_chain = {
            "🔴 Kritik": PALETTE["accent_red"],
            "🟡 Yüksek": PALETTE["accent_amber"],
            "🟢 Normal": PALETTE["accent_green"],
        }
        bar_colors_chain = [risk_colors_chain.get(r, PALETTE["accent_blue"]) for r in top20["risk"]]

        fig3 = go.Figure(go.Bar(
            x=top20["gecikme_orani"],
            y=top20["kombinasyon"],
            orientation="h",
            marker_color=bar_colors_chain,
            marker_line_width=0,
            text=top20["gecikme_orani"].apply(lambda v: f"%{v}"),
            textposition="outside",
        ))
        apply_dark_theme(fig3, 560)
        fig3.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig3, use_container_width=True)
        with st.expander("ℹ️ Yorumlama"):
            st.write(
                "Aynı partner, aynı bölge ve aynı hava koşullarının bir arada yarattığı gecikme baskısını gösterir. "
                "Tek bir faktör değil, üç faktörün kesişimi yüksek risk üretir — gecikme zinciri bu yüzden güçlü bir analiz aracıdır."
            )
        panel_end()

        # ── İnteraktif pivot ──
        panel_start("Çok Boyutlu Gecikme Pivot", "İki boyut seç — ortalama gecikme oranı ısı haritası")
        dim_options = ["delivery_partner", "region", "weather_condition"]
        dim_labels  = {"delivery_partner": "Partner", "region": "Bölge", "weather_condition": "Hava"}

        pc1, pc2 = st.columns(2)
        with pc1:
            dim1 = st.selectbox(
                "Satır (Y ekseni)",
                dim_options,
                format_func=lambda x: dim_labels[x],
                index=0, key="chain_dim1"
            )
        with pc2:
            remaining = [d for d in dim_options if d != dim1]
            dim2 = st.selectbox(
                "Sütun (X ekseni)",
                remaining,
                format_func=lambda x: dim_labels[x],
                index=0, key="chain_dim2"
            )

        pivot = pivot_chain(chain_df, dim1=dim1, dim2=dim2)
        if not pivot.empty:
            fig4 = px.imshow(
                pivot, text_auto=".1f", aspect="auto",
                color_continuous_scale=[
                    [0.0, "#F0FDF4"],
                    [0.4, "#FEF9C3"],
                    [0.7, "#FEE2E2"],
                    [1.0, "#991B1B"],
                ],
                labels=dict(
                    x=dim_labels[dim2],
                    y=dim_labels[dim1],
                    color="Gecikme %"
                ),
            )
            apply_dark_theme(fig4, 460)
            st.plotly_chart(fig4, use_container_width=True)
        with st.expander("ℹ️ Nasıl okunur?"):
            st.write(
                "Her hücre, o partner+bölge veya partner+hava kombinasyonunun ortalama gecikme oranını gösterir. "
                "Kırmızı hücreler öncelikli aksiyon alanlarıdır. Boyutları değiştirerek farklı açılardan analiz edebilirsin."
            )
        panel_end()

        # ── Tam tablo ──
        panel_start("Tüm Kombinasyonlar", f"{len(chain_df):,} kombinasyon · min. 5 gözlem")
        display_cols = ["kombinasyon", "toplam", "geciken", "gecikme_orani", "risk"]
        st.dataframe(
            chain_df[display_cols].rename(columns={
                "kombinasyon":    "Kombinasyon",
                "toplam":         "Toplam",
                "geciken":        "Geciken",
                "gecikme_orani":  "Gecikme %",
                "risk":           "Risk",
            }),
            use_container_width=True,
            height=400,
        )
        panel_end()

    panel_end()
