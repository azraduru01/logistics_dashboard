"""
data_loader.py — Veri yükleme katmanı
Önce MySQL'e bağlanmaya çalışır; başarısız olursa CSV'ye geçer.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from typing import Tuple

from config import MYSQL_CONFIG, MYSQL_TABLE, MYSQL_TABLE_FALLBACK, CSV_PATH, CSV_SEPARATOR


# ─────────────────────────────────────────────────────────
# INTERNAL: MySQL
# ─────────────────────────────────────────────────────────
def _try_mysql() -> Tuple[pd.DataFrame | None, str | None]:
    """MySQL'den veri çekmeyi dener. (df, hata_mesajı) döndürür."""
    try:
        import mysql.connector
        from mysql.connector import Error

        conn = mysql.connector.connect(**MYSQL_CONFIG)

        # Önce delivery_analysis'i dene, yoksa kitap2'ye geç
        try:
            df = pd.read_sql(f"SELECT * FROM {MYSQL_TABLE}", conn)
            source_label = f"MySQL → {MYSQL_TABLE}"
        except Exception:
            df = pd.read_sql(f"SELECT * FROM {MYSQL_TABLE_FALLBACK}", conn)
            source_label = f"MySQL → {MYSQL_TABLE_FALLBACK}"

        conn.close()
        return df, source_label

    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────
# INTERNAL: CSV
# ─────────────────────────────────────────────────────────
def _load_csv() -> Tuple[pd.DataFrame | None, str | None]:
    """CSV'den veri yükler. (df, hata_mesajı) döndürür."""
    try:
        df = pd.read_csv(CSV_PATH, sep=CSV_SEPARATOR)
        # İlk satır gerçek header değilse (Delivery_Logistics gibi) atla
        if df.columns[0].lower().startswith("delivery_logistics"):
            df = pd.read_csv(CSV_PATH, sep=CSV_SEPARATOR, skiprows=1)
        return df, "CSV"
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────
# PUBLIC: Temizleme
# ─────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Kolon adı normalleştirme
    df.columns = [c.strip().lower() for c in df.columns]

    # 'delayed' → 'is_delayed' dönüşümü (CSV'de farklı gelebilir)
    if "delayed" in df.columns and "is_delayed" not in df.columns:
        df["is_delayed"] = df["delayed"].map(
            lambda x: 1 if str(x).strip().lower() in ("yes", "1", "true") else 0
        )

    numeric_cols = [
        "distance_km", "package_weight_kg", "delivery_cost",
        "delivery_rating", "is_delayed",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # delivery_time_hours / expected_time_hours — zaman damgası olarak gelebilir
    for time_col in ["delivery_time_hours", "expected_time_hours"]:
        if time_col in df.columns:
            raw = df[time_col]
            if raw.dtype == object or str(raw.dtype).startswith("datetime"):
                # "1970-01-01 00:00:00.000000008" → nanosecond cinsinden int al
                try:
                    df[time_col] = pd.to_datetime(raw, errors="coerce").astype("int64") / 1e9
                except Exception:
                    df[time_col] = pd.to_numeric(raw, errors="coerce")
            else:
                df[time_col] = pd.to_numeric(raw, errors="coerce")

    text_cols = [
        "delivery_partner", "package_type", "vehicle_type",
        "delivery_mode", "region", "weather_condition", "delivery_status",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower()

    if "is_delayed" in df.columns:
        df["is_delayed"] = df["is_delayed"].fillna(0).astype(int)

    return df


# ─────────────────────────────────────────────────────────
# PUBLIC: SQL KPI (MySQL bağlantısı varsa)
# ─────────────────────────────────────────────────────────
def get_sql_kpis(table: str = MYSQL_TABLE) -> dict | None:
    try:
        import mysql.connector
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        query = f"""
        SELECT
            COUNT(*) AS toplam_veri,
            SUM(is_delayed) AS geciken_siparis,
            (SUM(is_delayed) * 100.0 / COUNT(*)) AS gecikme_orani,
            AVG(delivery_cost) AS ortalama_maliyet,
            AVG(delivery_rating) AS ortalama_puan,
            AVG(distance_km) AS ortalama_mesafe,
            MAX(delivery_cost) AS en_yuksek_maliyet,
            MIN(delivery_cost) AS en_dusuk_maliyet
        FROM {table}
        """
        result = pd.read_sql(query, conn).iloc[0].to_dict()
        conn.close()
        return result
    except Exception:
        return None


# ─────────────────────────────────────────────────────────
# PUBLIC: Ana yükleme fonksiyonu (cache ile)
# ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=300)
def load_data() -> Tuple[pd.DataFrame, str, str | None]:
    """
    Returns
    -------
    df          : Temizlenmiş DataFrame
    source      : Verinin kaynağı ('MySQL → delivery_analysis' | 'CSV')
    error_msg   : Eğer CSV'ye düştüyse MySQL hata mesajı, yoksa None
    """
    df_raw, mysql_error = _try_mysql()
    fell_back = False

    if df_raw is None:
        df_raw, csv_source = _load_csv()
        fell_back = True
        if df_raw is None:
            raise RuntimeError(
                f"Veri yüklenemedi.\n"
                f"MySQL hatası: {mysql_error}\n"
                f"CSV hatası: {csv_source}"
            )
        source = "CSV (Delivery_Logistics.csv)"
    else:
        source = mysql_error  # mysql_error burada source_label tutuyor

    df = clean_data(df_raw)
    return df, source, (mysql_error if fell_back else None)
