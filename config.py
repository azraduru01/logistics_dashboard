"""
config.py — Merkezi yapılandırma dosyası
"""

# ─── MySQL ───────────────────────────────────────────────
MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "azraduru5312",
    "database": "lojisticks",
}

MYSQL_TABLE          = "delivery_analysis"
MYSQL_TABLE_FALLBACK = "kitap2"

# ─── CSV ─────────────────────────────────────────────────
CSV_PATH      = "Delivery_Logistics.csv"
CSV_SEPARATOR = ";"

# ─── MODEL ───────────────────────────────────────────────
MODEL_FEATURES = [
    "delivery_partner",
    "package_type",
    "vehicle_type",
    "delivery_mode",
    "region",
    "weather_condition",
    "distance_km",
    "package_weight_kg",
]

MODEL_TARGET = "is_delayed"

RF_PARAMS = {
    "n_estimators":    200,
    "random_state":    42,
    "class_weight":    "balanced",
    "max_depth":       None,
    "min_samples_split": 5,
    "min_samples_leaf":  2,
    "n_jobs":          -1,
}

# ─── PALETTE — Modern SaaS / Indigo ──────────────────────
PALETTE = {
    # Zemin & yüzeyler
    "bg_dark":        "#F5F4FF",   # çok hafif lila-beyaz
    "bg_card":        "#FFFFFF",
    "bg_surface":     "#EDE9FE",   # lavanta yüzey

    # Vurgular
    "accent_blue":    "#6366F1",   # indigo-500
    "accent_cyan":    "#8B5CF6",   # violet-500
    "accent_amber":   "#F59E0B",
    "accent_green":   "#10B981",
    "accent_red":     "#EF4444",
    "accent_purple":  "#A855F7",   # purple-500

    # Metin
    "text_primary":   "#1E1B4B",   # indigo-950
    "text_secondary": "#6B7280",

    # Kenarlık
    "border":         "#DDD6FE",   # indigo-200
}

# Grafik renk dizisi — indigo/violet/purple ailesi + kontrastlar
CHART_COLORS = [
    "#6366F1", "#8B5CF6", "#A855F7", "#EC4899",
    "#10B981", "#F59E0B", "#EF4444", "#06B6D4",
]

# ─── UYGULAMA ────────────────────────────────────────────
APP_TITLE   = "Logistics Intelligence Dashboard"
APP_ICON    = "📦"
APP_VERSION = "2.0"
