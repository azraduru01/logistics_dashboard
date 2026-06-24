"""
ui_components.py — Modern SaaS / Indigo tema
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from config import PALETTE, CHART_COLORS


# ─────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────
def inject_css() -> None:
    P = PALETTE
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        /* ── Zemin ── */
        .stApp {{
            background: {P['bg_dark']};
        }}
        .block-container {{
            padding-top: 0.6rem;
            padding-bottom: 2rem;
            max-width: 98%;
        }}

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {{
            background: #1E1B4B;
            border-right: none;
        }}
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] * {{
            color: #E0E7FF !important;
        }}
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"] > div {{
            background: rgba(255,255,255,0.08) !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
            border-radius: 10px !important;
        }}
        [data-testid="stSidebar"] [data-baseweb="select"] *,
        [data-testid="stSidebar"] [data-baseweb="input"] * {{
            color: #E0E7FF !important;
        }}
        [data-testid="stSidebar"] [data-baseweb="select"] svg {{
            fill: #A5B4FC !important;
        }}
        [data-testid="stSidebar"] div[data-testid="stButton"] button {{
            background: rgba(99,102,241,0.25) !important;
            color: #E0E7FF !important;
            border: 1px solid rgba(99,102,241,0.4) !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            transition: all 0.2s;
        }}
        [data-testid="stSidebar"] div[data-testid="stButton"] button:hover {{
            background: rgba(99,102,241,0.45) !important;
        }}
        [data-testid="stSidebar"] .stDownloadButton > button {{
            background: rgba(99,102,241,0.18) !important;
            color: #C7D2FE !important;
            border: 1px solid rgba(99,102,241,0.35) !important;
            border-radius: 10px !important;
        }}

        /* ── Sidebar slider ── */
        [data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stSlider"] div {{
            color: #A5B4FC !important;
        }}

        /* ── Hero — düz koyu, büyük başlık ── */
        .hero {{
            background: #1E1B4B;
            border-radius: 20px;
            padding: 32px 36px 28px 36px;
            margin-bottom: 22px;
            position: relative;
            overflow: hidden;
        }}
        .hero::after {{
            content: '';
            position: absolute;
            bottom: -40px; right: -40px;
            width: 220px; height: 220px;
            background: radial-gradient(circle, {P['accent_blue']}33 0%, transparent 65%);
            border-radius: 50%;
            pointer-events: none;
        }}
        .hero-eyebrow {{
            display: inline-block;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: {P['accent_blue']};
            background: rgba(99,102,241,0.15);
            border: 1px solid rgba(99,102,241,0.3);
            border-radius: 999px;
            padding: 3px 12px;
            margin-bottom: 14px;
        }}
        .hero-title {{
            font-size: 36px;
            font-weight: 800;
            color: #FFFFFF;
            line-height: 1.1;
            letter-spacing: -0.02em;
            margin-bottom: 8px;
        }}
        .hero-subtitle {{
            font-size: 14px;
            color: #A5B4FC;
            margin-bottom: 18px;
            font-weight: 400;
        }}
        .hero-badge {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            background: rgba(255,255,255,0.07);
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 999px;
            padding: 5px 13px;
            font-size: 11px;
            font-weight: 600;
            color: #C7D2FE;
            margin-right: 8px;
            margin-top: 4px;
        }}

        /* ── KPI Card ── */
        .kpi-card {{
            background: {P['bg_card']};
            border: 1px solid {P['border']};
            border-radius: 16px;
            padding: 20px 22px;
            min-height: 116px;
            position: relative;
            overflow: hidden;
            transition: transform 0.15s, box-shadow 0.15s;
            box-shadow: 0 1px 4px rgba(99,102,241,0.06);
        }}
        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(99,102,241,0.13);
        }}
        .kpi-card-accent {{
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            border-radius: 16px 16px 0 0;
        }}
        .kpi-label {{
            font-size: 10.5px;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: {P['text_secondary']};
            margin-bottom: 10px;
            margin-top: 6px;
        }}
        .kpi-value {{
            font-size: 30px;
            font-weight: 800;
            color: {P['text_primary']};
            font-family: 'JetBrains Mono', monospace;
            line-height: 1;
            margin-bottom: 7px;
            letter-spacing: -0.02em;
        }}
        .kpi-note {{
            font-size: 12px;
            color: {P['text_secondary']};
        }}

        /* ── Panel ── */
        .panel {{
            background: {P['bg_card']};
            border: 1px solid {P['border']};
            border-radius: 18px;
            padding: 20px 20px 14px 20px;
            margin-bottom: 16px;
            box-shadow: 0 1px 4px rgba(99,102,241,0.05);
        }}
        .panel-title {{
            font-size: 15px;
            font-weight: 700;
            color: {P['text_primary']};
            margin-bottom: 2px;
        }}
        .panel-subtitle {{
            font-size: 12px;
            color: {P['text_secondary']};
            margin-bottom: 14px;
        }}

        /* ── Risk Badges ── */
        .risk-high {{
            background: #FFF5F5;
            border: 1px solid #FED7D7;
            color: #C53030;
            border-radius: 12px;
            padding: 14px 18px;
            font-size: 14px;
            font-weight: 600;
        }}
        .risk-medium {{
            background: #FFFBEB;
            border: 1px solid #FDE68A;
            color: #92400E;
            border-radius: 12px;
            padding: 14px 18px;
            font-size: 14px;
            font-weight: 600;
        }}
        .risk-low {{
            background: #F0FFF4;
            border: 1px solid #9AE6B4;
            color: #276749;
            border-radius: 12px;
            padding: 14px 18px;
            font-size: 14px;
            font-weight: 600;
        }}

        /* ── Source Badge ── */
        .source-badge {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            background: rgba(16,185,129,0.12);
            border: 1px solid rgba(16,185,129,0.3);
            color: #059669;
            border-radius: 999px;
            padding: 4px 12px;
            font-size: 11px;
            font-weight: 600;
        }}
        .source-badge-warn {{
            background: rgba(245,158,11,0.12);
            border-color: rgba(245,158,11,0.3);
            color: #D97706;
        }}

        /* ── Expander ── */
        details summary {{
            color: {P['text_secondary']} !important;
            font-size: 12px;
        }}
        details {{
            border: 1px solid {P['border']} !important;
            border-radius: 10px !important;
            background: {P['bg_surface']} !important;
        }}

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {{
            background: {P['bg_card']};
            border-radius: 14px;
            border: 1px solid {P['border']};
            padding: 5px;
            gap: 3px;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 10px !important;
            color: {P['text_secondary']} !important;
            font-size: 12.5px !important;
            font-weight: 600 !important;
            padding: 6px 14px !important;
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, {P['accent_blue']}, {P['accent_cyan']}) !important;
            color: white !important;
        }}
        .stTabs [data-baseweb="tab-panel"] {{
            padding-top: 18px;
        }}

        /* ── Genel metin ── */
        p, li {{
            color: {P['text_secondary']};
        }}
        h1, h2, h3, h4 {{
            color: {P['text_primary']};
        }}

        /* ── Input / Select — ana alan ── */
        [data-baseweb="input"] > div,
        [data-baseweb="select"] > div {{
            background: {P['bg_surface']} !important;
            border-color: {P['border']} !important;
            color: {P['text_primary']} !important;
            border-radius: 10px !important;
        }}
        input, textarea {{
            color: {P['text_primary']} !important;
        }}

        /* ── Buttons ── */
        .stButton > button {{
            background: linear-gradient(135deg, {P['accent_blue']}, {P['accent_cyan']}) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            letter-spacing: 0.01em;
            transition: opacity 0.2s, transform 0.15s;
        }}
        .stButton > button:hover {{
            opacity: 0.88;
            transform: translateY(-1px);
        }}

        /* ── Download button ── */
        .stDownloadButton > button {{
            background: {P['bg_surface']} !important;
            color: {P['accent_blue']} !important;
            border: 1px solid {P['border']} !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
        }}

        /* ── Dataframe ── */
        [data-testid="stDataFrame"] {{
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid {P['border']};
        }}

        /* ── Divider ── */
        hr {{
            border-color: {P['border']} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────
# KPI CARD
# ─────────────────────────────────────────────────────────
def kpi_card(label: str, value: str, note: str = "", accent: str = PALETTE["accent_blue"]) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-card-accent" style="background:linear-gradient(90deg,{accent},{accent}99);"></div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt(val, digits: int = 0, suffix: str = "") -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    if digits > 0:
        return f"{val:,.{digits}f}{suffix}"
    return f"{val:,.0f}{suffix}"


# ─────────────────────────────────────────────────────────
# PANEL WRAPPER
# ─────────────────────────────────────────────────────────
def panel_start(title: str, subtitle: str = "") -> None:
    sub_html = f"<div class='panel-subtitle'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"<div class='panel'><div class='panel-title'>{title}</div>{sub_html}",
        unsafe_allow_html=True,
    )


def panel_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# PLOTLY TEMA
# ─────────────────────────────────────────────────────────
def apply_dark_theme(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=PALETTE["text_secondary"],
        font_family="Inter",
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font_color=PALETTE["text_secondary"],
        ),
        xaxis=dict(
            gridcolor="#EDE9FE",
            linecolor=PALETTE["border"],
            tickfont_color=PALETTE["text_secondary"],
        ),
        yaxis=dict(
            gridcolor="#EDE9FE",
            linecolor=PALETTE["border"],
            tickfont_color=PALETTE["text_secondary"],
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────
def bar_chart(df, x, y, color=PALETTE["accent_blue"], height=370, text_auto=True):
    fig = px.bar(
        df, x=x, y=y,
        text_auto=".2f" if text_auto else False,
        color_discrete_sequence=[color],
    )
    fig.update_traces(marker_line_width=0, marker_color=color)
    return apply_dark_theme(fig, height)


def donut_chart(df, names, values, height=370):
    fig = px.pie(df, names=names, values=values, hole=0.62,
                 color_discrete_sequence=CHART_COLORS)
    fig.update_traces(textfont_color="white", textfont_size=12)
    fig.update_layout(showlegend=True)
    return apply_dark_theme(fig, height)


def scatter_chart(df, x, y, size, color, hover_data, height=390):
    fig = px.scatter(df, x=x, y=y, size=size, color=color,
                     hover_data=hover_data,
                     color_discrete_sequence=CHART_COLORS)
    return apply_dark_theme(fig, height)


def heatmap_chart(pivot_df, height=420):
    fig = px.imshow(
        pivot_df, text_auto=".2f", aspect="auto",
        color_continuous_scale=[
            [0.0, "#F5F3FF"],
            [0.5, "#6366F1"],
            [1.0, "#3730A3"],
        ],
    )
    return apply_dark_theme(fig, height)


def histogram_chart(series, nbins=35, height=370):
    fig = go.Figure(go.Histogram(
        x=series.dropna(), nbinsx=nbins,
        marker_color=PALETTE["accent_blue"],
        marker_line_width=0,
        opacity=0.85,
    ))
    return apply_dark_theme(fig, height)


def corr_heatmap(df, height=420):
    corr = df.select_dtypes(include="number").corr()
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto", zmin=-1, zmax=1,
        color_continuous_scale=[
            [0.0, "#F87171"],
            [0.5, "#F5F3FF"],
            [1.0, "#6366F1"],
        ],
    )
    return apply_dark_theme(fig, height)


def confusion_matrix_fig(cm, height=300):
    labels = ["Zamanında", "Gecikmiş"]
    fig = px.imshow(
        cm, x=labels, y=labels, text_auto=True, aspect="auto",
        color_continuous_scale=[[0, "#F5F3FF"], [1, "#6366F1"]],
        labels=dict(x="Tahmin", y="Gerçek"),
    )
    return apply_dark_theme(fig, height)


def feature_importance_fig(importance_df, top_n=15, height=400):
    top = importance_df.head(top_n).sort_values("importance")
    colors = [PALETTE["accent_blue"] if i % 2 == 0 else PALETTE["accent_cyan"]
              for i in range(len(top))]
    fig = go.Figure(go.Bar(
        x=top["importance"], y=top["feature"],
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
    ))
    return apply_dark_theme(fig, height)
