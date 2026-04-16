import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Potret Ketenagakerjaan D.I. Yogyakarta",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ===== Import Font ===== */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=DM+Serif+Display&display=swap');

    /* ===== CSS Variables ===== */
    :root {
        --navy:       #001D39;
        --deep:       #0A4174;
        --muted:      #49769F;
        --teal:       #4E8EA2;
        --dusty:      #6EA2B3;
        --sky:        #7BBDE8;
        --pale:       #BDD8E9;
        --pale-light: #D9EDF6;
        --bg-main:    #A8D4E8;
    }

    /* ===== Global Reset ===== */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ===== Background Utama ===== */
    .stApp {
        background: linear-gradient(150deg, #7BBDE8 0%, #A8D4E8 45%, #BDD8E9 100%) !important;
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(150deg, #7BBDE8 0%, #A8D4E8 50%, #BDD8E9 100%) !important;
    }
    [data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0) !important;
        backdrop-filter: none !important;
    }
    section[data-testid="stMain"] {
        background: transparent !important;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width:100% !important;
    }

    /* ===== Global Text — Dark Navy on light blue BG ===== */
    .stApp p, .stApp span, .stApp label, .stApp li, .stApp small {
        color: #001D39;
    }

    /* =====================================================
       CHART CONTAINER — box outline di setiap grafik
       ===================================================== */
    .chart-box {
        background: rgba(255, 255, 255, 0.55);
        border: 1.5px solid rgba(0, 29, 57, 0.22);
        border-radius: 14px;
        box-shadow: 0 4px 18px rgba(0, 29, 57, 0.12),
                    0 1px 4px rgba(0, 29, 57, 0.08);
        padding: 6px 10px 4px 10px;
        margin-bottom: 4px;
    }

    /* ===== Dashboard Header ===== */
    .dashboard-header {
        background: linear-gradient(135deg, #001D39 0%, #0A4174 60%, #49769F 100%);
        border: 1px solid rgba(189, 216, 233, 0.35);
        border-radius: 18px;
        padding: 28px 36px;
        margin-bottom: 24px;
        box-shadow: 0 10px 40px rgba(0, 29, 57, 0.35),
                    0 0 0 1px rgba(123, 189, 232, 0.12);
    }
    .header-title {
        font-family: 'DM Serif Display', serif;
        font-size: 28px;
        font-weight: 400;
        color: #BDD8E9 !important;
        margin: 0;
        line-height: 1.35;
        letter-spacing: -0.3px;
    }
    .header-subtitle {
        font-size: 14px;
        color: #7BBDE8 !important;
        margin-top: 8px;
        line-height: 1.6;
        opacity: 0.9;
    }
    .header-badge {
        display: inline-block;
        background: linear-gradient(135deg, #49769F, #4E8EA2);
        color: #BDD8E9 !important;
        font-size: 11px;
        font-weight: 600;
        padding: 5px 14px;
        border-radius: 20px;
        margin-top: 12px;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }

    /* ===== KPI Cards ===== */
    .kpi-card {
        background: linear-gradient(145deg, #001D39 0%, #0A4174 70%, #49769F 100%);
        border: 1px solid rgba(123, 189, 232, 0.25);
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 6px 24px rgba(0, 29, 57, 0.3),
                    inset 0 1px 0 rgba(189, 216, 233, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 32px rgba(0, 29, 57, 0.4);
    }
    .kpi-label {
        font-size: 10.5px;
        font-weight: 700;
        color: #7BBDE8 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-family: 'DM Serif Display', serif;
        font-size: 30px;
        font-weight: 400;
        color: #BDD8E9 !important;
        line-height: 1;
    }
    .kpi-unit {
        font-size: 12px;
        color: #6EA2B3 !important;
        margin-top: 5px;
    }
    .kpi-delta-pos {
        font-size: 11.5px;
        color: #7BBDE8 !important;
        margin-top: 5px;
        font-weight: 600;
    }
    .kpi-delta-neg {
        font-size: 11.5px;
        color: #BDD8E9 !important;
        margin-top: 5px;
        font-weight: 600;
        opacity: 0.7;
    }

    /* ===== Section Headers — Dark Navy ===== */
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #001D39 !important;
        border-left: 3px solid #4E8EA2;
        padding-left: 12px;
        margin: 22px 0 14px 0;
        letter-spacing: 0.2px;
    }

    /* ===== Insight Box ===== */
    .insight-box {
        background: linear-gradient(135deg, #001D39 0%, #0A4174 100%);
        border: 1px solid rgba(123, 189, 232, 0.2);
        border-left: 4px solid #4E8EA2;
        border-radius: 14px;
        padding: 20px 24px;
        margin: 12px 0;
        box-shadow: 0 4px 16px rgba(0, 29, 57, 0.25);
    }
    .insight-title {
        font-size: 11px;
        font-weight: 700;
        color: #7BBDE8 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }
    .insight-text {
        font-size: 13.5px;
        color: #BDD8E9 !important;
        line-height: 1.75;
    }
    .insight-highlight {
        color: #7BBDE8 !important;
        font-weight: 600;
    }
            

    /* =====================================================
       TABS — Perbaikan utama: active tab putih + navy bg
       ===================================================== */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(0, 29, 57, 0.1) !important;
        border: 1px solid rgba(0, 29, 57, 0.22) !important;
        border-radius: 12px !important;
        padding: 5px !important;
        gap: 4px !important;
        display: flex !important;
        justify-content: center !important;
        backdrop-filter: blur(4px) !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px !important;
        padding: 9px 20px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        letter-spacing: 0.2px !important;
        color: #001D39 !important;
        transition: color 0.15s ease, background 0.15s ease !important;
    }
    /* Tab AKTIF — background navy gelap, teks PUTIH */
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #001D39, #0A4174) !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 10px rgba(0, 29, 57, 0.3) !important;
    }
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span,
    .stTabs [aria-selected="true"] div {
        color: #FFFFFF !important;
    }
    /* Tab non-aktif hover */
    .stTabs [aria-selected="false"]:hover {
        background: rgba(0, 29, 57, 0.09) !important;
        color: #001D39 !important;
    }

    /* =====================================================
       SIDEBAR — Sticky, fixed-width, gelap
       ===================================================== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #001D39 0%, #021527 100%) !important;
        border-right: 1px solid rgba(123, 189, 232, 0.2) !important;
        min-width: 300px !important;
        max-width: 300px !important;
        /* Sticky sidebar */
        position: fixed !important;
        top: 0 !important;
        height: 100vh !important;
    }
    [data-testid="stSidebar"] > div {
        background: transparent !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        overflow-y: auto !important;
        scrollbar-width: none !important;
        -ms-overflow-style: none !important;
        padding: 0 16px !important;
        box-sizing: border-box !important;
        height: 100vh !important;
    }
    [data-testid="stSidebar"] > div:first-child::-webkit-scrollbar,
    [data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar {
        display: none !important;
        width: 0px !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        overflow-y: auto !important;
        scrollbar-width: none !important;
    }

    /* Teks sidebar — terang di atas bg gelap */
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown small,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #BDD8E9 !important;
        font-size: 13px !important;
        font-family: 'DM Sans', sans-serif !important;
        line-height: 1.5 !important;
    }
    [data-testid="stSidebar"] h3 {
        font-size: 14px !important;
        font-weight: 700 !important;
        color: #7BBDE8 !important;
        letter-spacing: 0.5px !important;
    }

    /* ===== Tombol Collapse Sidebar (panah) — kontras & terlihat ===== */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    button[data-testid="baseButton-headerNoPadding"] {
        background: rgba(10, 65, 116, 0.85) !important;
        border: 1.5px solid rgba(123, 189, 232, 0.5) !important;
        border-radius: 8px !important;
        color: #BDD8E9 !important;
        box-shadow: 0 2px 8px rgba(0, 29, 57, 0.3) !important;
    }
    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebarCollapsedControl"] svg,
    button[data-testid="baseButton-headerNoPadding"] svg {
        fill: #BDD8E9 !important;
        stroke: #BDD8E9 !important;
        width: 20px !important;
        height: 20px !important;
    }
    /* Tombol close di dalam sidebar */
    [data-testid="stSidebar"] button[kind="header"],
    [data-testid="stSidebar"] [data-testid="baseButton-headerNoPadding"] {
        background: rgba(10, 65, 116, 0.7) !important;
        border-radius: 8px !important;
        border: 1.5px solid rgba(123, 189, 232, 0.4) !important;
        padding: 6px !important;
    }
    [data-testid="stSidebar"] button[kind="header"] svg,
    [data-testid="stSidebar"] [data-testid="baseButton-headerNoPadding"] svg {
        fill: #BDD8E9 !important;
        color: #BDD8E9 !important;
    }
    /* Sembunyikan teks ghost 'keyboard_doubl...' */
    /* Tetap sembunyikan teks ghost 'keyboard_double...' tanpa mematikan seluruh header */
    [data-testid="stSidebar"] button span {
        display: none !important;
    }

    /* =====================================================
       SIDEBAR — Dropdown selectbox & multiselect (#0A4174)
       ===================================================== */
/* 1. MENGUBAH KOTAK INPUT (Yang tadinya putih jadi Gelap) */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div,
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] > div > div,
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #0A4174 !important; /* Biru Gelap */
        border: 1px solid rgba(123, 189, 232, 0.4) !important;
        color: #BDD8E9 !important;
    }
            /* Menghilangkan garis double lancip di tepi kotak */
[data-testid="stSidebar"] [data-testid="stMultiSelect"] {
    border: none !important; /* Menghapus border luar yang bikin lancip */
}

    /* Membuat kotak utama melengkung mulus */
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] > div > div {
        background-color: #0A4174 !important;
        border: 1.5px solid rgba(123, 189, 232, 0.4) !important; /* Border tipis saja */
        border-radius: 12px !important; /* Lengkungan lebih halus */
        padding: 2px !important;
    }

    /* Memastikan elemen internal juga melengkung agar tidak "tembus" */
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        border-radius: 12px !important;
        border: none !important;
        background-color: transparent !important;
    }
    /* 2. MENGUBAH PIL/TAG (Bantul, Sleman, dll) agar tidak terlihat pucat */
    [data-testid="stSidebar"] [data-baseweb="tag"] {
        background-color: #49769F !important; /* Biru medium agar kontras */
        border: none !important;
    }

    /* 3. MEMASTIKAN TEKS DI DALAMNYA TERANG */
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] span,
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] p,
    [data-testid="stSidebar"] [data-baseweb="tag"] span {
        color: #BDD8E9 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] span,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] p,
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] div,
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] span,
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] p {
        color: #BDD8E9 !important;
        font-size: 13px !important;
    }
    /* Ikon panah & tombol x — kontras putih */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] svg,
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] svg {
        fill: #BDD8E9 !important;
        stroke: #BDD8E9 !important;
        opacity: 1 !important;
    }
    /* Multiselect tag/pill */
    [data-testid="stSidebar"] [data-baseweb="tag"] {
        background-color: rgba(78, 142, 162, 0.5) !important;
        border: 1px solid rgba(123, 189, 232, 0.4) !important;
        border-radius: 6px !important;
    }
    [data-testid="stSidebar"] [data-baseweb="tag"] span {
        color: #BDD8E9 !important;
        font-size: 12px !important;
    }
    [data-testid="stSidebar"] [data-baseweb="tag"] [data-baseweb="icon"] {
        color: #BDD8E9 !important;
        fill: #BDD8E9 !important;
    }
    /* Dropdown list popup */
    [data-testid="stSidebar"] ul[data-baseweb="menu"],
    [data-testid="stSidebar"] [role="listbox"] {
        background-color: #0A4174 !important;
        border: 1px solid rgba(123, 189, 232, 0.3) !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] ul[data-baseweb="menu"] li,
    [data-testid="stSidebar"] [role="option"] {
        background-color: transparent !important;
        color: #BDD8E9 !important;
        font-size: 13px !important;
    }
    [data-testid="stSidebar"] ul[data-baseweb="menu"] li:hover,
    [data-testid="stSidebar"] [role="option"]:hover {
        background-color: rgba(123, 189, 232, 0.18) !important;
    }

    /* =====================================================
       SLIDER — label tahun muncul, warna biru
       ===================================================== */
    /* =====================================================
    SLIDER GLOBAL (Titik Thumb & Font)
    ===================================================== */
    [data-testid="stSlider"] [data-baseweb="thumb"] {
        background-color: #7BBDE8 !important;
        border: 2px solid #001D39 !important;
    }

    /* =====================================================
    1. SLIDER KHUSUS SIDEBAR (Garis Terang)
    ===================================================== */
    [data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div > div {
        background: rgba(189, 216, 233, 0.3) !important; /* Garis belum terisi (terang) */
    }

    [data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div > div > div {
        background: #7BBDE8 !important; /* Garis sudah terisi (biru muda) */
    }

    /* =====================================================
    2. SLIDER KHUSUS MAIN AREA (Garis Gelap)
    ===================================================== */
    section[data-testid="stMain"] [data-testid="stSlider"] > div > div > div > div {
        background: rgba(0, 29, 57, 0.2) !important; /* Garis belum terisi (gelap transparan) */
    }

    section[data-testid="stMain"] [data-testid="stSlider"] > div > div > div > div > div {
        background: linear-gradient(90deg, #001D39, #0A4174) !important; /* Garis terisi (navy gelap) */
    }

    /* Label Angka Tahun di Main Area agar tetap gelap */
    section[data-testid="stMain"] [data-testid="stSlider"] [data-baseweb="typography"] {
        color: #001D39 !important;
        font-weight: bold !important;
    }
    [data-testid="stSlider"] [data-baseweb="thumb"],
    [data-testid="stSlider"] [role="slider"] {
        background-color: #7BBDE8 !important;
        border: 2px solid #001D39 !important;
        box-shadow: 0 0 0 3px rgba(78, 142, 162, 0.4) !important;
        width: 18px !important;
        height: 18px !important;
    }
    /* Label di sidebar (bg gelap) → terang */
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stTickBarMin"],
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stTickBarMax"],
    [data-testid="stSidebar"] [data-testid="stSlider"] p,
    [data-testid="stSidebar"] [data-testid="stSlider"] span {
        color: #BDD8E9 !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        opacity: 1 !important;
    }
    /* Label Judul Slider di main area (Pilih Tahun...) */
    section[data-testid="stMain"] [data-testid="stSlider"] label p {
        color: #001D39 !important;
        font-size: 16px !important;      /* Ukuran font besar */
        font-weight: 600 !important;     /* Tebal/Bold */
        opacity: 1 !important;
        margin-bottom: 10px !important;  /* Jarak ke garis slider */
    }

    /* Angka tahun di ujung garis (Min/Max) dan teks kecil lainnya */
    section[data-testid="stMain"] [data-testid="stSlider"] div[data-testid="stTickBarMin"],
    section[data-testid="stMain"] [data-testid="stSlider"] div[data-testid="stTickBarMax"],
    section[data-testid="stMain"] [data-testid="stSlider"] span,
    section[data-testid="stMain"] [data-testid="stSlider"] small {
        color: #001D39 !important;
        font-size: 15px !important;      /* Sedikit lebih besar dari 12px biar jelas */
        font-weight: 500 !important;
        opacity: 1 !important;
    }
            
    /* Tooltip thumb */
    [data-testid="stSlider"] [data-baseweb="tooltip"],
    [data-testid="stSlider"] div[data-baseweb="popover"] {
        background: #001D39 !important;
        border-radius: 6px !important;
        padding: 4px 8px !important;
    }
    [data-testid="stSlider"] [data-baseweb="tooltip"] div,
    [data-testid="stSlider"] div[data-baseweb="popover"] div {
        color: #BDD8E9 !important;
        font-size: 12px !important;
        font-weight: 600 !important;
    }
    [data-testid="stSlider"] * {
        accent-color: #4E8EA2 !important;
    }
    

    /* ===== Selectbox di main area ===== */
    section[data-testid="stMain"] [data-testid="stSelectbox"] > div > div,
    section[data-testid="stMain"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background-color: rgba(217, 237, 246, 0.9) !important;
        border: 1px solid rgba(0, 29, 57, 0.25) !important;
        border-radius: 8px !important;
        color: #001D39 !important;
    }
    section[data-testid="stMain"] [data-testid="stSelectbox"] div,
    section[data-testid="stMain"] [data-testid="stSelectbox"] span {
        color: #001D39 !important;
    }
    section[data-testid="stMain"] [data-testid="stSelectbox"] svg {
        fill: #001D39 !important;
        opacity: 0.8 !important;
    }
            /* 1. MENGUNCI SIDEBAR (PATEN) */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #001D39 0%, #021527 100%) !important;
        position: fixed !important;
        height: 100vh !important;
        min-width: 300px !important;
        max-width: 300px !important;
        z-index: 999;
    }

    /* 2. MENGGESER DASHBOARD BIAR GAK KETUTUPAN SIDEBAR */
    section[data-testid="stMain"] {
        margin-left: 300px !important; 
        width: calc(100% - 300px) !important; 
    }

    /* 3. TETAPKAN WARNA LABEL SLIDER (Kode yang kamu tanya tadi) */
    section[data-testid="stMain"] [data-testid="stSlider"] div[data-testid="stTickBarMin"],
    section[data-testid="stMain"] [data-testid="stSlider"] div[data-testid="stTickBarMax"],
    section[data-testid="stMain"] [data-testid="stSlider"] p,
    section[data-testid="stMain"] [data-testid="stSlider"] span,
    section[data-testid="stMain"] [data-testid="stSlider"] small {
        color: #001D39 !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        opacity: 1 !important;
    }

    /* 4. Menghilangkan tombol hide secara permanen */
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }

    /* 5. Fixed Sumber Data di paling bawah */
    .source-container {
        position: absolute; 
        bottom: 20px;
        left: 15px;
        right: 15px;
        background: rgba(10, 65, 116, 0.95);
        padding: 12px;
        border-radius: 10px;
        border: 1px solid rgba(123, 189, 232, 0.2);
        z-index: 1000;
    }

    /* ===== Dataframe ===== */
    [data-testid="stDataFrame"] {
        background: rgba(189, 216, 233, 0.5) !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }
    .stDataFrame thead tr th {
        background: #0A4174 !important;
        color: #BDD8E9 !important;
        font-weight: 600 !important;
    }
    .stDataFrame tbody tr:nth-child(even) {
        background: rgba(78, 142, 162, 0.08) !important;
    }

    /* ===== Cluster Badge ===== */
    .cluster-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }

    /* ===== Divider & Source Badge ===== */
    .custom-divider {
        border: none;
        border-top: 1px solid rgba(0, 29, 57, 0.2);
        margin: 20px 0;
    }
    .source-badge {
        background: rgba(10, 65, 116, 0.5);
        border: 1px solid rgba(123, 189, 232, 0.25);
        border-radius: 10px;
        padding: 10px 14px;
        font-size: 11px;
        color: #BDD8E9 !important;
        margin-top: 4px;
        line-height: 1.6;
    }
    .source-badge b {
        color: #7BBDE8 !important;
    }

    /* ===== Metric container ===== */
    [data-testid="metric-container"] {
        background: rgba(189, 216, 233, 0.5) !important;
        border: 1px solid rgba(0, 29, 57, 0.2) !important;
        border-radius: 10px !important;
    }

    /* ===== Alert ===== */
    .stAlert {
        background: rgba(78, 142, 162, 0.15) !important;
        border: 1px solid rgba(78, 142, 162, 0.4) !important;
        color: #001D39 !important;
        border-radius: 10px !important;
    }

    /* ===== Small helper text ===== */
    small {
        color: #001D39 !important;
        opacity: 0.85 !important;
    }
</style>
""", unsafe_allow_html=True)


# PALETTE
WILAYAH_COLORS = {
    "Kulon Progo":     "#001D39",
    "Bantul":          "#0A4174",
    "Gunung Kidul":    "#49769F",
    "Sleman":          "#4E8EA2",
    "Kota Yogyakarta": "#6EA2B3",
    "DI Yogyakarta":   "#7BBDE8",
}


def chart_container_start():
    """Mulai chart-box wrapper."""
    st.markdown("<div class='chart-box'>", unsafe_allow_html=True)

def chart_container_end():
    """Tutup chart-box wrapper."""
    st.markdown("</div>", unsafe_allow_html=True)


# apply_theme
def apply_theme(fig):
    """
    Menerapkan tema Navy-Blue Palette ke semua grafik Plotly.
    - plot_bgcolor: putih transparan → warna bar gelap (#001D39, #0A4174) kontras
    - Axis teks & judul → #001D39 (navy gelap), tegas di atas bg terang
    - Legend → di luar plot area (kanan), tidak menutupi data
    - Gridlines → halus navy transparan
    """
    fig.update_layout(
        paper_bgcolor="rgba(0, 0, 0, 0)",
       
        plot_bgcolor="rgba(255, 255, 255, 0.10)",
        font=dict(
            color="#001D39",
            family="DM Sans, sans-serif",
            size=12
        ),
        xaxis=dict(
            gridcolor="rgba(0, 29, 57, 0.08)",
            zerolinecolor="rgba(0, 29, 57, 0.2)",
            linecolor="rgba(0, 29, 57, 0.35)",
            linewidth=1.5,
            mirror=False,
            showgrid=True,
            tickfont=dict(color="#001D39", size=11, family="DM Sans"),
            title_font=dict(size=12, color="#001D39", family="DM Sans"),
            tickmode="linear",
            dtick=1,
            ticks="outside",
            ticklen=4,
        ),
        yaxis=dict(
            gridcolor="rgba(0, 29, 57, 0.08)",
            zerolinecolor="rgba(0, 29, 57, 0.2)",
            linecolor="rgba(0, 29, 57, 0.35)",
            linewidth=1.5,
            mirror=False,
            showgrid=True,
            tickfont=dict(color="#001D39", size=11, family="DM Sans"),
            title_font=dict(size=12, color="#001D39", family="DM Sans"),
            ticks="outside",
            ticklen=4,
        ),
        
        legend=dict(
            bgcolor="rgba(0, 29, 57, 0.82)",
            bordercolor="rgba(123, 189, 232, 0.35)",
            borderwidth=1,
            font=dict(color="#BDD8E9", size=11, family="DM Sans"),
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
        ),
        hoverlabel=dict(
            bgcolor="#001D39",
            bordercolor="#4E8EA2",
            font=dict(color="#BDD8E9", size=12, family="DM Sans"),
        ),
        # Margin kanan diperlebar untuk legend
        margin=dict(t=55, b=45, l=60, r=145),
        title=dict(
            font=dict(color="#001D39", size=15, family="DM Sans"),
            x=0.01,
            xanchor="left",
        ),
    )
    return fig


#  DATA LOADING & CLUSTERING — TIDAK DIUBAH

@st.cache_data
def load_all_data():
    FILE_PATH = "kumpulan dataset 2.xlsx"

    df_tpt = pd.read_excel(FILE_PATH, sheet_name="TPT dan TPAK menurut KabupatenK")
    df_tpt.columns = ["Tahun", "Wilayah", "TPT", "TPAK"]
    df_tpt["Tahun"] = pd.to_numeric(df_tpt["Tahun"], errors="coerce")
    df_tpt = df_tpt.dropna(subset=["Tahun", "Wilayah"])
    df_tpt["Tahun"] = df_tpt["Tahun"].astype(int)

    df_pop = pd.read_excel(FILE_PATH, sheet_name="Jumlah Penduduk")
    df_pop.columns = ["Tahun", "Wilayah", "Jumlah_Penduduk", "Laju_Pertumbuhan",
                       "Persen_Penduduk", "Kepadatan", "Rasio_JK"]
    df_pop["Tahun"] = pd.to_numeric(df_pop["Tahun"], errors="coerce")
    df_pop = df_pop.dropna(subset=["Tahun", "Wilayah"])
    df_pop["Tahun"] = df_pop["Tahun"].astype(int)

    df_pendidikan = pd.read_excel(FILE_PATH, sheet_name="Penduduk Berumur 15 Tahun ke At")
    col_map = {
        df_pendidikan.columns[0]: "Tahun",
        df_pendidikan.columns[1]: "Pendidikan",
        df_pendidikan.columns[2]: "Bekerja",
        df_pendidikan.columns[3]: "Pengangguran_PernahBekerja",
        df_pendidikan.columns[4]: "Pengangguran_BelumBekerja",
        df_pendidikan.columns[5]: "Total_Pengangguran",
        df_pendidikan.columns[6]: "Total_AK",
        df_pendidikan.columns[7]: "Persen_Bekerja",
        df_pendidikan.columns[8]: "Sekolah",
        df_pendidikan.columns[9]: "Rumah_Tangga",
        df_pendidikan.columns[10]: "Lainnya",
        df_pendidikan.columns[11]: "Total_BAK",
        df_pendidikan.columns[12]: "Total",
        df_pendidikan.columns[13]: "TPAK_Pendidikan",
    }
    df_pendidikan = df_pendidikan.rename(columns=col_map)
    df_pendidikan["Tahun"] = pd.to_numeric(df_pendidikan["Tahun"], errors="coerce")
    df_pendidikan = df_pendidikan.dropna(subset=["Tahun"])
    df_pendidikan["Tahun"] = df_pendidikan["Tahun"].astype(int)

    df_pendapatan = pd.read_excel(FILE_PATH, sheet_name="rata rata pendapatan sektor inf")
    df_pendapatan.columns = ["Tahun", "Wilayah", "Pendapatan_TidakSekolah",
                              "Pendapatan_SD", "Pendapatan_SMP",
                              "Pendapatan_SMAkeAtas", "Pendapatan_Total"]
    df_pendapatan["Tahun"] = pd.to_numeric(df_pendapatan["Tahun"], errors="coerce")
    df_pendapatan = df_pendapatan.dropna(subset=["Tahun", "Wilayah"])
    df_pendapatan["Tahun"] = df_pendapatan["Tahun"].astype(int)
    df_pendapatan["Wilayah"] = df_pendapatan["Wilayah"].replace({"Gunungkidul": "Gunung Kidul"})

    df_tki = pd.read_excel(FILE_PATH, sheet_name="TKI")
    df_tki.columns = ["Tahun", "Wilayah", "Jumlah_TKI"]
    df_tki["Tahun"] = pd.to_numeric(df_tki["Tahun"], errors="coerce")
    df_tki = df_tki.dropna(subset=["Tahun", "Wilayah"])
    df_tki["Tahun"] = df_tki["Tahun"].astype(int)

    df_pendapatan_sektor = pd.read_excel(FILE_PATH, sheet_name="rata2 pendapatan informallapang")
    df_pendapatan_sektor.columns = ["Tahun", "Wilayah", "Upah_Pertanian",
                                    "Upah_Industri", "Upah_Jasa", "Upah_Total"]
    df_pendapatan_sektor["Tahun"] = pd.to_numeric(df_pendapatan_sektor["Tahun"], errors="coerce")
    df_pendapatan_sektor = df_pendapatan_sektor.dropna(subset=["Tahun", "Wilayah"])
    df_pendapatan_sektor["Tahun"] = df_pendapatan_sektor["Tahun"].astype(int)
    df_pendapatan_sektor["Wilayah"] = df_pendapatan_sektor["Wilayah"].replace({"Gunungkidul": "Gunung Kidul"})

    df_lapangan = pd.read_excel(FILE_PATH, sheet_name="Jumlah penduduk bekerja menurut")
    df_lapangan.columns = [
        "Tahun", "Wilayah", "Pertanian", "Pertambangan", "Industri",
        "Listrik_Gas", "Air_Sampah", "Konstruksi", "Perdagangan",
        "Transportasi", "Akomodasi_Makan", "Informasi_Komunikasi",
        "Jasa_Keuangan", "Real_Estat", "Jasa_Perusahaan",
        "Adm_Pemerintahan", "Jasa_Pendidikan", "Jasa_Kesehatan", "Jasa_Lain"
    ]
    df_lapangan["Tahun"] = pd.to_numeric(df_lapangan["Tahun"], errors="coerce")
    df_lapangan = df_lapangan.dropna(subset=["Tahun"])
    df_lapangan["Tahun"] = df_lapangan["Tahun"].astype(int)

    df_status = pd.read_excel(FILE_PATH, sheet_name="Sheet11")
    df_status.columns = ["Tahun", "Status", "Jumlah"]
    df_status["Tahun"] = pd.to_numeric(df_status["Tahun"], errors="coerce")
    df_status = df_status.dropna(subset=["Tahun", "Status"])
    df_status["Tahun"] = df_status["Tahun"].astype(int)

    df_pencaker = pd.read_excel(FILE_PATH, sheet_name="Pencari Kerja, Lowongan Kerja, ")
    pencaker_cols_full = [
        "Tahun", "Wilayah",
        "PencakerL", "PencakerP", "PencakerTotal",
        "LowonganL", "LowonganP", "LowonganTotal",
        "PenempatanL", "PenempatanP", "PenempatanTotal",
        "Extra"
    ]
    df_pencaker.columns = pencaker_cols_full[:len(df_pencaker.columns)]
    if "Extra" in df_pencaker.columns:
        df_pencaker = df_pencaker.drop(columns=["Extra"])
    df_pencaker["Tahun"] = pd.to_numeric(df_pencaker["Tahun"], errors="coerce")
    df_pencaker = df_pencaker.dropna(subset=["Tahun", "Wilayah"])
    df_pencaker["Tahun"] = df_pencaker["Tahun"].astype(int)

    return {
        "tpt": df_tpt, "pop": df_pop, "pendidikan": df_pendidikan,
        "pendapatan": df_pendapatan, "tki": df_tki,
        "pendapatan_sektor": df_pendapatan_sektor, "lapangan": df_lapangan,
        "status": df_status, "pencaker": df_pencaker
    }


@st.cache_data
def run_clustering(df_tpt, df_pendapatan, df_pencaker, tahun_cluster):
    wilayah_kab = ["Kulon Progo", "Bantul", "Gunung Kidul", "Sleman", "Kota Yogyakarta"]

    tpt_f = df_tpt[(df_tpt["Tahun"] == tahun_cluster) & (df_tpt["Wilayah"].isin(wilayah_kab))][
        ["Wilayah", "TPT", "TPAK"]].copy()
    pend_f = df_pendapatan[(df_pendapatan["Tahun"] == tahun_cluster) & (df_pendapatan["Wilayah"].isin(wilayah_kab))][
        ["Wilayah", "Pendapatan_Total"]].copy()

    pen_f = df_pencaker[(df_pencaker["Tahun"] == tahun_cluster) & (df_pencaker["Wilayah"].isin(wilayah_kab))].copy()
    if "PenempatanTotal" in pen_f.columns and "PencakerTotal" in pen_f.columns:
        pen_f["Rasio_Penempatan"] = (pen_f["PenempatanTotal"] / pen_f["PencakerTotal"].replace(0, np.nan)) * 100
        pen_f = pen_f[["Wilayah", "Rasio_Penempatan"]]
    else:
        pen_f = pd.DataFrame({"Wilayah": wilayah_kab, "Rasio_Penempatan": [np.nan]*5})

    df_cluster = tpt_f.merge(pend_f, on="Wilayah", how="outer")
    df_cluster = df_cluster.merge(pen_f, on="Wilayah", how="outer")
    df_cluster = df_cluster.dropna()
    if df_cluster.shape[0] < 3:
        return None, None

    features = ["TPT", "TPAK", "Pendapatan_Total", "Rasio_Penempatan"]
    X = df_cluster[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_cluster["Cluster"] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    df_cluster["PCA_1"] = coords[:, 0]
    df_cluster["PCA_2"] = coords[:, 1]

    cluster_tpt = df_cluster.groupby("Cluster")["TPT"].mean().sort_values()
    label_map = {}
    for i, (cl, _) in enumerate(cluster_tpt.items()):
        label_map[cl] = ["TPT Rendah", "TPT Sedang", "TPT Tinggi"][i]
    df_cluster["Label_Cluster"] = df_cluster["Cluster"].map(label_map)
    profile = df_cluster.groupby("Label_Cluster")[features].mean().reset_index()
    return df_cluster, profile


#  MAIN APP

def main():
    data = load_all_data()
    df_tpt = data["tpt"]; df_pop = data["pop"]; df_pendidikan = data["pendidikan"]
    df_pendapatan = data["pendapatan"]; df_tki = data["tki"]
    df_pendapatan_sektor = data["pendapatan_sektor"]; df_lapangan = data["lapangan"]
    df_status = data["status"]; df_pencaker = data["pencaker"]

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 0px 0 10px;'>
            <div style='font-family:"DM Serif Display",serif; font-size:18px; font-weight:400;
                        color:#BDD8E9; margin-top:-0px; line-height:1.3;'>
                Ketenagakerjaan DIY
            </div>
        </div>
        <hr style='border-color:rgba(123, 189, 232, 0.25); margin: 5px 0 10px;'>
        """, unsafe_allow_html=True)

        st.markdown("### Filter")
        st.markdown(
            "<small style='color:#7BBDE8; font-size:12px;'>"
            "Filter ini berlaku untuk semua visualisasi</small>",
            unsafe_allow_html=True
        )

        tahun_min = int(df_tpt["Tahun"].min())
        tahun_max = int(df_tpt["Tahun"].max())
        tahun_range = st.slider(
            "Rentang Tahun", min_value=tahun_min, max_value=tahun_max,
            value=(tahun_min, tahun_max), step=1
        )

        wilayah_opts = sorted([w for w in df_tpt["Wilayah"].unique() if w != "DI Yogyakarta"])
        wilayah_sel = st.multiselect("Pilih Wilayah", options=wilayah_opts, default=wilayah_opts)
        if not wilayah_sel:
            wilayah_sel = wilayah_opts

        var_opts = {
            "Tingkat Pengangguran Terbuka (TPT %)": "TPT",
            "Tingkat Partisipasi Angkatan Kerja (TPAK %)": "TPAK",
        }
        var_label = st.selectbox("Variabel Tren Utama", options=list(var_opts.keys()))
        var_col = var_opts[var_label]

        st.markdown("""
        <div style='position: fixed; bottom: 25px; width: 228px; z-index: 99;'>
            <div class='source-badge'>
                <b>Sumber Data</b><br>
                BPS Provinsi D.I. Yogyakarta<br>
                <small style='opacity:0.75;'>Data diakses: 2025</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── DASHBOARD HEADER ──
    st.markdown("""
    <div class='dashboard-header'>
        <p class='header-title' style='font-size:36px; line-height:1.25;'>
            Potret Ketenagakerjaan D.I. Yogyakarta:<br>
            Siapa Bekerja, Berapa Upah, dan Ke Mana Arahnya?
        </p>
        <p class='header-subtitle'>
            Eksplorasi komprehensif indikator ketenagakerjaan, demografi tenaga kerja, dan penyerapan
            lapangan kerja berbasis data resmi BPS Provinsi D.I. Yogyakarta
        </p>
        <span class='header-badge'>Official Statistics · BPS DIY · Kelompok 4</span>
    </div>
    """, unsafe_allow_html=True)

    # ── FILTER MASKS ──
    mask_tpt = (
        (df_tpt["Tahun"] >= tahun_range[0]) &
        (df_tpt["Tahun"] <= tahun_range[1]) &
        (df_tpt["Wilayah"].isin(wilayah_sel))
    )
    df_tpt_f = df_tpt[mask_tpt].copy()
    df_diy = df_tpt[(df_tpt["Wilayah"] == "DI Yogyakarta")].copy()
    latest_year = int(df_diy["Tahun"].max())
    all_years_diy = sorted(df_diy["Tahun"].unique())
    earliest_year = int(df_diy["Tahun"].min())

    # KPI YEAR SELECTOR
    col_kpi_header, col_kpi_year = st.columns([3, 1])
    with col_kpi_year:
        kpi_year = st.selectbox(
            "Tahun Indikator", options=list(reversed(all_years_diy)),
            index=0, key="kpi_year_select"
        )
    kpi_year = int(kpi_year)
    prev_year = kpi_year - 1
    is_earliest = (kpi_year == earliest_year)

    with col_kpi_header:
        warning_text = ""
        if is_earliest:
            warning_text = (
                '  ·  <span style="color:#6EA2B3; font-size:11px;">'
                'Tahun awal data — tidak ada pembanding tahun sebelumnya</span>'
            )
        st.markdown(
            f"<div style='padding-top:8px; font-size:16px; color:#001D39;'>"
            f"Menampilkan indikator kunci untuk tahun "
            f"<b style='color:#0A4174; font-size:15px;'>{kpi_year}</b>"
            f"{warning_text}</div>", unsafe_allow_html=True
        )

    # KPI VALUES
    tpt_cur   = df_diy[df_diy["Tahun"] == kpi_year]["TPT"].values
    tpt_prev  = df_diy[df_diy["Tahun"] == prev_year]["TPT"].values
    tpak_cur  = df_diy[df_diy["Tahun"] == kpi_year]["TPAK"].values
    tpak_prev = df_diy[df_diy["Tahun"] == prev_year]["TPAK"].values

    tpt_val       = float(tpt_cur[0])  if len(tpt_cur)  > 0 else 0
    tpt_prev_val  = float(tpt_prev[0]) if len(tpt_prev) > 0 else None
    tpak_val      = float(tpak_cur[0]) if len(tpak_cur) > 0 else 0
    tpak_prev_val = float(tpak_prev[0])if len(tpak_prev)> 0 else None

    pend_diy = df_pendapatan[df_pendapatan["Wilayah"] == "DI Yogyakarta"]
    pend_yr  = pend_diy[pend_diy["Tahun"] == kpi_year]
    if len(pend_yr) > 0:
        pend_terbaru = pend_yr.iloc[0]["Pendapatan_Total"]; pend_tahun = kpi_year
    elif len(pend_diy) > 0:
        pend_terbaru = pend_diy.sort_values("Tahun").iloc[-1]["Pendapatan_Total"]
        pend_tahun   = int(pend_diy.sort_values("Tahun").iloc[-1]["Tahun"])
    else:
        pend_terbaru, pend_tahun = 0, 0

    tki_diy = df_tki[df_tki["Wilayah"] == "DI Yogyakarta"]
    tki_yr  = tki_diy[tki_diy["Tahun"] == kpi_year]
    if len(tki_yr) > 0:
        tki_latest_val = int(tki_yr.iloc[0]["Jumlah_TKI"]); tki_latest_yr = kpi_year
    elif len(tki_diy) > 0:
        tki_latest_val = int(tki_diy.sort_values("Tahun").iloc[-1]["Jumlah_TKI"])
        tki_latest_yr  = int(tki_diy.sort_values("Tahun").iloc[-1]["Tahun"])
    else:
        tki_latest_val, tki_latest_yr = 0, 0

    pencaker_diy = df_pencaker[df_pencaker["Wilayah"] == "DI Yogyakarta"]
    pencaker_yr  = pencaker_diy[pencaker_diy["Tahun"] == kpi_year]
    if len(pencaker_yr) > 0:
        pencaker_latest = pencaker_yr.iloc[0]
        rasio_penempatan = (pencaker_latest["PenempatanTotal"] /
                            max(pencaker_latest["PencakerTotal"], 1)) * 100 \
                            if "PenempatanTotal" in pencaker_latest else 0
    elif len(pencaker_diy) > 0:
        pencaker_latest  = pencaker_diy.sort_values("Tahun").iloc[-1]
        rasio_penempatan = (pencaker_latest["PenempatanTotal"] /
                            max(pencaker_latest["PencakerTotal"], 1)) * 100 \
                            if "PenempatanTotal" in pencaker_latest else 0
    else:
        rasio_penempatan = 0

    tpt_delta  = (tpt_val  - tpt_prev_val)  if tpt_prev_val  is not None else None
    tpak_delta = (tpak_val - tpak_prev_val) if tpak_prev_val is not None else None

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if tpt_delta is not None:
            dc = "kpi-delta-neg" if tpt_delta > 0 else "kpi-delta-pos"
            di = "▲" if tpt_delta > 0 else "▼"
            dh = f"<div class='{dc}'>{di} {abs(tpt_delta):.2f}% vs {prev_year}</div>"
        else:
            dh = "<div class='kpi-delta-pos' style='color:#6EA2B3;'>Data awal tidak tersedia</div>"
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>TPT DIY</div>
            <div class='kpi-value'>{tpt_val:.2f}</div>
            <div class='kpi-unit'>persen</div>{dh}
        </div>""", unsafe_allow_html=True)

    with col2:
        if tpak_delta is not None:
            dc2 = "kpi-delta-pos" if tpak_delta > 0 else "kpi-delta-neg"
            di2 = "▲" if tpak_delta > 0 else "▼"
            dh2 = f"<div class='{dc2}'>{di2} {abs(tpak_delta):.2f}% vs {prev_year}</div>"
        else:
            dh2 = "<div class='kpi-delta-pos' style='color:#6EA2B3;'>Data awal tidak tersedia</div>"
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>TPAK DIY</div>
            <div class='kpi-value'>{tpak_val:.2f}</div>
            <div class='kpi-unit'>persen</div>{dh2}
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Pendapatan Informal ({pend_tahun})</div>
            <div class='kpi-value' style='font-size:22px;'>Rp {pend_terbaru/1e6:.2f}Jt</div>
            <div class='kpi-unit'>per bulan · DIY</div>
            <div class='kpi-delta-pos'>Pekerja informal sektor campuran</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>TKI Luar Negeri ({tki_latest_yr})</div>
            <div class='kpi-value'>{tki_latest_val:,}</div>
            <div class='kpi-unit'>jiwa</div>
            <div class='kpi-delta-pos'>Asal Provinsi DIY</div>
        </div>""", unsafe_allow_html=True)

    with col5:
        rp_color = "kpi-delta-pos" if rasio_penempatan >= 50 else "kpi-delta-neg"
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Rasio Penempatan Kerja</div>
            <div class='kpi-value'>{rasio_penempatan:.1f}</div>
            <div class='kpi-unit'>persen</div>
            <div class='{rp_color}'>Penempatan / Pencaker · DIY</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)

    # TABS
    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "Eksplorasi Demografi",
        "Eksplorasi Ketenagakerjaan",
        "Upah & Tenaga Kerja",
        "Analisis Clustering",
        "Insight & Temuan"
    ])

    #  TAB 0 — DEMOGRAFI
    with tab0:
        st.markdown("<div class='section-header'>Gambaran Kependudukan D.I. Yogyakarta</div>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<small style='color:#001D39'>Periode: <b style='color:#0A4174'>"
            f"{tahun_range[0]}–{tahun_range[1]}</b> · Wilayah terpilih</small>",
            unsafe_allow_html=True
        )

        df_pop_f = df_pop[
            (df_pop["Tahun"] >= tahun_range[0]) &
            (df_pop["Tahun"] <= tahun_range[1])
        ].copy()

        st.markdown("<div class='section-header'>Tren Jumlah & Laju Pertumbuhan Penduduk</div>",
                    unsafe_allow_html=True)
        pop_kab_f = df_pop_f[df_pop_f["Wilayah"].isin(wilayah_sel)].copy()

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            fig_pop = go.Figure()
            for w in wilayah_sel:
                dw = pop_kab_f[pop_kab_f["Wilayah"] == w].sort_values("Tahun")
                if dw.empty: continue
                color = WILAYAH_COLORS.get(w, "#7BBDE8")
                fig_pop.add_trace(go.Scatter(
                    x=dw["Tahun"], y=dw["Jumlah_Penduduk"],
                    mode="lines+markers", name=w,
                    line=dict(color=color, width=2.5),
                    marker=dict(size=7, color=color, line=dict(width=1.5, color="#BDD8E9")),
                    hovertemplate=f"<b>{w}</b><br>Tahun: %{{x}}<br>Penduduk: <b>%{{y:,.0f}} jiwa</b><extra></extra>"
                ))
            fig_pop.update_layout(
                title=dict(text="Jumlah Penduduk per Kabupaten/Kota", font=dict(size=15)),
                xaxis_title="Tahun", yaxis_title="Jumlah Penduduk (jiwa)", height=380,
            )
            fig_pop = apply_theme(fig_pop)
            chart_container_start()
            st.plotly_chart(fig_pop, use_container_width=True)
            chart_container_end()

        with col_d2:
            fig_laju = go.Figure()
            for w in wilayah_sel:
                dw = pop_kab_f[pop_kab_f["Wilayah"] == w].sort_values("Tahun")
                if dw.empty or "Laju_Pertumbuhan" not in dw.columns: continue
                color = WILAYAH_COLORS.get(w, "#7BBDE8")
                fig_laju.add_trace(go.Bar(
                    x=dw["Tahun"], y=dw["Laju_Pertumbuhan"],
                    name=w, marker_color=color, opacity=0.9,
                    hovertemplate=f"<b>{w}</b><br>Tahun: %{{x}}<br>Laju: <b>%{{y:.2f}}%</b><extra></extra>"
                ))
            fig_laju.update_layout(
                title=dict(text="Laju Pertumbuhan Penduduk (%)", font=dict(size=15)),
                xaxis_title="Tahun", yaxis_title="Laju Pertumbuhan (%)",
                barmode="group", height=380,
            )
            fig_laju = apply_theme(fig_laju)
            chart_container_start()
            st.plotly_chart(fig_laju, use_container_width=True)
            chart_container_end()

        st.markdown("<div class='section-header'>Rasio Jenis Kelamin & Kepadatan Penduduk</div>",
                    unsafe_allow_html=True)
        pop_tahun_avail = sorted(df_pop_f["Tahun"].unique())
        col_d3, col_d4 = st.columns(2)

        if pop_tahun_avail:
            tahun_demo = st.select_slider(
                    "Pilih Tahun untuk Snapshot Demografi",
                    options=pop_tahun_avail, value=pop_tahun_avail[-1],
                    key="tahun_demo_slider"
            )
            df_demo_yr = pop_kab_f[pop_kab_f["Tahun"] == tahun_demo].copy()
            col_d3, col_d4 = st.columns(2)


        with col_d3:
                if not df_demo_yr.empty and "Rasio_JK" in df_demo_yr.columns:
                    df_demo_yr = df_demo_yr.dropna(subset=["Rasio_JK", "Jumlah_Penduduk"])
                    df_demo_yr["Est_Perempuan"] = df_demo_yr["Jumlah_Penduduk"] / (1 + df_demo_yr["Rasio_JK"] / 100)
                    df_demo_yr["Est_LakiLaki"]  = df_demo_yr["Jumlah_Penduduk"] - df_demo_yr["Est_Perempuan"]
                    fig_gender = go.Figure()
                    fig_gender.add_trace(go.Bar(
                        x=df_demo_yr["Wilayah"], y=df_demo_yr["Est_LakiLaki"],
                        name="Laki-laki (estimasi)", marker_color="#001D39",
                        hovertemplate="%{x}<br>Laki-laki: <b>%{y:,.0f} jiwa</b><extra></extra>"
                    ))
                    fig_gender.add_trace(go.Bar(
                        x=df_demo_yr["Wilayah"], y=df_demo_yr["Est_Perempuan"],
                        name="Perempuan (estimasi)", marker_color="#6EA2B3",
                        hovertemplate="%{x}<br>Perempuan: <b>%{y:,.0f} jiwa</b><extra></extra>"
                    ))
                    fig_gender.update_layout(
                        title=dict(text=f"Estimasi Komposisi L/P per Wilayah ({tahun_demo})", font=dict(size=13)),
                        barmode="group", xaxis_title="Kabupaten/Kota",
                        yaxis_title="Jumlah (jiwa)", height=360,
                    )
                    fig_gender = apply_theme(fig_gender)
                    fig_gender.update_xaxes(tickmode="auto", dtick=None)
                    chart_container_start()
                    st.plotly_chart(fig_gender, use_container_width=True)
                    chart_container_end()

        with col_d4:
            if pop_tahun_avail and not df_demo_yr.empty and "Kepadatan" in df_demo_yr.columns:
                df_dens = df_demo_yr.dropna(subset=["Kepadatan"]).sort_values("Kepadatan", ascending=True)
                colors_dens = [WILAYAH_COLORS.get(w, "#7BBDE8") for w in df_dens["Wilayah"]]
                fig_dens = go.Figure(go.Bar(
                    y=df_dens["Wilayah"], x=df_dens["Kepadatan"],
                    orientation="h",
                    marker=dict(color=colors_dens, line=dict(color="rgba(0,29,57,0.25)", width=1)),
                    text=df_dens["Kepadatan"].round(0).astype(int).astype(str) + " jiwa/km²",
                    textposition="outside", textfont=dict(color="#001D39", size=10),
                    hovertemplate="%{y}<br>Kepadatan: <b>%{x:,.0f} jiwa/km²</b><extra></extra>"
                ))
                fig_dens.update_layout(
                    title=dict(text=f"Kepadatan Penduduk per Wilayah ({tahun_demo})", font=dict(size=13)),
                    xaxis_title="Kepadatan (jiwa/km²)", height=360,
                )
                fig_dens = apply_theme(fig_dens)
                fig_dens.update_xaxes(tickmode="auto", dtick=None)
                chart_container_start()
                st.plotly_chart(fig_dens, use_container_width=True)
                chart_container_end()

        st.markdown("<div class='section-header'>Proporsi & Distribusi Penduduk Antar Wilayah</div>",
                    unsafe_allow_html=True)
        col_d5, col_d6 = st.columns([1.2, 1])

        with col_d5:
            if pop_tahun_avail and not df_demo_yr.empty:
                df_pie_pop = df_demo_yr.dropna(subset=["Jumlah_Penduduk"])
                colors_pie_pop = [WILAYAH_COLORS.get(w, "#7BBDE8") for w in df_pie_pop["Wilayah"]]
                fig_pie_pop = go.Figure(data=[go.Pie(
                    labels=df_pie_pop["Wilayah"].replace({"Kota Yogyakarta": "Kota Yogya"}),
                    values=df_pie_pop["Jumlah_Penduduk"],
                    marker=dict(colors=colors_pie_pop, line=dict(color="#FFFFFF", width=2)),
                    hovertemplate="<b>%{label}</b><br>Penduduk: %{value:,} jiwa<br>(%{percent})<extra></extra>",
                    textinfo="percent+label", textfont=dict(size=11, color="#BDD8E9"), hole=0.45
                )])
                fig_pie_pop.update_layout(
                    title=dict(text=f"Proporsi Penduduk Antar Wilayah ({tahun_demo})", font=dict(size=13)),
                    height=380, showlegend=False,
                )
                fig_pie_pop = apply_theme(fig_pie_pop)
                chart_container_start()
                st.plotly_chart(fig_pie_pop, use_container_width=True)
                chart_container_end()

        with col_d6:
            if pop_tahun_avail and not df_demo_yr.empty and "Rasio_JK" in df_demo_yr.columns:
                df_rjk = df_demo_yr.dropna(subset=["Rasio_JK"]).sort_values("Rasio_JK")
                fig_rjk = go.Figure()
                fig_rjk.add_trace(go.Bar(
                    x=df_rjk["Rasio_JK"], y=df_rjk["Wilayah"], orientation="h",
                    marker=dict(
                        color=df_rjk["Rasio_JK"],
                        colorscale=[[0, "#001D39"], [0.5, "#49769F"], [1, "#95C7E6"]],
                        showscale=True,
                        colorbar=dict(title="Rasio JK", tickfont=dict(color="#001D39", size=9))
                    ),
                    text=df_rjk["Rasio_JK"].round(1).astype(str),
                    textposition="outside", textfont=dict(color="#001D39", size=10),
                    hovertemplate="%{y}<br>Rasio JK: <b>%{x:.1f}</b><br>(L per 100 P)<extra></extra>"
                ))
                fig_rjk.add_vline(
                    x=100, line_dash="dot", line_color="#001D39",
                    annotation_text="Seimbang (100)",
                    annotation_font_color="#001D39",
                    annotation_position="top right"
                )
                fig_rjk.update_layout(
                    title=dict(text=f"Rasio Jenis Kelamin ({tahun_demo})", font=dict(size=13)),
                    xaxis_title="Rasio JK (L per 100 P)", height=380,
                )
                fig_rjk = apply_theme(fig_rjk)
                fig_rjk.update_xaxes(tickmode="auto", dtick=None)
                chart_container_start()
                st.plotly_chart(fig_rjk, use_container_width=True)
                chart_container_end()

        st.markdown("<div class='section-header'>Distribusi Penduduk vs Kepadatan (Bubble Chart)</div>",
                    unsafe_allow_html=True)
        pop_trend_all = df_pop_f[df_pop_f["Wilayah"].isin(wilayah_sel)].dropna(
            subset=["Jumlah_Penduduk", "Kepadatan", "Laju_Pertumbuhan"]
        )
        if not pop_trend_all.empty:
            x_max = pop_trend_all["Kepadatan"].max() + 3000
            y_min = pop_trend_all["Laju_Pertumbuhan"].min() - 0.8
            y_max = pop_trend_all["Laju_Pertumbuhan"].max() + 0.8
            fig_bubble = px.scatter(
                pop_trend_all, x="Kepadatan", y="Laju_Pertumbuhan",
                size="Jumlah_Penduduk", color="Wilayah",
                animation_frame="Tahun", hover_name="Wilayah",
                color_discrete_map=WILAYAH_COLORS,
                size_max=60, range_x=[-1000, x_max], range_y=[y_min, y_max],
                labels={"Kepadatan": "Kepadatan Penduduk (jiwa/km²)",
                        "Laju_Pertumbuhan": "Laju Pertumbuhan (%)",
                        "Jumlah_Penduduk": "Jumlah Penduduk"}
            )
            fig_bubble.update_layout(
                title=dict(
                    text="Dinamika Kepadatan vs Laju Pertumbuhan<br>"
                         "<sup style='font-size:17px; color:#001D39'>* Ukuran bubble merepresentasikan total jumlah penduduk</sup>",
                    font=dict(size=14, color="#001D39")
                ),
                height=550,
            )
            fig_bubble = apply_theme(fig_bubble)
            fig_bubble.update_xaxes(tickmode="auto", dtick=None)
            fig_bubble.update_layout(margin=dict(t=85, r=145))
            chart_container_start()
            st.plotly_chart(fig_bubble, use_container_width=True)
            chart_container_end()

    #  TAB 1 — KETENAGAKERJAAN
    with tab1:
        st.markdown("<div class='section-header'>Tren TPT & TPAK per Kabupaten/Kota</div>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<small style='color:#001D39'>Variabel: <b style='color:#0A4174'>{var_label}</b> · "
            f"Periode: <b style='color:#0A4174'>{tahun_range[0]}–{tahun_range[1]}</b></small>",
            unsafe_allow_html=True
        )

        fig_line = go.Figure()
        for wilayah in wilayah_sel:
            df_w = df_tpt_f[df_tpt_f["Wilayah"] == wilayah].sort_values("Tahun")
            if df_w.empty: continue
            color = WILAYAH_COLORS.get(wilayah, "#7BBDE8")
            fig_line.add_trace(go.Scatter(
                x=df_w["Tahun"], y=df_w[var_col],
                mode="lines+markers", name=wilayah,
                line=dict(color=color, width=2.5),
                marker=dict(size=7, color=color, line=dict(width=1.5, color="#FFFFFF")),
                hovertemplate=f"<b>{wilayah}</b><br>Tahun: %{{x}}<br>{var_label}: <b>%{{y:.2f}}%</b><extra></extra>"
            ))
        df_diy_f = df_tpt[
            (df_tpt["Wilayah"] == "DI Yogyakarta") &
            (df_tpt["Tahun"] >= tahun_range[0]) &
            (df_tpt["Tahun"] <= tahun_range[1])
        ].sort_values("Tahun")
        fig_line.add_trace(go.Scatter(
            x=df_diy_f["Tahun"], y=df_diy_f[var_col],
            mode="lines", name="DIY (Rata-rata)",
            line=dict(color="#0A4174", width=2, dash="dot"),
            hovertemplate=f"<b>DI Yogyakarta</b><br>Tahun: %{{x}}<br>{var_label}: <b>%{{y:.2f}}%</b><extra></extra>"
        ))
        fig_line.update_layout(
            title=dict(text=f"Tren {var_label}", font=dict(size=15)),
            xaxis_title="Tahun", yaxis_title=f"{var_label} (%)", height=380,
        )
        fig_line = apply_theme(fig_line)
        chart_container_start()
        st.plotly_chart(fig_line, use_container_width=True)
        chart_container_end()

        st.markdown("<div class='section-header'>Perbandingan TPT & TPAK Antar Wilayah</div>",
                    unsafe_allow_html=True)
        tahun_bar_opts = sorted(df_tpt_f["Tahun"].unique(), reverse=True)
        if tahun_bar_opts:
            tahun_bar = st.select_slider(
                "Pilih Tahun untuk Perbandingan",
                options=sorted(df_tpt_f["Tahun"].unique()),
                value=tahun_bar_opts[0]
            )
            df_bar = df_tpt_f[df_tpt_f["Tahun"] == tahun_bar].copy()
            df_bar_sorted = df_bar.sort_values(var_col, ascending=False)
            colors_bar = [WILAYAH_COLORS.get(w, "#7BBDE8") for w in df_bar_sorted["Wilayah"]]
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=df_bar_sorted["Wilayah"], y=df_bar_sorted[var_col],
                marker=dict(color=colors_bar, line=dict(color="rgba(255,255,255,0.4)", width=1)),
                hovertemplate=f"<b>%{{x}}</b><br>{var_label}: <b>%{{y:.2f}}%</b><extra></extra>",
                text=df_bar_sorted[var_col].round(2).astype(str) + "%",
                textposition="outside", textfont=dict(color="#001D39", size=11)
            ))
            avg_diy = df_diy[df_diy["Tahun"] == tahun_bar][var_col].values
            if len(avg_diy) > 0:
                fig_bar.add_hline(
                    y=float(avg_diy[0]), line_dash="dot", line_color="#001D39",
                    annotation_text=f"Rata-rata DIY: {avg_diy[0]:.2f}%",
                    annotation_font_color="#001D39",
                    annotation_position="bottom right"
                )
            fig_bar.update_layout(
                title=dict(text=f"Perbandingan {var_label} Tahun {tahun_bar}", font=dict(size=13)),
                xaxis_title="Kabupaten/Kota", yaxis_title=f"{var_label} (%)", height=380,
            )
            fig_bar = apply_theme(fig_bar)
            fig_bar.update_xaxes(tickmode="auto", dtick=None)
            chart_container_start()
            st.plotly_chart(fig_bar, use_container_width=True)
            chart_container_end()

        st.markdown("<div class='section-header'>Pencari Kerja vs Lowongan vs Penempatan</div>",
                    unsafe_allow_html=True)
        df_penc_agg = df_pencaker[
            (df_pencaker["Tahun"] >= tahun_range[0]) &
            (df_pencaker["Tahun"] <= tahun_range[1]) &
            (df_pencaker["Wilayah"] == "DI Yogyakarta")
        ].sort_values("Tahun")
        if not df_penc_agg.empty:
            fig_penc = go.Figure()
            fig_penc.add_trace(go.Bar(
                x=df_penc_agg["Tahun"], y=df_penc_agg["PencakerTotal"],
                name="Pencari Kerja", marker_color="#001D39", opacity=0.9,
                hovertemplate="Tahun: %{x}<br>Pencari Kerja: <b>%{y:,}</b><extra></extra>"
            ))
            fig_penc.add_trace(go.Bar(
                x=df_penc_agg["Tahun"], y=df_penc_agg["LowonganTotal"],
                name="Lowongan Kerja", marker_color="#4E8EA2", opacity=0.9,
                hovertemplate="Tahun: %{x}<br>Lowongan Kerja: <b>%{y:,}</b><extra></extra>"
            ))
            fig_penc.add_trace(go.Scatter(
                x=df_penc_agg["Tahun"], y=df_penc_agg["PenempatanTotal"],
                mode="lines+markers", name="Penempatan Kerja",
                line=dict(color="#49769F", width=2.5),
                marker=dict(size=8, color="#49769F"),
                hovertemplate="Tahun: %{x}<br>Penempatan: <b>%{y:,}</b><extra></extra>"
            ))
            fig_penc.update_layout(
                title=dict(text="Dinamika Pasar Kerja D.I. Yogyakarta", font=dict(size=13)),
                barmode="group", xaxis_title="Tahun", yaxis_title="Jumlah (orang)", height=380,
            )
            fig_penc = apply_theme(fig_penc)
            chart_container_start()
            st.plotly_chart(fig_penc, use_container_width=True)
            chart_container_end()

        st.markdown("<div class='section-header'>Komposisi Status Pekerjaan Utama</div>",
                    unsafe_allow_html=True)
        df_status_f = df_status[
            (df_status["Tahun"] >= tahun_range[0]) &
            (df_status["Tahun"] <= tahun_range[1])
        ].copy()
        if not df_status_f.empty:
            df_status_pivot = df_status_f.pivot_table(
                index="Tahun", columns="Status", values="Jumlah", aggfunc="sum"
            ).reset_index()
            fig_area = go.Figure()
            status_cols = [c for c in df_status_pivot.columns if c != "Tahun"]
            color_palette = ["#001D39", "#0A4174", "#49769F", "#4E8EA2",
                             "#6EA2B3", "#7BBDE8", "#3D96CE"]
            for i, status in enumerate(status_cols):
                hex_color = color_palette[i % len(color_palette)]
                fig_area.add_trace(go.Scatter(
                    x=df_status_pivot["Tahun"],
                    y=df_status_pivot[status] / 1000 if df_status_pivot[status].max() > 10000 else df_status_pivot[status],
                    name=status, mode="lines+markers",
                    line=dict(width=2.5, color=hex_color),
                    marker=dict(size=7, color=hex_color, line=dict(width=1.5, color="#FFFFFF")),
                    hovertemplate="%{fullData.name}: <b>%{y:.1f} ribu</b><extra></extra>"
                ))
            fig_area.update_layout(
                title=dict(text="Tren Status Pekerjaan Utama D.I. Yogyakarta (Ribu Jiwa)", font=dict(size=13)),
                xaxis_title="Tahun", yaxis_title="Jumlah Pekerja (ribu jiwa)", height=380,
            )
            fig_area = apply_theme(fig_area)
            chart_container_start()
            st.plotly_chart(fig_area, use_container_width=True)
            chart_container_end()

    #  TAB 2 — UPAH & TENAGA KERJA
    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("<div class='section-header'>Tren Pendapatan Informal per Wilayah</div>",
                        unsafe_allow_html=True)
            df_pend_f = df_pendapatan[
                (df_pendapatan["Tahun"] >= tahun_range[0]) &
                (df_pendapatan["Tahun"] <= tahun_range[1]) &
                (df_pendapatan["Wilayah"].isin(wilayah_sel))
            ].copy()
            if not df_pend_f.empty:
                fig_pend = go.Figure()
                for w in wilayah_sel:
                    dw = df_pend_f[df_pend_f["Wilayah"] == w].sort_values("Tahun")
                    if dw.empty: continue
                    fig_pend.add_trace(go.Scatter(
                        x=dw["Tahun"], y=dw["Pendapatan_Total"],
                        mode="lines+markers", name=w,
                        line=dict(color=WILAYAH_COLORS.get(w, "#7BBDE8"), width=2.5),
                        marker=dict(size=7),
                        hovertemplate=f"<b>{w}</b><br>Tahun: %{{x}}<br>Pendapatan: <b>Rp %{{y:,.0f}}</b><extra></extra>"
                    ))
                fig_pend.update_layout(
                    title=dict(text="Rata-rata Pendapatan Bersih Pekerja Informal", font=dict(size=13)),
                    xaxis_title="Tahun", yaxis_title="Pendapatan (Rp)", height=350,
                )
                fig_pend = apply_theme(fig_pend)
                chart_container_start()
                st.plotly_chart(fig_pend, use_container_width=True)
                chart_container_end()

        with col_b:
            st.markdown("<div class='section-header'>Pendapatan Informal per Sektor</div>",
                        unsafe_allow_html=True)
            df_sek_diy = df_pendapatan_sektor[
                (df_pendapatan_sektor["Wilayah"] == "DI Yogyakarta") &
                (df_pendapatan_sektor["Tahun"] >= tahun_range[0]) &
                (df_pendapatan_sektor["Tahun"] <= tahun_range[1])
            ].sort_values("Tahun")
            if not df_sek_diy.empty:
                fig_sek = go.Figure()
                sektor_info = [
                    ("Upah_Pertanian", "Pertanian", "#49769F"),
                    ("Upah_Industri",  "Industri",  "#0A4174"),
                    ("Upah_Jasa",      "Jasa",       "#6EA2B3"),
                ]
                for col_s, label_s, color_s in sektor_info:
                    if col_s in df_sek_diy.columns:
                        fig_sek.add_trace(go.Bar(
                            x=df_sek_diy["Tahun"], y=df_sek_diy[col_s],
                            name=label_s, marker_color=color_s,
                            hovertemplate=f"<b>{label_s}</b><br>Tahun: %{{x}}<br>Upah: <b>Rp %{{y:,.0f}}</b><extra></extra>"
                        ))
                fig_sek.update_layout(
                    title=dict(text="Upah Informal per Sektor (DIY)", font=dict(size=13)),
                    barmode="group", xaxis_title="Tahun", yaxis_title="Upah (Rp)", height=350,
                )
                fig_sek = apply_theme(fig_sek)
                chart_container_start()
                st.plotly_chart(fig_sek, use_container_width=True)
                chart_container_end()

        st.markdown("<div class='section-header'>Tren Tenaga Kerja Indonesia (TKI) Luar Negeri</div>",
                    unsafe_allow_html=True)
        df_tki_f = df_tki[
            (df_tki["Tahun"] >= tahun_range[0]) &
            (df_tki["Tahun"] <= tahun_range[1])
        ].copy()
        df_tki_f["Wilayah"] = df_tki_f["Wilayah"].replace({
            "Kulonprogo": "Kulon Progo",
            "Gunungkidul": "Gunung Kidul",
            "Yogyakarta": "Kota Yogyakarta",
        })
        df_tki_kab = df_tki_f[df_tki_f["Wilayah"].isin(wilayah_sel)].copy()
        if not df_tki_kab.empty:
            col_tki1, col_tki2 = st.columns([1.2, 1])
            with col_tki1:
                fig_tki = go.Figure()
                for w in wilayah_sel:
                    dw = df_tki_kab[df_tki_kab["Wilayah"] == w].sort_values("Tahun")
                    if dw.empty: continue
                    fig_tki.add_trace(go.Scatter(
                        x=dw["Tahun"], y=dw["Jumlah_TKI"],
                        mode="lines+markers", name=w,
                        line=dict(color=WILAYAH_COLORS.get(w, "#7BBDE8"), width=2.5),
                        marker=dict(size=7),
                        hovertemplate=f"<b>{w}</b><br>Tahun: %{{x}}<br>TKI: <b>%{{y:,}} jiwa</b><extra></extra>"
                    ))
                fig_tki.update_layout(
                    title=dict(text="Jumlah TKI Luar Negeri per Kabupaten/Kota", font=dict(size=13)),
                    xaxis_title="Tahun", yaxis_title="Jumlah (jiwa)", height=400,
                )
                fig_tki = apply_theme(fig_tki)
                chart_container_start()
                st.plotly_chart(fig_tki, use_container_width=True)
                chart_container_end()

            with col_tki2:
                tki_terbaru_yr = int(df_tki_kab["Tahun"].max())
                df_tki_pie = df_tki_kab[df_tki_kab["Tahun"] == tki_terbaru_yr].copy()
                if not df_tki_pie.empty:
                    labels_pie = df_tki_pie["Wilayah"].replace({"Kota Yogyakarta": "Kota Yogya"})
                    colors_pie = [WILAYAH_COLORS.get(w, "#7BBDE8") for w in df_tki_pie["Wilayah"]]
                    font_sizes = [10 if label == "Kota Yogya" else 11 for label in labels_pie]
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=labels_pie, values=df_tki_pie["Jumlah_TKI"],
                        marker=dict(colors=colors_pie),
                        hovertemplate="<b>%{label}</b><br>TKI: %{value:,}<br>(%{percent})<extra></extra>",
                        textinfo="percent+label", textposition="inside",
                        textfont=dict(size=font_sizes, color="#BDD8E9"),
                        hole=0.4
                    )])
                    fig_pie.update_layout(
                        title=dict(text=f"Distribusi TKI ({tki_terbaru_yr})", font=dict(size=13)),
                        height=400, showlegend=False, margin=dict(t=80, b=30, l=30, r=30),
                    )
                    fig_pie = apply_theme(fig_pie)
                    chart_container_start()
                    st.plotly_chart(fig_pie, use_container_width=True)
                    chart_container_end()

        st.markdown("<div class='section-header'>Angkatan Kerja Berdasarkan Pendidikan (D.I. Yogyakarta)</div>",
                    unsafe_allow_html=True)
        df_pend_edu = df_pendidikan[
            (df_pendidikan["Tahun"] >= tahun_range[0]) &
            (df_pendidikan["Tahun"] <= tahun_range[1])
        ].copy()
        if not df_pend_edu.empty:
            tahun_edu_opts = sorted(df_pend_edu["Tahun"].unique())
            tahun_edu_sel  = st.select_slider(
                "Pilih Tahun untuk Analisis Pendidikan",
                options=tahun_edu_opts, value=tahun_edu_opts[-1]
            )
            df_edu_yr = df_pend_edu[df_pend_edu["Tahun"] == tahun_edu_sel].copy()
            edu_map = {
                "Tidak pernah sekolah/Tidak tamat SD": "Tdk Sekolah/Tdk Tamat SD",
                "SD": "SD", "SMP": "SMP", "SMA": "SMA", "SMK": "SMK",
                "Diploma I/II/III": "D1/D2/D3",
                "Diploma IV/S1/S2/S3": "D4/S1/S2/S3",
            }
            df_edu_yr["Pendidikan"] = df_edu_yr["Pendidikan"].replace(edu_map)

            col_e1, col_e2 = st.columns(2)
            with col_e1:
                fig_edu = go.Figure()
                fig_edu.add_trace(go.Bar(
                    y=df_edu_yr["Pendidikan"], x=df_edu_yr["Bekerja"],
                    name="Bekerja", orientation="h", marker_color="#0A4174",
                    hovertemplate="%{y}<br>Bekerja: <b>%{x:,.1f} ribu</b><extra></extra>"
                ))
                fig_edu.add_trace(go.Bar(
                    y=df_edu_yr["Pendidikan"], x=df_edu_yr["Total_Pengangguran"],
                    name="Pengangguran", orientation="h", marker_color="#49769F",
                    hovertemplate="%{y}<br>Pengangguran: <b>%{x:,.1f} ribu</b><extra></extra>"
                ))
                fig_edu.update_layout(
                    title=dict(text=f"Bekerja vs Pengangguran Menurut Pendidikan ({tahun_edu_sel})", font=dict(size=12)),
                    barmode="group", xaxis_title="Ribu Jiwa", height=350,
                )
                fig_edu = apply_theme(fig_edu)
                fig_edu.update_xaxes(tickmode="auto", dtick=None)
                chart_container_start()
                st.plotly_chart(fig_edu, use_container_width=True)
                chart_container_end()

            st.markdown("""
            <div class='insight-box' style='width: 100%; padding: 15px 25px;'>
                <div class='insight-title' style='margin-bottom: 10px; font-size: 14px; letter-spacing: 1.5px;'>
                    DETAIL KLASIFIKASI PENDIDIKAN
                </div>
                <div class='insight-text' style='font-size: 14px; line-height: 1.6;'>
                    <span style='margin-right: 25px; color: #7BBDE8;'><b>0. Tidak Sekolah/Tamat SD:</b> <span style='color: #BDD8E9;'>Tidak sekolah/belum tamat SD.</span></span>
                    <span style='margin-right: 25px; color: #7BBDE8;'><b>1. SD/SMP/SMA/SMK:</b> <span style='color: #BDD8E9;'>Dasar & Menengah.</span></span>
                    <span style='margin-right: 25px; color: #7BBDE8;'><b>2. D1/D2/D3:</b> <span style='color: #BDD8E9;'>Diploma/Vokasi.</span></span>
                    <span style='color: #7BBDE8;'><b>3. D4/S1/S2/S3:</b> <span style='color: #BDD8E9;'>Sarjana s.d Doktor.</span></span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with col_e2:
                fig_pct = go.Figure(go.Bar(
                    y=df_edu_yr["Pendidikan"], x=df_edu_yr["Persen_Bekerja"],
                    orientation="h",
                    marker=dict(
                        color=df_edu_yr["Persen_Bekerja"],
                        colorscale=[[0, "#001D39"], [0.5, "#49769F"], [1, "#75B1D6"]],
                        showscale=True,
                        colorbar=dict(title="%", tickfont=dict(color="#001D39"))
                    ),
                    hovertemplate="%{y}<br>% Bekerja: <b>%{x:.1f}%</b><extra></extra>"
                ))
                fig_pct.update_layout(
                    title=dict(text=f"Persentase Bekerja Terhadap AK per Pendidikan ({tahun_edu_sel})", font=dict(size=12)),
                    xaxis_title="% Bekerja", height=350,
                )
                fig_pct = apply_theme(fig_pct)
                fig_pct.update_xaxes(tickmode="auto", dtick=None)
                chart_container_start()
                st.plotly_chart(fig_pct, use_container_width=True)
                chart_container_end()

        st.markdown("<div class='section-header'>Lapangan Usaha Utama Pekerja D.I. Yogyakarta</div>",
                    unsafe_allow_html=True)
        df_lap_f = df_lapangan[
            (df_lapangan["Tahun"] >= tahun_range[0]) &
            (df_lapangan["Tahun"] <= tahun_range[1])
        ].copy()
        if not df_lap_f.empty:
            lap_cols = ["Pertanian", "Industri", "Konstruksi", "Perdagangan",
                        "Akomodasi Makan", "Jasa Pendidikan", "Jasa Kesehatan",
                        "Transportasi", "Informasi Komunikasi"]
            lap_cols_exist = [c for c in lap_cols if c in df_lap_f.columns]
            df_lap_melt = df_lap_f[["Tahun"] + lap_cols_exist].melt(
                id_vars="Tahun", var_name="Sektor", value_name="Pekerja"
            )
            sektor_colors = [
                "#001D39", "#0A4174", "#49769F", "#4E8EA2",
                "#6EA2B3", "#7BBDE8", "#7EB1D0", "#001D39", "#0A4174"
            ][:len(lap_cols_exist)]
            fig_lap = px.bar(
                df_lap_melt, x="Tahun", y="Pekerja", color="Sektor",
                barmode="stack", color_discrete_sequence=sektor_colors,
                custom_data=["Sektor"],
            )
            fig_lap.update_traces(
                hovertemplate="Tahun: %{x}<br>%{customdata[0]}: <b>%{y:.2f} ribu</b><extra></extra>"
            )
            fig_lap.update_layout(
                title=dict(text="Komposisi Pekerja Menurut Lapangan Usaha (ribu jiwa)", font=dict(size=13)),
                xaxis_title="Tahun", yaxis_title="Ribu Jiwa", height=400,
            )
            fig_lap = apply_theme(fig_lap)
            chart_container_start()
            st.plotly_chart(fig_lap, use_container_width=True)
            chart_container_end()

    #  TAB 3 — CLUSTERING
    with tab3:
        st.markdown("""
        <div class='insight-box' style='margin-bottom:20px;'>
            <div class='insight-title'>Metodologi Analisis Clustering</div>
            <div class='insight-text'>
                Analisis <span class='insight-highlight'>K-Means Clustering</span> digunakan untuk
                mengelompokkan 5 kabupaten/kota di D.I. Yogyakarta berdasarkan profil ketenagakerjaan.
                <br><br>
                <b style='color:#7BBDE8;'>Fitur yang digunakan:</b><br>
                · <span class='insight-highlight'>TPT</span> – Tingkat Pengangguran Terbuka (%)<br>
                · <span class='insight-highlight'>TPAK</span> – Tingkat Partisipasi Angkatan Kerja (%)<br>
                · <span class='insight-highlight'>Pendapatan Informal</span> – Rata-rata pendapatan bersih (Rp/bulan)<br>
                · <span class='insight-highlight'>Rasio Penempatan</span> – Proporsi pencari kerja yang berhasil ditempatkan (%)
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Pilih Tahun Analisis</div>", unsafe_allow_html=True)
        st.markdown("<small style='color:#001D39;'>Pilih tahun spesifik untuk melihat klaster wilayah:</small>",
                    unsafe_allow_html=True)

        tahun_cluster_opts = sorted(
            [t for t in df_tpt["Tahun"].unique()
             if t in df_pendapatan["Tahun"].unique() and t in df_pencaker["Tahun"].unique()]
        )
        col_drop1, col_drop2 = st.columns([1, 3])
        with col_drop1:
            if tahun_cluster_opts:
                tahun_cluster = st.selectbox(
                    "Tahun", options=tahun_cluster_opts,
                    index=len(tahun_cluster_opts) - 1,
                    label_visibility="collapsed", key="cluster_year_select"
                )
            else:
                tahun_cluster = 2023

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
        df_cluster_result, cluster_profile = run_clustering(
            df_tpt, df_pendapatan, df_pencaker, tahun_cluster
        )

        if df_cluster_result is not None:
            st.markdown(f"<div class='section-header'>Hasil Clustering Tahun {tahun_cluster}</div>",
                        unsafe_allow_html=True)
            col_c1, col_c2 = st.columns([1.5, 1])
            with col_c1:
                cluster_colors = {
                    "TPT Rendah": "#7BBDE8",
                    "TPT Sedang": "#4E8EA2",
                    "TPT Tinggi": "#0A4174",
                }
                fig_clust = go.Figure()
                for label, color in cluster_colors.items():
                    mask = df_cluster_result["Label_Cluster"] == label
                    sub  = df_cluster_result[mask]
                    if sub.empty: continue
                    fig_clust.add_trace(go.Scatter(
                        x=sub["PCA_1"], y=sub["PCA_2"],
                        mode="markers+text", name=label,
                        marker=dict(size=30, color=color, opacity=0.88,
                                    line=dict(width=2, color="#FFFFFF")),
                        text=sub["Wilayah"],
                        textposition="top center",
                        textfont=dict(size=11, color="#001D39"),
                        hovertemplate=(
                            "<b>%{text}</b><br>Cluster: " + label +
                            "<br>PCA1: %{x:.3f} | PCA2: %{y:.3f}<extra></extra>"
                        )
                    ))
                fig_clust.update_layout(
                    title=dict(text=f"Peta Profil Klaster Ketenagakerjaan ({tahun_cluster}) · PCA 2D",
                               font=dict(size=13)),
                    xaxis_title="Komponen Utama 1 (PCA)",
                    yaxis_title="Komponen Utama 2 (PCA)", height=400,
                )
                fig_clust = apply_theme(fig_clust)
                fig_clust.update_xaxes(tickmode="auto", dtick=None)
                chart_container_start()
                st.plotly_chart(fig_clust, use_container_width=True)
                chart_container_end()

            with col_c2:
                st.markdown("<div class='section-header'>Keanggotaan Cluster</div>", unsafe_allow_html=True)
                rows = [df_cluster_result.iloc[0:3], df_cluster_result.iloc[3:5]]
                label_color = "#87CEEB"
                for row_data in rows:
                    cols = st.columns(len(row_data))
                    for idx, (_, row) in enumerate(row_data.iterrows()):
                        
                        color_map = {
                            "TPT Rendah": "#4DD0E1", # Cyan muda
                            "TPT Sedang": "#81C784", # Hijau muda
                            "TPT Tinggi": "#FF8A65"  # Oranye/Salmon soft
                        }
                        accent = color_map.get(row["Label_Cluster"], "#7BBDE8")
                   
                        cols[idx].markdown(f"""
                        <div style='background: linear-gradient(135deg, #1A4D7C, #2C5E8F);
                                    border: 1px solid {accent}80; 
                                    border-top: 4px solid {accent};
                                    border-radius: 12px; 
                                    padding: 18px 15px; 
                                    margin-bottom: 15px;
                                    min-height: 180px;
                                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                            <div style='font-size: 18px; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;'>
                                {row['Wilayah']}
                            </div>
                            <div style='font-size: 14px; font-weight: 600; color: {accent}; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px;'>
                                {row['Label_Cluster']}
                            </div>
                            <div style='font-size: 13px; color: #E3F2FD; line-height: 1.6;'>
                                <span style='color: {label_color}; font-weight: 600;'>TPT:</span> <b>{row['TPT']:.2f}%</b><br>
                                <span style='color: {label_color}; font-weight: 600;'>TPAK:</span> <b>{row['TPAK']:.2f}%</b><br>
                                <div style='margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.1);'>
                                    <span style='color: {label_color}; font-weight: 600;'>Pendapatan:</span><br>
                                    <b style='font-size: 14px; color: #FFD54F;'>Rp {row['Pendapatan_Total']:,.0f}</b><br>
                                    <span style='color: {label_color}; font-weight: 600;'>Penempatan:</span> <b>{row['Rasio_Penempatan']:.1f}%</b>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            if cluster_profile is not None and not cluster_profile.empty:
                st.markdown("<div class='section-header'>Profil Rata-rata Tiap Cluster</div>",
                            unsafe_allow_html=True)
                features_radar = ["TPT", "TPAK", "Pendapatan_Total", "Rasio_Penempatan"]
                profile_norm = cluster_profile.copy()
                for f in features_radar:
                    if f in profile_norm.columns:
                        min_v = profile_norm[f].min()
                        max_v = profile_norm[f].max()
                        profile_norm[f] = ((profile_norm[f] - min_v) / (max_v - min_v)) \
                                           if max_v > min_v else 0.5

                label_display = ["TPT (%)", "TPAK (%)", "Pendapatan", "Rasio Penempatan (%)"]
                fig_radar = go.Figure()
                radar_colors_map = {
                    "TPT Rendah": "#7BBDE8",
                    "TPT Sedang": "#4E8EA2",
                    "TPT Tinggi": "#0A4174"
                }
                for _, row in profile_norm.iterrows():
                    vals = [row[f] for f in features_radar if f in row]
                    hex_color = radar_colors_map.get(row["Label_Cluster"], "#49769F")
                    h = hex_color.lstrip('#')
                    r, g, b = tuple(int(h[j:j+2], 16) for j in (0, 2, 4))
                    safe_rgba = f"rgba({r}, {g}, {b}, 0.2)"
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=label_display + [label_display[0]],
                        fill="toself", name=row["Label_Cluster"],
                        line=dict(color=hex_color, width=2),
                        fillcolor=safe_rgba,
                        hovertemplate="%{theta}: <b>%{r:.3f}</b> (ternormalisasi)<extra></extra>"
                    ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor="rgba(255, 255, 255, 0.15)",
                        radialaxis=dict(
                            visible=True, range=[0, 1],
                            gridcolor="rgba(0, 29, 57, 0.15)",
                            tickfont=dict(color="#001D39", size=9)
                        ),
                        angularaxis=dict(tickfont=dict(color="#001D39", size=11))
                    ),
                    showlegend=True,
                    title=dict(text="Radar Profil Cluster (Nilai Ternormalisasi 0–1)",
                               font=dict(size=13, color="#001D39")),
                    height=420,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(
                        bgcolor="rgba(0, 29, 57, 0.82)",
                        bordercolor="rgba(123, 189, 232, 0.35)",
                        borderwidth=1, font=dict(color="#BDD8E9")
                    )
                )
                chart_container_start()
                st.plotly_chart(fig_radar, use_container_width=True)
                chart_container_end()

                st.markdown("<div class='section-header'>Tabel Rata-rata Indikator per Cluster</div>",
                            unsafe_allow_html=True)
                profile_display = cluster_profile.copy()
                profile_display["Pendapatan_Total"] = profile_display["Pendapatan_Total"].apply(
                    lambda x: f"Rp {x:,.0f}"
                )
                profile_display["TPT"]  = profile_display["TPT"].apply(lambda x: f"{x:.2f}%")
                profile_display["TPAK"] = profile_display["TPAK"].apply(lambda x: f"{x:.2f}%")
                profile_display["Rasio_Penempatan"] = profile_display["Rasio_Penempatan"].apply(
                    lambda x: f"{x:.1f}%"
                )
                profile_display.columns = ["Cluster", "TPT", "TPAK",
                                            "Pendapatan Informal (Rp/bln)", "Rasio Penempatan (%)"]
                st.dataframe(profile_display, use_container_width=True, hide_index=True)
        else:
            st.warning(f"Data clustering tidak tersedia untuk tahun {tahun_cluster}. "
                       "Silakan pilih tahun lain.")

    #  TAB 4 — INSIGHT & TEMUAN
    with tab4:
        st.markdown("<div class='section-header'>Insight & Temuan Spesifik Berdasarkan Filter</div>",
                    unsafe_allow_html=True)
        col_i1, col_i2 = st.columns(2)

        if len(wilayah_sel) == 1:
            w_fokus = wilayah_sel[0]
            df_w = df_tpt_f[df_tpt_f["Wilayah"] == w_fokus]
            if not df_w.empty:
                tpt_max_yr  = int(df_w.loc[df_w["TPT"].idxmax(), "Tahun"])
                tpt_max_val = df_w["TPT"].max()
                tpt_min_yr  = int(df_w.loc[df_w["TPT"].idxmin(), "Tahun"])
                tpt_min_val = df_w["TPT"].min()
                tpt_terbaru = df_w[df_w["Tahun"] == df_w["Tahun"].max()]["TPT"].values[0]
            else:
                tpt_max_yr = tpt_max_val = tpt_min_yr = tpt_min_val = tpt_terbaru = 0

            df_pend_w    = df_pendapatan[(df_pendapatan["Wilayah"] == w_fokus)]
            pend_terbaru = df_pend_w[df_pend_w["Tahun"] == df_pend_w["Tahun"].max()]["Pendapatan_Total"].values[0] \
                           if not df_pend_w.empty else 0

            if df_cluster_result is not None and w_fokus in df_cluster_result["Wilayah"].values:
                w_cluster = df_cluster_result[df_cluster_result["Wilayah"] == w_fokus]["Label_Cluster"].values[0]
            else:
                w_cluster = "Data tidak tersedia untuk tahun klaster"

            with col_i1:
                st.markdown(f"""
                <div class='insight-box'>
                    <div class='insight-title'>Dinamika Pengangguran {w_fokus}</div>
                    <div class='insight-text'>
                        Sepanjang periode yang dipilih, tingkat pengangguran (TPT) di <b>{w_fokus}</b>
                        sempat menyentuh angka tertinggi sebesar <span class='insight-highlight'>{tpt_max_val:.2f}% pada {tpt_max_yr}</span>.
                        Kondisi paling optimal terjadi pada tahun <span class='insight-highlight'>{tpt_min_yr}</span>
                        dengan TPT terendah di angka <span class='insight-highlight'>{tpt_min_val:.2f}%</span>.
                        Saat ini, TPT terakhir tercatat di angka <b>{tpt_terbaru:.2f}%</b>.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class='insight-box'>
                    <div class='insight-title'>Posisi Klaster {w_fokus}</div>
                    <div class='insight-text'>
                        Berdasarkan analisis K-Means terakhir, <b>{w_fokus}</b> masuk ke dalam kategori:
                        <br><br>
                        <span class='insight-highlight' style='font-size:15px;'>{w_cluster}</span>
                        <br><br>
                        Hal ini menandakan bahwa wilayah ini membutuhkan intervensi spesifik yang sesuai dengan
                        tingkat partisipasi angkatan kerja dan kemampuan penyerapan pasarnya.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_i2:
                st.markdown(f"""
                <div class='insight-box'>
                    <div class='insight-title'>Karakteristik Pekerja & Upah</div>
                    <div class='insight-text'>
                        Rata-rata pendapatan pekerja informal di wilayah <b>{w_fokus}</b> pada data terbaru
                        adalah sebesar <span class='insight-highlight'>Rp {pend_terbaru:,.0f}/bulan</span>.
                        Angka ini sangat dipengaruhi oleh struktur lapangan usaha utama yang mendominasi di wilayah ini,
                        di mana wilayah dengan dominasi jasa cenderung memiliki basis upah lebih tinggi dibanding wilayah agraris.
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            tpt_min_yr  = int(df_tpt_f.loc[df_tpt_f["TPT"].idxmin(), "Tahun"]) if not df_tpt_f.empty else 0
            tpt_max_yr  = int(df_tpt_f.loc[df_tpt_f["TPT"].idxmax(), "Tahun"]) if not df_tpt_f.empty else 0
            tpt_min_val = df_tpt_f["TPT"].min() if not df_tpt_f.empty else 0
            tpt_max_val = df_tpt_f["TPT"].max() if not df_tpt_f.empty else 0

            max_yr_avail = df_tpt_f["Tahun"].max()
            df_kab_latest = df_tpt_f[df_tpt_f["Tahun"] == max_yr_avail]
            if not df_kab_latest.empty:
                kab_tpt_tinggi = df_kab_latest.sort_values("TPT", ascending=False).iloc[0]
                kab_tpt_rendah = df_kab_latest.sort_values("TPT").iloc[0]
            else:
                kab_tpt_tinggi = kab_tpt_rendah = None

            df_pend_f = df_pendapatan[
                (df_pendapatan["Tahun"] == df_pendapatan["Tahun"].max()) &
                (df_pendapatan["Wilayah"].isin(wilayah_sel))
            ]
            if not df_pend_f.empty:
                pend_max = df_pend_f.sort_values("Pendapatan_Total", ascending=False).iloc[0]
                pend_min = df_pend_f.sort_values("Pendapatan_Total").iloc[0]
            else:
                pend_max = pend_min = None

            with col_i1:
                st.markdown(f"""
                <div class='insight-box'>
                    <div class='insight-title'>Tren Pengangguran Regional</div>
                    <div class='insight-text'>
                        Pada wilayah yang Anda pilih, rekor TPT tertinggi terjadi pada tahun
                        <span class='insight-highlight'>{tpt_max_yr} ({tpt_max_val:.2f}%)</span>,
                        yang bertepatan dengan dampak pandemi. Pasca pandemi, pemulihan pasar kerja
                        sempat menekan TPT hingga <span class='insight-highlight'>{tpt_min_val:.2f}% pada {tpt_min_yr}</span>.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div class='insight-box'>
                    <div class='insight-title'>Insight Clustering</div>
                    <div class='insight-text'>
                        Analisis klaster mengkonfirmasi bahwa wilayah dengan basis ekonomi agraris
                        (seperti Gunung Kidul/Kulon Progo) cenderung menutupi angka pengangguran terbuka
                        karena tingginya partisipasi kerja informal. Sebaliknya, wilayah urban (Kota Yogya/Sleman)
                        punya TPT lebih tinggi namun menawarkan upah yang jauh lebih kompetitif.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_i2:
                if pend_max is not None:
                    st.markdown(f"""
                    <div class='insight-box'>
                        <div class='insight-title'>Ketimpangan Pendapatan Informal</div>
                        <div class='insight-text'>
                            Terdapat kesenjangan signifikan. Pekerja informal di
                            <span class='insight-highlight'>{pend_max['Wilayah']}</span>
                            rata-rata berpendapatan
                            <span class='insight-highlight'>Rp {pend_max['Pendapatan_Total']:,.0f}/bulan</span>,
                            jauh di atas <span class='insight-highlight'>{pend_min['Wilayah']}</span>
                            yang hanya <span class='insight-highlight'>Rp {pend_min['Pendapatan_Total']:,.0f}/bulan</span>.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                if kab_tpt_tinggi is not None:
                    st.markdown(f"""
                    <div class='insight-box'>
                        <div class='insight-title'>Disparitas Pengangguran Antar Wilayah</div>
                        <div class='insight-text'>
                            Dari pilihan Anda pada tahun {max_yr_avail},
                            <span class='insight-highlight'>{kab_tpt_tinggi['Wilayah']}</span>
                            mencatat TPT tertinggi (<b>{kab_tpt_tinggi['TPT']:.2f}%</b>), sementara
                            <span class='insight-highlight'>{kab_tpt_rendah['Wilayah']}</span>
                            memiliki TPT terendah (<b>{kab_tpt_rendah['TPT']:.2f}%</b>).
                            Kesenjangan ini mengindikasikan perlunya kebijakan asimetris.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Rekomendasi Kebijakan</div>", unsafe_allow_html=True)
        reko_cols = st.columns(3)
        rekomendasi = [
            {
                "judul": "Penguatan Bursa Kerja",
                "deskripsi": "Rasio penempatan masih rendah di beberapa wilayah. Perlu optimalisasi job fair berbasis digital dan penguatan Disnaker.",
                "target": "Wilayah dengan TPT Sedang/Tinggi"
            },
            {
                "judul": "Peningkatan Kompetensi",
                "deskripsi": "Pekerja dengan pendidikan SMA ke atas mendominasi penyerapan. Program vokasi DUDI perlu diperluas.",
                "target": "Semua Wilayah"
            },
            {
                "judul": "Formalisasi Sektor Informal",
                "deskripsi": "Upah informal agraris tertinggal jauh. Transformasi pekerja menuju UMKM dan koperasi berbadan hukum sangat mendesak.",
                "target": "Wilayah Agraris (Kulon Progo, Bantul, GK)"
            }
        ]
        for i, reko in enumerate(rekomendasi):
            with reko_cols[i]:
                st.markdown(f"""
                <div class='insight-box' style='height:280px; display:flex; flex-direction:column; justify-content:space-between;'>
                    <div>
                        <div class='insight-title'>{reko['judul']}</div>
                        <div class='insight-text'>{reko['deskripsi']}</div>
                    </div>
                    <div style='margin-top:auto; font-size:11px;
                                background:rgba(78, 142, 162, 0.2);
                                padding:6px 12px; border-radius:8px;
                                color:#7BBDE8; border:1px solid rgba(123, 189, 232, 0.25);'>
                        Target: {reko['target']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Snapshot Indikator Kunci Berdasarkan Filter</div>",
                    unsafe_allow_html=True)
        df_snapshot = df_tpt_f.copy()
        df_snapshot_latest = df_snapshot[df_snapshot["Tahun"] == df_snapshot["Tahun"].max()].copy()
        df_snapshot_latest = df_snapshot_latest.sort_values("TPT", ascending=False)
        pend_latest = df_pendapatan[df_pendapatan["Tahun"] == df_pendapatan["Tahun"].max()][
            ["Wilayah", "Pendapatan_Total"]].copy()
        df_snap = df_snapshot_latest.merge(pend_latest, on="Wilayah", how="left")
        df_snap["TPT"]  = df_snap["TPT"].apply(lambda x: f"{x:.2f}%")
        df_snap["TPAK"] = df_snap["TPAK"].apply(lambda x: f"{x:.2f}%")
        df_snap["Pendapatan_Total"] = df_snap["Pendapatan_Total"].apply(
            lambda x: f"Rp {x:,.0f}" if pd.notna(x) else "N/A"
        )
        df_snap = df_snap.rename(columns={
            "Wilayah": "Kabupaten/Kota",
            "TPT": "TPT (%)",
            "TPAK": "TPAK (%)",
            "Pendapatan_Total": "Pend. Informal (Rp/bln)"
        })[["Kabupaten/Kota", "TPT (%)", "TPAK (%)", "Pend. Informal (Rp/bln)"]]
        st.dataframe(df_snap, use_container_width=True, hide_index=True)

    # ── FOOTER ──
    st.markdown("<hr style='border-color:rgba(0, 29, 57, 0.2); margin-top:32px;'>",
                unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; color:#001D39; font-size:12px; padding:12px 0 24px;'>
        <b>Potret Ketenagakerjaan D.I. Yogyakarta</b><br>
        Sumber Data: Badan Pusat Statistik (BPS) Provinsi D.I. Yogyakarta · Diakses 2025<br>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()