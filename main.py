import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="2024 인구증감 × 의료이용 (Pastel)", page_icon="🌿", layout="wide")

# --- Pastel/minimal styling ---
st.markdown(
    """
    <style>
      :root {
        --bg: #fbfbff;
        --card: rgba(255,255,255,0.75);
        --stroke: rgba(49, 51, 63, 0.14);
      }
      .stApp { background: var(--bg); }
      .block-container { padding-top: 2rem; padding-bottom: 2rem; }
      .kpi-card {
        border: 1px solid var(--stroke);
        border-radius: 18px;
        padding: 14px 16px;
        background: var(--card);
        box-shadow: 0 8px 26px rgba(18, 18, 28, 0.06);
      }
      .kpi-title { font-size: 0.85rem; opacity: 0.78; margin-bottom: 4px; }
      .kpi-value { font-size: 1.6rem; font-weight: 750; line-height: 1.15; }
      .kpi-sub { font-size: 0.8rem; opacity: 0.7; margin-top: 6px; }
      .section-title { font-size: 1.05rem; font-weight: 750; margin: 0.2rem 0 0.6rem; }
      .hint { font-size: 0.92rem; opacity: 0.78; }
      .pill {
        display:inline-block; padding: 3px 10px; border-radius: 999px;
        border: 1px solid var(--stroke); font-size:.78rem; opacity:.85;
        background: rgba(255,255,255,0.6); margin-right: 6px;
      }
      .soft { opacity:.78; }
    </style>
    """,
    unsafe_allow_html=True,
)

FILES = {
    "pop_tidy": "population_change_2024_tidy.csv",
    "merged": "merged_sido_population_x_hira_2024.csv",
}

MODE = st.sidebar.radio("📦 데이터 불러오기", ["폴더에서 읽기(기본)", "파일 업로드"])


@st.cache_data
def load_local():
    pop_tidy = pd.read_csv(FILES["pop_tidy"])
    merged = pd.read_csv(FILES["merged"])
    return pop_tidy, merged


def load_upload():
    f1 = st.sidebar.file_uploader(f"업로드: {FILES['pop_tidy']}", type=["csv"], key=FILES["pop_tidy"])
    f2 = st.sidebar.file_uploader(f"업로드: {FILES['merged']}", type=["csv"], key=FILES["merged"])
    if (f1 is None) or (f2 is None):
        st.sidebar.info("⬆️ 업로드 모드에서는 2개 CSV를 모두 올려야 해요.")
        return None
    return pd.read_csv(f1), pd.read_csv(f2)


if MODE.startswith("폴더"):
    pop_tidy, merged = load_local()
else:
    loaded = load_upload()
    if loaded is None:
        st.stop()
    pop_tidy, merged = loaded

# ✅ 핵심 방어: 중복 컬럼명 제거 (Streamlit/pyarrow + pandas sort 에러 예방)
merged = merged.loc[:, ~merged.columns.duplicated()].copy()


def add_simple_regression_line(df, x, y):
    d = df[[x, y]].dropna()
    if len(d) < 2:
        return None
    xv = d[x].astype(float).values
    yv = d[y].astype(float).values
    a, b = np.polyfit(xv, yv, 1)
    r = float(np.corrcoef(xv, yv)[0, 1])
    xs = np.array([float(xv.min()), float(xv.max())])
    ys = a * xs + b
    line = go.Scatter(x=xs, y=ys, mode="lines", name=f"회귀선 (r={r:.2f})")
    return a, b, r, line


def kpi_card(title, value, sub=""):
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


label = {
    "pop_change_2024": "2024 인구증감(명)",
    "pop_avg_2024": "2024 평균 인구(명)",
    "patients_per_1k": "인구 1천명당 환자수",
    "amount_per_capita": "1인당 의료비(원/인)",
    "amount_per_patient": "환자 1인당 의료비(원/명)",
    "total_amount_2024": "총 의료비(원)",
}

# --- Approximate representative coordinates (centers / major city) ---
coords = {
    "서울특별시": (37.5665, 126.9780),
    "부산광역시": (35.1796, 129.0756),
    "대구광역시": (35.8714, 128.6014),
    "인천광역시": (37.4563, 126.7052),
    "광주광역시": (35.1595, 126.8526),
    "대전광역시": (36.3504, 127.3845),
    "울산광역시": (35.5384, 129.3114),
    "세종특별자치시": (36.4801, 127.2890),
    "경기도": (37.4138, 127.5183),
    "강원특별자치도": (37.8228, 128.1555),
    "충청북도": (36.6357, 127.4917),
    "충청남도": (36.5184, 126.8000),
    "전북특별자치도": (35.7175, 127.1530),
    "전라남도": (34.8161, 126.4629),
    "경상북도": (36.4919, 128.8889),
    "경상남도": (35.4606, 128.2132),
    "제주특별자치도": (33.4996, 126.5312),
}

# ensure coordinates exist
merged["lat"] = merged["sido"].map(lambda x: coords.get(x, (np.nan, np.nan))[0])
merged["lon"] = merged["sido"].map(lambda x: coords.get(x, (np.nan, np.nan))[1])

st.markdown("## 🌿 2024 인구증감 × 의료이용 (Pastel Dashboard)")
st.markd

