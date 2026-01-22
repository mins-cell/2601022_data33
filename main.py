import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="2024 인구증감 × 의료이용 (Pastel)", page_icon="🌿", layout="wide")

# --- Pastel/minimal styling ---
st.markdown(
    '''
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
    ''',
    unsafe_allow_html=True,
)

FILES = {
    "pop_tidy": "population_change_2024_tidy.csv",
    "merged":   "merged_sido_population_x_hira_2024.csv",
}

MODE = st.sidebar.radio("📦 데이터 불러오기", ["폴더에서 읽기(기본)", "파일 업로드"])

@st.cache_data
def load_local():
    pop_tidy = pd.read_csv(FILES["pop_tidy"])
    merged   = pd.read_csv(FILES["merged"])
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
        f'''
        <div class="kpi-card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>
        ''',
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
merged = merged.copy()
merged["lat"] = merged["sido"].map(lambda x: coords.get(x, (np.nan, np.nan))[0])
merged["lon"] = merged["sido"].map(lambda x: coords.get(x, (np.nan, np.nan))[1])

st.markdown("## 🌿 2024 인구증감 × 의료이용 (Pastel Dashboard)")
st.markdown('<span class="pill">파스텔·미니멀</span><span class="pill">관계 중심</span><span class="pill">지도(버블)</span>', unsafe_allow_html=True)
st.markdown('<div class="hint">관계를 또렷하게 보려면: <b>사분면</b>과 <b>지도</b>를 같이 보세요.</div>', unsafe_allow_html=True)

# --- Sidebar controls ---
st.sidebar.markdown("### 🔎 관계 설정")
x_key = st.sidebar.selectbox("X(인구/구조)", ["pop_change_2024", "pop_avg_2024"], index=0)
y_key = st.sidebar.selectbox("Y(의료이용)", ["amount_per_capita", "patients_per_1k", "amount_per_patient", "total_amount_2024"], index=0)
use_log_y = st.sidebar.checkbox("Y를 log10로 변환", value=False)
split_basis = st.sidebar.radio("사분면 기준", ["중앙값(median)", "평균(mean)"], index=0)

df_rel = merged.copy()
if use_log_y:
    df_rel = df_rel[df_rel[y_key] > 0].copy()
    df_rel[y_key + "_log10"] = np.log10(df_rel[y_key].astype(float))
    y_plot = y_key + "_log10"
    y_label = label[y_key] + " (log10)"
else:
    y_plot = y_key
    y_label = label[y_key]
x_label = label[x_key]

# --- KPIs ---
k1, k2, k3, k4 = st.columns(4)
with k1: kpi_card("시도 수", f"{df_rel['sido'].nunique():,}", "분석 대상")
with k2: kpi_card("인구증감 평균", f"{df_rel['pop_change_2024'].mean():,.0f} 명", "시도 평균")
with k3: kpi_card("1인당 의료비(중앙)", f"{df_rel['amount_per_capita'].median():,.0f} 원/인", "중앙값")
with k4: kpi_card("환자/1천명(중앙)", f"{df_rel['patients_per_1k'].median():,.1f}", "중앙값")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🧭 관계", "🧩 사분면", "🗺️ 지도(버블)", "🌡️ 표준화", "📅 시도 상세"])

with tab1:
    st.markdown('<div class="section-title">산점도 + 회귀선 + r</div>', unsafe_allow_html=True)
    cA, cB = st.columns([2, 1])

    fig = px.scatter(
        df_rel,
        x=x_key,
        y=y_plot,
        hover_name="sido",
        size="pop_avg_2024",
        hover_data={
            "pop_change_2024": True,
            "pop_avg_2024": ":,.0f",
            "patients_per_1k": ":,.1f",
            "amount_per_capita": ":,.0f",
            "amount_per_patient": ":,.0f",
            "total_amount_2024": ":,.0f",
        },
        title=f"{x_label} ↔ {y_label} (버블=평균 인구)",
        template="plotly_white",
    )
    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label, height=520)

    reg = add_simple_regression_line(df_rel, x_key, y_plot)
    if reg is not None:
        a, b, r, line = reg
        fig.add_trace(line)

    with cA:
        st.plotly_chart(fig, use_container_width=True)
    with cB:
        st.markdown('<div class="section-title">요약</div>', unsafe_allow_html=True)
        if reg is not None:
            kpi_card("상관계수 r", f"{r:.2f}", "0에 가까울수록 선형 관계 약함")
            kpi_card("회귀식", f"y = {a:.4g}·x + {b:.4g}", "단순선형(참고용)")
        st.markdown('<div class="section-title">TOP 5 (Y 기준)</div>', unsafe_allow_html=True)
        show_cols = ["sido", x_key, y_key, "patients_per_1k", "amount_per_capita"]
        st.dataframe(merged.sort_values(y_key, ascending=False)[show_cols].head(5), use_container_width=True, height=240)

with tab2:
    st.markdown('<div class="section-title">사분면: “인구↓ / 의료↑” 지역 찾기</div>', unsafe_allow_html=True)
    x_cut = df_rel[x_key].median() if split_basis.startswith("중앙") else df_rel[x_key].mean()
    y_cut = df_rel[y_plot].median() if split_basis.startswith("중앙") else df_rel[y_plot].mean()

    q = df_rel.copy()
    q["quadrant"] = np.select(
        [
            (q[x_key] >= x_cut) & (q[y_plot] >= y_cut),
            (q[x_key] <  x_cut) & (q[y_plot] >= y_cut),
            (q[x_key] <  x_cut) & (q[y_plot] <  y_cut),
            (q[x_key] >= x_cut) & (q[y_plot] <  y_cut),
        ],
        ["Q1 인구↑/의료↑", "Q2 인구↓/의료↑ (관심)", "Q3 인구↓/의료↓", "Q4 인구↑/의료↓"],
        default="",
    )

    fig = px.scatter(
        q, x=x_key, y=y_plot, color="quadrant",
        hover_name="sido", size="pop_avg_2024",
        title="사분면 분류 (기준선=평균/중앙값)",
        template="plotly_white",
    )
    fig.add_vline(x=float(x_cut))
    fig.add_hline(y=float(y_cut))
    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label, height=540)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Q2(인구↓/의료↑) 리스트</div>', unsafe_allow_html=True)
    st.dataframe(
        q[q["quadrant"].str.startswith("Q2")][["sido", x_key, y_plot, "patients_per_1k", "amount_per_capita"]]
        .sort_values(y_plot, ascending=False),
        use_container_width=True,
    )

with tab3:
    st.markdown('<div class="section-title">지도(버블): 값이 큰 지역이 어디인지 한눈에</div>', unsafe_allow_html=True)
    map_metric = st.selectbox("지도 색상 지표", ["amount_per_capita", "patients_per_1k", "pop_change_2024", "amount_per_patient"], index=0)
    size_metric = st.selectbox("버블 크기 지표", ["pop_avg_2024", "total_amount_2024", "patients_2024"], index=0)

    m = merged.dropna(subset=["lat","lon"]).copy()
    fig = px.scatter_mapbox(
        m,
        lat="lat",
        lon="lon",
        size=size_metric,
        color=map_metric,
        hover_name="sido",
        hover_data={
            "pop_change_2024": ":,.0f",
            "pop_avg_2024": ":,.0f",
            "patients_per_1k": ":,.1f",
            "amount_per_capita": ":,.0f",
            "amount_per_patient": ":,.0f",
            "total_amount_2024": ":,.0f",
        },
        zoom=5,
        center={"lat": 36.3, "lon": 127.8},
        height=620,
    )
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=40,b=0), title=f"지도: {label[map_metric]}")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("※ 버블 맵은 시도 대표 좌표(대략)로 표시됩니다. 행정경계(폴리곤) 지도도 원하면 GeoJSON 추가해서 바꿀 수 있어요.")

with tab4:
    st.markdown('<div class="section-title">표준화(z-score) 히트맵</div>', unsafe_allow_html=True)
    z = merged[["sido", "pop_change_2024", "patients_per_1k", "amount_per_capita", "amount_per_patient"]].copy()
    for c in ["pop_change_2024", "patients_per_1k", "amount_per_capita", "amount_per_patient"]:
        sd = z[c].std(ddof=0)
        z[c] = (z[c] - z[c].mean()) / (sd if sd != 0 else 1)

    metric_name = {
        "pop_change_2024": "인구증감(z)",
        "patients_per_1k": "환자/1천명(z)",
        "amount_per_capita": "1인당 의료비(z)",
        "amount_per_patient": "환자1인당 의료비(z)",
    }

    heat = px.imshow(
        z.set_index("sido").rename(columns=metric_name),
        aspect="auto",
        title="시도×지표 z-score (0보다 크면 평균보다 큼)",
        template="plotly_white",
    )
    heat.update_layout(height=560)
    st.plotly_chart(heat, use_container_width=True)

with tab5:
    st.markdown('<div class="section-title">시도 선택 → 월별 인구증감 + 의료 KPI</div>', unsafe_allow_html=True)
    sido_list = merged["sido"].sort_values().unique().tolist()
    selected = st.selectbox("시도", sido_list, index=0)

    row = merged[merged["sido"] == selected].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("2024 인구증감 합계", f"{row['pop_change_2024']:,.0f} 명")
    with c2: kpi_card("1인당 의료비", f"{row['amount_per_capita']:,.0f} 원/인")
    with c3: kpi_card("인구 1천명당 환자수", f"{row['patients_per_1k']:,.1f}")
    with c4: kpi_card("환자 1인당 의료비", f"{row['amount_per_patient']:,.0f} 원/명")

    d = pop_tidy[pop_tidy["sido"] == selected].copy()
    d["month_num"] = d["month"].str.extract(r"2024년(\d+)월").astype(int)
    d = d.sort_values("month_num")

    fig1 = px.line(d, x="month", y="pop_change", markers=True, title=f"{selected} 월별 인구증감(2024)", template="plotly_white")
    fig1.update_layout(xaxis_title="", yaxis_title="인구증감(명)", height=420)
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.line(d, x="month", y="pop_end", markers=True, title=f"{selected} 월말 인구(2024)", template="plotly_white")
    fig2.update_layout(xaxis_title="", yaxis_title="월말 인구(명)", height=420)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.caption("※ 환자수/명세서건수는 의료행위별 통계를 시도 단위로 합산한 값이라 '고유 인원'과 다를 수 있습니다. "
           "지역 비교를 위한 탐색적 지표로 활용하세요.")
