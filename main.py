
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="2023–2024 인구·의료 비교", page_icon="📊", layout="wide")

# --- Clean UI styles (kept simple) ---
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
      .card {
        border: 1px solid var(--stroke);
        border-radius: 18px;
        padding: 14px 16px;
        background: var(--card);
        box-shadow: 0 8px 26px rgba(18, 18, 28, 0.06);
      }
      .card-title { font-size: 0.9rem; opacity: 0.78; margin-bottom: 6px; }
      .card-value { font-size: 1.55rem; font-weight: 750; line-height: 1.15; }
      .card-sub { font-size: 0.8rem; opacity: 0.7; margin-top: 6px; }
      .section-title { font-size: 1.05rem; font-weight: 750; margin: 0.2rem 0 0.6rem; }
      .hint { font-size: 0.92rem; opacity: 0.78; }
      .pill {
        display:inline-block; padding: 3px 10px; border-radius: 999px;
        border: 1px solid var(--stroke); font-size:.78rem; opacity:.85;
        background: rgba(255,255,255,0.6); margin-right: 6px;
      }
      .small { font-size: .85rem; opacity: .78; }
    </style>
    ''',
    unsafe_allow_html=True,
)

FILES = {
    "wide": "compare_2023_2024_wide.csv",
    "long": "compare_2023_2024_long.csv",
    "pop_month": "population_month_tidy_2023_2024.csv",
}

MODE = st.sidebar.radio("데이터 불러오기", ["폴더에서 읽기(기본)", "파일 업로드"])

@st.cache_data
def load_local():
    wide = pd.read_csv(FILES["wide"])
    long = pd.read_csv(FILES["long"])
    pop_month = pd.read_csv(FILES["pop_month"])
    return wide, long, pop_month

def load_upload():
    f_wide = st.sidebar.file_uploader(f"업로드: {FILES['wide']}", type=["csv"], key="wide")
    f_long = st.sidebar.file_uploader(f"업로드: {FILES['long']}", type=["csv"], key="long")
    f_popm = st.sidebar.file_uploader(f"업로드: {FILES['pop_month']}", type=["csv"], key="pop_month")
    if (f_wide is None) or (f_long is None) or (f_popm is None):
        st.sidebar.info("업로드 모드에서는 CSV 3개를 모두 올려야 해요.")
        return None
    return pd.read_csv(f_wide), pd.read_csv(f_long), pd.read_csv(f_popm)

if MODE.startswith("폴더"):
    wide, long, pop_month = load_local()
else:
    loaded = load_upload()
    if loaded is None:
        st.stop()
    wide, long, pop_month = loaded

# ✅ safety: remove duplicate columns if any
wide = wide.loc[:, ~wide.columns.duplicated()].copy()
long = long.loc[:, ~long.columns.duplicated()].copy()
pop_month = pop_month.loc[:, ~pop_month.columns.duplicated()].copy()

# --- Helpers ---
def card(title, value, sub=""):
    st.markdown(
        f'''
        <div class="card">
          <div class="card-title">{title}</div>
          <div class="card-value">{value}</div>
          <div class="card-sub">{sub}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

def add_reg_line(df, x, y):
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

def uniq_cols(cols):
    return list(dict.fromkeys(cols))

# --- Labels ---
THEMES = {
    "delta_pop_change": ("인구증감 변화", "2024 - 2023 (명)"),
    "delta_patients_per_1k": ("인구 1천명당 환자수 변화", "2024 - 2023"),
    "delta_amount_per_capita": ("1인당 의료비 변화", "2024 - 2023 (원)"),
    "delta_amount_total": ("총 의료비 변화", "2024 - 2023 (원)"),
}

# --- Header ---
st.markdown("## 2023–2024 시도별 인구증감과 의료이용 비교")
st.markdown('<span class="pill">연도 비교</span><span class="pill">시도 단위</span><span class="pill">관계 분석</span>', unsafe_allow_html=True)
st.markdown(
    '<div class="hint">메인 화면에서 테마별 상위 지역을 빠르게 확인하고, '
    '탭에서 관계(산점도/사분면), 지도, 시도별 월별 추이를 더 자세히 볼 수 있어요.</div>',
    unsafe_allow_html=True
)

# --- Top cards (overall KPIs) ---
c1, c2, c3, c4 = st.columns(4)
with c1:
    card("시도 수", f"{wide['sido'].nunique():,}", "분석 대상 지역 수")
with c2:
    card("인구증감 변화(평균)", f"{wide['delta_pop_change'].mean():,.0f}", "2024 - 2023 평균")
with c3:
    card("1인당 의료비 변화(중앙값)", f"{wide['delta_amount_per_capita'].median():,.0f}", "2024 - 2023 중앙값")
with c4:
    card("환자수/1천명 변화(중앙값)", f"{wide['delta_patients_per_1k'].median():,.1f}", "2024 - 2023 중앙값")

st.markdown("---")

tab_home, tab_rel, tab_quad, tab_map, tab_detail = st.tabs(
    ["🏠 메인", "📈 관계", "🧭 사분면", "🗺️ 지도", "📅 시도 상세"]
)

# ---------------- Home: top 5 per theme ----------------
with tab_home:
    st.markdown('<div class="section-title">테마별 상위 5개 지역</div>', unsafe_allow_html=True)
    st.markdown('<div class="small">기준: 각 테마의 변화량(2024 − 2023)이 큰 순서</div>', unsafe_allow_html=True)

    cols = st.columns(2)
    theme_keys = list(THEMES.keys())

    # show in a grid of cards with mini tables
    grid = st.columns(2)
    for i, key in enumerate(theme_keys):
        title, sub = THEMES[key]
        target_col = grid[i % 2]
        with target_col:
            top5 = wide.sort_values(key, ascending=False)[["sido", key]].head(5).copy()
            fig = px.bar(top5.sort_values(key, ascending=True), x=key, y="sido", orientation="h",
                         title=f"{title} (상위 5)", template="plotly_white")
            fig.update_layout(height=320, margin=dict(l=10,r=10,t=40,b=10), xaxis_title=sub, yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">전체 표</div>', unsafe_allow_html=True)
    st.dataframe(wide.sort_values("delta_amount_per_capita", ascending=False), use_container_width=True)

# ---------------- Relationship ----------------
with tab_rel:
    st.markdown('<div class="section-title">변화량 간 관계 (산점도 + 회귀선)</div>', unsafe_allow_html=True)
    x_key = st.selectbox("X 축", ["delta_pop_change", "delta_patients_per_1k"], index=0)
    y_key = st.selectbox("Y 축", ["delta_amount_per_capita", "delta_amount_total"], index=0)

    x_label = THEMES.get(x_key, (x_key, ""))[0]
    y_label = THEMES.get(y_key, (y_key, ""))[0]

    df = wide.copy()

    fig = px.scatter(df, x=x_key, y=y_key, hover_name="sido",
                     title=f"{x_label} ↔ {y_label}", template="plotly_white")
    reg = add_reg_line(df, x_key, y_key)
    if reg is not None:
        a, b, r, line = reg
        fig.add_trace(line)
        st.caption(f"상관계수 r = {r:.2f} (단순선형 기준)")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Quadrants ----------------
with tab_quad:
    st.markdown('<div class="section-title">사분면으로 관심 지역 찾기</div>', unsafe_allow_html=True)
    st.markdown('<div class="small">기준선: 선택한 분할 기준(중앙값/평균)</div>', unsafe_allow_html=True)

    x_key = st.selectbox("X(분할)", ["delta_pop_change", "delta_patients_per_1k"], index=0, key="q_x")
    y_key = st.selectbox("Y(분할)", ["delta_amount_per_capita", "delta_amount_total"], index=0, key="q_y")
    basis = st.radio("분할 기준", ["중앙값(median)", "평균(mean)"], index=0, horizontal=True)

    df = wide.copy()
    x_cut = df[x_key].median() if basis.startswith("중앙") else df[x_key].mean()
    y_cut = df[y_key].median() if basis.startswith("중앙") else df[y_key].mean()

    df["quadrant"] = np.select(
        [
            (df[x_key] >= x_cut) & (df[y_key] >= y_cut),
            (df[x_key] <  x_cut) & (df[y_key] >= y_cut),
            (df[x_key] <  x_cut) & (df[y_key] <  y_cut),
            (df[x_key] >= x_cut) & (df[y_key] <  y_cut),
        ],
        ["Q1: X↑/Y↑", "Q2: X↓/Y↑", "Q3: X↓/Y↓", "Q4: X↑/Y↓"],
        default="",
    )

    fig = px.scatter(df, x=x_key, y=y_key, color="quadrant", hover_name="sido",
                     title="사분면 분류", template="plotly_white")
    fig.add_vline(x=float(x_cut))
    fig.add_hline(y=float(y_cut))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Q2: X↓ / Y↑ 지역</div>', unsafe_allow_html=True)
    cols = uniq_cols(["sido", x_key, y_key, "delta_amount_per_capita", "delta_patients_per_1k", "delta_pop_change"])
    q2 = df[df["quadrant"].str.startswith("Q2")][cols].copy()
    st.dataframe(q2.sort_values(y_key, ascending=False), use_container_width=True)

# ---------------- Map ----------------
with tab_map:
    st.markdown('<div class="section-title">지역 분포 보기</div>', unsafe_allow_html=True)
    st.markdown('<div class="small">지도는 시도 대표 좌표(대략)에 표시됩니다.</div>', unsafe_allow_html=True)

    # Representative coordinates (approx)
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

    df = wide.copy()
    df["lat"] = df["sido"].map(lambda x: coords.get(x, (np.nan, np.nan))[0])
    df["lon"] = df["sido"].map(lambda x: coords.get(x, (np.nan, np.nan))[1])
    df = df.dropna(subset=["lat","lon"]).copy()

    color_metric = st.selectbox("색상 지표", list(THEMES.keys()), index=2)
    size_metric = st.selectbox("크기 지표", ["pop_avg_year_2024", "amount_year_2024", "patients_year_2024"], index=0)

    fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        color=color_metric,
        size=size_metric,
        hover_name="sido",
        hover_data={
            "delta_pop_change": ":,.0f",
            "delta_patients_per_1k": ":,.1f",
            "delta_amount_per_capita": ":,.0f",
            "delta_amount_total": ":,.0f",
        },
        zoom=5,
        center={"lat": 36.3, "lon": 127.8},
        height=640,
    )
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=40,b=0),
                      title=f"지도: {THEMES[color_metric][0]}")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Detail ----------------
with tab_detail:
    st.markdown('<div class="section-title">시도별 월별 인구증감(2023/2024)</div>', unsafe_allow_html=True)
    sido_list = sorted(pop_month["sido"].unique().tolist())
    selected = st.selectbox("시도 선택", sido_list, index=0)

    d = pop_month[pop_month["sido"]==selected].copy()
    d["month_num"] = d["month"].str.extract(r"(\d{4})년(\d{2})월").apply(lambda x: int(x[0])*100 + int(x[1]), axis=1)
    d = d.sort_values("month_num")

    fig = px.line(d, x="month", y="pop_change", color="year", markers=True,
                  title=f"{selected} 월별 인구증감", template="plotly_white")
    fig.update_layout(xaxis_title="", yaxis_title="인구증감(명)", height=420)
    st.plotly_chart(fig, use_container_width=True)

    # show key deltas for selected
    row = wide[wide["sido"]==selected].iloc[0]
    c1, c2, c3 = st.columns(3)
    with c1:
        card("인구증감 변화", f"{row['delta_pop_change']:,.0f}", "2024 - 2023")
    with c2:
        card("환자수/1천명 변화", f"{row['delta_patients_per_1k']:,.1f}", "2024 - 2023")
    with c3:
        card("1인당 의료비 변화", f"{row['delta_amount_per_capita']:,.0f}", "2024 - 2023")

st.markdown("---")
st.caption("※ 환자수/명세서건수는 의료행위별 통계를 시도 단위로 합산한 값이라 '고유 인원'과 다를 수 있습니다. "
           "비교·탐색 목적의 지표로 활용하세요.")
