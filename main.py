# main.py
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import pydeck as pdk
import json
import requests


st.set_page_config(page_title="지역별 의료행위 × 인구증감 대시보드", layout="wide")

# -----------------------------
# Utilities
# -----------------------------
def read_csv_auto(file) -> pd.DataFrame:
    """Try common Korean CSV encodings automatically."""
    # Streamlit UploadedFile supports getvalue(); read bytes into buffer via pandas
    raw = file.getvalue()
    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            return pd.read_csv(pd.io.common.BytesIO(raw), encoding=enc)
        except Exception:
            continue
    # last resort: let pandas guess
    return pd.read_csv(pd.io.common.BytesIO(raw))

def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")

def normalize_sido_from_pop(행정구역: str) -> str:
    """
    Convert population '행정구역' like '서울특별시  (1100000000)' -> '서울'
    and match HIRA sido labels: 서울, 부산, ... 경기, 강원, 충북, ...
    """
    if pd.isna(행정구역):
        return np.nan
    # remove code in parentheses
    name = re.sub(r"\s*\(.*?\)\s*", "", str(행정구역)).strip()
    name = re.sub(r"\s+", " ", name)

    # remove suffixes
    name = (
        name.replace("특별시", "")
            .replace("광역시", "")
            .replace("특별자치시", "")
            .replace("특별자치도", "")
            .replace("자치도", "")
            .replace("도", "")
    ).strip()

    # match HIRA abbreviations
    mapping = {
        "서울": "서울",
        "부산": "부산",
        "대구": "대구",
        "인천": "인천",
        "광주": "광주",
        "대전": "대전",
        "울산": "울산",
        "세종": "세종",
        "경기": "경기",
        "강원": "강원",
        "충청북": "충북",
        "충청남": "충남",
        "전북": "전북",
        "전라북": "전북",
        "전남": "전남",
        "전라남": "전남",
        "경북": "경북",
        "경상북": "경북",
        "경남": "경남",
        "경상남": "경남",
        "제주": "제주",
    }
    return mapping.get(name, name)

# Rough centroids for a scatter map (approx.)
SIDO_CENTROIDS = {
    "서울": (37.5665, 126.9780),
    "부산": (35.1796, 129.0756),
    "대구": (35.8714, 128.6014),
    "인천": (37.4563, 126.7052),
    "광주": (35.1595, 126.8526),
    "대전": (36.3504, 127.3845),
    "울산": (35.5384, 129.3114),
    "세종": (36.4800, 127.2890),
    "경기": (37.4138, 127.5183),
    "강원": (37.8228, 128.1555),
    "충북": (36.6357, 127.4917),
    "충남": (36.6588, 126.6728),
    "전북": (35.7175, 127.1530),
    "전남": (34.8161, 126.4629),
    "경북": (36.4919, 128.8889),
    "경남": (35.4606, 128.2132),
    "제주": (33.4996, 126.5312),
}

# -----------------------------
# Load & preprocess: Population
# -----------------------------
def preprocess_population(pop_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Expected columns like:
    - 행정구역
    - 2025년12월_전월인구수_계
    - 2025년12월_당월인구수_계
    - 2025년12월_인구증감_계
    """
    df = pop_raw.copy()

    if "행정구역" not in df.columns:
        raise ValueError("인구 데이터에 '행정구역' 컬럼이 없습니다.")

    df["시도"] = df["행정구역"].apply(normalize_sido_from_pop)

    # identify month prefix: e.g., '2025년12월_'
    month_prefix = None
    for c in df.columns:
        m = re.match(r"(\d{4}년\d{1,2}월)_", str(c))
        if m:
            month_prefix = m.group(1)
            break
    if not month_prefix:
        raise ValueError("인구 데이터에서 'YYYY년M월_' 형태의 컬럼 접두어를 찾지 못했습니다.")

    # target columns (계)
    prev_col = f"{month_prefix}_전월인구수_계"
    curr_col = f"{month_prefix}_당월인구수_계"
    diff_col = f"{month_prefix}_인구증감_계"

    missing = [c for c in [prev_col, curr_col, diff_col] if c not in df.columns]
    if missing:
        raise ValueError(f"인구 데이터에 필요한 컬럼이 없습니다: {missing}")

    df["전월인구"] = to_numeric_safe(df[prev_col])
    df["당월인구"] = to_numeric_safe(df[curr_col])
    df["인구증감"] = to_numeric_safe(df[diff_col])
    df["인구증감률(%)"] = np.where(df["전월인구"] > 0, (df["인구증감"] / df["전월인구"]) * 100, np.nan)

    # parse year/month
    ym = re.match(r"(\d{4})년(\d{1,2})월", month_prefix)
    year = int(ym.group(1)) if ym else None
    month = int(ym.group(2)) if ym else None
    df["인구기준연도"] = year
    df["인구기준월"] = month

    # drop 전국 row if exists (optional)
    df = df[df["시도"].notna()].copy()
    df = df[df["시도"] != "전국"].copy()

    # add lat/lon
    df["lat"] = df["시도"].map(lambda x: SIDO_CENTROIDS.get(x, (np.nan, np.nan))[0])
    df["lon"] = df["시도"].map(lambda x: SIDO_CENTROIDS.get(x, (np.nan, np.nan))[1])

    return df[["시도", "전월인구", "당월인구", "인구증감", "인구증감률(%)", "인구기준연도", "인구기준월", "lat", "lon"]]

# -----------------------------
# Load & preprocess: HIRA medical acts
# -----------------------------
def preprocess_hira(hira_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Expected columns:
    진료년도, 시도, 행위코드, 환자수, 명세서건수, 의료행위총사용량, 의료행위청구금액
    """
    required = ["진료년도", "시도", "행위코드", "환자수", "명세서건수", "의료행위총사용량", "의료행위청구금액"]
    missing = [c for c in required if c not in hira_raw.columns]
    if missing:
        raise ValueError(f"심평원 데이터에 필요한 컬럼이 없습니다: {missing}")

    df = hira_raw.copy()
    # numeric
    for c in ["환자수", "명세서건수", "의료행위총사용량", "의료행위청구금액"]:
        df[c] = to_numeric_safe(df[c])
    df["진료년도"] = to_numeric_safe(df["진료년도"]).astype("Int64")
    df["시도"] = df["시도"].astype(str).str.strip()

    return df

def aggregate_hira_by_sido(hira_df: pd.DataFrame, year: int, fillna_zero: bool=True) -> pd.DataFrame:
    df = hira_df[hira_df["진료년도"] == year].copy()
    if fillna_zero:
        df[["환자수", "명세서건수", "의료행위총사용량", "의료행위청구금액"]] = df[["환자수", "명세서건수", "의료행위총사용량", "의료행위청구금액"]].fillna(0)

    agg = df.groupby("시도", as_index=False).agg(
        환자수=("환자수", "sum"),
        명세서건수=("명세서건수", "sum"),
        의료행위총사용량=("의료행위총사용량", "sum"),
        의료행위청구금액=("의료행위청구금액", "sum"),
        행위코드종류수=("행위코드", "nunique"),
    )
    agg["진료년도"] = year
    return agg

# -----------------------------
# App UI
# -----------------------------
st.title("📍 지역별 의료행위(심평원) × 인구증감(주민등록) 대시보드")

with st.sidebar:
    st.header("1) 파일 업로드")
    hira_file = st.file_uploader("심평원 의료행위 CSV 업로드", type=["csv"])
    pop_file = st.file_uploader("주민등록 인구증감 CSV 업로드", type=["csv"])

    st.divider()
    st.header("2) 설정")
    fillna_zero = st.checkbox("결측치를 0으로 처리(권장)", value=True)
    top_n = st.slider("Top N (표/랭킹)", 5, 30, 15)

if not hira_file or not pop_file:
    st.info("왼쪽 사이드바에서 **심평원 CSV**와 **인구증감 CSV**를 업로드하면 대시보드가 생성됩니다.")
    st.stop()

# Load data
try:
    hira_raw = read_csv_auto(hira_file)
    pop_raw = read_csv_auto(pop_file)
except Exception as e:
    st.error(f"CSV 로딩 오류: {e}")
    st.stop()

# Preprocess
try:
    hira = preprocess_hira(hira_raw)
    pop = preprocess_population(pop_raw)
except Exception as e:
    st.error(f"전처리 오류: {e}")
    st.stop()

# Year selector from HIRA
years = sorted([int(y) for y in hira["진료년도"].dropna().unique()])
default_year = years[-1] if years else 2024

colA, colB = st.columns([1, 2])
with colA:
    year = st.selectbox("진료년도 선택", options=years, index=years.index(default_year) if default_year in years else 0)
with colB:
    st.caption("※ 인구 데이터는 업로드된 파일의 월(예: 2025년 12월)을 기준으로 계산됩니다. (전월↔당월 증감률)")

# Aggregate HIRA by sido
hira_sido = aggregate_hira_by_sido(hira, year=year, fillna_zero=fillna_zero)

# Merge with population
merged = hira_sido.merge(pop, on="시도", how="left")

# Per-capita metrics (per 10,000 people, using '당월인구')
merged["인구1만명당_총사용량"] = np.where(merged["당월인구"] > 0, (merged["의료행위총사용량"] / merged["당월인구"]) * 10000, np.nan)
merged["인구1만명당_청구금액"] = np.where(merged["당월인구"] > 0, (merged["의료행위청구금액"] / merged["당월인구"]) * 10000, np.nan)
merged["환자당_청구금액"] = np.where(merged["환자수"] > 0, merged["의료행위청구금액"] / merged["환자수"], np.nan)

# National average index (standardized)
national_avg = np.nanmean(merged["인구1만명당_총사용량"])
merged["표준화지수(총사용량)"] = merged["인구1만명당_총사용량"] / national_avg if national_avg and not np.isnan(national_avg) else np.nan

# Filters
all_sidos = [s for s in merged["시도"].dropna().unique().tolist()]
sel_sidos = st.multiselect("표시할 시도 선택(미선택 시 전체)", options=all_sidos, default=all_sidos)

view = merged[merged["시도"].isin(sel_sidos)].copy()

# -----------------------------
# KPI row
# -----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("시도 수", f"{view['시도'].nunique()}개")
k2.metric("의료행위 청구금액 합계", f"{view['의료행위청구금액'].sum():,.0f}")
k3.metric("인구 1만명당 총사용량(평균)", f"{np.nanmean(view['인구1만명당_총사용량']):,.2f}")
k4.metric("인구 증감률(평균, %)", f"{np.nanmean(view['인구증감률(%)']):,.3f}")

st.divider()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🫧 버블(인구증감 × 의료이용)", "🗺️ 지도(Scatter map)", "📋 랭킹/테이블"])

# 1) Bubble
with tab1:
    metric_choice = st.radio(
        "Y축 지표 선택",
        ["인구1만명당_총사용량", "인구1만명당_청구금액", "환자당_청구금액"],
        horizontal=True
    )

    bubble = view.copy()
    bubble["버블크기(인구)"] = bubble["당월인구"]
    bubble["라벨"] = bubble["시도"]

    fig = px.scatter(
        bubble,
        x="인구증감률(%)",
        y=metric_choice,
        size="버블크기(인구)",
        hover_name="라벨",
        hover_data={
            "당월인구": ":,",
            "의료행위총사용량": ":,",
            "의료행위청구금액": ":,",
            "인구1만명당_총사용량": ":.2f",
            "인구1만명당_청구금액": ":.2f",
            "환자당_청구금액": ":.2f",
        },
        labels={
            "인구증감률(%)": "인구 증감률(%) (전월→당월)",
            metric_choice: metric_choice,
        },
        title="인구 변화 vs 의료이용(인구보정) — 버블 크기=인구"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("해석 팁: 좌상단(인구↓, 의료이용↑)은 고령화/만성질환/공급구조 등의 가능성을 시사할 수 있어요.")

# 2) Map (scatter)
with tab2:
    map_metric = st.selectbox(
        "지도에서 색으로 표현할 지표",
        ["인구1만명당_총사용량", "인구1만명당_청구금액", "인구증감률(%)", "표준화지수(총사용량)"]
    )
    map_df = view.dropna(subset=["lat", "lon"]).copy()

    # Normalize for radius
    pop_max = np.nanmax(map_df["당월인구"]) if len(map_df) else 1
    map_df["radius"] = np.where(map_df["당월인구"].notna(), (map_df["당월인구"] / pop_max) * 80000 + 20000, 30000)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[lon, lat]",
        get_radius="radius",
        get_fill_color="[200, 30, 0, 140]",  # fixed color; metric shown via tooltip + optional legend in table
        pickable=True,
    )

    tooltip = {
        "html": """
        <b>{시도}</b><br/>
        인구(당월): {당월인구}<br/>
        인구증감률(%): {인구증감률(%)}
        <hr/>
        인구1만명당 총사용량: {인구1만명당_총사용량}<br/>
        인구1만명당 청구금액: {인구1만명당_청구금액}<br/>
        표준화지수(총사용량): {표준화지수(총사용량)}
        """,
        "style": {"backgroundColor": "white", "color": "black"},
    }

    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(latitude=36.3, longitude=127.8, zoom=6),
            layers=[layer],
            tooltip=tooltip,
        ),
        use_container_width=True,
    )

    st.info(
        "지도는 시도 중심점(대략 좌표) 기반 Scatter map 입니다. "
        "정확한 행정경계 채색(choropleth)을 원하면 '시도 GeoJSON'을 추가로 붙여서 확장할 수 있어요."
    )

    # Show a colored table to reflect chosen metric
    show_cols = ["시도", "당월인구", "인구증감률(%)", "인구1만명당_총사용량", "인구1만명당_청구금액", "표준화지수(총사용량)"]
    st.dataframe(
        map_df[show_cols].sort_values(map_metric, ascending=False).head(top_n),
        use_container_width=True
    )

# 3) Ranking/table
with tab3:
    rank_metric = st.selectbox(
        "랭킹 기준",
        ["의료행위청구금액", "의료행위총사용량", "명세서건수", "환자수", "인구1만명당_총사용량", "인구1만명당_청구금액", "환자당_청구금액"]
    )
    ranked = view.sort_values(rank_metric, ascending=False).copy()

    st.subheader(f"Top {top_n} 시도 — {rank_metric}")
    show_cols = [
        "시도", "진료년도",
        "당월인구", "인구증감", "인구증감률(%)",
        "환자수", "명세서건수", "의료행위총사용량", "의료행위청구금액",
        "인구1만명당_총사용량", "인구1만명당_청구금액", "환자당_청구금액",
        "행위코드종류수", "표준화지수(총사용량)"
    ]
    st.dataframe(ranked[show_cols].head(top_n), use_container_width=True)

    csv = ranked[show_cols].to_csv(index=False).encode("utf-8-sig")
    st.download_button("집계 테이블 CSV 다운로드", data=csv, file_name=f"merged_hira_pop_{year}.csv", mime="text/csv")

st.caption("ⓘ 환자수는 행위코드별 중복 집계 가능성이 있어, 지역 비교는 '총사용량/청구금액/인구보정 지표' 중심을 권장합니다.")
