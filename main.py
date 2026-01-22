# app.py
# Streamlit: 지역별 의료행위(심평원) × 인구증감(주민등록) + 시도 경계 지도(Choropleth)
# 실행: streamlit run app.py

import re
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="지역별 의료행위 × 인구증감 대시보드", layout="wide")


# -----------------------------
# CSV loader (auto-encoding)
# -----------------------------
def read_csv_from_path(path: str) -> pd.DataFrame:
    # 로컬 파일은 보통 utf-8-sig 또는 cp949가 많아서 둘 다 시도
    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)



def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def normalize_sido_from_pop(행정구역: str) -> str:
    """
    예: '서울특별시  (1100000000)' -> '서울'
    심평원 시도 표기(서울, 부산, ... 충북, 충남 등)와 매칭되도록 단순화
    """
    if pd.isna(행정구역):
        return np.nan

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
        "전라북": "전북",
        "전라남": "전남",
        "경상북": "경북",
        "경상남": "경남",
        "제주": "제주",
    }
    return mapping.get(name, name)


# -----------------------------
# GeoJSON (no file needed)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_korea_sido_geojson():
    """
    시도(1단계 행정구역) 경계 GeoJSON을 웹에서 자동으로 가져옵니다.
    네트워크 제한/실패 시 예외를 던지며, 호출부에서 fallback 처리합니다.
    """
    url = "https://simplemaps.com/static/svg/country/kr/admin1/kr.json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


# Simplemaps admin1 id ↔ 시도명 매핑
GEO_ID_BY_SIDO = {
    "서울": "KR11",
    "부산": "KR26",
    "대구": "KR27",
    "인천": "KR28",
    "광주": "KR29",
    "대전": "KR30",
    "울산": "KR31",
    "경기": "KR41",
    "강원": "KR42",
    "충북": "KR43",
    "충남": "KR44",
    "전북": "KR45",
    "전남": "KR46",
    "경북": "KR47",
    "경남": "KR48",
    "제주": "KR49",
    "세종": "KR50",
}


# (fallback용) 시도 중심점 좌표(대략)
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
# Population preprocessing
# -----------------------------
def preprocess_population(pop_raw: pd.DataFrame) -> pd.DataFrame:
    """
    기대 컬럼(예):
    - 행정구역
    - 2025년12월_전월인구수_계
    - 2025년12월_당월인구수_계
    - 2025년12월_인구증감_계
    """
    df = pop_raw.copy()

    if "행정구역" not in df.columns:
        raise ValueError("인구 데이터에 '행정구역' 컬럼이 없습니다.")

    df["시도"] = df["행정구역"].apply(normalize_sido_from_pop)

    # detect month prefix 'YYYY년M월_'
    month_prefix = None
    for c in df.columns:
        m = re.match(r"(\d{4}년\d{1,2}월)_", str(c))
        if m:
            month_prefix = m.group(1)
            break
    if not month_prefix:
        raise ValueError("인구 데이터에서 'YYYY년M월_' 형태의 컬럼 접두어를 찾지 못했습니다.")

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

    ym = re.match(r"(\d{4})년(\d{1,2})월", month_prefix)
    df["인구기준연도"] = int(ym.group(1)) if ym else np.nan
    df["인구기준월"] = int(ym.group(2)) if ym else np.nan

    # remove 전국 row if exists
    df = df[df["시도"].notna()].copy()
    df = df[df["시도"] != "전국"].copy()

    # fallback lat/lon
    df["lat"] = df["시도"].map(lambda x: SIDO_CENTROIDS.get(x, (np.nan, np.nan))[0])
    df["lon"] = df["시도"].map(lambda x: SIDO_CENTROIDS.get(x, (np.nan, np.nan))[1])

    return df[["시도", "전월인구", "당월인구", "인구증감", "인구증감률(%)", "인구기준연도", "인구기준월", "lat", "lon"]]


# -----------------------------
# HIRA preprocessing & aggregation
# -----------------------------
def preprocess_hira(hira_raw: pd.DataFrame) -> pd.DataFrame:
    required = ["진료년도", "시도", "행위코드", "환자수", "명세서건수", "의료행위총사용량", "의료행위청구금액"]
    missing = [c for c in required if c not in hira_raw.columns]
    if missing:
        raise ValueError(f"심평원 데이터에 필요한 컬럼이 없습니다: {missing}")

    df = hira_raw.copy()
    for c in ["환자수", "명세서건수", "의료행위총사용량", "의료행위청구금액"]:
        df[c] = to_numeric_safe(df[c])

    df["진료년도"] = to_numeric_safe(df["진료년도"]).astype("Int64")
    df["시도"] = df["시도"].astype(str).str.strip()
    df["행위코드"] = df["행위코드"].astype(str).str.strip()
    return df


def aggregate_hira_by_sido(hira_df: pd.DataFrame, year: int, fillna_zero: bool = True) -> pd.DataFrame:
    df = hira_df[hira_df["진료년도"] == year].copy()
    if fillna_zero:
        cols = ["환자수", "명세서건수", "의료행위총사용량", "의료행위청구금액"]
        df[cols] = df[cols].fillna(0)

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
# UI
# -----------------------------
st.title("📍 지역별 의료행위(심평원) × 인구증감(주민등록) 대시보드")

with st.sidebar:
    st.header("1) 파일 설정")

    use_repo_files = st.checkbox("GitHub(레포) 내 CSV를 기본으로 사용", value=True)

    hira_file = st.file_uploader("심평원 의료행위 CSV 업로드(선택)", type=["csv"])
    pop_file = st.file_uploader("주민등록 인구증감 CSV 업로드(선택)", type=["csv"])
# (레포에 들어있는 기본 파일 경로) - 실제 파일명에 맞게 수정!
DEFAULT_HIRA_PATH = "건강보험심사평가원_의료행위별 시도별 건강보험 진료 통계_20241231.csv"
DEFAULT_POP_PATH  = "202512_202512_주민등록인구기타현황(인구증감)_월간.csv"

try:
    if use_repo_files and (hira_file is None) and (pop_file is None):
        hira_raw = read_csv_from_path(DEFAULT_HIRA_PATH)
        pop_raw  = read_csv_from_path(DEFAULT_POP_PATH)
        st.sidebar.success("레포 내 기본 CSV를 로드했습니다.")
    else:
        if (hira_file is None) or (pop_file is None):
            st.info("왼쪽 사이드바에서 CSV를 업로드하거나, '레포 내 CSV 사용'을 켜주세요.")
            st.stop()

        hira_raw = read_csv_auto(hira_file)
        pop_raw  = read_csv_auto(pop_file)

except FileNotFoundError as e:
    st.error(
        "레포 내 CSV 파일을 찾지 못했습니다.\n"
        "1) 파일명이 코드의 DEFAULT_*_PATH와 동일한지\n"
        "2) 파일이 app.py와 같은 폴더(또는 지정한 경로)에 있는지 확인해주세요.\n\n"
        f"에러: {e}"
    )
    st.stop()
except Exception as e:
    st.error(f"CSV 로딩 오류: {e}")
    st.stop()

# Load
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

# Year selector
years = sorted([int(y) for y in hira["진료년도"].dropna().unique()])
if not years:
    st.error("심평원 데이터에서 '진료년도'를 찾지 못했습니다.")
    st.stop()

default_year = years[-1]
left, right = st.columns([1, 2])
with left:
    year = st.selectbox("진료년도 선택", options=years, index=years.index(default_year))
with right:
    st.caption("※ 인구 데이터는 업로드된 파일의 월(전월→당월 증감률)을 기준으로 계산됩니다.")

# Aggregate & merge
hira_sido = aggregate_hira_by_sido(hira, year=year, fillna_zero=fillna_zero)
merged = hira_sido.merge(pop, on="시도", how="left")

# Derived metrics
merged["인구1만명당_총사용량"] = np.where(merged["당월인구"] > 0, (merged["의료행위총사용량"] / merged["당월인구"]) * 10000, np.nan)
merged["인구1만명당_청구금액"] = np.where(merged["당월인구"] > 0, (merged["의료행위청구금액"] / merged["당월인구"]) * 10000, np.nan)
merged["환자당_청구금액"] = np.where(merged["환자수"] > 0, merged["의료행위청구금액"] / merged["환자수"], np.nan)

national_avg = np.nanmean(merged["인구1만명당_총사용량"])
merged["표준화지수(총사용량)"] = (
    merged["인구1만명당_총사용량"] / national_avg
    if national_avg and not np.isnan(national_avg)
    else np.nan
)

# Map id for choropleth
merged["geo_id"] = merged["시도"].map(GEO_ID_BY_SIDO)

# Filters
all_sidos = [s for s in merged["시도"].dropna().unique().tolist()]
sel_sidos = st.multiselect("표시할 시도 선택(미선택 시 전체)", options=all_sidos, default=all_sidos)
view = merged[merged["시도"].isin(sel_sidos)].copy()

# KPI
k1, k2, k3, k4 = st.columns(4)
k1.metric("시도 수", f"{view['시도'].nunique()}개")
k2.metric("의료행위 청구금액 합계", f"{view['의료행위청구금액'].sum():,.0f}")
k3.metric("인구 1만명당 총사용량(평균)", f"{np.nanmean(view['인구1만명당_총사용량']):,.2f}")
k4.metric("인구 증감률(평균, %)", f"{np.nanmean(view['인구증감률(%)']):,.3f}")

st.divider()

tab1, tab2, tab3 = st.tabs(["🫧 버블(인구증감 × 의료이용)", "🗺️ 시도 경계 지도(Choropleth)", "📋 랭킹/테이블"])

# -----------------------------
# Tab 1: Bubble
# -----------------------------
with tab1:
    metric_choice = st.radio(
        "Y축 지표 선택",
        ["인구1만명당_총사용량", "인구1만명당_청구금액", "환자당_청구금액"],
        horizontal=True,
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
            "표준화지수(총사용량)": ":.2f",
        },
        labels={
            "인구증감률(%)": "인구 증감률(%) (전월→당월)",
            metric_choice: metric_choice,
        },
        title="인구 변화 vs 의료이용(인구보정) — 버블 크기=인구",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("해석 팁: 좌상단(인구↓, 의료이용↑)은 고령화/만성질환/공급 구조 등의 가능성을 시사할 수 있어요.")

# -----------------------------
# Tab 2: Choropleth map (no geojson file needed)
# -----------------------------
 # -----------------------------
# Tab 2: Map
# -----------------------------
with tab2:
    st.subheader("🗺️ 지도")

    safe_mode = st.checkbox("지도 안전모드(외부 GeoJSON 다운로드 안 함)", value=True)

    map_metric = st.selectbox(
        "지도에서 색으로 표현할 지표",
        ["인구1만명당_총사용량", "인구1만명당_청구금액", "인구증감률(%)", "표준화지수(총사용량)"],
    )

    map_df = view.copy()

    if safe_mode:
        fallback = map_df.copy()
        fallback["lat"] = fallback["시도"].map(lambda x: SIDO_CENTROIDS.get(x, (np.nan, np.nan))[0])
        fallback["lon"] = fallback["시도"].map(lambda x: SIDO_CENTROIDS.get(x, (np.nan, np.nan))[1])
        fallback = fallback.dropna(subset=["lat", "lon"])

        fig2 = px.scatter_mapbox(
            fallback,
            lat="lat",
            lon="lon",
            size="당월인구",
            color=map_metric,
            hover_name="시도",
            zoom=5.5,
            height=650,
        )
        fig2.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.stop()

# -----------------------------
# Tab 3: Table / ranking
# -----------------------------
with tab3:
    rank_metric = st.selectbox(
        "랭킹 기준",
        ["의료행위청구금액", "의료행위총사용량", "명세서건수", "환자수",
         "인구1만명당_총사용량", "인구1만명당_청구금액", "환자당_청구금액"],
    )

    ranked = view.sort_values(rank_metric, ascending=False).copy()

    st.subheader(f"Top {top_n} 시도 — {rank_metric}")

    cols = [
        "시도", "진료년도",
        "당월인구", "인구증감", "인구증감률(%)",
        "환자수", "명세서건수", "의료행위총사용량", "의료행위청구금액",
        "인구1만명당_총사용량", "인구1만명당_청구금액", "환자당_청구금액",
        "행위코드종류수", "표준화지수(총사용량)"
    ]
    st.dataframe(ranked[cols].head(top_n), use_container_width=True)

    csv = ranked[cols].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "집계 테이블 CSV 다운로드",
        data=csv,
        file_name=f"merged_hira_pop_{year}.csv",
        mime="text/csv",
    )

st.caption("ⓘ 환자수는 행위코드별 중복 집계 가능성이 있어, 지역 비교는 '총사용량/청구금액/인구보정 지표' 중심을 권장합니다.")
