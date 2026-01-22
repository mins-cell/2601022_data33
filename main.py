# main.py
# Streamlit: 지역별 의료행위(심평원) × 인구증감(주민등록) + 시도 경계 지도(Choropleth)
# 실행: streamlit run main.py

import os
import re
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="지역별 의료행위 × 인구증감 대시보드", layout="wide")


# -----------------------------
# CSV loader (auto-encoding)
# -----------------------------
def read_csv_auto(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            return pd.read_csv(pd.io.common.BytesIO(raw), encoding=enc)
        except Exception:
            continue
    return pd.read_csv(pd.io.common.BytesIO(raw))


def read_csv_from_path(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def normalize_sido_from_pop(행정구역: str) -> str:
    if pd.isna(행정구역):
        return np.nan

    name = re.sub(r"\s*\(.*?\)\s*", "", str(행정구역)).strip()
    name = re.sub(r"\s+", " ", name)

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
    url = "https://simplemaps.com/static/svg/country/kr/admin1/kr.json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


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
# Auto-detect repo CSV paths
# -----------------------------
def list_repo_csv_files(base_dir="."):
    try:
        return sorted([f for f in os.listdir(base_dir) if f.lower().endswith(".csv")])
    except Exception:
        return []


def detect_default_paths(csv_files):
    # heuristic keyword matching
    hira_keys = ["의료행위", "심사평가원", "진료", "건강보험", "심평원"]
    pop_keys = ["주민등록", "인구증감", "인구기타현황", "월간"]

    def score(name, keys):
        s = 0
        for k in keys:
            if k in name:
                s += 1
        return s

    # choose best match by score
    hira_best = max(csv_files, key=lambda x: score(x, hira_keys), default=None)
    pop_best = max(csv_files, key=lambda x: score(x, pop_keys), default=None)

    # ensure score >= 1, else None
    hira_best = hira_best if hira_best and score(hira_best, hira_keys) >= 1 else None
    pop_best = pop_best if pop_best and score(pop_best, pop_keys) >= 1 else None
    return hira_best, pop_best


# -----------------------------
# Population preprocessing
# -----------------------------
def preprocess_population(pop_raw: pd.DataFrame) -> pd.DataFrame:
    df = pop_raw.copy()

    if "행정구역" not in df.columns:
        raise ValueError("인구 데이터에 '행정구역' 컬럼이 없습니다.")

    df["시도"] = df["행정구역"].apply(normalize_sido_from_pop)

    month_prefix = None
    for c in df.columns:
        m = re.match(r"(\d{4}년\d{1,2}월)_", str(c))
        if m:
            month_prefix = m.group(1)
            break
    if not month_prefix:
        raise ValueError("인구 데이터에서 'YYYY년M월_' 형태의 컬럼 접두어를 찾지 못했습니다.")

    # robust column finding
    def find_col(cols, must_include):
        for c in cols:
            ok = True
            for k in must_include:
                if k not in str(c):
                    ok = False
                    break
            if ok:
                return c
        return None

    prev_col = find_col(df.columns, [month_prefix, "전월", "인구"])
    curr_col = find_col(df.columns, [month_prefix, "당월", "인구"])
    diff_col = find_col(df.columns, [month_prefix, "증감"])

    if (prev_col is None) or (curr_col is None) or (diff_col is None):
        raise ValueError(
            "인구 데이터 컬럼 자동 탐지 실패입니다.\n"
            f"- 탐지된 month_prefix: {month_prefix}\n"
            f"- 컬럼 일부: {list(df.columns)[:25]}"
        )

    df["전월인구"] = to_numeric_safe(df[prev_col])
    df["당월인구"] = to_numeric_safe(df[curr_col])
    df["인구증감"] = to_numeric_safe(df[diff_col])
    df["인구증감률(%)"] = np.where(df["전월인구"] > 0, (df["인구증감"] / df["전월인구"]) * 100, np.nan)

    ym = re.match(r"(\d{4})년(\d{1,2})월", month_prefix)
    df["인구기준연도"] = int(ym.group(1)) if ym else np.nan
    df["인구기준월"] = int(ym.group(2)) if ym else np.nan

    df = df[df["시도"].notna()].copy()
    df = df[df["시도"] != "전국"].copy()

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

repo_csvs = list_repo_csv_files(".")
auto_hira, auto_pop = detect_default_paths(repo_csvs)

with st.sidebar:
    st.header("1) 데이터 소스")

    # 디버그용: 현재 폴더 파일 확인(필요없으면 주석 처리)
    with st.expander("📁 현재 폴더 CSV 목록 보기", expanded=False):
        st.write(repo_csvs if repo_csvs else "CSV 파일이 없습니다. (레포에 업로드 되었는지 확인)")

    source_mode = st.radio(
        "CSV 로딩 방식",
        ["레포 파일 자동 로드(추천)", "업로드(file_uploader)로 사용"],
        index=0,
    )

    st.divider()
    st.header("2) 업로드(선택)")
    hira_up = st.file_uploader("심평원 의료행위 CSV 업로드", type=["csv"])
    pop_up = st.file_uploader("주민등록 인구증감 CSV 업로드", type=["csv"])

    st.divider()
    st.header("3) 옵션")
    fillna_zero = st.checkbox("결측치를 0으로 처리(권장)", value=True)
    top_n = st.slider("Top N (랭킹/표)", 5, 30, 15)

# Resolve paths (this "prints" defaults accurately)
if source_mode == "레포 파일 자동 로드(추천)":
    # If user uploaded, prefer uploaded
    if hira_up is not None and pop_up is not None:
        st.sidebar.success("업로드 파일을 우선 사용합니다.")
        hira_raw = read_csv_auto(hira_up)
        pop_raw = read_csv_auto(pop_up)
        DEFAULT_HIRA_PATH = "(uploaded)"
        DEFAULT_POP_PATH = "(uploaded)"
    else:
        # If not uploaded, use auto-detected repo files (or user pick)
        if not repo_csvs:
            st.error("레포 폴더에서 CSV 파일을 찾지 못했습니다. main.py와 같은 위치에 CSV가 있는지 확인해주세요.")
            st.stop()

        # Let user override if detection failed / or want explicit choice
        st.sidebar.subheader("자동 탐지된 기본 파일")
        st.sidebar.write("DEFAULT_HIRA_PATH:", auto_hira or "탐지 실패")
        st.sidebar.write("DEFAULT_POP_PATH:", auto_pop or "탐지 실패")

        hira_choice = st.sidebar.selectbox(
            "심평원 CSV 선택(레포 내)",
            options=repo_csvs,
            index=repo_csvs.index(auto_hira) if auto_hira in repo_csvs else 0,
        )
        pop_choice = st.sidebar.selectbox(
            "인구증감 CSV 선택(레포 내)",
            options=repo_csvs,
            index=repo_csvs.index(auto_pop) if auto_pop in repo_csvs else min(1, len(repo_csvs) - 1),
        )

        DEFAULT_HIRA_PATH = hira_choice
        DEFAULT_POP_PATH = pop_choice

        try:
            hira_raw = read_csv_from_path(DEFAULT_HIRA_PATH)
            pop_raw = read_csv_from_path(DEFAULT_POP_PATH)
            st.sidebar.success("레포 내 CSV를 로드했습니다.")
        except Exception as e:
            st.error(f"레포 CSV 로딩 오류: {e}")
            st.stop()

else:
    # upload mode
    if hira_up is None or pop_up is None:
        st.info("왼쪽 사이드바에서 심평원 CSV와 인구증감 CSV를 업로드하면 대시보드가 생성됩니다.")
        st.stop()

    hira_raw = read_csv_auto(hira_up)
    pop_raw = read_csv_auto(pop_up)
    DEFAULT_HIRA_PATH = "(uploaded)"
    DEFAULT_POP_PATH = "(uploaded)"

# Show resolved defaults in main page too
st.caption(f"📌 사용 중인 파일: 심평원={DEFAULT_HIRA_PATH} / 인구={DEFAULT_POP_PATH}")

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
    st.caption("※ 인구 데이터는 업로드/레포 파일의 월(전월→당월 증감률)을 기준으로 계산됩니다.")

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

# Tab 1: Bubble
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
    st.caption("해석 팁: 좌상단(인구↓, 의료이용↑)은 고령화/만성질환/공급 구조 가능성을 시사할 수 있어요.")

# Tab 2: Choropleth
with tab2:
    st.subheader("🗺️ 시도 경계 지도(Choropleth)")

    map_metric = st.selectbox(
        "지도에서 색으로 표현할 지표",
        ["인구1만명당_총사용량", "인구1만명당_청구금액", "인구증감률(%)", "표준화지수(총사용량)"],
    )

    map_df = view.dropna(subset=["geo_id"]).copy()

    # 안전모드: 외부 GeoJSON 다운로드를 막고, 점 지도 사용
    safe_mode = st.checkbox("지도 안전모드(외부 경계 다운로드 없이 점 지도)", value=False)

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
        fig2.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        try:
            geojson = load_korea_sido_geojson()

            figm = px.choropleth(
                map_df,
                geojson=geojson,
                locations="geo_id",
                featureidkey="properties.id",
                color=map_metric,
                hover_name="시도",
                hover_data={
                    "당월인구": ":,",
                    "인구증감": ":,",
                    "인구증감률(%)": ":.3f",
                    "의료행위총사용량": ":,",
                    "의료행위청구금액": ":,",
                    "인구1만명당_총사용량": ":.2f",
                    "인구1만명당_청구금액": ":.2f",
                    "표준화지수(총사용량)": ":.2f",
                },
                labels={map_metric: map_metric},
            )
            figm.update_geos(fitbounds="locations", visible=False)
            figm.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(figm, use_container_width=True)
            st.caption("※ 시도 경계는 실행 시 웹에서 자동 로드됩니다(별도 파일 불필요).")

        except Exception as e:
            st.warning(f"경계 GeoJSON 로드 실패 → 점 지도(fallback)로 표시합니다.\n- 오류: {e}")

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
            fig2.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig2, use_container_width=True)

    show_cols = ["시도", "당월인구", "인구증감률(%)", "인구1만명당_총사용량", "인구1만명당_청구금액", "표준화지수(총사용량)"]
    st.dataframe(
        map_df[show_cols].sort_values(map_metric, ascending=False).head(top_n),
        use_container_width=True,
    )

# Tab 3: Table / ranking
with tab3:
    rank_metric = st.selectbox(
        "랭킹 기준",
        [
            "의료행위청구금액",
            "의료행위총사용량",
            "명세서건수",
            "환자수",
            "인구1만명당_총사용량",
            "인구1만명당_청구금액",
            "환자당_청구금액",
        ],
    )

    ranked = view.sort_values(rank_metric, ascending=False).copy()

    st.subheader(f"Top {top_n} 시도 — {rank_metric}")

    cols = [
        "시도",
        "진료년도",
        "당월인구",
        "인구증감",
        "인구증감률(%)",
        "환자수",
        "명세서건수",
        "의료행위총사용량",
        "의료행위청구금액",
        "인구1만명당_총사용량",
        "인구1만명당_청구금액",
        "환자당_청구금액",
        "행위코드종류수",
        "표준화지수(총사용량)",
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
