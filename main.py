# app.py
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# =========================
# 0) 기본 설정
# =========================
st.set_page_config(page_title="2024 지역(시도) 인구-진료 대시보드", layout="wide")

POP_PATH = "/mnt/data/202312_202512_주민등록인구기타현황(인구증감)_월간.csv"
HIRA_PATH = "/mnt/data/건강보험심사평가원_의료행위별 시도별 건강보험 진료 통계_20241231.csv"
CANCER_PATH = "/mnt/data/국립암센터_암발생 통계 정보_20260120.csv"

# =========================
# 1) GeoJSON 자동 로더 (다운로드 + 캐시 + fallback)
# =========================
@st.cache_data(show_spinner=False)
def load_korea_sido_geojson() -> dict:
    """
    Korea 시도(17개) 경계 GeoJSON을 자동으로 가져옵니다.
    - 1순위: southkorea-maps (kostat 2018 provinces geojson)
    - 2순위: southkorea-maps (gadm provinces geojson)
    - 실패 시: 예외 발생 (지도 대신 안내문 출력)
    """
    # 신뢰도 높은 공개 리포지토리들에서 RAW GeoJSON을 직접 가져옵니다.
    # (southkorea/southkorea-maps는 시도/시군구 등 여러 레벨 제공)
    candidates = [
        "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2018/json/skorea-provinces-2018-geo.json",
        "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/gadm/json/skorea-provinces-geo.json",
    ]

    last_err = None
    for url in candidates:
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            gj = r.json()
            if isinstance(gj, dict) and "features" in gj and len(gj["features"]) >= 10:
                return gj
        except Exception as e:
            last_err = e

    raise RuntimeError(f"GeoJSON 로드 실패: {last_err}")

def geojson_sido_property_key(geojson_obj: dict) -> str:
    """
    GeoJSON feature의 시도명 속성 key를 추정합니다.
    주로 properties.name 또는 properties.NAME_1 등의 형태를 가짐.
    """
    sample_props = geojson_obj["features"][0].get("properties", {})
    # 흔한 후보들
    candidates = ["name", "NAME_1", "CTP_KOR_NM", "sidonm", "sido_nm"]
    for k in candidates:
        if k in sample_props:
            return k
    # 아무거나 문자열 값인 첫 키
    for k, v in sample_props.items():
        if isinstance(v, str) and len(v) >= 2:
            return k
    return "name"

# =========================
# 2) CSV 로더 (인코딩 자동-ish)
# =========================
def read_csv_flexible(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        st.error(f"파일을 찾을 수 없어요: {path}")
        return pd.DataFrame()

    # 한국 공공데이터는 cp949/euc-kr가 많음
    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue

    # 최후: 오류 무시
    return pd.read_csv(path, encoding="cp949", errors="ignore")

# =========================
# 3) 시도명 표준화 (매핑 + 정규화)
# =========================
SIMPLE_TO_FULL = {
    "서울": "서울특별시",
    "부산": "부산광역시",
    "대구": "대구광역시",
    "인천": "인천광역시",
    "광주": "광주광역시",
    "대전": "대전광역시",
    "울산": "울산광역시",
    "세종": "세종특별자치시",
    "경기": "경기도",
    "강원": "강원도",
    "충북": "충청북도",
    "충남": "충청남도",
    "전북": "전북특별자치도",  # 2023.6 전북특별자치도 (데이터마다 전라북도일 수 있어 처리)
    "전남": "전라남도",
    "경북": "경상북도",
    "경남": "경상남도",
    "제주": "제주특별자치도",
}

# 데이터에 "전라북도"로 들어오는 경우도 많아서 같이 흡수
ALIASES = {
    "전라북도": "전북특별자치도",
    "전북": "전북특별자치도",
    "강원특별자치도": "강원도",  # GeoJSON/통계 데이터가 강원도/강원특별자치도 혼용할 수 있어 보정
    "강원": "강원도",
}

def normalize_sido_name(x: str) -> str:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()

    # "서울특별시 종로구" 같이 내려오는 경우 첫 토큰만
    s = s.split()[0]

    # 괄호/특수문자 제거
    s = re.sub(r"[()]", "", s)

    # 단축형 → 정식
    if s in SIMPLE_TO_FULL:
        s = SIMPLE_TO_FULL[s]

    if s in ALIASES:
        s = ALIASES[s]

    return s

# =========================
# 4) 데이터 전처리: 인구 (2024 집계)
# =========================
def prep_population_2024(df_pop: pd.DataFrame) -> pd.DataFrame:
    """
    주민등록 인구증감(월간)에서 2024년 시도 단위로:
    - pop_avg_2024: 2024 월별 총인구 평균
    - pop_change_2024: 2024 연간 증감(월 증감 합 또는 (12월-1월))
    """
    if df_pop.empty:
        return pd.DataFrame()

    # 컬럼 후보 추정
    col_region = next((c for c in df_pop.columns if "행정" in c or "지역" in c or "시도" in c), None)
    col_ym = next((c for c in df_pop.columns if "년월" in c or "기준" in c and "월" in c), None)
    col_pop = next((c for c in df_pop.columns if ("총" in c and "인구" in c) or c.endswith("총인구수")), None)
    col_delta = next((c for c in df_pop.columns if "증감" in c), None)

    # 최소 요건
    if not (col_region and col_ym):
        # 그래도 최대한 찾기
        st.warning("인구 데이터에서 '행정구역/년월' 컬럼을 자동으로 찾지 못해 일부 기능이 제한될 수 있어요.")
        return pd.DataFrame()

    df = df_pop.copy()
    df["sido"] = df[col_region].apply(normalize_sido_name)

    # 년월 → YYYYMM 형태로 파싱
    def parse_ym(v):
        s = str(v)
        s = re.sub(r"[^0-9]", "", s)
        if len(s) >= 6:
            return int(s[:6])
        return np.nan

    df["ym"] = df[col_ym].apply(parse_ym)
    df = df.dropna(subset=["sido", "ym"])
    df["year"] = (df["ym"] // 100).astype(int)
    df_2024 = df[df["year"] == 2024].copy()

    # 숫자화
    if col_pop and col_pop in df_2024.columns:
        df_2024["pop"] = pd.to_numeric(df_2024[col_pop], errors="coerce")
    else:
        df_2024["pop"] = np.nan

    if col_delta and col_delta in df_2024.columns:
        df_2024["delta"] = pd.to_numeric(df_2024[col_delta], errors="coerce")
    else:
        df_2024["delta"] = np.nan

    agg = df_2024.groupby("sido", as_index=False).agg(
        pop_avg_2024=("pop", "mean"),
        pop_change_2024=("delta", "sum"),
    )

    # 보조: 월별 시계열용은 별도 리턴이 편하지만, 여기선 agg만
    return agg

def pop_timeseries_2024(df_pop: pd.DataFrame) -> pd.DataFrame:
    if df_pop.empty:
        return pd.DataFrame()

    col_region = next((c for c in df_pop.columns if "행정" in c or "지역" in c or "시도" in c), None)
    col_ym = next((c for c in df_pop.columns if "년월" in c or "기준" in c and "월" in c), None)
    col_delta = next((c for c in df_pop.columns if "증감" in c), None)
    if not (col_region and col_ym and col_delta):
        return pd.DataFrame()

    df = df_pop.copy()
    df["sido"] = df[col_region].apply(normalize_sido_name)

    def parse_ym(v):
        s = re.sub(r"[^0-9]", "", str(v))
        if len(s) >= 6:
            return int(s[:6])
        return np.nan

    df["ym"] = df[col_ym].apply(parse_ym)
    df = df.dropna(subset=["sido", "ym"])
    df["year"] = (df["ym"] // 100).astype(int)
    df = df[df["year"] == 2024].copy()
    df["delta"] = pd.to_numeric(df[col_delta], errors="coerce")
    df["month"] = (df["ym"] % 100).astype(int)

    return df[["sido", "year", "month", "ym", "delta"]].sort_values(["sido", "ym"])

# =========================
# 5) 데이터 전처리: 심평원 (2024 시도 집계 + Top N 의료행위)
# =========================
def prep_hira_2024(df_hira: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    심평원 진료통계에서:
    - sido_summary_2024: 시도별 총 청구금액/환자수(가능하면) 집계
    - top_actions_2024: 시도별 의료행위 TopN 뽑을 수 있는 long 형태
    """
    if df_hira.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 컬럼 후보
    col_year = next((c for c in df_hira.columns if "년도" in c or "연도" in c), None)
    col_sido = next((c for c in df_hira.columns if "시도" in c), None)
    col_action_name = next((c for c in df_hira.columns if "행위" in c and ("명" in c or "내용" in c)), None)
    col_action_code = next((c for c in df_hira.columns if "행위" in c and ("코드" in c or "code" in c.lower())), None)

    # 금액/환자수 추정
    col_amount = next((c for c in df_hira.columns if ("청구" in c or "진료" in c or "금액" in c) and ("금" in c or "비" in c)), None)
    col_patients = next((c for c in df_hira.columns if "환자" in c and ("수" in c or "인원" in c)), None)

    if not (col_year and col_sido):
        st.warning("심평원 데이터에서 '년도/시도' 컬럼을 자동으로 찾지 못했어요. 컬럼명을 확인해 주세요.")
        return pd.DataFrame(), pd.DataFrame()

    df = df_hira.copy()
    df["year"] = pd.to_numeric(df[col_year], errors="coerce")
    df = df[df["year"] == 2024].copy()

    df["sido"] = df[col_sido].apply(normalize_sido_name)

    if col_amount:
        df["amount"] = pd.to_numeric(df[col_amount], errors="coerce")
    else:
        df["amount"] = np.nan

    if col_patients:
        df["patients"] = pd.to_numeric(df[col_patients], errors="coerce")
    else:
        df["patients"] = np.nan

    # 시도 요약
    sido_summary = df.groupby("sido", as_index=False).agg(
        amount_2024=("amount", "sum"),
        patients_2024=("patients", "sum"),
    )

    # 의료행위 TopN용 long 형태
    action_label = None
    if col_action_name:
        action_label = col_action_name
    elif col_action_code:
        action_label = col_action_code

    if action_label:
        top_actions = df.groupby(["sido", action_label], as_index=False).agg(
            amount=("amount", "sum"),
            patients=("patients", "sum"),
        )
        top_actions = top_actions.rename(columns={action_label: "action"})
    else:
        top_actions = pd.DataFrame()

    return sido_summary, top_actions

# =========================
# 6) 결합 + 지도용 데이터 준비
# =========================
def merge_for_map(pop_sido: pd.DataFrame, hira_sido: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(pop_sido, hira_sido, on="sido", how="outer")

    # 결합 지표
    df["amount_per_capita"] = df["amount_2024"] / df["pop_avg_2024"]
    df["patients_per_1000"] = (df["patients_2024"] / df["pop_avg_2024"]) * 1000
    return df

# =========================
# UI
# =========================
st.title("2024 시도별 인구증감 × 진료통계 대시보드")
st.caption("GeoJSON 없이도 실행 시 자동으로 시도 경계 GeoJSON을 불러와 지도(Choropleth)를 그립니다.")

with st.sidebar:
    st.header("필터")
    metric = st.selectbox(
        "지도 지표",
        [
            "amount_per_capita",
            "patients_per_1000",
            "amount_2024",
            "patients_2024",
            "pop_change_2024",
            "pop_avg_2024",
        ],
        format_func=lambda x: {
            "amount_per_capita": "인당 청구금액",
            "patients_per_1000": "인구 1,000명당 환자수",
            "amount_2024": "총 청구금액(2024)",
            "patients_2024": "총 환자수(2024)",
            "pop_change_2024": "연간 인구증감(합)",
            "pop_avg_2024": "연평균 인구(월평균)",
        }[x],
    )
    top_n = st.slider("의료행위 Top N", 5, 30, 10)
    top_by = st.radio("Top 기준", ["amount", "patients"], horizontal=True)
    st.divider()
    st.info("※ 암발생 통계 파일은 지역(시도) 컬럼이 없어 지도 결합이 어렵고, 연도도 2024가 아닐 수 있어 별도 탭으로 두는 것을 권장합니다.")

# 데이터 로드
with st.spinner("데이터 로딩 중..."):
    df_pop_raw = read_csv_flexible(POP_PATH)
    df_hira_raw = read_csv_flexible(HIRA_PATH)
    df_cancer_raw = read_csv_flexible(CANCER_PATH)

pop_sido_2024 = prep_population_2024(df_pop_raw)
pop_ts_2024 = pop_timeseries_2024(df_pop_raw)
hira_sido_2024, hira_actions_2024 = prep_hira_2024(df_hira_raw)
merged = merge_for_map(pop_sido_2024, hira_sido_2024)

# GeoJSON 로드 (자동)
geojson_ok = True
try:
    korea_geojson = load_korea_sido_geojson()
    prop_key = geojson_sido_property_key(korea_geojson)
except Exception as e:
    geojson_ok = False
    st.error(f"시도 GeoJSON 자동 로드에 실패했어요. (지도 대신 표/차트만 표시합니다)\n\n에러: {e}")

# GeoJSON의 시도명과 merged의 시도명을 최대한 맞추기 위한 보정
# southkorea-maps의 properties.name은 보통 "서울특별시" 같은 형태가 많아 normalize만 해줘도 꽤 맞습니다.
if geojson_ok:
    # GeoJSON 안의 시도명 목록 확인 (디버그용)
    geo_names = sorted({f["properties"].get(prop_key, "") for f in korea_geojson["features"]})
    # 강원도/전북특별자치도 등 혼용 보정(geojson이 "강원도"로만 오거나 "전라북도"로 올 수 있음)
    merged["sido_for_geo"] = merged["sido"].replace({
        "강원특별자치도": "강원도",
        "전라북도": "전북특별자치도",
        "전북특별자치도": "전북특별자치도",
    })
else:
    merged["sido_for_geo"] = merged["sido"]

# 시도 목록
all_sido = sorted([s for s in merged["sido"].dropna().unique()])
default_sido = ["서울특별시", "경기도"] if "서울특별시" in all_sido and "경기도" in all_sido else all_sido[:2]
selected_sido = st.multiselect("시도 선택 (차트/표 연동)", options=all_sido, default=default_sido)

# =========================
# 1행: 지도 + KPI
# =========================
c1, c2 = st.columns([1.25, 1])

with c1:
    st.subheader("시도별 지도 (2024)")
    if geojson_ok and not merged.empty:
        # plotly choropleth (mapbox 토큰 없이 가능)
        # featureidkey는 properties.<prop_key> 로 설정
        # location은 merged의 sido_for_geo
        # projection은 mercator 등
        fig_map = px.choropleth(
            merged,
            geojson=korea_geojson,
            locations="sido_for_geo",
            featureidkey=f"properties.{prop_key}",
            color=metric,
            hover_name="sido",
            hover_data={
                "pop_avg_2024": ":,.0f",
                "pop_change_2024": ":,.0f",
                "amount_2024": ":,.0f",
                "patients_2024": ":,.0f",
                "amount_per_capita": ":,.2f",
                "patients_per_1000": ":,.2f",
                "sido_for_geo": False,
            },
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("지도는 GeoJSON 로드가 필요해요. 현재는 표/차트만 표시합니다.")

with c2:
    st.subheader("2024 핵심 지표 (선택 시도 합/평균)")
    if merged.empty or len(selected_sido) == 0:
        st.info("왼쪽에서 시도를 선택해 주세요.")
    else:
        sel = merged[merged["sido"].isin(selected_sido)].copy()

        # KPI 계산(합/가중)
        pop = sel["pop_avg_2024"].sum(skipna=True)
        amt = sel["amount_2024"].sum(skipna=True)
        pat = sel["patients_2024"].sum(skipna=True)
        popchg = sel["pop_change_2024"].sum(skipna=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("연평균 인구(합)", f"{pop:,.0f}" if pd.notna(pop) else "NA")
        k2.metric("연간 인구증감(합)", f"{popchg:,.0f}" if pd.notna(popchg) else "NA")
        k3.metric("총 청구금액(합)", f"{amt:,.0f}" if pd.notna(amt) else "NA")
        k4.metric("총 환자수(합)", f"{pat:,.0f}" if pd.notna(pat) else "NA")

        st.divider()
        if pop and pop > 0:
            st.metric("인당 청구금액(선택 시도)", f"{(amt/pop):,.2f}")
            st.metric("인구 1,000명당 환자수(선택 시도)", f"{(pat/pop*1000):,.2f}")
        else:
            st.info("인구 값이 없어 인당 지표 계산이 제한됩니다.")

# =========================
# 2행: 월별 인구증감 + 의료행위 TopN
# =========================
c3, c4 = st.columns([1, 1])

with c3:
    st.subheader("2024 월별 인구증감 (선택 시도)")
    if pop_ts_2024.empty or len(selected_sido) == 0:
        st.info("인구 시계열을 만들 수 없거나 시도를 선택하지 않았습니다.")
    else:
        ts = pop_ts_2024[pop_ts_2024["sido"].isin(selected_sido)].copy()
        if ts.empty:
            st.info("선택 시도에 대한 인구 월별 데이터가 없습니다.")
        else:
            fig_line = px.line(ts, x="month", y="delta", color="sido", markers=True)
            fig_line.update_layout(xaxis_title="월", yaxis_title="인구 증감(명)", margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_line, use_container_width=True)

with c4:
    st.subheader(f"의료행위 Top {top_n} (선택 시도)")
    if hira_actions_2024.empty or len(selected_sido) == 0:
        st.info("의료행위 TopN을 만들 수 없거나 시도를 선택하지 않았습니다.")
    else:
        act = hira_actions_2024[hira_actions_2024["sido"].isin(selected_sido)].copy()
        if act.empty:
            st.info("선택 시도에 대한 의료행위 데이터가 없습니다.")
        else:
            # 선택 시도 전체에서 Top N
            grp = act.groupby("action", as_index=False).agg(
                amount=("amount", "sum"),
                patients=("patients", "sum"),
            )
            grp[top_by] = pd.to_numeric(grp[top_by], errors="coerce")
            grp = grp.sort_values(top_by, ascending=False).head(top_n)

            fig_bar = px.bar(grp, x=top_by, y="action", orientation="h")
            fig_bar.update_layout(xaxis_title="값", yaxis_title="", margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_bar, use_container_width=True)

            st.dataframe(grp, use_container_width=True, hide_index=True)

# =========================
# 3행: 결합 테이블 (디버그/검증용)
# =========================
st.subheader("시도별 결합 테이블 (2024)")
st.dataframe(
    merged.sort_values(metric, ascending=False) if metric in merged.columns else merged,
    use_container_width=True,
    hide_index=True
)

# (옵션) 암발생 데이터 안내
with st.expander("암발생 통계(국립암센터) — 왜 지도 결합이 어려운가?"):
    st.write(
        "업로드된 암발생 통계 파일은 일반적으로 연도/성별/암종/연령군 중심이며, "
        "시도(지역) 컬럼이 없어서 시도 지도와 직접 결합이 어렵습니다. "
        "따라서 별도 탭에서 연도/성별/암종/연령군 기반 차트로 구성하는 것을 추천합니다."
    )
    st.write("파일 상위 5행 미리보기:")
    st.dataframe(df_cancer_raw.head(), use_container_width=True)
