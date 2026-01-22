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

def normalize_sido_name(x: str
