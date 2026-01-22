import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="2024 인구변화 × 의료이용 (시도)", layout="wide")
st.title("2024년 시도별 인구 변화 × 건강보험 의료이용(심평원)")

st.caption(
    "데이터: 주민등록 인구증감(월간) + 심평원 의료행위별 시도별 진료통계(2024). "
    "이 앱은 같은 폴더의 CSV 4개를 읽습니다. (업로드 모드도 지원)"
)

MODE = st.sidebar.radio("데이터 불러오기", ["폴더에서 읽기(기본)", "파일 업로드"])

FILES = {
    "인구 월별(2024) tidy": "population_change_2024_tidy.csv",
    "인구 요약(2024)": "population_change_2024_summary.csv",
    "의료 요약(2024)": "hira_medical_stats_2024_sido_summary.csv",
    "결합(2024)": "merged_sido_population_x_hira_2024.csv",
}

@st.cache_data
def load_local():
    pop_tidy = pd.read_csv(FILES["인구 월별(2024) tidy"])
    pop_sum  = pd.read_csv(FILES["인구 요약(2024)"])
    hira_sum = pd.read_csv(FILES["의료 요약(2024)"])
    merged   = pd.read_csv(FILES["결합(2024)"])
    return pop_tidy, pop_sum, hira_sum, merged

def load_upload():
    uploaded = {}
    for _, fname in FILES.items():
        f = st.sidebar.file_uploader(f"업로드: {fname}", type=["csv"], key=fname)
        if f is None:
            st.sidebar.warning(f"{fname} 업로드 필요")
            return None
        uploaded[fname] = pd.read_csv(f)

    pop_tidy = uploaded[FILES["인구 월별(2024) tidy"]]
    pop_sum  = uploaded[FILES["인구 요약(2024)"]]
    hira_sum = uploaded[FILES["의료 요약(2024)"]]
    merged   = uploaded[FILES["결합(2024)"]]
    return pop_tidy, pop_sum, hira_sum, merged

if MODE.startswith("폴더"):
    try:
        pop_tidy, pop_sum, hira_sum, merged = load_local()
    except Exception as e:
        st.error("같은 폴더에 CSV 파일 4개가 있는지 확인해 주세요.")
        st.exception(e)
        st.stop()
else:
    loaded = load_upload()
    if loaded is None:
        st.stop()
    pop_tidy, pop_sum, hira_sum, merged = loaded

# Sidebar controls
sido_list = merged["sido"].sort_values().unique().tolist()
selected_sido = st.sidebar.selectbox("시도 선택(상세)", ["(전체)"] + sido_list)
metric = st.sidebar.radio("지표", ["amount_per_capita", "patients_per_1k", "pop_change_2024", "amount_per_patient"])
metric_labels = {
    "amount_per_capita": "1인당 의료비(원/인)",
    "patients_per_1k": "인구 1천명당 환자수",
    "pop_change_2024": "2024 인구증감(명)",
    "amount_per_patient": "환자 1인당 의료비(원/명)",
}

tab1, tab2, tab3 = st.tabs(["요약(막대/버블)", "관계(산점도)", "시도 상세(월별)"])

with tab1:
    st.subheader("시도별 요약")
    st.dataframe(merged.sort_values(metric, ascending=False), use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        fig = px.bar(merged.sort_values(metric, ascending=False), x="sido", y=metric, title=metric_labels[metric])
        fig.update_layout(xaxis_title="", yaxis_title=metric_labels[metric])
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        fig2 = px.scatter(
            merged,
            x="pop_change_2024",
            y="amount_per_capita",
            hover_name="sido",
            size="pop_avg_2024",
            title="인구증감 vs 1인당 의료비 (버블=평균인구)",
        )
        fig2.update_layout(xaxis_title="2024 인구증감(명)", yaxis_title="1인당 의료비(원/인)")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("변수 관계 탐색")
    x = st.selectbox("X축", ["pop_change_2024", "pop_avg_2024", "patients_per_1k"], index=0)
    y = st.selectbox("Y축", ["amount_per_capita", "total_amount_2024", "amount_per_patient"], index=0)
    fig = px.scatter(merged, x=x, y=y, hover_name="sido", size="pop_avg_2024")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("시도 상세(월별 인구증감/인구)")
    if selected_sido == "(전체)":
        st.info("왼쪽에서 시도를 선택하면 월별 추이가 나옵니다.")
    else:
        df = pop_tidy[pop_tidy["sido"] == selected_sido].copy()
        df["month_num"] = df["month"].str.extract(r"2024년(\d+)월").astype(int)
        df = df.sort_values("month_num")

        fig = px.line(df, x="month", y="pop_change", markers=True, title=f"{selected_sido} 월별 인구증감(2024)")
        fig.update_layout(xaxis_title="", yaxis_title="인구증감(명)")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(df, x="month", y="pop_end", markers=True, title=f"{selected_sido} 월말 인구(2024)")
        fig2.update_layout(xaxis_title="", yaxis_title="월말 인구(명)")
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.caption(
    "※ 환자수/명세서건수는 의료행위별 통계를 시도 단위로 합산한 값이라 '고유 인원'과 다를 수 있습니다. "
    "지역 비교를 위한 탐색적 지표로 활용하세요."
)
