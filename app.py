"""
KAC-PPI Dashboard — Streamlit 시제품
====================================
2026 국토교통 데이터활용 경진대회 — 제품·서비스 개발 부문 출품작

실행: streamlit run app.py
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis as A

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="KAC-PPI 대시보드",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
  .stMetric { background:#F7F9FC; border-radius:8px; padding:12px; border:1px solid #E1E5EB; }
  .ppi-high { color:#E03131; font-weight:600; }
  .ppi-mid { color:#F59F00; font-weight:600; }
  .ppi-low { color:#2F9E44; font-weight:600; }
  .card { background:#FFFFFF; border:1px solid #E1E5EB; border-radius:10px; padding:16px; margin-bottom:12px; }
  div[data-testid="stSidebarNav"] li div a span { font-size:1.05rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def _csv(name: str) -> Path:
    """CSV를 data/ 폴더에서 먼저 찾고, 없으면 app.py 같은 폴더(root)에서 찾는다.
    → GitHub 드래그 업로드로 폴더 구조가 펴져도 동작하도록."""
    p = DATA_DIR / name
    return p if p.exists() else BASE_DIR / name


@st.cache_data(show_spinner=False)
def load_all():
    """
    원본 VOC(xlsx)가 있으면 실시간 계산, 없으면 사전계산 CSV로 폴백.
    → 배포 시 민감한 원본 VOC를 공개하지 않고 집계 결과만으로 동작 가능.
    """
    raw_path = Path(A.VOC_FILE)
    if raw_path.exists():
        # --- 실시간 계산 모드 ---
        df = A.load_voc()
        df = A.add_pseudo_user(df)
        ppi, components, share, norm, growth, sentiment = A.compute_ppi(df)
        clusters = A.cluster_airports(ppi)
        keywords = A.extract_keywords(df)
        summary = pd.DataFrame({
            "종합_PPI": ppi.mean(axis=1).round(1),
            "1위_카테고리": ppi.idxmax(axis=1),
            "1위_점수": ppi.max(axis=1).round(1),
            "총_불편불만": df[df["VOC유형"] == "불편불만"]
                .groupby("대상공항").size().reindex(A.TARGET_AIRPORTS, fill_value=0),
        }).sort_values("종합_PPI", ascending=False)
        n_voc = len(df)
        n_users = df["pseudo_user"].nunique()
        mode = "live"
    else:
        # --- 사전계산 CSV 폴백 모드 (배포용) ---
        ppi = pd.read_csv(_csv("ppi_table.csv"), index_col=0, encoding="utf-8-sig")
        summary = pd.read_csv(_csv("airport_summary.csv"), index_col=0, encoding="utf-8-sig")
        clusters = pd.read_csv(_csv("cluster_table.csv"), index_col=0,
                               encoding="utf-8-sig").squeeze("columns")
        keywords = pd.read_csv(_csv("category_keywords.csv"), encoding="utf-8-sig")
        components = {}
        n_voc = int(summary["총_불편불만"].sum())
        try:
            n_users = sum(1 for _ in open(_csv("pseudo_user_behavior.csv"),
                                          encoding="utf-8-sig")) - 1
        except FileNotFoundError:
            n_users = 0
        df = None
        mode = "csv"

    return {
        "df": df, "ppi": ppi, "components": components,
        "clusters": clusters, "keywords": keywords, "summary": summary,
        "n_voc": n_voc, "n_users": n_users, "mode": mode,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ppi_label(score: float) -> str:
    if score >= 60: return "★★★ 심각 (즉시 개선)"
    if score >= 40: return "★★ 주의 (단기 개선)"
    return "★ 안정 (모니터링)"

def action_for(category: str, score: float) -> list[str]:
    """카테고리별 권장 액션 매핑."""
    book = {
        "직원관련": [
            "보안검색·ID체크 직원 응대 매뉴얼 재정비 및 정기 모니터링",
            "친절 인센티브 + 불친절 다발 시간대 별도 교육",
            "VIP 응대 표준화 SOP 도입",
        ],
        "터미널": [
            "혼잡 시간대 동선·표지 재정비",
            "휴게/편의시설(의자·화장실·실내온습도) 점검 주기 단축",
            "교통약자·임산부 시설 확대 검토",
        ],
        "주차장": [
            "주차예약제 UX 개선(앱·웹사이트 결제 단순화)",
            "다자녀가구·장애인 할인 안내 강화",
            "성수기 임시주차장 확보, 주차장 실시간 만차 알림",
        ],
        "보안검색": [
            "보안검색 대기시간 실시간 표시 + 분산 유도",
            "기내반입 금지품목 사전 안내 강화(앱 푸시)",
            "노약자·임산부 우선 레인 운영",
        ],
        "ID체크": [
            "ID체크 대기줄 동선 재설계",
            "비대면(셀프) ID체크 단말 시범 도입",
            "피크 시간대 인력 가변 배치",
        ],
        "연계교통": [
            "택시·셔틀버스 배차 정보 통합 안내(K-MaaS 연계)",
            "심야·새벽 교통편 확대",
            "환승 결제 단순화(QR/모바일)",
        ],
        "상업시설": [
            "식음료·기내반입 정보 일원화 안내판",
            "공항 내 가격 정책 모니터링",
            "심야 운영 매장 확대 검토",
        ],
        "공사업무": [
            "공사 일정·구간 사전 공지 강화",
            "공사 중 우회 동선 명확화",
            "소음·분진 저감 대책 공시",
        ],
        "항공사": [
            "항공사 카운터 위치·운영시간 안내 정확도 개선",
            "결항·지연 시 공항 측 정보 제공 체계 정비",
        ],
    }
    return book.get(category, ["카테고리 맞춤 개선책 협의 필요"])


# ---------------------------------------------------------------------------
# Sidebar — navigation
# ---------------------------------------------------------------------------
DATA = load_all()
ppi, summary, clusters, keywords, df = DATA["ppi"], DATA["summary"], DATA["clusters"], DATA["keywords"], DATA["df"]

st.sidebar.title("✈️ KAC-PPI")
st.sidebar.caption("공항 페인포인트 진단지표")
page = st.sidebar.radio(
    "페이지",
    ["① 전국 진단", "② 공항별 PPI 카드", "③ 클러스터링 뷰", "④ 처방 추천"],
)
st.sidebar.markdown("---")
st.sidebar.metric("분석 대상 VOC", f"{DATA['n_voc']:,}건")
st.sidebar.metric("가명화 pseudo-user", f"{DATA['n_users']:,}명")
st.sidebar.metric("기간", "2023.01 ~ 2025.12")
st.sidebar.metric("공항 수", f"{len(A.TARGET_AIRPORTS)}개")
if DATA["mode"] == "csv":
    st.sidebar.caption("📊 사전계산 데이터 모드 (원본 VOC 비공개)")
st.sidebar.markdown(
    "<small style='color:#6b7280'>2026 국토교통 데이터활용 경진대회<br>제품·서비스 개발 부문 출품작</small>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Page 1 — 전국 진단
# ---------------------------------------------------------------------------
if page == "① 전국 진단":
    st.title("① 전국 공항 페인포인트 진단")
    st.caption("KAC VOC 10,378건 기반 — 공항별 페인포인트 상대 비교")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("최고 PPI 공항", summary.index[0], f"{summary['종합_PPI'].iloc[0]:.1f}")
    c2.metric("최다 불편불만 공항", summary["총_불편불만"].idxmax(),
              f"{int(summary['총_불편불만'].max()):,}건")
    c3.metric("2023→2025 VOC 증가율", "+22.8%", "공항 전체")
    c4.metric("국민신문고 유입 증가", "2.4배", "2023→2025")

    st.subheader("공항별 종합 PPI 순위")
    fig = px.bar(
        summary.reset_index(),
        x="종합_PPI", y="대상공항", orientation="h",
        color="종합_PPI", color_continuous_scale="Reds",
        text="종합_PPI", height=380,
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=0, r=0, t=10, b=10))
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    st.plotly_chart(fig, width="stretch")

    st.subheader("PPI 히트맵 — 공항 × 카테고리")
    fig2 = px.imshow(
        ppi, text_auto=".0f", aspect="auto",
        color_continuous_scale="Reds", height=380,
        labels=dict(x="카테고리", y="공항", color="PPI"),
    )
    st.plotly_chart(fig2, width="stretch")

    st.info(
        "**해석 가이드** — PPI는 카테고리별 4가지 컴포넌트(불편불만 상대 점유 0.4 / "
        "1만명당 정규화 건수 0.3 / 시계열 증가율 0.2 / 감정 가중치 0.1)의 가중합이며, "
        "카테고리 내 공항 간 상대비교 형태로 0~100 스케일로 정규화한다."
    )

# ---------------------------------------------------------------------------
# Page 2 — 공항별 PPI 카드
# ---------------------------------------------------------------------------
elif page == "② 공항별 PPI 카드":
    st.title("② 공항별 페인포인트 진단 카드")
    airport = st.selectbox("공항 선택", A.TARGET_AIRPORTS, index=A.TARGET_AIRPORTS.index("청주공항"))

    row = ppi.loc[airport].sort_values(ascending=False)
    top_cat = row.index[0]
    top_score = row.iloc[0]
    overall = summary.loc[airport, "종합_PPI"]

    c1, c2, c3 = st.columns(3)
    c1.metric("종합 PPI", f"{overall:.1f}", ppi_label(overall))
    c2.metric("1위 페인포인트", top_cat, f"PPI {top_score:.1f}")
    c3.metric("총 불편불만(3년)", f"{int(summary.loc[airport, '총_불편불만']):,}건")

    st.subheader("페인포인트 레이더 차트")
    fig = go.Figure(data=go.Scatterpolar(
        r=row.values.tolist() + [row.values[0]],
        theta=row.index.tolist() + [row.index[0]],
        fill='toself', line=dict(color="#E03131"),
        name=airport,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False, height=400, margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader(f"{top_cat} 상위 키워드 (내용분류3)")
    kw = keywords[keywords["카테고리"] == top_cat].head(5)
    if not kw.empty:
        st.dataframe(kw, hide_index=True, width="stretch")
    else:
        st.write("해당 카테고리 키워드 데이터 없음")

    st.subheader("권장 액션 (Top 3)")
    for i, a in enumerate(action_for(top_cat, top_score), 1):
        st.markdown(f"**{i}.** {a}")

# ---------------------------------------------------------------------------
# Page 3 — 클러스터링 뷰
# ---------------------------------------------------------------------------
elif page == "③ 클러스터링 뷰":
    st.title("③ 페인포인트 패턴 유사 공항 클러스터링")
    st.caption("K-Means(k=4)로 공항 9곳을 4개 그룹으로 분류 → 그룹 단위 표준 처방 도출")

    cluster_df = clusters.reset_index()
    cluster_df.columns = ["공항", "클러스터"]
    cluster_df["종합_PPI"] = cluster_df["공항"].map(summary["종합_PPI"])
    cluster_df["1위_카테고리"] = cluster_df["공항"].map(summary["1위_카테고리"])
    for cl in sorted(cluster_df["클러스터"].unique()):
        members = cluster_df[cluster_df["클러스터"] == cl]
        with st.expander(f"🟢 {cl} — {len(members)}개 공항", expanded=True):
            st.dataframe(members.drop(columns=["클러스터"]), hide_index=True, width="stretch")

    st.subheader("클러스터별 평균 PPI 프로필")
    cluster_df["클러스터_그룹"] = cluster_df["클러스터"]
    profile = ppi.copy()
    profile["클러스터"] = clusters
    avg_by_cluster = profile.groupby("클러스터").mean()
    fig = px.imshow(
        avg_by_cluster, text_auto=".0f", aspect="auto",
        color_continuous_scale="Reds", height=320,
        labels=dict(x="카테고리", y="클러스터", color="평균 PPI"),
    )
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Page 4 — 처방 추천
# ---------------------------------------------------------------------------
else:
    st.title("④ 데이터 기반 서비스 개선 처방 추천")
    st.caption("PPI 점수 60점 이상 공항×카테고리를 자동 추출 → 권장 액션 + 우선순위")

    threshold = st.slider("PPI 임계값", 40, 90, 60, step=5)
    flat = ppi.reset_index().melt(id_vars="대상공항", var_name="카테고리", value_name="PPI")
    flat = flat[flat["PPI"] >= threshold].sort_values("PPI", ascending=False)
    flat["우선순위"] = flat["PPI"].apply(lambda v: "★★★ 즉시" if v >= 75 else ("★★ 단기" if v >= 60 else "★ 모니터링"))

    st.metric("처방 대상 (공항×카테고리)", f"{len(flat)}건")
    if flat.empty:
        st.warning("임계값을 낮춰 보세요.")
    else:
        for _, r in flat.iterrows():
            with st.container():
                st.markdown(f"### {r['대상공항']} — {r['카테고리']} `PPI {r['PPI']:.1f}` {r['우선순위']}")
                actions = action_for(r["카테고리"], r["PPI"])
                for i, a in enumerate(actions, 1):
                    st.markdown(f"- **액션 {i}.** {a}")
                st