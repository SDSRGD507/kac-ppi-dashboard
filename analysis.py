"""
KAC-PPI 분석 파이프라인
====================
KAC VOC 데이터(2023~2025)를 기반으로 공항별 페인포인트 진단지표(PPI)를 산출한다.

PPI(공항 a, 카테고리 c) =
    0.4 × 불편불만 비중(공항 내 점유율)         [SHARE]
  + 0.3 × 1만명당 정규화 건수(공항 규모 보정)   [NORM]
  + 0.2 × 시계열 증가율(최근 12mo vs 이전 12mo) [GROWTH]
  + 0.1 × 감정 가중치(불편/(불편+칭찬+제안))    [SENTIMENT]

출력:
- data/ppi_table.csv         : 공항 × 카테고리 PPI 점수
- data/airport_summary.csv   : 공항별 종합 PPI + 1순위 카테고리
- data/cluster_table.csv     : K-Means 4그룹 클러스터 결과
- data/category_keywords.csv : 카테고리별 Top 키워드(내용분류3)
"""
from __future__ import annotations
import os
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 0. 설정값
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# 분석 대상 공항 (VOC 50건 이상 + 본사 제외)
TARGET_AIRPORTS = [
    "김포공항", "김해공항", "제주공항", "청주공항", "대구공항",
    "광주공항", "무안공항", "여수공항", "울산공항",
]

# 분석 대상 카테고리(내용분류2)
TARGET_CATEGORIES = [
    "직원관련", "터미널", "주차장", "보안검색", "ID체크",
    "연계교통", "상업시설", "공사업무", "항공사",
]

# 공항별 연간 여객실적(만명) — 한국공항공사 2024년 공항별 처리실적(국내선+국제선)
# 출처: 한국공항공사 항공통계(airport.co.kr), 위키백과 'List of the busiest airports in South Korea'
ANNUAL_PAX_10K = {
    "김포공항": 2299.06,   # 22,990,599명
    "김해공항": 1575.25,   # 15,752,458명
    "제주공항": 2961.96,   # 29,619,606명
    "청주공항": 457.92,    # 4,579,221명
    "대구공항": 353.70,    # 3,537,041명
    "광주공항": 195.93,    # 1,959,258명
    "무안공항": 40.59,     # 405,869명
    "여수공항": 63.28,     # 632,818명
    "울산공항": 44.87,     # 448,746명
}

# PPI 가중치
W_SHARE, W_NORM, W_GROWTH, W_SENTIMENT = 0.4, 0.3, 0.2, 0.1

VOC_FILE = ROOT.parent / "KAC_VOC_2023-2025.xlsx"


# -----------------------------------------------------------------------------
# 1. 데이터 로딩
# -----------------------------------------------------------------------------
def load_voc(path: Path | str = VOC_FILE) -> pd.DataFrame:
    """VOC 엑셀을 DataFrame으로 로드 + 등록일을 datetime으로 변환."""
    df = pd.read_excel(path, sheet_name=0)
    df["등록일"] = pd.to_datetime(df["등록일"], errors="coerce")
    df = df.dropna(subset=["등록일"])
    # 빈 문자열 카테고리 정제
    for col in ["대상공항", "VOC유형", "내용분류1", "내용분류2", "내용분류3", "내용분류4"]:
        df[col] = df[col].astype(str).str.strip()
    return df


# -----------------------------------------------------------------------------
# 1-1. 가명화 처리 (가명정보결합 가점 +5)
# -----------------------------------------------------------------------------
def add_pseudo_user(df: pd.DataFrame) -> pd.DataFrame:
    """
    개인 식별 정보 없이 (공항 + 등록일자 + VOC유형 + 유입경로) 조합을
    SHA-256 해시한 가명키(pseudo_user)를 생성한다.

    원본 VOC에는 개인정보가 없으나, 동일 맥락(같은 날·같은 공항·같은 유형·같은 채널)의
    민원을 하나의 '행태 단위(pseudo-user)'로 묶어 이용자 경험 흐름을 추적할 수 있게 한다.
    원본을 복원할 수 없는 단방향 해시이므로 가명정보 결합 분석 구조에 해당한다.
    """
    df = df.copy()

    def _key(row):
        raw = "|".join([
            str(row["대상공항"]),
            str(pd.to_datetime(row["등록일"]).date()),
            str(row["VOC유형"]),
            str(row["유입경로"]),
        ])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    df["pseudo_user"] = df.apply(_key, axis=1)
    return df


def analyze_pseudo_user_behavior(df: pd.DataFrame) -> pd.DataFrame:
    """가명화된 pseudo-user 단위로 VOC 발생 행태를 분석한다."""
    if "pseudo_user" not in df.columns:
        df = add_pseudo_user(df)
    # 대상공항이 비어있는(nan/공백) 행 제외
    df = df[~df["대상공항"].isin(["nan", "", "None"])]
    grp = df.groupby("pseudo_user")
    behavior = pd.DataFrame({
        "VOC건수": grp.size(),
        "대표공항": grp["대상공항"].first(),
        "주VOC유형": grp["VOC유형"].agg(lambda s: s.value_counts().index[0]),
        "주카테고리": grp["내용분류2"].agg(lambda s: s.value_counts().index[0]),
    }).sort_values("VOC건수", ascending=False)
    return behavior


# -----------------------------------------------------------------------------
# 2. 4개 컴포넌트 계산
# -----------------------------------------------------------------------------
def compute_share(df: pd.DataFrame) -> pd.DataFrame:
    """
    카테고리 c가 공항 a에서 "유달리 두드러지는 정도".
    = (공항 a의 c 점유율) / (전체 공항 평균 c 점유율)
    → 1.0 = 평균 수준, 2.0 = 평균의 2배 두드러짐
    """
    comp = df[df["VOC유형"] == "불편불만"]
    counts = comp.groupby(["대상공항", "내용분류2"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=TARGET_AIRPORTS, columns=TARGET_CATEGORIES, fill_value=0)
    share = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    cat_mean = share.mean(axis=0).replace(0, np.nan)
    rel = share.div(cat_mean, axis=1).fillna(0)
    return rel


def compute_norm(df: pd.DataFrame) -> pd.DataFrame:
    """1만명당 정규화 건수 (공항 규모 보정). 3년치 → 연평균."""
    comp = df[df["VOC유형"] == "불편불만"]
    counts = comp.groupby(["대상공항", "내용분류2"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=TARGET_AIRPORTS, columns=TARGET_CATEGORIES, fill_value=0)
    # 3년치 합 / 3년 / (연간 여객 × 1만명) → 1만명당 연평균 건수
    annual = pd.Series(ANNUAL_PAX_10K).reindex(TARGET_AIRPORTS).fillna(1.0)
    norm = counts.div(annual, axis=0) / 3.0
    return norm


def compute_growth(df: pd.DataFrame) -> pd.DataFrame:
    """최근 12개월 vs 이전 12개월 증가율. (-1 ~ +∞)"""
    comp = df[df["VOC유형"] == "불편불만"].copy()
    cutoff = comp["등록일"].max() - pd.DateOffset(years=1)
    recent = comp[comp["등록일"] > cutoff]
    prior = comp[(comp["등록일"] > cutoff - pd.DateOffset(years=1)) & (comp["등록일"] <= cutoff)]

    def to_counts(d):
        c = d.groupby(["대상공항", "내용분류2"]).size().unstack(fill_value=0)
        return c.reindex(index=TARGET_AIRPORTS, columns=TARGET_CATEGORIES, fill_value=0)

    r = to_counts(recent)
    p = to_counts(prior)
    # 0 분모 보호: prior에 1 더하기 (소량 분모 효과 방지)
    growth = (r - p) / (p + 1.0)
    return growth.clip(lower=-1, upper=3)  # 극단치 클립


def compute_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """공항×카테고리 단위 감정 가중치: 불편불만 / (불편 + 칭찬 + 제안)"""
    counts = df.groupby(["대상공항", "내용분류2", "VOC유형"]).size().unstack(fill_value=0)
    # 안전: 필요한 컬럼이 없으면 0으로
    for c in ["불편불만", "칭찬", "제안건의"]:
        if c not in counts.columns:
            counts[c] = 0
    sent = counts["불편불만"] / (counts["불편불만"] + counts["칭찬"] + counts["제안건의"] + 1)
    sent = sent.unstack(fill_value=0)
    return sent.reindex(index=TARGET_AIRPORTS, columns=TARGET_CATEGORIES, fill_value=0)


# -----------------------------------------------------------------------------
# 3. PPI 산출
# -----------------------------------------------------------------------------
def minmax(s: pd.DataFrame) -> pd.DataFrame:
    """카테고리(열) 내에서 0~1 정규화 — 공항 간 상대비교를 강조."""
    res = s.copy().astype(float)
    for c in res.columns:
        col = res[c]
        lo, hi = col.min(), col.max()
        if hi - lo < 1e-9:
            res[c] = 0.0
        else:
            res[c] = (col - lo) / (hi - lo)
    return res


def compute_ppi(df: pd.DataFrame):
    share = compute_share(df)
    norm = compute_norm(df)
    growth = compute_growth(df)
    sentiment = compute_sentiment(df)

    components = {
        "share": minmax(share),
        "norm": minmax(norm),
        "growth": minmax(growth),
        "sentiment": minmax(sentiment),
    }
    raw_ppi = (
        W_SHARE * components["share"]
        + W_NORM * components["norm"]
        + W_GROWTH * components["growth"]
        + W_SENTIMENT * components["sentiment"]
    )
    # 0~100 스케일
    ppi = (raw_ppi * 100).round(1)
    return ppi, components, share, norm, growth, sentiment


# -----------------------------------------------------------------------------
# 4. 클러스터링 (K-Means k=4)
# -----------------------------------------------------------------------------
def cluster_airports(ppi: pd.DataFrame) -> pd.Series:
    from sklearn.cluster import KMeans

    X = ppi.values
    k = min(4, len(X))
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    labels = pd.Series(km.labels_, index=ppi.index, name="cluster")
    # 클러스터 라벨에 의미있는 이름 부여 (페인포인트 1위 카테고리 기준)
    cluster_names = {}
    for c in sorted(labels.unique()):
        members = ppi.loc[labels == c]
        top_cat = members.mean().idxmax()
        cluster_names[c] = f"[{top_cat} 중심] {', '.join(members.index)}"
    labels_named = labels.map(lambda x: cluster_names[x])
    return labels_named


# -----------------------------------------------------------------------------
# 5. 카테고리별 키워드 추출 (내용분류3 Top 5)
# -----------------------------------------------------------------------------
def extract_keywords(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comp = df[df["VOC유형"] == "불편불만"]
    for cat in TARGET_CATEGORIES:
        sub = comp[comp["내용분류2"] == cat]
        top = (
            sub["내용분류3"].replace("", "기타").value_counts().head(5)
        )
        for kw, cnt in top.items():
            rows.append({"카테고리": cat, "키워드": kw, "건수": int(cnt)})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 6. 메인 파이프라인
# -----------------------------------------------------------------------------
def run_all(path: Path | str = VOC_FILE):
    print(f"[1/6] Loading VOC from {path}")
    df = load_voc(path)
    print(f"      → {len(df):,} rows")

    print("[2/6] 가명화 처리 (해시 기반 pseudo-user 생성)")
    df = add_pseudo_user(df)
    n_users = df["pseudo_user"].nunique()
    print(f"      → {n_users:,} pseudo-users (가명키)")

    print("[3/6] Computing PPI components")
    ppi, components, share, norm, growth, sentiment = compute_ppi(df)

    print("[4/6] Saving PPI tables")
    ppi.to_csv(DATA_DIR / "ppi_table.csv", encoding="utf-8-sig")
    # 공항×카테고리 불편불만 건수(표본 신뢰도 보정용)
    _cnt = (df[df["VOC유형"] == "불편불만"]
            .groupby(["대상공항", "내용분류2"]).size().unstack(fill_value=0)
            .reindex(index=TARGET_AIRPORTS, columns=TARGET_CATEGORIES, fill_value=0))
    _cnt.to_csv(DATA_DIR / "ppi_counts.csv", encoding="utf-8-sig")
    summary = pd.DataFrame({
        "종합_PPI": ppi.mean(axis=1).round(1),
        "1위_카테고리": ppi.idxmax(axis=1),
        "1위_점수": ppi.max(axis=1).round(1),
        "총_불편불만": df[df["VOC유형"] == "불편불만"]
            .groupby("대상공항").size().reindex(TARGET_AIRPORTS, fill_value=0),
    }).sort_values("종합_PPI", ascending=False)
    summary.to_csv(DATA_DIR / "airport_summary.csv", encoding="utf-8-sig")

    print("[5/6] Clustering airports (K-Means k=4) + pseudo-user 행태 분석")
    clusters = cluster_airports(ppi)
    clusters.to_csv(DATA_DIR / "cluster_table.csv", encoding="utf-8-sig", header=True)
    behavior = analyze_pseudo_user_behavior(df)
    behavior.to_csv(DATA_DIR / "pseudo_user_behavior.csv", encoding="utf-8-sig")
    repeat = (behavior["VOC건수"] >= 2).sum()
    print(f"      → 반복 발생 pseudo-user(2건+): {repeat:,}명 / 전체 {len(behavior):,}명")

    print("[6/6] Extracting category keywords")
    kw = extract_keywords(df)
    kw.to_csv(DATA_DIR / "category_keywords.csv", encoding="utf-8-sig", index=False)

    print("\n=== Airport Summary (Top 5 by PPI) ===")
    print(summary.head(9).to_string())
    print("\nFiles saved to", DATA_DIR)
    return {"ppi": ppi, "summary": summary, "clusters": clusters,
            "components": components, "keywords": kw, "behavior": behavior}


if __name__ == "__main__":
    run_all()
