import streamlit as st
import pandas as pd

# =========================
# PLO4 EV Viewer / Range Tool
# =========================
# CSVの最低限の形:
# hand,ev
# AdAc8h3h,1.75
# KhKdThTd,1.74
# AdAc4h2h,1.73

RANKS = "23456789TJQKA"
RANK_VALUE = {r: i for i, r in enumerate(RANKS, start=2)}
SUITS = "cdhs"


def parse_hand(hand: str):
    """AdAc8h3h -> ['Ad', 'Ac', '8h', '3h']"""
    hand = str(hand).strip()
    if len(hand) != 8:
        return []
    cards = [hand[i:i+2] for i in range(0, 8, 2)]
    valid = all(len(c) == 2 and c[0] in RANKS and c[1] in SUITS for c in cards)
    return cards if valid else []


def suits_of(cards):
    return [c[1] for c in cards]


def is_aaaa(cards):
    return len(cards) == 4 and all(c[0] == "A" for c in cards)


def is_aaxx(cards):
    return len(cards) == 4 and sum(c[0] == "A" for c in cards) >= 2 and not is_aaaa(cards)


def is_aa_exact(cards):
    return len(cards) == 4 and sum(c[0] == "A" for c in cards) == 2


def is_aaa(cards):
    return len(cards) == 4 and sum(c[0] == "A" for c in cards) == 3


def side_cards_for_aa(cards):
    side = [c for c in cards if c[0] != "A"]
    side = sorted(side, key=lambda c: RANK_VALUE[c[0]], reverse=True)
    return "".join(c[0] for c in side)


def suit_pattern(cards):
    counts = sorted(pd.Series(suits_of(cards)).value_counts().tolist(), reverse=True)

    if counts == [4]:
        return "mono"
    if counts == [3, 1]:
        return "ts"
    if counts == [2, 2]:
        return "ds"
    if counts == [2, 1, 1]:
        return "ss"
    if counts == [1, 1, 1, 1]:
        return "rainbow"
    return "unknown"


def ace_suited_count(cards):
    """
    Aが同スートの非Aカードと何本つながっているか。

    AdAc8h3h = 0
    AdAh8d3h = 2
    AdAc8d3h = 1
    """
    aces = [c for c in cards if c[0] == "A"]
    non_aces = [c for c in cards if c[0] != "A"]

    count = 0
    for a in aces:
        if any(x[1] == a[1] for x in non_aces):
            count += 1

    return count


def category(cards):
    if is_aaaa(cards):
        return "AAAA"
    if is_aaa(cards):
        return "AAAx"
    if is_aa_exact(cards):
        return "AAxx"
    return "non-AA"


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hand"] = df["hand"].astype(str).str.strip()
    df["cards"] = df["hand"].apply(parse_hand)

    df = df[df["cards"].apply(lambda x: len(x) == 4)].copy()

    df["category"] = df["cards"].apply(category)
    df["is_aaxx"] = df["cards"].apply(is_aaxx)
    df["is_aaaa"] = df["cards"].apply(is_aaaa)
    df["side_cards"] = df["cards"].apply(
        lambda c: side_cards_for_aa(c) if is_aaxx(c) else ""
    )
    df["suit_pattern"] = df["cards"].apply(suit_pattern)
    df["ace_suited_count"] = df["cards"].apply(ace_suited_count)
    df["ev"] = pd.to_numeric(df["ev"], errors="coerce")

    return df.drop(columns=["cards"])


def assign_open_by_ev(df: pd.DataFrame, target_pct: float) -> pd.DataFrame:
    """
    EV降順で上位target_pctをopenにする。
    ただしAAxxはAAAA以外すべてopen固定。
    """
    df = df.copy()

    total = len(df)
    target_n = round(total * target_pct / 100)

    df["open_auto"] = 0

    # AAxxは強制open、AAAAはfold
    df.loc[df["is_aaxx"], "open_auto"] = 1
    df.loc[df["is_aaaa"], "open_auto"] = 0

    forced_open_n = int(df["open_auto"].sum())
    remaining_n = max(target_n - forced_open_n, 0)

    candidates = df[
        (df["open_auto"] == 0)
        & (~df["is_aaaa"])
        & (df["ev"].notna())
    ].copy()

    candidates = candidates.sort_values("ev", ascending=False)
    selected_idx = candidates.head(remaining_n).index

    df.loc[selected_idx, "open_auto"] = 1

    return df


st.set_page_config(page_title="PLO4 EV Viewer", layout="wide")

st.title("PLO4 EV Viewer / UTG Range Tool")

st.markdown(
    """
CSVを読み込んで、EV順・AAxx分類・UTGオープン判定を確認するためのツールです。

最低限、CSVには `hand` と `ev` の列が必要です。
"""
)

uploaded = st.file_uploader("EV CSVをアップロード", type=["csv"])

target_pct = st.sidebar.number_input(
    "UTG open %",
    min_value=0.0,
    max_value=100.0,
    value=18.1,
    step=0.1,
)

only_open = st.sidebar.checkbox("openのみ表示", value=False)
only_aaxx = st.sidebar.checkbox("AAxxのみ表示", value=False)
search = st.sidebar.text_input("ハンド検索 例: AdAc8h3h / AA / KKTT")

if uploaded is None:
    st.info("CSVをアップロードしてください。例: hand,ev の2列")
    st.code(
        "hand,ev\nAdAc8h3h,1.75\nKhKdThTd,1.74\nAdAc4h2h,1.73",
        language="csv",
    )
    st.stop()

raw = pd.read_csv(uploaded)

if "hand" not in raw.columns or "ev" not in raw.columns:
    st.error("CSVには hand と ev の列が必要です。")
    st.stop()

df = enrich(raw)
df = assign_open_by_ev(df, target_pct)

view = df.copy()

if only_open:
    view = view[view["open_auto"] == 1]

if only_aaxx:
    view = view[view["is_aaxx"]]

if search:
    s = search.strip()
    view = view[
        view["hand"].str.contains(s, case=False, na=False)
        | view["side_cards"].str.contains(s.replace("AA", ""), case=False, na=False)
        | view["category"].str.contains(s, case=False, na=False)
    ]

col1, col2, col3, col4 = st.columns(4)

col1.metric("総ハンド数", f"{len(df):,}")
col2.metric("open数", f"{int(df['open_auto'].sum()):,}")
col3.metric("open率", f"{df['open_auto'].mean() * 100:.2f}%")
col4.metric("AAxx数", f"{int(df['is_aaxx'].sum()):,}")

st.subheader("EV順テーブル")

view = view.sort_values("ev", ascending=False, na_position="last")

st.dataframe(
    view[
        [
            "hand",
            "ev",
            "open_auto",
            "category",
            "side_cards",
            "suit_pattern",
            "ace_suited_count",
        ]
    ],
    use_container_width=True,
    height=520,
)

st.subheader("カテゴリ集計")

summary = (
    df.groupby(["category", "suit_pattern", "ace_suited_count"], dropna=False)
    .agg(
        count=("hand", "count"),
        avg_ev=("ev", "mean"),
        open_count=("open_auto", "sum"),
    )
    .reset_index()
)

summary["open_rate"] = summary["open_count"] / summary["count"]

st.dataframe(summary, use_container_width=True)

st.subheader("ダウンロード")

out = df[
    [
        "hand",
        "ev",
        "open_auto",
        "category",
        "side_cards",
        "suit_pattern",
        "ace_suited_count",
    ]
].sort_values("ev", ascending=False)

st.download_button(
    "open判定つきCSVをダウンロード",
    data=out.to_csv(index=False).encode("utf-8-sig"),
    file_name="plo4_utg_range_output.csv",
    mime="text/csv",
)
