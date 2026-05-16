import streamlit as st
import pandas as pd
from math import sqrt

# =========================
# PLO4 EV Memory / Predictor Tool
# =========================
# 目的:
# 1. position, hand, ev を1つずつ手入力して覚えさせる
# 2. 登録済みEVをCSVで保存・再アップロードできる
# 3. 登録済みデータをもとに、同じポジション内で未入力ハンドのEVを近似予測する
#
# requirements.txt:
# streamlit
# pandas

RANKS = "23456789TJQKA"
RANK_VALUE = {r: i for i, r in enumerate(RANKS, start=2)}
SUITS = "cdhs"
POSITIONS = ["UTG", "HJ", "CO", "BTN", "SB"]
DEFAULT_OPEN_PCT = {
    "UTG": 18.1,
    "HJ": 24.0,
    "CO": 36.0,
    "BTN": 60.0,
    "SB": 45.0,
}


# ---------- hand parsing ----------

def parse_hand(hand: str):
    """AdAc8h3h -> ['Ad', 'Ac', '8h', '3h']"""
    hand = str(hand).strip()
    if len(hand) != 8:
        return []
    cards = [hand[i:i+2] for i in range(0, 8, 2)]
    valid = all(len(c) == 2 and c[0] in RANKS and c[1] in SUITS for c in cards)
    if not valid:
        return []
    if len(set(cards)) != 4:
        return []
    return cards


def ranks_of(cards):
    return [c[0] for c in cards]


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


def pair_count(cards):
    counts = pd.Series(ranks_of(cards)).value_counts().tolist()
    return sum(1 for x in counts if x >= 2)


def max_duplicate(cards):
    return max(pd.Series(ranks_of(cards)).value_counts().tolist())


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


def rank_values_sorted(cards):
    vals = sorted([RANK_VALUE[c[0]] for c in cards], reverse=True)
    return vals


def connectedness_score(cards):
    vals = sorted(set([RANK_VALUE[c[0]] for c in cards]))
    if len(vals) <= 1:
        return 0
    gaps = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
    return sum(max(0, 5 - g) for g in gaps)


def high_card_score(cards):
    return sum(RANK_VALUE[c[0]] for c in cards)


def canonical_key_from_cards(cards):
    """
    スート名の違いを無視して、同じ構成のハンドを同じキーにする。

    例:
    AsAcKsKc と AhAcKhKc はどちらも同じ canonical_key になる。

    考え方:
    1. ランクの高い順にカードを並べる
    2. 出てきたスートに順番に a,b,c,d を割り当てる
    3. ランク + 正規化スートで表現する
    """
    sorted_cards = sorted(
        cards,
        key=lambda c: (-RANK_VALUE[c[0]], c[1])
    )

    suit_map = {}
    next_label_ord = ord("a")
    parts = []

    for c in sorted_cards:
        rank = c[0]
        suit = c[1]
        if suit not in suit_map:
            suit_map[suit] = chr(next_label_ord)
            next_label_ord += 1
        parts.append(rank + suit_map[suit])

    return "".join(parts)


def canonical_key(hand: str):
    cards = parse_hand(hand)
    if not cards:
        return ""
    return canonical_key_from_cards(cards)


def feature_row(hand: str):
    cards = parse_hand(hand)
    if not cards:
        return None

    vals = rank_values_sorted(cards)
    while len(vals) < 4:
        vals.append(0)

    sp = suit_pattern(cards)
    cat = category(cards)

    return {
        "hand": hand.strip(),
        "canonical_key": canonical_key_from_cards(cards),
        "category": cat,
        "is_aaxx": int(is_aaxx(cards)),
        "is_aaaa": int(is_aaaa(cards)),
        "side_cards": side_cards_for_aa(cards) if is_aaxx(cards) else "",
        "suit_pattern": sp,
        "ace_suited_count": ace_suited_count(cards),
        "pair_count": pair_count(cards),
        "max_duplicate": max_duplicate(cards),
        "r1": vals[0],
        "r2": vals[1],
        "r3": vals[2],
        "r4": vals[3],
        "high_card_score": high_card_score(cards),
        "connectedness": connectedness_score(cards),
        "sp_ds": int(sp == "ds"),
        "sp_ss": int(sp == "ss"),
        "sp_ts": int(sp == "ts"),
        "sp_mono": int(sp == "mono"),
        "sp_rainbow": int(sp == "rainbow"),
        "cat_AAxx": int(cat == "AAxx"),
        "cat_AAAx": int(cat == "AAAx"),
        "cat_AAAA": int(cat == "AAAA"),
        "cat_nonAA": int(cat == "non-AA"),
    }


def normalize_position(pos):
    pos = str(pos).strip().upper()
    aliases = {
        "MP": "HJ",
        "UTG1": "HJ",
        "BU": "BTN",
        "BUTTON": "BTN",
        "BB": "BB",
    }
    return aliases.get(pos, pos)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        hand = str(r.get("hand", "")).strip()
        ev = r.get("ev", None)
        position = normalize_position(r.get("position", "UTG"))
        if position not in POSITIONS:
            position = "UTG"

        fr = feature_row(hand)
        if fr is None:
            continue
        fr["position"] = position
        fr["ev"] = pd.to_numeric(ev, errors="coerce")
        rows.append(fr)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------- prediction ----------

FEATURE_COLS = [
    "is_aaxx", "is_aaaa", "ace_suited_count", "pair_count", "max_duplicate",
    "r1", "r2", "r3", "r4", "high_card_score", "connectedness",
    "sp_ds", "sp_ss", "sp_ts", "sp_mono", "sp_rainbow",
    "cat_AAxx", "cat_AAAx", "cat_AAAA", "cat_nonAA",
]

WEIGHTS = {
    "is_aaxx": 5.0,
    "is_aaaa": 8.0,
    "ace_suited_count": 2.5,
    "pair_count": 2.0,
    "max_duplicate": 1.5,
    "r1": 0.5,
    "r2": 0.5,
    "r3": 0.6,
    "r4": 0.6,
    "high_card_score": 0.08,
    "connectedness": 0.8,
    "sp_ds": 2.5,
    "sp_ss": 1.2,
    "sp_ts": 1.0,
    "sp_mono": 0.8,
    "sp_rainbow": 1.2,
    "cat_AAxx": 4.0,
    "cat_AAAx": 4.0,
    "cat_AAAA": 5.0,
    "cat_nonAA": 4.0,
}


def distance(a: dict, b: pd.Series):
    total = 0.0
    for col in FEATURE_COLS:
        w = WEIGHTS.get(col, 1.0)
        av = float(a.get(col, 0))
        bv = float(b.get(col, 0))
        total += w * ((av - bv) ** 2)
    return sqrt(total)


def predict_ev(target_hand: str, learned: pd.DataFrame, position: str, k: int = 8):
    target = feature_row(target_hand)
    if target is None:
        return None, None, "invalid_hand"

    position = normalize_position(position)
    data = learned[(learned["ev"].notna()) & (learned["position"] == position)].copy()

    if len(data) == 0:
        return None, None, "no_position_data"

    target_key = target.get("canonical_key", "")
    same = data[data["canonical_key"] == target_key]
    if len(same) > 0:
        return float(same.iloc[0]["ev"]), same.assign(distance=0.0).head(1), "exact"

    data["distance"] = data.apply(lambda row: distance(target, row), axis=1)
    nearest = data.sort_values("distance", ascending=True).head(k).copy()
    nearest["weight"] = 1 / (nearest["distance"] + 0.001)
    pred = (nearest["ev"] * nearest["weight"]).sum() / nearest["weight"].sum()
    return float(pred), nearest, "predicted"


def assign_open_by_ev(df: pd.DataFrame, target_pct: float, position: str) -> pd.DataFrame:
    df = df[df["position"] == position].copy()
    total = len(df)
    target_n = round(total * target_pct / 100)

    df["open_auto"] = 0

    # ここは今までの思想を残している。AAxxはAAAA以外open固定。
    # ポジション別に厳密化するなら、後でpositionごとにルールを分ける。
    df.loc[df["is_aaxx"] == 1, "open_auto"] = 1
    df.loc[df["is_aaaa"] == 1, "open_auto"] = 0

    forced_open_n = int(df["open_auto"].sum())
    remaining_n = max(target_n - forced_open_n, 0)

    candidates = df[(df["open_auto"] == 0) & (df["is_aaaa"] == 0) & (df["ev"].notna())]
    selected_idx = candidates.sort_values("ev", ascending=False).head(remaining_n).index
    df.loc[selected_idx, "open_auto"] = 1
    return df


# ---------- Streamlit UI ----------

st.set_page_config(page_title="PLO4 EV Memory", layout="wide")
st.title("PLO4 EV Memory / Position EV Predictor")

st.markdown(
    """
EVを **ポジション別** に1つずつ登録して、登録済みデータから未入力ハンドのEVを近似します。  
重要: この予測はGTOソルバーではなく、あなたが入力したEVデータに基づく近似です。

### 基本運用
1. 最初に、前回保存した `plo4_ev_memory.csv` を左側から読み込む  
2. `position` を選び、新しい `hand` と `ev` を登録する  
3. 最後に必ず `登録済みEVをCSVで保存` を押す  
4. 次回は、その保存した `plo4_ev_memory.csv` をまた読み込む  

保存しないで閉じると、その回に入力したEVは消えます。
"""
)

if "manual_rows" not in st.session_state:
    st.session_state.manual_rows = []

# 既存CSVの読み込み
st.sidebar.header("データ")
uploaded = st.sidebar.file_uploader("登録済みCSVを読み込む", type=["csv"])

if uploaded is not None:
    raw = pd.read_csv(uploaded)
    if "hand" in raw.columns and "ev" in raw.columns:
        if "position" not in raw.columns:
            raw["position"] = "UTG"
            st.sidebar.warning("古いCSV形式だったため、全データをUTGとして読み込みました。")
        loaded = raw[["position", "hand", "ev"]].dropna(subset=["hand"])
        loaded["position"] = loaded["position"].apply(normalize_position)
        st.session_state.manual_rows = loaded.to_dict("records")
        st.sidebar.success(f"{len(loaded)}件読み込みました")
    else:
        st.sidebar.error("CSVには hand と ev の列が必要です")

# 表示・登録対象ポジション
selected_position = st.sidebar.selectbox("表示・予測するポジション", POSITIONS, index=0)

with st.form("add_ev_form"):
    st.subheader("EVを1つ登録")
    c0, c1, c2, c3 = st.columns([1, 2, 1, 1])
    pos_input = c0.selectbox("position", POSITIONS, index=POSITIONS.index(selected_position))
    hand_input = c1.text_input("hand", placeholder="例: AdAc8h3h")
    ev_input = c2.number_input("ev", value=0.0, step=0.01, format="%.4f")
    submitted = c3.form_submit_button("登録")

    if submitted:
        fr = feature_row(hand_input)
        if fr is None:
            st.error("ハンド形式が不正です。例: AdAc8h3h")
        else:
            rows = st.session_state.manual_rows
            # position + hand が同じなら上書き
            rows = [
                r for r in rows
                if not (
                    normalize_position(r.get("position", "UTG")) == pos_input
                    and canonical_key(str(r.get("hand", ""))) == canonical_key(hand_input)
                )
            ]
            rows.append({"position": pos_input, "hand": hand_input.strip(), "ev": ev_input})
            st.session_state.manual_rows = rows
            st.success(f"登録しました: {pos_input} {hand_input.strip()} = {ev_input:.4f}")

learned_raw = pd.DataFrame(st.session_state.manual_rows)

if len(learned_raw) == 0:
    st.info("まずは position, hand, ev を登録してください。例: UTG / AdAc8h3h / 1.75")
    st.code("position,hand,ev\nUTG,AdAc8h3h,1.75\nUTG,KhKdThTd,1.74\nUTG,AdAc4h2h,1.73", language="csv")
    st.stop()

learned = enrich(learned_raw)
learned = learned.sort_values(["position", "hand"]).drop_duplicates(subset=["position", "canonical_key"], keep="last")
current = learned[learned["position"] == selected_position].copy()

st.divider()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("総登録数", f"{len(learned):,}")
col2.metric(f"{selected_position}登録数", f"{len(current):,}")
col3.metric(f"{selected_position} AAxx数", f"{int(current['is_aaxx'].sum()) if len(current) else 0:,}")
col4.metric(f"{selected_position} 平均EV", f"{current['ev'].mean():.4f}" if len(current) else "-")
col5.metric(f"{selected_position} 最高EV", f"{current['ev'].max():.4f}" if len(current) else "-")

st.subheader("ポジション別 登録数")
pos_summary = learned.groupby("position").agg(count=("hand", "count"), avg_ev=("ev", "mean"), max_ev=("ev", "max")).reset_index()
st.dataframe(pos_summary, use_container_width=True)

# 予測
st.subheader("未入力ハンドのEVを近似")
p1, p2, p3 = st.columns([1, 2, 1])
predict_position = p1.selectbox("予測position", POSITIONS, index=POSITIONS.index(selected_position))
predict_hand = p2.text_input("予測したいhand", placeholder="例: AdAc7h2h")
k = p3.slider("近いハンドを何個使うか", min_value=1, max_value=30, value=8)

if predict_hand:
    pred, nearest, status = predict_ev(predict_hand, learned, position=predict_position, k=k)
    if status == "invalid_hand":
        st.warning("ハンド形式が不正です。例: AdAc8h3h")
    elif status == "no_position_data":
        st.warning(f"{predict_position} の登録データがまだありません。このポジションのEVを先に登録してください。")
    elif pred is not None:
        label = "登録済みEV" if status == "exact" else "推定EV"
        st.metric(f"{predict_position} {label}", f"{pred:.4f}")
        st.caption("同じポジション内の近いハンドを加重平均して推定しています。")
        st.dataframe(
            nearest[["position", "hand", "ev", "distance", "category", "side_cards", "suit_pattern", "ace_suited_count", "connectedness"]],
            use_container_width=True,
        )

st.divider()

# 削除機能
st.subheader("登録済みEVを削除")
st.caption("間違って登録した場合は、position と hand を指定して1件削除できます。")

with st.form("delete_ev_form"):
    d0, d1, d2 = st.columns([1, 2, 1])
    delete_position = d0.selectbox("削除position", POSITIONS, index=POSITIONS.index(selected_position))
    delete_hand = d1.text_input("削除するhand", placeholder="例: AdAc8h3h")
    delete_submitted = d2.form_submit_button("削除")

    if delete_submitted:
        fr = feature_row(delete_hand)
        if fr is None:
            st.error("ハンド形式が不正です。例: AdAc8h3h")
        else:
            before = len(st.session_state.manual_rows)
            st.session_state.manual_rows = [
                r for r in st.session_state.manual_rows
                if not (
                    normalize_position(r.get("position", "UTG")) == delete_position
                    and canonical_key(str(r.get("hand", ""))) == canonical_key(delete_hand)
                )
            ]
            after = len(st.session_state.manual_rows)
            if after < before:
                st.success(f"削除しました: {delete_position} {delete_hand.strip()}")
                st.rerun()
            else:
                st.warning(f"該当データが見つかりません: {delete_position} {delete_hand.strip()}")

st.divider()

# 登録済みテーブル
st.subheader(f"登録済みEV: {selected_position}")
search = st.text_input("検索", placeholder="例: AA / AdAc / KKTT")
view = current.copy()

if search:
    s = search.strip()
    view = view[
        view["hand"].str.contains(s, case=False, na=False)
        | view["category"].str.contains(s, case=False, na=False)
        | view["side_cards"].str.contains(s.replace("AA", ""), case=False, na=False)
    ]

if len(view) > 0:
    view = view.sort_values("ev", ascending=False, na_position="last")
    st.dataframe(
        view[["position", "hand", "ev", "category", "side_cards", "suit_pattern", "ace_suited_count", "pair_count", "connectedness"]],
        use_container_width=True,
        height=420,
    )
else:
    st.info(f"{selected_position} の登録データがありません。")

# open判定
st.subheader(f"EV順で{selected_position} open判定")
default_pct = DEFAULT_OPEN_PCT.get(selected_position, 18.1)
target_pct = st.number_input("open %", min_value=0.0, max_value=100.0, value=float(default_pct), step=0.1)

if len(current) > 0:
    range_df = assign_open_by_ev(learned, target_pct, selected_position)
    st.dataframe(
        range_df.sort_values("ev", ascending=False)[["position", "hand", "ev", "open_auto", "category", "side_cards", "suit_pattern", "ace_suited_count"]],
        use_container_width=True,
        height=360,
    )
else:
    range_df = pd.DataFrame()
    st.info("このポジションのデータがないため、open判定はできません。")

# カテゴリ集計
st.subheader(f"カテゴリ集計: {selected_position}")
if len(current) > 0:
    summary = (
        current.groupby(["category", "suit_pattern", "ace_suited_count"], dropna=False)
        .agg(count=("hand", "count"), avg_ev=("ev", "mean"), max_ev=("ev", "max"), min_ev=("ev", "min"))
        .reset_index()
    )
    st.dataframe(summary, use_container_width=True)

# ダウンロード
st.subheader("保存")
out = learned[["position", "hand", "ev"]].sort_values(["position", "hand"])
st.download_button(
    "登録済みEVをCSVで保存",
    data=out.to_csv(index=False).encode("utf-8-sig"),
    file_name="plo4_ev_memory.csv",
    mime="text/csv",
)

if len(range_df) > 0:
    range_out = range_df[["position", "hand", "ev", "open_auto", "category", "side_cards", "suit_pattern", "ace_suited_count"]].sort_values("ev", ascending=False)
    st.download_button(
        f"{selected_position} open判定つきCSVを保存",
        data=range_out.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"plo4_{selected_position.lower()}_range_output.csv",
        mime="text/csv",
    )

if st.button("登録データを画面上からリセット"):
    st.session_state.manual_rows = []
    st.rerun()
