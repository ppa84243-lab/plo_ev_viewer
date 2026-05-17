import streamlit as st
import pandas as pd
from math import sqrt

# =========================
# PLO4 EV Memory / Predictor Tool
# =========================
# requirements.txt:
# streamlit
# pandas

# 表示・ソート順は A K Q J T 9 8 7 6 5 4 3 2
RANKS = "AKQJT98765432"
RANK_VALUE = {
    "A": 14,
    "K": 13,
    "Q": 12,
    "J": 11,
    "T": 10,
    "9": 9,
    "8": 8,
    "7": 7,
    "6": 6,
    "5": 5,
    "4": 4,
    "3": 3,
    "2": 2,
}
SUITS = "cdhs"
POSITIONS = ["UTG", "HJ", "CO", "BTN", "SB"]
TOTAL_PLO4_COMBOS = 270725

DEFAULT_OPEN_PCT = {
    "UTG": 18.1,
    "HJ": 22.9,
    "CO": 30.9,
    "BTN": 47.6,
    "SB": 30.3,
}


# ---------- hand parsing ----------

def parse_hand(hand: str):
    """AdAc8h3h -> ['Ad', 'Ac', '8h', '3h']"""
    hand = str(hand).strip()
    if len(hand) != 8:
        return []
    cards = [hand[i:i + 2] for i in range(0, 8, 2)]
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
    return sorted([RANK_VALUE[c[0]] for c in cards], reverse=True)


def connectedness_score(cards):
    vals = sorted(set([RANK_VALUE[c[0]] for c in cards]))
    if len(vals) <= 1:
        return 0
    gaps = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    return sum(max(0, 5 - g) for g in gaps)


def high_card_score(cards):
    return sum(RANK_VALUE[c[0]] for c in cards)


def rank_string_sort_value(rank_string: str):
    """AKQJT98765432順で文字列ランクを数値化する。例: QQ > TT"""
    if not rank_string:
        return 0
    value = 0
    for r in rank_string:
        value = value * 15 + RANK_VALUE.get(r, 0)
    return value


def hand_sort_value(cards):
    """ハンド全体を A K Q J T 9 ... 2 の順でソートするための数値。"""
    vals = sorted([RANK_VALUE[c[0]] for c in cards], reverse=True)
    value = 0
    for v in vals:
        value = value * 15 + v
    return value


def canonical_key_from_cards(cards):
    """
    スート名の違いを無視して、同じ構成のハンドを同じキーにする。
    実スート c/d/h/s を a/b/c/d に置き換える全パターンを試し、
    その中で一番小さい表現を採用する。
    """
    from itertools import permutations

    if not cards:
        return ""

    labels = "abcd"
    possible_keys = []

    for perm in permutations(labels, 4):
        suit_map = dict(zip(SUITS, perm))
        converted = []
        for card in cards:
            rank = card[0]
            suit = card[1]
            converted.append(rank + suit_map[suit])

        converted = sorted(converted, key=lambda c: (-RANK_VALUE[c[0]], c[1]))
        possible_keys.append("".join(converted))

    return min(possible_keys)


def canonical_key(hand: str):
    cards = parse_hand(hand)
    if not cards:
        return ""
    return canonical_key_from_cards(cards)


def normalize_hand_order(cards):
    """表示用にランク降順、同ランクはスート順で並べる。"""
    ordered = sorted(cards, key=lambda c: (-RANK_VALUE[c[0]], c[1]))
    return "".join(ordered)


def generate_equivalent_hands(hand: str):
    """
    入力ハンドとスート構造が同じハンドをすべて生成する。
    ランク構造とスート接続構造を保ったまま、c/d/h/sを入れ替える。
    """
    from itertools import permutations

    cards = parse_hand(hand)
    if not cards:
        return []

    generated = set()
    for perm in permutations(SUITS, 4):
        suit_map = dict(zip(SUITS, perm))
        new_cards = []
        for card in cards:
            rank = card[0]
            suit = card[1]
            new_cards.append(rank + suit_map[suit])
        generated.add(normalize_hand_order(new_cards))

    return sorted(generated, key=lambda h: (canonical_key(h), h))


def feature_row(hand: str):
    cards = parse_hand(hand)
    if not cards:
        return None

    vals = rank_values_sorted(cards)
    while len(vals) < 4:
        vals.append(0)

    sp = suit_pattern(cards)
    cat = category(cards)
    side_cards = side_cards_for_aa(cards) if is_aaxx(cards) else ""

    return {
        "hand": hand.strip(),
        "canonical_key": canonical_key_from_cards(cards),
        "category": cat,
        "is_aaxx": int(is_aaxx(cards)),
        "is_aaaa": int(is_aaaa(cards)),
        "side_cards": side_cards,
        "side_cards_sort": rank_string_sort_value(side_cards),
        "hand_sort": hand_sort_value(cards),
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


def assign_open_by_positive_ev(df: pd.DataFrame, position: str) -> pd.DataFrame:
    df = df[df["position"] == position].copy()
    df["open_auto"] = (df["ev"] > 0).astype(int)
    return df


# ---------- Streamlit UI ----------

st.set_page_config(page_title="PLO4 EV Memory", layout="wide")
st.title("PLO4 EV Memory / Position EV Predictor")

st.markdown(
    """
EVを **ポジション別** に登録して、登録済みデータから未入力ハンドのEVを近似します。  
重要: この予測はGTOソルバーではなく、あなたが入力したEVデータに基づく近似です。

### 基本運用
1. 前回保存したCSVを左側から追加読み込みする  
2. `position` を選び、新しい `hand` と `ev` を登録する  
3. 最後に必ず `登録済みEVをCSVで保存` を押す  
4. 次回は、その保存したCSVをまた読み込む  

保存しないで閉じると、その回に入力したEVは消えます。
"""
)

if "manual_rows" not in st.session_state:
    st.session_state.manual_rows = []

if "loaded_csv_signatures" not in st.session_state:
    st.session_state.loaded_csv_signatures = set()

# 既存CSVの読み込み
st.sidebar.header("データ")
uploaded_files = st.sidebar.file_uploader(
    "登録済みCSVを追加読み込み",
    type=["csv"],
    accept_multiple_files=True,
    help="分けて保存したCSVを複数選択できます。読み込んだデータは現在の登録データに追加・統合されます。",
)

if uploaded_files:
    added_total = 0
    rows = st.session_state.manual_rows

    for uploaded in uploaded_files:
        csv_signature = f"{uploaded.name}_{uploaded.size}"

        # 同じCSVを画面更新のたびに再読み込みしない
        if csv_signature in st.session_state.loaded_csv_signatures:
            continue

        raw = pd.read_csv(uploaded)
        if "hand" in raw.columns and "ev" in raw.columns:
            if "position" not in raw.columns:
                raw["position"] = "UTG"
                st.sidebar.warning(f"{uploaded.name}: 古いCSV形式だったため、全データをUTGとして読み込みました。")

            loaded = raw[["position", "hand", "ev"]].dropna(subset=["hand"])
            loaded["position"] = loaded["position"].apply(normalize_position)

            for _, item in loaded.iterrows():
                pos = normalize_position(item["position"])
                hand = str(item["hand"]).strip()

                try:
                    ev = round(float(item["ev"]), 2)
                except ValueError:
                    continue

                key = canonical_key(hand)
                if pos not in POSITIONS or key == "":
                    continue

                # CSV読み込み時は position + hand が完全一致したものだけ上書きする。
                # canonical_keyで潰すと、同型スート違いハンドが1件に減ってしまう。
                rows = [
                    r for r in rows
                    if not (
                        normalize_position(r.get("position", "UTG")) == pos
                        and str(r.get("hand", "")).strip().lower() == hand.lower()
                    )
                ]

                rows.append({"position": pos, "hand": hand, "ev": ev})
                added_total += 1

            st.session_state.loaded_csv_signatures.add(csv_signature)
        else:
            st.sidebar.error(f"{uploaded.name}: CSVには hand と ev の列が必要です")

    st.session_state.manual_rows = rows
    if added_total > 0:
        st.sidebar.success(f"CSVから追加読み込み: {added_total}件")

# 表示・登録対象ポジション
selected_position = st.sidebar.selectbox("表示・予測するポジション", POSITIONS, index=0)

with st.form("add_ev_form"):
    st.subheader("EVを1つ登録")
    c0, c1, c2, c3 = st.columns([1, 2, 1, 1])
    pos_input = c0.selectbox("position", POSITIONS, index=POSITIONS.index(selected_position))
    hand_input = c1.text_input("hand", placeholder="例: AdAc8h3h")
    ev_input = c2.number_input("ev", value=0.00, step=0.01, format="%.2f")
    submitted = c3.form_submit_button("登録")

    if submitted:
        fr = feature_row(hand_input)
        if fr is None:
            st.error("ハンド形式が不正です。例: AdAc8h3h")
        else:
            rows = st.session_state.manual_rows
            # 登録時は同型ハンドをまとめて上書きする
            rows = [
                r for r in rows
                if not (
                    normalize_position(r.get("position", "UTG")) == pos_input
                    and canonical_key(str(r.get("hand", ""))) == canonical_key(hand_input)
                )
            ]
            equivalent_hands = generate_equivalent_hands(hand_input)
            for h in equivalent_hands:
                rows.append({"position": pos_input, "hand": h, "ev": round(float(ev_input), 2)})
            st.session_state.manual_rows = rows
            st.success(
                f"登録しました: {pos_input} {hand_input.strip()} = {ev_input:.2f} / "
                f"同型ハンド {len(equivalent_hands)}件を自動追加"
            )

st.subheader("EVをまとめて登録")
st.caption("複数行を貼り付けて一括登録できます。形式は `position,hand,ev` または `hand,ev` です。`hand,ev` の場合は選択中のポジションとして登録します。")

with st.form("bulk_add_ev_form"):
    bulk_text = st.text_area(
        "一括入力",
        placeholder="""例:
UTG,AdAcTdTc,4.33
UTG,AdAcQdQc,4.41
HJ,AsAcKsKc,4.80

または
AdAcTdTc,4.33
AdAcQdQc,4.41""",
        height=180,
    )
    bulk_submitted = st.form_submit_button("まとめて登録")

    if bulk_submitted:
        rows = st.session_state.manual_rows
        added_count = 0
        error_lines = []

        for line_no, line in enumerate(bulk_text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue

            if "," in line:
                parts = [p.strip() for p in line.split(",")]
            elif "\t" in line:
                parts = [p.strip() for p in line.split("\t")]
            else:
                parts = [p.strip() for p in line.split()]

            if len(parts) == 3:
                pos_bulk, hand_bulk, ev_bulk = parts
                pos_bulk = normalize_position(pos_bulk)
            elif len(parts) == 2:
                pos_bulk = selected_position
                hand_bulk, ev_bulk = parts
            else:
                error_lines.append(f"{line_no}行目: 形式が不正")
                continue

            if pos_bulk not in POSITIONS:
                error_lines.append(f"{line_no}行目: positionが不正: {pos_bulk}")
                continue

            fr = feature_row(hand_bulk)
            if fr is None:
                error_lines.append(f"{line_no}行目: handが不正: {hand_bulk}")
                continue

            try:
                ev_bulk = round(float(ev_bulk), 2)
            except ValueError:
                error_lines.append(f"{line_no}行目: evが不正: {ev_bulk}")
                continue

            # 一括登録時も同型ハンドをまとめて上書きする
            rows = [
                r for r in rows
                if not (
                    normalize_position(r.get("position", "UTG")) == pos_bulk
                    and canonical_key(str(r.get("hand", ""))) == canonical_key(hand_bulk)
                )
            ]

            equivalent_hands = generate_equivalent_hands(hand_bulk)
            for h in equivalent_hands:
                rows.append({"position": pos_bulk, "hand": h, "ev": ev_bulk})
                added_count += 1

        st.session_state.manual_rows = rows

        if added_count > 0:
            st.success(f"一括登録しました: {added_count}件")
        if error_lines:
            st.warning("一部登録できない行がありました。")
            st.code("\n".join(error_lines), language="text")

st.subheader("選択中ポジションへ連続登録")
st.caption("GTO Wizardで現在見ているポジションに合わせて、hand と ev を連続で登録します。登録先は左側の『表示・予測するポジション』です。")

with st.form("quick_add_current_position_form"):
    q1, q2, q3 = st.columns([2, 1, 1])
    quick_hand = q1.text_input("hand", placeholder="例: AsAcKsKc", key="quick_hand_current_position")
    quick_ev_text = q2.text_input("ev", placeholder="例: 4.33", key="quick_ev_current_position")
    quick_submitted = q3.form_submit_button(f"{selected_position}に登録")

    if quick_submitted:
        fr = feature_row(quick_hand)
        if fr is None:
            st.error("ハンド形式が不正です。例: AsAcKsKc")
        else:
            try:
                ev_value = round(float(str(quick_ev_text).strip()), 2)
                rows = st.session_state.manual_rows

                rows = [
                    r for r in rows
                    if not (
                        normalize_position(r.get("position", "UTG")) == selected_position
                        and canonical_key(str(r.get("hand", ""))) == canonical_key(quick_hand)
                    )
                ]

                equivalent_hands = generate_equivalent_hands(quick_hand)
                for h in equivalent_hands:
                    rows.append({"position": selected_position, "hand": h, "ev": ev_value})

                st.session_state.manual_rows = rows
                st.success(
                    f"登録しました: {selected_position} {quick_hand.strip()} = {ev_value:.2f} / "
                    f"同型ハンド {len(equivalent_hands)}件を自動追加"
                )
            except ValueError:
                st.error("EV形式が不正です。例: 4.33")

learned_raw = pd.DataFrame(st.session_state.manual_rows)

if len(learned_raw) == 0:
    st.info("まずは position, hand, ev を登録してください。例: UTG / AdAc8h3h / 1.75")
    st.code("position,hand,ev\nUTG,AdAc8h3h,1.75\nUTG,KhKdThTd,1.74\nUTG,AdAc4h2h,1.73", language="csv")
    st.stop()

learned = enrich(learned_raw)
learned = learned.sort_values(["position", "hand"]).drop_duplicates(subset=["position", "hand"], keep="last")
current = learned[learned["position"] == selected_position].copy()

st.divider()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("総登録数", f"{len(learned):,}")
col2.metric(f"{selected_position}登録数", f"{len(current):,}")
col3.metric(f"{selected_position} AAxx数", f"{int(current['is_aaxx'].sum()) if len(current) else 0:,}")
col4.metric(f"{selected_position} 平均EV", f"{current['ev'].mean():.2f}" if len(current) else "-")
col5.metric(f"{selected_position} 最高EV", f"{current['ev'].max():.2f}" if len(current) else "-")

st.subheader("ポジション別 登録数")
pos_summary = learned.groupby("position").agg(
    count=("hand", "count"),
    avg_ev=("ev", "mean"),
    max_ev=("ev", "max"),
).reset_index()
pos_summary["total_pct"] = pos_summary["count"] / TOTAL_PLO4_COMBOS * 100
pos_summary = pos_summary[["position", "count", "total_pct", "avg_ev", "max_ev"]]
st.dataframe(
    pos_summary,
    use_container_width=True,
    column_config={
        "position": "position",
        "count": "登録数",
        "total_pct": st.column_config.NumberColumn("全体%", format="%.4f%%"),
        "avg_ev": st.column_config.NumberColumn("平均EV", format="%.2f"),
        "max_ev": st.column_config.NumberColumn("最高EV", format="%.2f"),
    },
)

# ハンド別・ポジション別EV確認
st.subheader("ハンド別 ポジションEV / R・F")
st.caption("ハンドを入力すると、各ポジションの登録済みEVと、EVがプラスならR、0以下ならFを表示します。スート違いの同型ハンドも同じものとして検索します。")

lookup_hand = st.text_input("確認したいhand", placeholder="例: AsAcKsKc", key="lookup_hand_by_position")

if lookup_hand:
    lookup_key = canonical_key(lookup_hand)
    if lookup_key == "":
        st.warning("ハンド形式が不正です。例: AsAcKsKc")
    else:
        rows = []
        for pos in POSITIONS:
            pos_data = learned[
                (learned["position"] == pos)
                & (learned["canonical_key"] == lookup_key)
            ].copy()

            if len(pos_data) > 0:
                hit = pos_data.sort_values("ev", ascending=False).iloc[0]
                ev = float(hit["ev"])
                action = "R" if ev > 0 else "F"
                rows.append({
                    "position": pos,
                    "registered_hand": hit["hand"],
                    "ev": ev,
                    "R/F": action,
                    "status": "registered",
                })
            else:
                rows.append({
                    "position": pos,
                    "registered_hand": "",
                    "ev": None,
                    "R/F": "-",
                    "status": "not registered",
                })

        lookup_df = pd.DataFrame(rows)
        st.dataframe(
            lookup_df,
            use_container_width=True,
            column_config={
                "position": "position",
                "registered_hand": "登録ハンド",
                "ev": st.column_config.NumberColumn("EV", format="%.2f"),
                "R/F": "R/F",
                "status": "状態",
            },
        )

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
        st.metric(f"{predict_position} {label}", f"{pred:.2f}")
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
            # 削除時は同型ハンドをまとめて消す
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
    view = view.sort_values(
        ["ev", "side_cards_sort", "hand_sort", "hand"],
        ascending=[False, False, False, True],
        na_position="last",
    )
    st.dataframe(
        view[["position", "hand", "ev", "category", "side_cards", "suit_pattern", "ace_suited_count", "pair_count", "connectedness"]],
        use_container_width=True,
        height=420,
        column_config={
            "ev": st.column_config.NumberColumn("EV", format="%.2f"),
        },
    )
else:
    st.info(f"{selected_position} の登録データがありません。")

# R/F確認は「ハンド別 ポジションEV / R・F」に集約する。
range_df = assign_open_by_positive_ev(learned, selected_position) if len(current) > 0 else pd.DataFrame()

# カテゴリ集計
st.subheader(f"カテゴリ集計: {selected_position}")
if len(current) > 0:
    summary = (
        current.groupby(["category", "suit_pattern", "ace_suited_count"], dropna=False)
        .agg(count=("hand", "count"), avg_ev=("ev", "mean"), max_ev=("ev", "max"), min_ev=("ev", "min"))
        .reset_index()
    )
    st.dataframe(
        summary,
        use_container_width=True,
        column_config={
            "avg_ev": st.column_config.NumberColumn("平均EV", format="%.2f"),
            "max_ev": st.column_config.NumberColumn("最高EV", format="%.2f"),
            "min_ev": st.column_config.NumberColumn("最低EV", format="%.2f"),
        },
    )

# ダウンロード
st.subheader("保存")
out = learned[["position", "hand", "ev"]].sort_values(["position", "hand"])

st.download_button(
    "登録済みEVをCSVで保存（全ポジション統合）",
    data=out.to_csv(index=False).encode("utf-8-sig"),
    file_name="plo4_ev_memory.csv",
    mime="text/csv",
)

st.caption("ポジションごとに分けて保存したい場合はこちら。")
for pos in POSITIONS:
    pos_out = out[out["position"] == pos]
    if len(pos_out) == 0:
        continue
    st.download_button(
        f"{pos}だけCSVで保存",
        data=pos_out.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"plo4_ev_memory_{pos.lower()}.csv",
        mime="text/csv",
    )

if st.button("登録データを画面上からリセット"):
    st.session_state.manual_rows = []
    st.session_state.loaded_csv_signatures = set()
    st.rerun()
