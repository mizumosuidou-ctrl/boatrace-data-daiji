"""実戦の成績を、過去の検証（ev-check）と同じ物差しで見る表（試験中の買い目。見るだけ）。

  python -m minamo live-check
- 材料は web/data/日付/場-R.json（締切前に決めた買い目・その時の確率とオッズ・結果と配当・照合）。
- 3連単（ev）と2連単（ex）それぞれに：
  回収率と、レースを入れ替えて1000回数え直した幅（下5%〜上95%）・100%を超えた割合、
  過去の検証の数字との比べ、100%超えをはっきり言うのに要るレース数、
  締切順の連敗、平掛けとケリー1/4で資金10万円から買っていたらどうなったか、日ごとの成績。
"""
from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from . import store
from .ml.wind_table import _pad

# 選び方を変えた日（この日から新しい選び方）と、その選び方の過去の検証（回収率・幅。幅が無ければ None）
#   補正なし・最大6点：8/24〜9/10 の2,469R。補正B：後半3,014R（10/3 夜から。10/4 から1日通して）。最大9点は10/4 夕方から（10/5 から1日通して）
#   10/4 までは締切の0〜4分前（平均1.5分前）に決め直した組（人が買うには遅い）。10/5 からは締切の約5分前に決めて固定した組
#   （ev-check「11.」：5分前のオッズで決めると3連単124.2%・2連単120.5%）
RULES = {"ev": [("", "補正なし・最大6点", (94.9, None, None)), ("20261004", "補正B・最大6点", (118.1, 98.8, 137.5)),
                ("20261005", "補正B・最大9点・5分前に固定", (124.2, 104.3, 147.4)),
                # 10/6 から：②が①より0.5以上速いレースは見送り（ev-check「13-5.」3連単128.3%・2連単120.5%、幅は未計算）
                ("20261006", "＋②が速いレースは見送り", (128.3, None, None)),
                # 10/7 から：選んだ組のうちオッズ15〜120倍だけ（ev-check「19.」の前後 121.8・120.0%）
                ("20261007", "＋オッズ15〜120倍の組だけ", (120.9, None, None)),
                # 10/11 から：1点しか残らないレースは見送り（ev-check「22.」の前・後 111.7・146.3%）
                ("20261011", "＋1点だけのレースは見送り", (128.2, None, None))],
         "ex": [("", "補正B・最大3点・直前", (120.0, 109.4, 131.0)), ("20261005", "補正B・最大3点・5分前に固定", (120.5, 110.2, 132.0)),
                ("20261006", "＋②が速いレースは見送り", (120.5, None, None)),
                # 10/7 から：選んだ組のうちオッズ10〜80倍だけ（ev-check「19.」の前後 121.2・113.1%）
                ("20261007", "＋オッズ10〜80倍の組だけ", (117.0, None, None))],
         # 組は3連単と同じ。金額を合成オッズ配分に（ev-check「12-2.」：121.9%・幅98.7〜149.3%）
         "co": [("", "3連単と同じ組・合成オッズ配分", (121.9, 98.7, 149.3)),
                ("20261007", "＋オッズ15〜120倍の組だけ（検証なし）", (None, None, None)),
                ("20261011", "＋1点だけのレースは見送り（検証なし）", (None, None, None))],
         # TIME予想（ユーザーの予想方法）：過去の検証は無い。500レースまでは判断しない
         "time": [("", "TIME予想", (None, None, None))],
         # 隊形①-②（ev-check「15-5.」）：A＝①〈③②④で2連単①-②（609R・115.2%）、B＝②が速く①〈②④③（102R・164.8%、記録だけ）
         "fm": [("", "①〈③②④ → 2連単①-②", (115.2, None, None))],
         "fmb": [("", "②が速い＋①〈②④③（記録だけ）", (164.8, None, None))],
         # 一致（rtm-learn「4.」）：RTMの1番手が①以外で MINAMO も35%以上 → 2連単 その艇-①（レース前の予想 DEEP 134.3%・NORMAL 130.2%）
         "ag": [("", "RTMとMINAMOが①以外の同じ艇 → 2連単 その艇-①", (130.2, None, None))],
         # 記録だけ：同じルールで締切の2分前・1分前に決め直した組（10/7〜。5分前に固定した ev・ex と同じレースで比べる）
         "ev3": [("", "3連単（試し）を3分前に決め直す", (None, None, None)), ("20261011", "＋1点だけのレースは見送り", (None, None, None))],
         "ex3": [("", "2連単（試し）を3分前に決め直す", (None, None, None))],
         "ev2": [("", "3連単（試し）を2分前に決め直す", (None, None, None)), ("20261011", "＋1点だけのレースは見送り", (None, None, None))],
         "ev1": [("", "3連単（試し）を1分前に決め直す", (None, None, None)), ("20261011", "＋1点だけのレースは見送り", (None, None, None))],
         "ex2": [("", "2連単（試し）を2分前に決め直す", (None, None, None))],
         "ex1": [("", "2連単（試し）を1分前に決め直す", (None, None, None))],
         # 2連単（全レース。見送りなし・確率の上位3点）：ユーザーの希望（10/6）。過去の検証は ev-check「6.」の確率上位3点
         "xa": [("", "全レース・MINAMOの確率の上位3点", (None, None, None))],
         # コツコツ当てる君（10/11〜、記録だけ）：普通の予想のうち補正Bの期待値1.2以上（live-check --ev-filter：実戦 103.4%・104.7%）
         "kk": [("", "普通の予想のうち期待値1.2以上", (None, None, None))],
         # 🍒穴狙い🍒（記録だけ。ev-check「16.」：A 74.2%・B 75.7%）
         "ch": [("", "🍒 A：MINAMOの上位12点（①頭以外）", (74.2, None, None))],
         "chb": [("", "🍒 B：攻める艇とその外の頭で12点", (75.7, None, None))]}
TIME_MIN_RACES = 500
LABEL = {"ev": "3連単（期待値1.2以上）", "ex": "2連単（期待値1.2以上）", "co": "3連単（合成オッズ配分。1レースの投資は同じ）",
         "time": "TIME予想（あなたの予想方法・仮想資金）", "fm": "隊形①-②（①〈③②④ → 2連単①-②）",
         "fmb": "隊形①-② B（②が速い＋①〈②④③ → 2連単①-②。記録だけ）",
         "ag": "一致（RTMの1番手が①以外で、MINAMOもその艇を35%以上 → 2連単 その艇-①）",
         "ev3": "3連単（試し）締切3分前に決め直す（記録だけ）", "ex3": "2連単（試し）締切3分前に決め直す（記録だけ）",
         "ev2": "3連単（試し）締切2分前に決め直す（記録だけ）", "ev1": "3連単（試し）締切1分前に決め直す（記録だけ）",
         "ex2": "2連単（試し）締切2分前に決め直す（記録だけ）", "ex1": "2連単（試し）締切1分前に決め直す（記録だけ）",
         "kk": "コツコツ当てる君（普通の予想のうち割安な組だけ。記録だけ）",
         "xa": "2連単（全レース。見送りなし・MINAMOの確率の上位3点）",
         "ch": "🍒穴狙い🍒 A（外の方がスタートが0.4以上速い所があるレースで、イン逃しだけ12点。記録だけ）",
         "chb": "🍒穴狙い🍒 B（攻める艇とその外の頭で12点。記録だけ）"}
# ev-check「10.」の目安：資金10万円・平掛けの1点・ケリー1/4の1点の上限
PLANS = {"ev": (100, 1_000), "ex": (300, 3_000)}
Z90 = 1.645


def rows(data_dir: Path, k: str) -> list[dict]:
    """試験中の買い目を買って結果の出たレースを、締切順に。stake・ret は1点100円で数えた円。
    レースごとのファイル（日付/場-R.json）の settle を読む（day.json の一覧は、古い日だと払戻の欄が無いことがある）。"""
    out, src = [], {"co": "ev", "fmb": "fm", "chb": "ch"}.get(k, k)  # 合成オッズ配分の組・オッズ・時刻は3連単のもの
    exa = k in ("ex", "fm", "fmb", "ag", "xa", "ex3", "ex2", "ex1")  # 2連単
    for path in sorted(Path(data_dir).glob("*/[0-9][0-9]-[0-9][0-9].json")):
        race = store.read_json(path) or {}
        st, res = race.get("settle") or {}, race.get("result") or {}
        if race.get("demo") or res.get("cancelled") or not st.get(f"{k}_bought") or st.get(f"{k}_stake") is None:
            continue
        date, combo = race.get("date") or path.parent.name, res.get("exacta" if exa else "trifecta")
        tp = race.get("time_pick") or {}
        if not combo:
            continue
        out.append({"key": (date, race.get("deadline") or "99:99", race.get("jcd", ""), race.get("rno", 0)),
                    "date": date, "label": f"{date[4:6]}/{date[6:]} {(race.get('venue') or {}).get('name', '')}{race.get('rno', '')}R",
                    "stake": float(st[f"{k}_stake"]), "ret": float(st.get(f"{k}_return") or 0), "hit": bool(st.get(f"{k}_hit")),
                    "result": combo, "pay": (res.get("exacta_payout" if exa else "payout") or 0),
                    "items": [] if k in ("time", "fmb", "chb") else race.get(f"{src}_items") or [],
                    "mins": mins_before(date, race.get("deadline"), tp.get("at") if k == "time" else
                                        (race.get(f"{src}_pick") or {}).get("at") if src in ("fm", "ag", "ch", "xa", "kk") else race.get(f"{src}_at")),
                    "n": len(tp.get("combos") or []) if k == "time" else None,
                    "s12": float(st.get("time12_stake") or 0), "r12": float(st.get("time12_return") or 0),
                    "in_escape": st.get("time_in_escape"), "edge": (race.get("ch_pick") or {}).get("edge")})
    out.sort(key=lambda x: x["key"])
    return out


def mins_before(date: str, deadline: str | None, at: str | None) -> float | None:
    """買い目を決めた時刻が、締切の何分前か（分からなければ None）。"""
    try:
        hh, mm = map(int, (deadline or "").split(":"))
        when = datetime.fromisoformat(at)
    except (ValueError, TypeError):
        return None
    dl = datetime.strptime(date, "%Y%m%d").replace(hour=hh, minute=mm, tzinfo=store.JST)
    return (dl - when.astimezone(store.JST)).total_seconds() / 60


def boot(st: np.ndarray, rt: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float, float, float]:
    """回収率・下5%・上95%・100%を超えた割合（レースを入れ替えて n 回数え直す）。"""
    if not len(st):
        return (float("nan"),) * 4
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(st), size=(n, len(st)))
    roi = 100 * rt[idx].sum(axis=1) / st[idx].sum(axis=1)
    return 100 * rt.sum() / st.sum(), float(np.percentile(roi, 5)), float(np.percentile(roi, 95)), float((roi > 100).mean() * 100)


def races_needed(st: np.ndarray, rt: np.ndarray, roi: float) -> float:
    """回収率が roi% のとき、下5%が100%を超えるのに要るレース数の目安（1レースの収支のばらつきは実戦のもの）。"""
    if len(st) < 2 or roi <= 100:
        return float("nan")
    sd = float(np.std(rt - st, ddof=1))
    return (Z90 * sd / (st.mean() * (roi / 100 - 1))) ** 2


def streaks(rs: list[dict]) -> tuple[int, int, str]:
    """一番長い連敗・今の連敗・一番長い連敗の始まり。"""
    best, cur, start, best_from = 0, 0, "", ""
    for r in rs:
        if r["hit"]:
            cur = 0
            continue
        if cur == 0:
            start = r["label"]
        cur += 1
        if cur > best:
            best, best_from = cur, start
    return best, cur, best_from


def sim_rows(rs: list[dict]) -> list[list[tuple[float, float, bool, float]]]:
    """ev_check._simulate 用：[(確率, 決めたときのオッズ, 当たり, 確定オッズ)]。確率かオッズの無い組は外す。"""
    out = []
    for r in rs:
        out.append([(x["p"], x["odds"], x["combo"] == r["result"], r["pay"] / 100 if x["combo"] == r["result"] else 0.0)
                    for x in r["items"] if x.get("p") is not None and x.get("odds")])
    return out


def section(rs: list[dict], k: str) -> list[str]:
    lines = [f"\n■ {LABEL[k]}"]
    if not rs:
        return lines + ["  まだ結果の出たレースがありません"]
    st, rt = np.array([r["stake"] for r in rs]), np.array([r["ret"] for r in rs])
    roi, lo, hi, over = boot(st, rt)
    hits = sum(r["hit"] for r in rs)
    pts = np.mean([r["n"] if k == "time" else len(r["items"]) if k == "co" else r["stake"] / 100 for r in rs])
    unit = "1レース" if k == "co" else "仮想資金" if k == "time" else "1点"
    lines.append(f"  {rs[0]['date']}〜{rs[-1]['date']}  {len(rs):,}R  平均{pts:.1f}点  的中 {hits}R（{100 * hits / len(rs):.1f}%）"
                 f"  投資 {st.sum() * 10:,.0f}円 → 払戻 {rt.sum() * 10:,.0f}円（{unit if k == 'time' else unit + '1,000円'}）")
    lines.append(f"  通算の回収率 {roi:.1f}%  幅 {lo:.1f}〜{hi:.1f}%  100%超え {over:.1f}%")
    lines.append("  選び方ごと（過去の検証は、その選び方で学習に使っていない期間を買ったときの回収率）")
    rules = RULES[k]
    for i, (since, name, (b_roi, b_lo, b_hi)) in enumerate(rules):
        until = rules[i + 1][0] if i + 1 < len(rules) else "99999999"
        part = [r for r in rs if since <= r["date"] < until]
        if not part:
            lines.append(f"    {_pad(name, 28)}まだ結果の出たレースがありません" + (f"（過去の検証 {b_roi:.1f}%）" if b_roi is not None else ""))
            continue
        ps, pr = np.array([r["stake"] for r in part]), np.array([r["ret"] for r in part])
        p_roi, p_lo, p_hi, p_over = boot(ps, pr)
        if b_roi is None:  # 過去の検証が無い（TIME予想）
            lines.append(f"    {_pad(name, 28)}{len(part):>5}R  回収率 {p_roi:6.1f}%  幅 {p_lo:5.1f}〜{p_hi:5.1f}%  100%超え {p_over:5.1f}%")
            if k == "time":
                lines += time_extra(part)
            continue
        band = f"幅 {b_lo:.1f}〜{b_hi:.1f}%" if b_lo is not None else "幅なし"
        lines.append(f"    {_pad(name, 28)}{len(part):>5}R  回収率 {p_roi:6.1f}%  幅 {p_lo:5.1f}〜{p_hi:5.1f}%  100%超え {p_over:5.1f}%"
                     f"  ｜ 過去の検証 {b_roi:.1f}%（{band}）")
        verdict = ("実戦の幅の中に、過去の検証の数字が入っている（今のところ検証どおりと言える範囲）" if p_lo <= b_roi <= p_hi
                   else "実戦の幅の外に、過去の検証の数字がある（検証より悪い。数が増えても続くなら見直す）" if p_hi < b_roi
                   else "実戦の幅の外に、過去の検証の数字がある（検証より良い。たまたまのこともあるので、数が増えるまで様子を見る）")
        lines.append(f"      → {verdict}")
        if k in ("ch", "chb"):
            lines += cherry_extra(part)
        need = races_needed(ps, pr, b_roi)
        if need == need:
            lines.append(f"      回収率が検証どおり{b_roi:.0f}%なら、100%超えをはっきり言うには約{math.ceil(need):,}R（いま {len(part):,}R）")
    best, cur, best_from = streaks(rs)
    lines.append(f"  連敗（締切順）：一番長い {best}連敗（{best_from}〜）  今 {cur}連敗")
    # 買い目を決めた時刻（最後に決め直した時刻。締切の何分前か）
    mins = sorted(r["mins"] for r in rs if r["mins"] is not None)
    if mins:
        late = sum(m <= 5 for m in mins)
        lines.append(f"  買い目を決めた時刻：締切の平均 {np.mean(mins):.1f}分前（真ん中 {np.median(mins):.1f}分前、早い {mins[-1]:.1f}・遅い {mins[0]:.1f}。"
                     f"5分前より後 {100 * late / len(mins):.0f}%、{len(mins)}R）")
        # 当たった組の、決めたときのオッズと確定オッズ
    moves = [(r["pay"] / 100) / x["odds"] for r in rs if r["hit"] for x in r["items"] if x["combo"] == r["result"] and x.get("odds")]
    if moves:
        lines.append(f"  当たった組のオッズ：確定 ÷ 決めたとき＝平均 {np.mean(moves):.2f}倍（{len(moves)}本。1より小さいと、締切までに下がっている）")
    # 資金10万円から（合成オッズ配分は、1レースの金額が決まっているので出さない）
    from .ml.ev_check import _simulate

    if k not in PLANS:
        return lines + daily(rs)
    sims = sim_rows(rs)
    n_items = sum(1 for s in sims if s)
    flat, cap = PLANS[k]
    lines.append(f"  資金10万円から（確率とオッズの残っている {n_items:,}R。払戻は確定オッズ）")
    for name, how, kw in ((f"平掛け 1点{flat:,}円", "flat", {"unit": flat}), (f"ケリー1/4 1点{cap:,}円まで", "kelly", {"cap": cap})):
        r = _simulate([s for s in sims if s], how, **kw)
        lines.append(f"    {_pad(name, 24)}最後の資金 {r['bank']:>10,.0f}円  一番少ないとき {r['low']:>9,.0f}円  一番減ったとき −{r['dd']:4.1f}%")
    return lines + daily(rs)


def time_extra(rs: list[dict]) -> list[str]:
    """TIME予想：12点運用との比べ・イン逃げレースかどうか・500レースまでの残り。"""
    s12, r12 = sum(r["s12"] for r in rs), sum(r["r12"] for r in rs)
    lines = [f"      12点運用（1点1,000円）なら 回収率 {100 * r12 / s12:6.1f}%" if s12 else "      12点運用：記録なし"]
    for flag, name in ((True, "イン逃げレース"), (False, "非イン逃げレース")):
        g = [r for r in rs if r["in_escape"] is flag]
        if g:
            s, t = sum(r["stake"] for r in g), sum(r["ret"] for r in g)
            lines.append(f"      {_pad(name, 18)}{len(g):>5}R 的中{sum(r['hit'] for r in g):>4}R  回収率 {100 * t / s if s else float('nan'):6.1f}%")
    left = TIME_MIN_RACES - len(rs)
    lines.append(f"      → {'検証運用中（あと' + str(left) + 'Rで500R。それまでは有効性を判断しない）' if left > 0 else '500Rに届きました'}")
    return lines


def cherry_extra(rs: list[dict]) -> list[str]:
    """🍒：MINAMOが①を市場より弱いと見たレース（市場の①の見立て − MINAMOの①の見立て）だけにしたら（ev-check「16-2.」と同じ物差し）。"""
    lines = []
    for e in (0.0, 0.05, 0.10, 0.15):
        g = [r for r in rs if r.get("edge") is not None and r["edge"] >= e]
        if g:
            s, t = sum(r["stake"] for r in g), sum(r["ret"] for r in g)
            lines.append(f"      MINAMOが①を市場より{100 * e:>2.0f}ポイント以上弱く見たレース {len(g):>4}R 的中{sum(r['hit'] for r in g):>3}R"
                         f"  回収率 {100 * t / s if s else float('nan'):6.1f}%")
    return lines


def daily(rs: list[dict]) -> list[str]:
    """日ごとの成績と通算（1点1,000円。合成オッズ配分は1レース1,000円）。"""
    lines = ["  日ごと（1点1,000円）"]
    cum_s = cum_r = 0.0
    for d in sorted({r["date"] for r in rs}):
        part = [r for r in rs if r["date"] == d]
        s, t = sum(r["stake"] for r in part), sum(r["ret"] for r in part)
        cum_s, cum_r = cum_s + s, cum_r + t
        lines.append(f"    {d[4:6]}/{d[6:]} {len(part):>4}R 的中{sum(r['hit'] for r in part):>3}R  回収率 {100 * t / s:6.1f}%"
                     f"  収支 {10 * (t - s):>+10,.0f}円  通算 {100 * cum_r / cum_s:6.1f}%")
    return lines


def build(data_dir: Path = store.DATA_DIR) -> str:
    lines = ["実戦の成績（試験中の買い目。締切前に決めた組を、その時のオッズで。幅はレースを入れ替えて1000回数え直した下5%〜上95%）"]
    for k in ("ev", "ev3", "ev2", "ev1", "co", "ex", "ex3", "ex2", "ex1", "time", "fm", "fmb", "ag", "xa", "ch", "chb"):
        lines += section(rows(data_dir, k), k)
    return "\n".join(lines)


LABELS = ("T10", "T5", "T1", "FINAL")


def _race_key(date: str, jcd: str, rno) -> str:
    return f"{date}-{str(jcd).zfill(2)}-{int(rno):02d}"


def odds_compare(data_dir: Path, raw: Path) -> str:
    """実戦で買い目を決めたときのオッズと、検証に使うデータベースのオッズ（10分前・5分前・1分前・確定）を、同じ組で比べる。
    実戦÷5分前 が1に近ければ同じ物差し（実戦の不調は数の少なさ）。1よりはっきり大きければ、実戦はデータベースの5分前より早い・薄いオッズで決めている。"""
    import pandas as pd

    from .ml import odds_history

    path = Path(raw) / "odds_hist.csv"
    if not path.exists():
        return "odds_hist.csv がありません（db_export.sh で書き出してください）"
    live = {}
    for f in sorted(Path(data_dir).glob("*/[0-9][0-9]-[0-9][0-9].json")):
        race = store.read_json(f) or {}
        if race.get("demo") or not race.get("deadline"):
            continue
        date = race.get("date") or f.parent.name
        items = {"trifecta": race.get("ev_items") or [], "exacta": race.get("ex_items") or []}
        if any(items.values()):
            live[_race_key(date, race.get("jcd") or f.stem[:2], race.get("rno") or int(f.stem[3:]))] = (race, items)
    if not live:
        return "実戦の買い目がありません"
    snaps = pd.read_csv(path, dtype=str)
    snaps["race"] = snaps["race_date"].str.replace("-", "").str[:8] + "-" + snaps["venue"].str.zfill(2) + "-" + \
        pd.to_numeric(snaps["race_no"], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(2)
    snaps = snaps[snaps["race"].isin(set(live))]
    if "captured_at" in snaps:
        snaps = snaps.sort_values("captured_at", na_position="first")
    snaps = snaps.drop_duplicates(["race", "label"], keep="last")
    db: dict[str, dict] = {}
    for row in snaps.itertuples(index=False):
        db.setdefault(row.race, {})[row.label] = {"trifecta": odds_history._parse(row.trifecta), "exacta": odds_history._parse(getattr(row, "exacta", "")),
                                                  "at": getattr(row, "captured_at", None)}
    lines = [f"実戦で決めたときのオッズと、データベース（検証の材料）のオッズを同じ組で比べる（実戦の買い目のあるレース {len(live):,}R、"
             f"データベースにもあるレース {len(db):,}R）"]
    if not db:
        lines.append("  データベースの書き出しに、まだ実戦の日が入っていません（db_export.sh を流してから、もう一度）")
        return "\n".join(lines)
    for key, tag in (("trifecta", "3連単（試し）"), ("exacta", "2連単（試し）")):
        ratio = {lab: [] for lab in LABELS}
        fin_live, fin_t5, mins_live, mins_db = [], [], [], {lab: [] for lab in LABELS}
        n = 0
        for rk, (race, items) in live.items():
            snap = db.get(rk)
            if not snap or not items[key]:
                continue
            date = rk[:8]
            at = (race.get("ev_at") if key == "trifecta" else race.get("ex_at"))
            m = mins_before(date, race.get("deadline"), at)
            if m is not None:
                mins_live.append(m)
            for lab in LABELS:
                if lab in snap:
                    mm = mins_before(date, race.get("deadline"), snap[lab]["at"])
                    if mm is not None:
                        mins_db[lab].append(mm)
            for it in items[key]:
                o = it.get("odds")
                if not o:
                    continue
                n += 1
                for lab in LABELS:
                    v = (snap.get(lab) or {}).get(key, {}).get(it["combo"])
                    if v:
                        ratio[lab].append(o / v)
                fin = (snap.get("FINAL") or {}).get(key, {}).get(it["combo"])
                t5 = (snap.get("T5") or {}).get(key, {}).get(it["combo"])
                if fin:
                    fin_live.append(fin / o)
                    if t5:
                        fin_t5.append(fin / t5)
        if not n:
            continue
        med = lambda xs: f"{np.median(xs):.2f}" if xs else "--"
        avg = lambda xs: f"{np.mean(xs):.2f}" if xs else "--"
        lines.append(f"\n■ {tag}：{n:,}組")
        lines.append(f"  決めた時刻：実戦は締切の平均 {avg(mins_live)}分前" + "".join(
            f"・データベースの{lab} {avg(mins_db[lab])}分前" for lab in LABELS if lab != "FINAL" and mins_db[lab]))
        for lab in LABELS:
            if ratio[lab]:
                lines.append(f"  実戦のオッズ ÷ データベースの{_pad(lab, 6)} 中央値 {med(ratio[lab])}  平均 {avg(ratio[lab])}（{len(ratio[lab]):,}組）")
        lines.append(f"  確定 ÷ 実戦のオッズ       中央値 {med(fin_live)}  平均 {avg(fin_live)}（{len(fin_live):,}組）")
        lines.append(f"  確定 ÷ データベースのT5  中央値 {med(fin_t5)}  平均 {avg(fin_t5)}（{len(fin_t5):,}組。検証はこの下がり方で数えている）")
    lines.append("\n見方：「実戦÷T5」が1に近い＝検証と同じ物差し。1よりはっきり大きい＝実戦は検証より早い（薄い）オッズで決めていて、検証の回収率が甘く出ている")
    return "\n".join(lines)


def _band(v, edges, fmt):
    for lo, hi in zip(edges, edges[1:]):
        if lo <= v < hi:
            return fmt(lo, hi)
    return fmt(edges[-1], None)


def breakdown(data_dir: Path) -> str:
    """実戦の試し買い（3連単・2連単）を、買った組ひとつずつに分けて、どこで勝ち負けしているかを見る。
    組ごとの回収率＝その組を1点100円で買い続けたときの払戻÷投資。数が少ない区分は偶然が大きい。"""
    rows = {k: [] for k in ("ev", "ev3", "ev2", "ev1", "ex", "ex3", "ex2", "ex1")}
    for f in sorted(Path(data_dir).glob("*/[0-9][0-9]-[0-9][0-9].json")):
        race = store.read_json(f) or {}
        res = race.get("result") or {}
        if race.get("demo") or res.get("cancelled") or not res.get("trifecta"):
            continue
        esc = ((race.get("prediction") or {}).get("escape") or {}).get("index")
        boats = (race.get("prediction") or {}).get("boats") or []
        courses = [(b.get("boat"), b.get("course")) for b in boats]
        entry = None if len(courses) != 6 or any(c is None for _, c in courses) else any(b != c for b, c in courses)
        for k in rows:
            items = race.get(f"{k}_items") or []
            exa = k.startswith("ex")
            won, pay = (res.get("exacta"), res.get("exacta_payout")) if exa else (res.get("trifecta"), res.get("payout"))
            for i, it in enumerate(items):
                if not it.get("odds"):
                    continue
                hit = it["combo"] == won
                rows[k].append({"date": race.get("date") or f.parent.name, "odds": float(it["odds"]), "ev": float(it.get("ev") or 0),
                                "p": float(it.get("p") or 0), "rank": i + 1, "n": len(items), "head": it["combo"].split("-")[0],
                                "esc": esc, "entry": entry, "hit": hit, "pay": (pay or 0) if hit else 0})
    out = ["実戦の試し買いを、買った組ひとつずつで分けた成績（1点100円。回収率＝払戻÷投資。当たりが少ない区分は偶然が大きい）"]

    def line(tag, g):
        if not g:
            return None
        n, h, ret = len(g), sum(x["hit"] for x in g), sum(x["pay"] for x in g)
        return f"    {_pad(tag, 22)}{n:>5}組 当たり{h:>3} 回収率 {ret / n:6.1f}%"

    groups = [
        ("決めたときのオッズ", lambda x: _band(x["odds"], [0, 10, 15, 20, 30, 50, 80, 120, 200], lambda a, b: f"{a:g}〜{b:g}倍" if b else f"{a:g}倍〜")),
        ("期待値", lambda x: _band(x["ev"], [0, 1.2, 1.3, 1.5, 2.0, 3.0], lambda a, b: f"{a:g}〜{b:g}" if b else f"{a:g}〜")),
        ("確率", lambda x: _band(x["p"], [0, 0.01, 0.02, 0.03, 0.05, 0.08], lambda a, b: f"{100 * a:g}〜{100 * b:g}%" if b else f"{100 * a:g}%〜")),
        ("そのレースの何点目（確率順）", lambda x: f"{x['rank']}点目" if x["rank"] <= 4 else "5点目以降"),
        ("そのレースの点数", lambda x: "1点" if x["n"] == 1 else "2〜3点" if x["n"] <= 3 else "4〜6点" if x["n"] <= 6 else "7点以上"),
        ("頭（1着）の艇", lambda x: "①頭" if x["head"] == "1" else "①以外の頭"),
        ("進入（展示のコース）", lambda x: "--" if x.get("entry") is None else "進入が変わった" if x["entry"] else "枠なり"),
        ("イン逃げ指数", lambda x: "--" if x["esc"] is None else _band(x["esc"], [0, 40, 55, 70, 85], lambda a, b: f"{a}〜{b}" if b else f"{a}〜")),
    ]
    for k, name in (("ev", "3連単（試し）5分前"), ("ev3", "3連単 3分前"), ("ev2", "3連単 2分前"), ("ev1", "3連単 1分前"),
                    ("ex", "2連単（試し）5分前"), ("ex3", "2連単 3分前"), ("ex2", "2連単 2分前"), ("ex1", "2連単 1分前")):
        rs = rows[k]
        if not rs:
            continue
        out.append(f"\n■ {name}：{len(rs):,}組（{rs[0]['date']}〜{rs[-1]['date']}）")
        out.append(line("全部", rs))
        for gname, key in groups:
            out.append(f"  {gname}")
            seen = {}
            for x in rs:
                seen.setdefault(key(x), []).append(x)
            for tag in sorted(seen, key=lambda t: (float(re.match(r"[\d.]+", t).group()) if re.match(r"[\d.]+", t) else 999, t)):
                out.append(line(tag, seen[tag]))
    return "\n".join(x for x in out if x)


REVIEW_KINDS = (("picks", "普通の予想"), ("ev", "3連単（試し）"), ("ex", "2連単（試し）"), ("ex2", "2連単 2分前"), ("xa", "2連単 全R"),
                ("fm", "隊形"), ("ag", "一致"), ("ch", "🍒A"))


def day_review(data_dir: Path, date: str, jcd: str | None = None) -> str:
    """その日のレースを1つずつ振り返る（負けが続いた日の敗因探し。見るだけ）。
    結果の人気と払戻・MINAMOがその結果を120通りの何番目に見ていたか・1着のコース・イン逃げ指数・買い方ごとの組と当たり。
    jcd を省くと、場ごとのまとめだけ（その場がほかの場と比べて荒れていたか）。"""
    files = sorted((Path(data_dir) / date).glob("[0-9][0-9]-[0-9][0-9].json"))
    races = []
    for f in files:
        race = store.read_json(f) or {}
        res = race.get("result") or {}
        if race.get("demo") or res.get("cancelled") or not res.get("trifecta"):
            continue
        tri_all = race.get("tri_all") or {}
        ranked = sorted(tri_all, key=tri_all.get, reverse=True)
        res_tri = res["trifecta"]
        boats = {b.get("boat"): b for b in (race.get("prediction") or {}).get("boats") or []}
        win_boat = int(res_tri.split("-")[0])
        races.append({"jcd": f.stem[:2], "rno": int(f.stem[3:]), "venue": (race.get("venue") or {}).get("name", f.stem[:2]),
                      "race": race, "res": res, "rank": ranked.index(res_tri) + 1 if res_tri in ranked else None,
                      "p": tri_all.get(res_tri), "win_course": (boats.get(win_boat) or {}).get("course"),
                      "esc": ((race.get("prediction") or {}).get("escape") or {})})
    if not races:
        return f"{date} の結果の出たレースがありません"

    def summary(g):
        n = len(g)
        in1 = sum(r["win_course"] == 1 for r in g)
        top10 = sum(r["rank"] is not None and r["rank"] <= 10 for r in g)
        pops = [r["res"].get("popularity") for r in g if r["res"].get("popularity")]
        big = sum((r["res"].get("payout") or 0) >= 10000 for r in g)
        return (f"{n:>2}R ①（1コース）1着 {in1:>2}R（{100 * in1 / n:3.0f}%） 結果がMINAMOの上位10組 {top10:>2}R（{100 * top10 / n:3.0f}%）"
                f" 結果の人気 平均{np.mean(pops) if pops else float('nan'):5.1f}番 万舟 {big}R")

    out = [f"{date} の振り返り（見るだけ。予想は変えない）", "\n■ 場ごと（荒れ方の比べ）"]
    by = {}
    for r in races:
        by.setdefault((r["jcd"], r["venue"]), []).append(r)
    for (j, name), g in sorted(by.items()):
        out.append(f"  {_pad(name, 6)}{summary(g)}")
    out.append(f"  {_pad('全場', 6)}{summary(races)}")
    if not jcd:
        return "\n".join(out)
    g = sorted([r for r in races if r["jcd"] == jcd.zfill(2)], key=lambda r: r["rno"])
    if not g:
        return "\n".join(out + [f"\n場 {jcd} のレースがありません"])
    out.append(f"\n■ {g[0]['venue']} のレースごと")
    tally = {k: [0, 0, 0, 0] for k, _ in REVIEW_KINDS}  # 買ったレース・当たり・投資・払戻
    for r in g:
        race, res, st = r["race"], r["res"], r["race"].get("settle") or {}
        esc = r["esc"]
        rank = f"{r['rank']}番目（{100 * r['p']:.1f}%）" if r["rank"] else "--"
        out.append(f"  {r['rno']:>2}R 結果 {res['trifecta']}（{res.get('popularity') or '-'}番人気 {res.get('payout') or 0:,}円）"
                   f" 2連単 {res.get('exacta')}（{res.get('exacta_popularity') or '-'}番人気） {res.get('kimarite') or ''}"
                   f" 1着 {r['win_course'] or '-'}コース")
        out.append(f"      MINAMO：イン逃げ指数 {esc.get('index', '-')}（{esc.get('label', '')}） 結果は120通りの {rank}")
        for k, name in REVIEW_KINDS:
            if k == "picks":
                combos = [p["combo"] for p in (race.get("ai") or {}).get("picks") or []]
                hit, stake, ret = st.get("trifecta_hit"), st.get("stake"), st.get("return")
            else:
                if not st.get(f"{k}_bought"):
                    continue
                combos = [x.get("combo") for x in race.get(f"{k}_items") or []]
                hit, stake, ret = st.get(f"{k}_hit"), st.get(f"{k}_stake"), st.get(f"{k}_return")
            if not combos:
                continue
            t = tally[k]
            t[0] += 1
            t[1] += bool(hit)
            t[2] += stake or 0
            t[3] += ret or 0
            shown = " ".join(combos[:9]) + (" …" if len(combos) > 9 else "")
            out.append(f"      {'◎' if hit else '✕'} {_pad(name, 14)}{shown}")
    out.append(f"\n■ {g[0]['venue']} の買い方ごと（この日）")
    for k, name in REVIEW_KINDS:
        n, h, s, ret = tally[k]
        if n:
            out.append(f"  {_pad(name, 14)}{n:>2}R 当たり{h:>2} 回収率 {100 * ret / s if s else 0:6.1f}%")
    return "\n".join(out)


MORE_FIX_MIN = 5.0  # 3連複・2連複・拡連複の買い目を決める時刻（締切の何分前より前の、最後に記録したオッズ）


def _more_snap(rows: list[dict]) -> dict | None:
    """そのレースのオッズ履歴から、締切 MORE_FIX_MIN 分前より前で最後の、ほかの券種のオッズのある1行。"""
    got = [r for r in rows if r.get("more") and r.get("min") is not None and r["min"] >= MORE_FIX_MIN and r.get("t3")]
    return max(got, key=lambda r: r["at"]) if got else None


def _unordered(tri: dict[str, float], k: int) -> dict[str, float]:
    """3連単の確率を、着順を問わない k 艇の組（2連複 k=2・3連複 k=3）に足し合わせる。組の名前は公式と同じ「1=2」。"""
    out: dict[str, float] = {}
    for c, p in tri.items():
        boats = c.split("-")[:k]
        key = "=".join(sorted(boats, key=int))
        out[key] = out.get(key, 0.0) + p
    return out


def _wide_probs(tri: dict[str, float]) -> dict[str, float]:
    """拡連複：2艇がどちらも3着以内に入る確率。"""
    out: dict[str, float] = {}
    for c, p in tri.items():
        a, b, d = sorted(c.split("-"), key=int)
        for x, y in ((a, b), (a, d), (b, d)):
            out[f"{x}={y}"] = out.get(f"{x}={y}", 0.0) + p
    return out


MORE_RULES = {
    "trio": [("期待値1.2以上・最大3点", 1.2, 3), ("期待値1.2以上・最大5点", 1.2, 5), ("期待値1.0以上・最大5点", 1.0, 5),
             ("確率上位3点（オッズ見ない）", None, 3), ("確率上位5点（オッズ見ない）", None, 5)],
    "quinella": [("期待値1.2以上・最大2点", 1.2, 2), ("期待値1.2以上・最大3点", 1.2, 3), ("期待値1.0以上・最大3点", 1.0, 3),
                 ("確率上位2点（オッズ見ない）", None, 2), ("確率上位3点（オッズ見ない）", None, 3)],
    "wide": [("期待値1.2以上・最大2点（下限オッズ）", 1.2, 2), ("期待値1.0以上・最大3点（下限オッズ）", 1.0, 3),
             ("確率上位2点（オッズ見ない）", None, 2)],
}
MORE_NAME = {"trio": "3連複", "quinella": "2連複", "wide": "拡連複"}
MORE_MIN_P = {"trio": 0.01, "quinella": 0.02, "wide": 0.05}


def more_check(data_dir: Path, state_dir: Path) -> str:
    """3連複・2連複・拡連複を、記録してあるオッズ（締切5分前より前の最後の記録）と払戻で、買い方ごとに数える（見るだけ）。
    確率は MINAMO の3連単の確率を、3連単の試し買いと同じ補正B（その時の3連単オッズ）で直してから足し合わせる。
    3連単の試し買い（ev）も同じレースで並べる。前半・後半の日に分ける。"""
    import json

    calib = store.ev_calib()
    hist: dict[str, list[dict]] = {}
    for f in sorted((Path(state_dir) / "odds").glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("more"):
                hist.setdefault(row["race"], []).append(row)
    rows = []
    for f in sorted(Path(data_dir).glob("*/[0-9][0-9]-[0-9][0-9].json")):
        race = store.read_json(f) or {}
        res = race.get("result") or {}
        pays = res.get("payouts") or {}
        date = race.get("date") or f.parent.name
        key = f"{date}-{f.stem[:2]}-{f.stem[3:]}"
        snap = _more_snap(hist.get(key) or [])
        if race.get("demo") or res.get("cancelled") or not res.get("trifecta") or not race.get("tri_all") or not snap or not pays:
            continue
        tri = sorted(race["tri_all"].items(), key=lambda kv: -kv[1])
        if calib:
            tri = store.calibrate(tri, snap["t3"], *calib)
        tri = dict(tri)
        st = race.get("settle") or {}
        rows.append({"date": date, "probs": {"trio": _unordered(tri, 3), "quinella": _unordered(tri, 2), "wide": _wide_probs(tri)},
                     "odds": {k: snap["more"].get(k) or {} for k in MORE_RULES}, "pays": pays,
                     "ev": (st.get("ev_stake"), st.get("ev_return")) if st.get("ev_bought") else None})
    if not rows:
        return "3連複・2連複・拡連複のオッズを記録したレースがありません（10/4 から記録しています）"
    days = sorted({r["date"] for r in rows})
    mid = days[len(days) // 2]
    out = [f"3連複・2連複・拡連複の買い方の比べ（記録したオッズ {len(rows):,}R、{days[0]}〜{days[-1]}。"
           f"締切{MORE_FIX_MIN:g}分前より前の最後のオッズで決め、払戻で数える。1点100円。前半＝{mid}より前・後半＝{mid}から）"]

    def picks(r, kind, th, k):
        p, odds = r["probs"][kind], r["odds"][kind]
        cs = sorted(p, key=p.get, reverse=True)
        if th is None:
            return cs[:k]
        lo = lambda c: (odds[c][0] if isinstance(odds.get(c), (list, tuple)) else odds.get(c)) or 0
        return [c for c in cs if p[c] >= MORE_MIN_P[kind] and lo(c) and p[c] * lo(c) >= th][:k]

    def cell(g, kind, th, k):
        n = st = hits = 0
        pays = []
        for r in g:
            b = picks(r, kind, th, k)
            if not b:
                continue
            n += 1
            st += 100 * len(b)
            won = [r["pays"].get(kind, {}).get(c, 0) for c in b]
            got = sum(won)
            hits += got > 0
            pays.append(got)
        if not st:
            return _pad("（買うレースなし）", 50)
        tot = sum(pays)
        return _pad(f"{n:>4}R {st / 100 / n:3.1f}点 的中{100 * hits / n:5.1f}% 回収率{100 * tot / st:6.1f}%（最大除く{100 * (tot - max(pays)) / st:6.1f}%）", 50)
    a = [r for r in rows if r["date"] < mid]
    b = [r for r in rows if r["date"] >= mid]
    for kind, rules in MORE_RULES.items():
        out.append(f"\n■ {MORE_NAME[kind]}")
        out.append(f"  {_pad('', 34)}{_pad('前半', 50)}後半")
        for name, th, k in rules:
            out.append(f"  {_pad(name, 34)}{cell(a, kind, th, k)}{cell(b, kind, th, k)}")

    def ev_cell(g):
        bs = [r["ev"] for r in g if r["ev"] and r["ev"][0]]
        if not bs:
            return _pad("（買うレースなし）", 50)
        st, rt = sum(x for x, _ in bs), sum(y or 0 for _, y in bs)
        return _pad(f"{len(bs):>4}R 的中{100 * sum(1 for _, y in bs if y) / len(bs):5.1f}% 回収率{100 * rt / st:6.1f}%", 50)
    out.append("\n■ くらべ：3連単（試し）5分前（同じレース）")
    out.append(f"  {_pad('', 34)}{ev_cell(a)}{ev_cell(b)}")
    out.append("\n前半・後半とも100%を超え、一番大きい払戻を除いても100%前後なら、記録だけの試し買いに足す候補")
    return "\n".join(out)


def _odds_snaps(state_dir: Path) -> dict[str, list[dict]]:
    """var/state/odds/*.jsonl（取り直すたびのオッズ）をレースごとに。"""
    import json

    out: dict[str, list[dict]] = {}
    for f in sorted((Path(state_dir) / "odds").glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("race") and (row.get("t3") or row.get("t2")):
                out.setdefault(row["race"], []).append(row)
    return out


def _snap_at(rows: list[dict], fix_min: float = 5.0) -> dict | None:
    """締切 fix_min 分前より前で、最後に取ったオッズ。"""
    got = [r for r in rows if r.get("min") is not None and r["min"] >= fix_min and r.get("t3")]
    return max(got, key=lambda r: r["at"]) if got else None


# オッズを見ずに決まった形で選ぶ買い方（と、くらべの試し買い）。（名前, 2連単か, 組の取り出し方）
FILTER_KINDS = [
    ("fm", "隊形①-②", True, lambda race: [x.get("combo") for x in race.get("fm_items") or []]),
    ("ag", "一致", True, lambda race: [x.get("combo") for x in race.get("ag_items") or []]),
    ("xa", "2連単 全R", True, lambda race: [x.get("combo") for x in race.get("xa_items") or []]),
    ("time", "TIME予想", False, lambda race: list((race.get("time_pick") or {}).get("combos") or [])),
    ("ch", "🍒 A", False, lambda race: [x.get("combo") for x in race.get("ch_items") or []]),
    ("chb", "🍒 B", False, lambda race: list((race.get("ch_pick") or {}).get("combos_b") or [])),
    ("main", "普通の予想", False, lambda race: [p.get("combo") for p in (race.get("ai") or {}).get("picks") or []]),
    ("ev", "くらべ：3連単（試し）", False, lambda race: [x.get("combo") for x in race.get("ev_items") or []]),
    ("ex", "くらべ：2連単（試し）", True, lambda race: [x.get("combo") for x in race.get("ex_items") or []]),
]


def ev_filter_check(data_dir: Path, state_dir: Path) -> str:
    """決まった形の買い方（隊形・一致・全R・TIME・🍒・普通の予想）の組を、MINAMOの期待値で絞ったら回収率は上がるか（見るだけ）。
    期待値＝補正Bの確率（3連単の試し買いと同じ。2連単は3着を足し合わせる）×締切5分前より前の最後のオッズ。1点100円、払戻は結果の配当。
    前半・後半の日に分ける。外す組（期待値1.0未満）の成績も出す。"""
    calib = store.ev_calib()
    snaps = _odds_snaps(state_dir)
    rows = {k: [] for k, *_ in FILTER_KINDS}
    for f in sorted(Path(data_dir).glob("*/[0-9][0-9]-[0-9][0-9].json")):
        race = store.read_json(f) or {}
        res = race.get("result") or {}
        if race.get("demo") or res.get("cancelled") or not res.get("trifecta") or not race.get("tri_all"):
            continue
        date = race.get("date") or f.parent.name
        snap = _snap_at(snaps.get(f"{date}-{f.stem[:2]}-{f.stem[3:]}") or [])
        if not snap:
            continue
        tri = sorted(race["tri_all"].items(), key=lambda kv: -kv[1])
        tri = dict(store.calibrate(tri, snap["t3"], *calib) if calib else tri)
        exa: dict[str, float] = {}
        for c, p in tri.items():
            k2 = c.rsplit("-", 1)[0]
            exa[k2] = exa.get(k2, 0.0) + p
        for k, _, is_ex, get in FILTER_KINDS:
            combos = [c for c in get(race) if c]
            if not combos:
                continue
            probs, odds = (exa, snap.get("t2") or {}) if is_ex else (tri, snap["t3"])
            won, pay = (res.get("exacta"), res.get("exacta_payout")) if is_ex else (res["trifecta"], res.get("payout"))
            for c in combos:
                o = odds.get(c)
                rows[k].append({"date": date, "ev": probs.get(c, 0.0) * o if o else None, "hit": c == won, "pay": (pay or 0) if c == won else 0})
    days = sorted({x["date"] for v in rows.values() for x in v})
    if not days:
        return "オッズの記録と結果のそろったレースがありません"
    mid = days[len(days) // 2]
    out = [f"決まった形の買い方を、MINAMOの期待値で絞ったら（{days[0]}〜{days[-1]}。期待値＝補正Bの確率×締切5分前のオッズ。"
           f"1点100円・組ごとに数える。前半＝{mid}より前・後半＝{mid}から）"]

    def cell(g):
        if not g:
            return _pad("（なし）", 46)
        pays = [x["pay"] for x in g if x["hit"]]
        tot = sum(pays)
        cut = (tot - max(pays)) / len(g) if pays else 0.0
        return _pad(f"{len(g):>5}組 当たり{len(pays):>4} 回収率{tot / len(g):6.1f}%（最大除く{cut:6.1f}%）", 46)
    rules = [("全部（今のまま）", lambda x: True), ("期待値1.0以上だけ", lambda x: x["ev"] is not None and x["ev"] >= 1.0),
             ("期待値1.2以上だけ", lambda x: x["ev"] is not None and x["ev"] >= 1.2),
             ("外す組（期待値1.0未満）", lambda x: x["ev"] is not None and x["ev"] < 1.0)]
    for k, name, *_ in FILTER_KINDS:
        g = rows[k]
        if not g:
            continue
        out.append(f"\n■ {name}")
        out.append(f"  {_pad('', 26)}{_pad('前半', 46)}後半")
        for tag, keep in rules:
            out.append(f"  {_pad(tag, 26)}{cell([x for x in g if x['date'] < mid and keep(x)])}{cell([x for x in g if x['date'] >= mid and keep(x)])}")
    out.append("\n「期待値1.0以上だけ」が前半・後半とも「全部」より良く、100%に近づくなら、その買い方にオッズの絞りを足す候補")
    return "\n".join(out)
