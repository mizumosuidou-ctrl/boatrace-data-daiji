"""買い方の「多重比較」の点検：複数の買い方をまとめて判定すると、偶然よく見えるものが混ざる。補正をかけても「100%を超えた」と言えるか。

  python -m minamo multi-check            サーバーの data フォルダ（MINAMO_DATA_DIR）の直近60日を点検
  python -m minamo multi-check --url      公開サイトの直近60日を点検
やること（読むだけ。買い方も予想も変えない）：
  ・買い方ごとに回収率と95%区間（買ったレースを引き直す）
  ・「100%を超えている」「でたらめに買った約75%より良い」の片側 p 値を出し、買い方の数で Holm 法で補正
  ・補正後に有意になるまで、あと何レース必要か（今の回収率が続くとして）
  ・N通りを試して一番良いものを選ぶと、何も無くても平均でどれだけ良く見えるか（バックテストの見方）
"""
from __future__ import annotations

from statistics import NormalDist
from typing import Optional

import numpy as np

from .audit import NAMES, STRATS, TOTAL_KEYS

MIN_BETS = 30       # これより少ない買い方は、判定しない
RANDOM_ROI = 0.75   # でたらめに買ったときの回収率（払戻率）
EXTRA = {"co": ("co_bought", "co_stake", "co_return", "3連単 合成"), "time": ("time_bought", "time_stake", "time_return", "TIME予想（仮想資金）")}


def collect(days: dict[str, dict], used_days: Optional[dict[str, int]] = None) -> dict[str, list[tuple[float, float]]]:
    """買い方ごとの（投資, 払戻）をレースごとに集める。日ごとに「レースごとの合計」と「その日の合計」が一致する日だけ使う
    （一覧に払戻が入る前の古い日は、払戻が空で回収率が低く出てしまう）。"""
    out: dict[str, list[tuple[float, float]]] = {}
    keys = {k: (v[0], v[2], v[3]) for k, v in STRATS.items()}                    # 名前 → (買ったか, 投資, 払戻)
    keys |= {k: (v[0], v[1], v[2]) for k, v in EXTRA.items()}
    for day in days.values():
        totals = day.get("totals") or {}
        for k, (bk, sk, rk) in keys.items():
            tsk, trk = TOTAL_KEYS.get(k, (sk, rk))
            pairs = []
            for v in day.get("venues", []):
                for r in v.get("races", []):
                    if r.get("cancelled"):
                        continue
                    stake, ret = r.get(sk), r.get(rk)
                    bought = (isinstance(stake, (int, float)) and stake > 0) if bk is None or k == "main" else bool(r.get(bk))
                    if bought and isinstance(stake, (int, float)) and stake > 0 and r.get("result"):
                        pairs.append((float(stake), float(ret or 0)))
            s, t = sum(a for a, _ in pairs), sum(b for _, b in pairs)
            if pairs and sk in totals or tsk in totals:
                if pairs and s == (totals.get(tsk) or 0) and t == (totals.get(trk) or 0):
                    out.setdefault(k, []).extend(pairs)
                    if used_days is not None:
                        used_days[k] = used_days.get(k, 0) + 1
    return out


def holm(ps: list[float]) -> list[float]:
    """Holm 法で補正した p 値（元の並びで返す）。"""
    k = len(ps)
    order = sorted(range(k), key=lambda i: ps[i])
    adj, running = [0.0] * k, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (k - rank) * ps[i]))
        adj[i] = running
    return adj


def expected_best_of(n: int) -> float:
    """標準正規分布の乱数 n 個のうち、いちばん大きいものの期待値（σ単位）。n 通り試して最良を選ぶと、何も無くてもこれだけ良く見える。"""
    if n <= 1:
        return 0.0
    ln = np.log(n)
    return float(np.sqrt(2 * ln) - (np.log(ln) + np.log(4 * np.pi)) / (2 * np.sqrt(2 * ln)))


def required_bets(pairs: list[tuple[float, float]], alpha: float, power: float = 0.8) -> Optional[int]:
    """今の回収率が続くとして、100%超えを（片側 alpha で）判定できるまでの買ったレース数。今の回収率が100%以下なら None。"""
    s = np.array([a for a, _ in pairs])
    r = np.array([b for _, b in pairs])
    roi = r.sum() / s.sum()
    if roi <= 1.0:
        return None
    se0 = float(np.sqrt(np.sum((r - roi * s) ** 2)) / s.sum())  # 回収率の標準誤差（デルタ法）
    z = NormalDist().inv_cdf(1 - alpha) + NormalDist().inv_cdf(power)
    return int(np.ceil(len(pairs) * (z * se0 / (roi - 1.0)) ** 2))


def analyse(pairs_by: dict[str, list[tuple[float, float]]], n_boot: int = 4000, seed: int = 0, min_bets: int = MIN_BETS) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for k, pairs in pairs_by.items():
        if len(pairs) < min_bets:
            continue
        s = np.array([a for a, _ in pairs])
        r = np.array([b for _, b in pairs])
        idx = rng.integers(0, len(pairs), size=(n_boot, len(pairs)))
        boot = r[idx].sum(axis=1) / s[idx].sum(axis=1)
        rows.append({"key": k, "n": len(pairs), "stake": float(s.sum()), "roi": float(r.sum() / s.sum()),
                     "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)),
                     "p_profit": max(float(np.mean(boot <= 1.0)), 1 / n_boot), "p_random": max(float(np.mean(boot <= RANDOM_ROI)), 1 / n_boot)})
    for col in ("p_profit", "p_random"):
        adj = holm([x[col] for x in rows])
        for x, a in zip(rows, adj):
            x[col + "_adj"] = a
    alpha = 0.05 / max(len(rows), 1)
    for x in rows:
        x["need"] = required_bets(pairs_by[x["key"]], alpha)
    return rows


def label(k: str) -> str:
    return NAMES.get(k) or EXTRA.get(k, (0, 0, 0, k))[3]


def se_per_bet(x: dict) -> float:
    """1レース分に直した回収率の標準誤差（％ポイント）。買ったレース数が n なら、回収率の標準誤差は これ ÷ √n。"""
    return 100.0 * (x["hi"] - x["lo"]) / 3.92 * float(np.sqrt(x["n"]))


def bets_to_detect(sd_pt: float, edge_pt: float, k: int, power: float = 0.8) -> int:
    """本当の回収率が（でたらめや100%より）edge_pt 良いとき、k通りの補正（Bonferroni）の片側5%で、検出力 power で見分けるのに要る買ったレース数。
    sd_pt は1レースあたりのぶれ（pt）。"""
    z = NormalDist().inv_cdf(1 - 0.05 / max(k, 1)) + NormalDist().inv_cdf(power)
    return int(np.ceil((z * sd_pt / edge_pt) ** 2))


def build(days: dict[str, dict], n_candidates: int = 300, n_boot: int = 4000) -> str:
    used: dict[str, int] = {}
    pairs = collect(days, used)
    rows = sorted(analyse(pairs, n_boot=n_boot), key=lambda x: -x["roi"])
    k = len(rows)
    lines = [f"買い方の多重比較の点検（{len(days)}日分・{k}通りをまとめて判定。読むだけ）",
             f"  でたらめに買ったときの回収率は約{RANDOM_ROI * 100:.0f}%。補正後 p は、{k}通りを同時に見ることを差し引いた値（Holm 法。0.05未満で「偶然ではない」）", ""]
    fmt = lambda p: "<0.001" if p <= 0.001 else f"{p:.3f}"
    for x in rows:
        need = x["need"]
        if need is None:
            more = "回収率が100%以下なので、判定の対象外"
        elif need <= x["n"]:
            more = "判定できる量はある"
        elif need > 50_000:
            more = "100%との差が小さすぎて、ほぼ判定できない"
        else:
            more = f"今の回収率が続くなら、あと約{need - x['n']:,}R で判定できる"
        lines.append(f"・{label(x['key'])}：{x['n']:,}R・回収率 {100 * x['roi']:.1f}%（95%区間 {100 * x['lo']:.0f}〜{100 * x['hi']:.0f}%）")
        lines.append(f"　　100%超 補正後p {fmt(x['p_profit_adj'])}／約75%超 補正後p {fmt(x['p_random_adj'])}／{more}")
        need10 = bets_to_detect(se_per_bet(x), 10.0, k)
        pace = x["n"] / max(used.get(x["key"], 1), 1)
        lines.append(f"　　本当に +10pt 良いとき、見分けるのに約{need10:,}R（いまのペース 1日{pace:.0f}R なら約{need10 / max(pace, 1):.0f}日）")
    prof = [label(x["key"]) for x in rows if x["p_profit_adj"] < 0.05 and x["lo"] > 1.0]
    skill = [label(x["key"]) for x in rows if x["p_random_adj"] < 0.05]
    lines += ["", f"→ 補正しても「100%を超えている」と言える買い方：{'・'.join(prof) or 'なし'}",
              f"→ 補正しても「でたらめ（約75%）より良い」と言える買い方：{'・'.join(skill) or 'なし'}"]
    best = expected_best_of(n_candidates)
    lines += ["", f"バックテストの見方：{n_candidates}通りを試して最良を選ぶと、何も無くても平均で +{best:.1f}σ（標準誤差の{best:.1f}倍）良く見える。"
              "標準誤差は、買い方のぶれの大きさ（当たりが少なく、たまに大きい）と、買ったレース数 n で決まる（÷√n）："]
    for x in [r for r in rows if r["n"] >= 100][:4]:
        sb = se_per_bet(x)
        cells = "・".join(f"{n:,}R なら +{best * sb / np.sqrt(n):.0f}pt" for n in (1000, 3000, 6000))
        lines.append(f"　{label(x['key'])}（1レースのぶれ {sb:.0f}pt）：最良選びの見かけの上積み {cells}")
    lines += ["", "→ 判断：バックテストの高い回収率だけで金額を上げない。実戦（これから）の成績で、補正後 p が 0.05 未満、かつ95%区間の下限が100%を超えてから。"]
    return "\n".join(lines)
