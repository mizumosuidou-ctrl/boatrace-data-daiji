"""当日の出走表・直前情報から、学習済みLightGBMで1着確率を出す。"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .. import wind as wind_mod
from . import dataset as ds

log = logging.getLogger(__name__)

ML_DIR = Path(os.environ.get("MINAMO_ML_DIR", Path(__file__).resolve().parents[2] / "var" / "ml"))
# 学習のあとで見つけたモーター交換日（pipeline が出走表から見つけて書く）
SWAP_FILE = Path(os.environ.get("MINAMO_STATE_DIR", Path(__file__).resolve().parents[2] / "var" / "state")) / "motor_swaps.json"


def _live_swaps() -> dict:
    try:
        return json.loads(SWAP_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


class MLPredictor:
    def __init__(self, out_dir: Path = ML_DIR):
        import lightgbm as lgb

        self.dir = Path(out_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self.pre = lgb.Booster(model_file=str(self.dir / "model_pre.txt"))
        post = self.dir / "model_post.txt"
        self.post = lgb.Booster(model_file=str(post)) if post.exists() and self.meta.get("post_features") else None
        self.pc = pd.read_csv(self.dir / "stats_course.csv.gz", dtype={"toban": str}, parse_dates=["date"])
        self.pa = pd.read_csv(self.dir / "stats_racer.csv.gz", dtype={"toban": str}, parse_dates=["date"])
        self.stats_date = self.pc["date"].iloc[0] if len(self.pc) else pd.Timestamp("2000-01-01")
        # 当地成績・最近の調子・モーター実績（修正4。無ければ空）
        self.extra = {}
        for name, keys in (("local", ["toban", "venue"]), ("form", ["toban"]), ("motor", ["venue", "motor_no"])):
            path = self.dir / f"stats_{name}.csv.gz"
            t = pd.read_csv(path, dtype={k: str for k in keys}) if path.exists() else pd.DataFrame(columns=keys)
            self.extra[name] = t.assign(date=self.stats_date)
        # 修正7：F持ちのスタートのずれ・壁（無ければ使わない）
        self.new = {}
        for name in ("fhold", "wall"):
            path = self.dir / f"stats_{name}.csv.gz"
            if path.exists():
                self.new[name] = pd.read_csv(path, dtype={"toban": str}).assign(date=self.stats_date)
        # 2着・3着の専用モデル（採用されたときだけ）
        self.place = {}
        for name, info in (self.meta.get("place") or {}).items():
            files = [self.dir / f"model_top2_{name}.txt", self.dir / f"model_top3_{name}.txt"]
            if info.get("adopt") and all(f.exists() for f in files):
                self.place[name] = (lgb.Booster(model_file=str(files[0])), lgb.Booster(model_file=str(files[1])), float(info["w"]))
        self.mtime = (self.dir / "meta.json").stat().st_mtime

    @property
    def adopted(self) -> bool:
        return bool(self.meta.get("adopt"))

    def frame(self, card, before=None) -> pd.DataFrame:
        be = {b.boat: b for b in (before.entries if before else [])}
        courses = {b.boat: b.course for b in (before.entries if before else []) if b.course}
        entries = [e for e in card.entries if not e.absent]
        if len(courses) < len(entries) or len(set(courses.values())) != len(courses):
            courses = {e.boat: e.boat for e in entries}
        rt = getattr(card, "racetime", None) or {}
        rt_racers = rt.get("racers") or {}
        era = self.motor_era(card.jcd, card.date)
        wind = wind_mod.components(before.wind_dir, before.wind_speed) if before else (np.nan, np.nan)
        rows = []
        for e in entries:
            b = be.get(e.boat)
            r = rt_racers.get(e.toban)
            rows.append({
                "race_id": "live", "date": self.stats_date, "venue": card.jcd, "lane": e.boat, "race_no": float(card.rno),
                "course": courses[e.boat], "toban": e.toban, "grade_o": ds.GRADE_ORD.get(e.grade, np.nan),
                "motor_no": f"{e.motor_no}@{era}" if e.motor_no else np.nan,
                "rt_day": rt.get("day", np.nan),
                "rt_n": r[1] if r else (0.0 if rt else np.nan),
                "rt_best": r[0] if r else np.nan,
                "rt_series_rank": r[2] if r else np.nan,
                "rt_series_n": r[3] if r else np.nan,
                "motor_2": e.motor_2, "f_recent": float(e.f_count or 0), "f_hold": float(e.f_count or 0),
                "wind_tail": wind[0], "wind_cross": wind[1],
                "wave_cm": getattr(before, "wave_cm", None) if before else np.nan,
                "ex_time": b.exhibition_time if b else np.nan,
                "ex_st": b.start_st if b else np.nan,
                "tilt": b.tilt if b else np.nan,
                "lap_time": getattr(b, "lap_time", None) if b else np.nan,
                "turn_time": getattr(b, "turn_time", None) if b else np.nan,
                "straight_time": getattr(b, "straight_time", None) if b else np.nan,
            })
        df = pd.DataFrame(rows)
        for c in ("motor_2", "ex_time", "ex_st", "tilt", "lap_time", "turn_time", "straight_time",
                  "rt_day", "rt_n", "rt_best", "rt_series_rank", "rt_series_n", "wave_cm"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for c, (lo, hi) in ds.ORIG_BOUNDS.items():
            df.loc[~df[c].between(lo, hi), c] = np.nan
        df = ds.apply_stats(df, self.pc, self.pa, self.meta["priors"])
        df = ds.apply_extra(df, self.extra)
        df = ds.apply_new(df, self.new, self.meta["priors"])
        return df

    def motor_era(self, jcd: str, date: str) -> int:
        """学習と同じ数え方で「何回目のモーターか」。学習のあとに交換されていれば、その分も足す。"""
        known = sorted((self.meta.get("priors") or {}).get("motor_swaps", {}).get(jcd, []))
        era = int(np.searchsorted(np.array(known, dtype=str), str(date), side="right")) if known else 0
        last = max(known + [str((self.meta.get("data_range") or ["", ""])[1]).replace("-", "")[:8]])
        era += sum(1 for d in _live_swaps().get(jcd, []) if last < d <= str(date))
        return era

    def predict(self, card, before=None) -> Optional[dict]:
        """{boat: {"p":..., "factors":{...}, "start_order":...}} と使ったモデル名。"""
        try:
            df = self.frame(card, before)
            use_post = bool(self.post is not None and before is not None and before.complete)
            df = ds.add_race_features(df, with_ex=use_post)
            feats = self.meta["post_features"] if use_post else self.meta["pre_features"]
            booster = self.post if use_post else self.pre
            X = df.reindex(columns=feats).astype(float)
            raw = booster.predict(X)
            p = np.clip(raw, 1e-6, 1 - 1e-6)
            p = p / p.sum()
            contrib = booster.predict(X, pred_contrib=True)[:, :-1]
            q, place_w = None, 0.0
            pl = self.place.get("post" if use_post else "pre")
            if pl:
                from .train import place_q

                q = place_q(df, p, pl[0].predict(X), pl[1].predict(X))
                place_w = pl[2]
        except Exception:  # noqa: BLE001 — 予想は統計モデルで続行できる
            log.exception("ML prediction failed for %s%02d", card.jcd, card.rno)
            return None
        out = {}
        for i, boat in enumerate(df["lane"]):
            groups = {}
            for g, cols in ds.FACTOR_GROUPS.items():
                idx = [feats.index(c) for c in cols if c in feats]
                if idx:
                    groups[g] = float(contrib[i, idx].sum())
            order = "pred_start_order"  # 展示STは使わず、平均スタート順位だけで並べる
            out[int(boat)] = {
                "p": float(p[i]),
                "factors": groups,
                "start_order": float(df[order].iloc[i]),
                "motor_kp": _num_or_none(df["motor_kp_raw"].iloc[i]),
                "n_c": int(df["n_c"].iloc[i]),
            }
            if q is not None:
                out[int(boat)]["q"] = (float(q[i, 0]), float(q[i, 1]))  # ちょうど2着・ちょうど3着
        return {"engine": "lightgbm-post" if use_post else "lightgbm-pre", "boats": out, "pl_decay": self.meta.get("pl_decay"),
                "place_w": place_w,
                "wind": "wind_tail" in feats}  # 風をモデルが使っていれば、場の風の表では補正しない


def _num_or_none(v) -> Optional[float]:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else round(v, 2)


_cached: Optional[MLPredictor] = None


def get() -> Optional[MLPredictor]:
    """学習済みモデルがあれば読み込む（再学習されたら読み直す）。"""
    global _cached
    meta = ML_DIR / "meta.json"
    if not meta.exists():
        return None
    try:
        if _cached is None or meta.stat().st_mtime != _cached.mtime:
            _cached = MLPredictor(ML_DIR)
    except Exception:  # noqa: BLE001
        log.exception("failed to load ML model")
        _cached = None
    return _cached
