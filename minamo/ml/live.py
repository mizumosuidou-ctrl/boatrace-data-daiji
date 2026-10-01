"""当日の出走表・直前情報から、学習済みLightGBMで1着確率を出す。"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import dataset as ds

log = logging.getLogger(__name__)

ML_DIR = Path(os.environ.get("MINAMO_ML_DIR", Path(__file__).resolve().parents[2] / "var" / "ml"))


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
        rows = []
        for e in entries:
            b = be.get(e.boat)
            r = rt_racers.get(e.toban)
            rows.append({
                "race_id": "live", "date": self.stats_date, "venue": card.jcd, "lane": e.boat,
                "course": courses[e.boat], "toban": e.toban, "grade_o": ds.GRADE_ORD.get(e.grade, np.nan),
                "motor_no": str(e.motor_no) if e.motor_no else np.nan,
                "rt_day": rt.get("day", np.nan),
                "rt_n": r[1] if r else (0.0 if rt else np.nan),
                "rt_best": r[0] if r else np.nan,
                "rt_series_rank": r[2] if r else np.nan,
                "rt_series_n": r[3] if r else np.nan,
                "motor_2": e.motor_2, "f_recent": float(e.f_count or 0),
                "ex_time": b.exhibition_time if b else np.nan,
                "ex_st": b.start_st if b else np.nan,
                "tilt": b.tilt if b else np.nan,
                "lap_time": getattr(b, "lap_time", None) if b else np.nan,
                "turn_time": getattr(b, "turn_time", None) if b else np.nan,
                "straight_time": getattr(b, "straight_time", None) if b else np.nan,
            })
        df = pd.DataFrame(rows)
        for c in ("motor_2", "ex_time", "ex_st", "tilt", "lap_time", "turn_time", "straight_time",
                  "rt_day", "rt_n", "rt_best", "rt_series_rank", "rt_series_n"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for c, (lo, hi) in ds.ORIG_BOUNDS.items():
            df.loc[~df[c].between(lo, hi), c] = np.nan
        df = ds.apply_stats(df, self.pc, self.pa, self.meta["priors"])
        df = ds.apply_extra(df, self.extra)
        return df

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
            order = "combo_start_order" if use_post else "pred_start_order"
            out[int(boat)] = {
                "p": float(p[i]),
                "factors": groups,
                "start_order": float(df[order].iloc[i]),
                "motor_kp": _num_or_none(df["motor_kp_raw"].iloc[i]),
                "n_c": int(df["n_c"].iloc[i]),
            }
        return {"engine": "lightgbm-post" if use_post else "lightgbm-pre", "boats": out, "pl_decay": self.meta.get("pl_decay")}


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
