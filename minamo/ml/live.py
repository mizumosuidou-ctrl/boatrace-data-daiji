"""当日の出走表・直前情報から、学習済みLightGBMで1着確率を出す。"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .. import abilities as abilities_mod
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
        for name in ("fhold", "wall", "kimarite"):
            path = self.dir / f"stats_{name}.csv.gz"
            if path.exists():
                self.new[name] = pd.read_csv(path, dtype={"toban": str}).assign(date=self.stats_date)
        # ファン手帳（選手ごとの最新の期。無ければ使わない）
        path = self.dir / "stats_fan.csv.gz"
        self.fan = pd.read_csv(path, dtype={"toban": str}, parse_dates=["eff"]) if path.exists() else None
        # 画面のデータ欄：選手×コースの期間別（半年・1年・全期間）と F持ちのときの成績
        self.profile = {}
        path = self.dir / "stats_profile.csv.gz"
        if path.exists():
            for r in pd.read_csv(path, dtype={"toban": str}).to_dict("records"):
                self.profile.setdefault((r["toban"], int(r["course"])), {})[r["scope"]] = ds.profile_row(r)
        # 選手別アビリティの自動発見（学習のたびに直近1年から）
        self.found = {}
        path = self.dir / "stats_found.csv.gz"
        if path.exists():
            for r in pd.read_csv(path, dtype={"toban": str}).to_dict("records"):
                self.found.setdefault(r["toban"], []).append(r)
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
        day_first, day_last = ds.day_flags(getattr(card, "day_label", None))
        rows = []
        for e in entries:
            b = be.get(e.boat)
            r = rt_racers.get(e.toban)
            rows.append({
                "race_id": "live", "date": self.stats_date, "venue": card.jcd, "lane": e.boat, "race_no": float(card.rno),
                "day_first": day_first, "day_last": day_last,
                "course": courses[e.boat], "toban": e.toban, "grade_o": ds.GRADE_ORD.get(e.grade, np.nan),
                "motor_no": f"{e.motor_no}@{era}" if e.motor_no else np.nan,
                "rt_day": rt.get("day", np.nan),
                "rt_n": r[1] if r else (0.0 if rt else np.nan),
                "rt_best": r[0] if r else np.nan,
                "rt_series_rank": r[2] if r else np.nan,
                "rt_series_n": r[3] if r else np.nan,
                **self._series(rt, e.toban),
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
                  "rt_day", "rt_n", "rt_best", "rt_series_rank", "rt_series_n", "wave_cm", "ss_n", "ss_avg", "ss_wins"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for c, (lo, hi) in ds.ORIG_BOUNDS.items():
            df.loc[~df[c].between(lo, hi), c] = np.nan
        df = ds.apply_stats(df, self.pc, self.pa, self.meta["priors"])
        df = ds.apply_extra(df, self.extra)
        df = ds.apply_new(df, self.new, self.meta["priors"])
        return ds.apply_fan(df, self.fan)

    @staticmethod
    def _series(rt: dict, toban: str) -> dict:
        """今節成績（racetime.table の "series"：[走った数, 得点の合計, 1着の数]）。節の情報が無ければ NaN。"""
        if not rt or "series" not in rt:
            return {"ss_n": np.nan, "ss_avg": np.nan, "ss_wins": np.nan}
        s = rt["series"].get(toban)
        if not s or not s[0]:
            return {"ss_n": 0.0, "ss_avg": np.nan, "ss_wins": 0.0}
        return {"ss_n": float(s[0]), "ss_avg": s[1] / s[0], "ss_wins": float(s[2])}

    def motor_era(self, jcd: str, date: str) -> int:
        """学習と同じ数え方で「何回目のモーターか」。学習のあとに交換されていれば、その分も足す。"""
        known = sorted((self.meta.get("priors") or {}).get("motor_swaps", {}).get(jcd, []))
        era = int(np.searchsorted(np.array(known, dtype=str), str(date), side="right")) if known else 0
        last = max(known + [str((self.meta.get("data_range") or ["", ""])[1]).replace("-", "")[:8]])
        era += sum(1 for d in _live_swaps().get(jcd, []) if last < d <= str(date))
        return era

    def abilities(self, df: pd.DataFrame, use_post: bool, before=None) -> tuple[dict, dict, list]:
        """選手別アビリティ（展示後の進入コースで判定。欠けている値は None のまま、0や推測で埋めない）。"""
        boats = []
        for i in range(len(df)):
            r = df.iloc[i]
            toban, course = str(r["toban"]), int(r["course"])
            wn = _num_or_none(r.get("disp_wall_n"), 0)
            boats.append({
                "boat": int(r["lane"]), "toban": toban, "course": course,
                "prof": (self.profile.get((toban, course)) or {}).get("1y") or {},
                "wall": _num_or_none(r.get("disp_wall"), 3), "wall_n": int(wn) if wn else 0,
                "motor_2": _num_or_none(r.get("motor_2"), 2),
                "rt_series_rank": _num_or_none(r.get("rt_series_rank"), 0),
                "lap_time": _num_or_none(r.get("lap_time"), 2) if use_post else None,
                "found": [a for a in self.found.get(toban, []) if int(a["course"]) in (0, course)],
            })
        try:
            return abilities_mod.evaluate(boats, prelim=not (before is not None and before.complete))
        except Exception:  # noqa: BLE001 — アビリティが出せなくても予想は続ける
            log.exception("abilities failed")
            return {}, {}, []

    def place_mult(self, name: str) -> Optional[list]:
        info = (self.meta.get("place") or {}).get(name) or {}
        return info.get("mult") if info.get("mult_adopt") else None

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
            # 選手別アビリティ（展示後の進入コースで判定）。買い目反映ありのものだけ1着の強さに効かせる
            found, mult, keep = self.abilities(df, use_post, before)
            if mult:
                p = p * np.array([mult.get(int(b), 1.0) for b in df["lane"]])
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
                "stats": {**_stats(df.iloc[i]), "sr_model": _num_or_none(df["sr_c"].iloc[i], 3),  # 学習と同じ平均スタート順位（試し買いの見送りに使う）
                          "profile": self.profile.get((str(df["toban"].iloc[i]), int(df["course"].iloc[i])), {}),
                          "abilities": found.get(int(boat), [])},
            }
            if q is not None:
                out[int(boat)]["q"] = (float(q[i, 0]), float(q[i, 1]))  # ちょうど2着・ちょうど3着
        return {"engine": "lightgbm-post" if use_post else "lightgbm-pre", "boats": out, "pl_decay": self.meta.get("pl_decay"),
                "keep": keep,  # 買い目反映ありの報告登録アビリティ：買い目内に残す組の指示
                "place_w": place_w,
                "place_mult": self.place_mult("post" if use_post else "pre"),  # 2着・3着の残りやすさ（コース別の倍率。採用したときだけ）
                "wind": "wind_tail" in feats}  # 風をモデルが使っていれば、場の風の表では補正しない


def _stats(row) -> dict:
    """画面の「実力」欄：進入コースでの1着・2連対・3連対率、平均スタート順位、トップスタート率、壁率（前日まで・全場）。"""
    def g(k, d=3):
        v = row.get(k) if hasattr(row, "get") else None
        return _num_or_none(v, d)
    return {"n_c": int(row.get("n_c", 0) or 0), "win_c": g("disp_win_c"), "top2_c": g("disp_top2_c"), "top3_c": g("disp_top3_c"),
            "sr_c": g("disp_sr_c", 2), "top_st": g("disp_top_st"), "wall": g("disp_wall"),
            "wall_n": int(row.get("disp_wall_n") or 0) if _num_or_none(row.get("disp_wall_n")) is not None else None,
            # 決まり手（直近1年・その進入コース）。1コース：逃げ・差され・まくられ・まくられ差し、2コース：逃し、2〜6コース：差し・まくり・まくり差し
            "km": {k: g(f"disp_{k}") for k in ("km_nige", "km_sasare", "km_makurare", "km_makusasare", "km_nogashi",
                                                 "km_sashi", "km_makuri", "km_makurisashi") if g(f"disp_{k}") is not None},
            "km_n": int(row.get("disp_km_n") or 0) if _num_or_none(row.get("disp_km_n")) is not None else None}


def _num_or_none(v, digits: int = 2) -> Optional[float]:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else round(v, digits)


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
