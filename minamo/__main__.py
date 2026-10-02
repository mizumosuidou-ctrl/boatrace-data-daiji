"""MINAMO コマンドライン。

  python -m minamo run              常駐（毎分：取得・予想・結果照合）
  python -m minamo sync [YYYYMMDD]  指定日の出走表を取得して事前予想
  python -m minamo tick [YYYYMMDD]  1回だけ直前情報・結果を更新
  python -m minamo demo [--days N]  架空データでサイトを確認
  python -m minamo serve [--port]   web/ をローカル配信
  python -m minamo ml-train         LightGBMを学習（var/ml/raw のCSVから）
  python -m minamo ml-backfill      過去の展示データを公式サイトから取り寄せる
  python -m minamo ml-original      過去のオリジナル展示をボートレース日和から取り寄せる
  python -m minamo wind-table       場ごとの風の表を、データベースの過去の天気から作る
"""
from __future__ import annotations

import argparse
import functools
import json
import http.server
import logging
from pathlib import Path

from . import store


def main() -> None:
    ap = argparse.ArgumentParser(prog="minamo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--interval", type=int, default=60)
    run.add_argument("--no-ai", action="store_true")
    for name in ("sync", "tick"):
        p = sub.add_parser(name)
        p.add_argument("date", nargs="?")
        p.add_argument("--no-ai", action="store_true")
    demo = sub.add_parser("demo")
    demo.add_argument("--days", type=int, default=3)
    demo.add_argument("--venues", type=int, default=14)
    mlt = sub.add_parser("ml-train", help="LightGBMを学習・検証して var/ml に保存")
    mlt.add_argument("--raw", default=None, help="書き出したCSVの場所（既定 var/ml/raw）")
    mlt.add_argument("--test-days", type=int, default=90)
    sub.add_parser("ml-synthetic", help="動作確認用の架空CSVを var/ml/raw に作る")
    mlb = sub.add_parser("ml-backfill", help="過去の展示データを公式サイトから取り寄せる（1秒1件）")
    mlb.add_argument("--from", dest="date_from", default="20250101")
    mlb.add_argument("--to", dest="date_to", default="20991231")
    mlb.add_argument("--limit", type=int, default=None)
    mlo = sub.add_parser("ml-original", help="過去のオリジナル展示（一周・まわり足・直線）をボートレース日和から取り寄せる（3〜5秒に1件）")
    mlo.add_argument("--days", type=int, default=183, help="さかのぼる日数（既定 183＝約6か月）")
    mlo.add_argument("--limit", type=int, default=None)
    mls = sub.add_parser("ml-series", help="過去の開催（大会名・グレード）を公式サイトから取り寄せる（1日1件）")
    mls.add_argument("--raw", default=None)
    mlf = sub.add_parser("formation", help="スタート隊形トゥエルブの分布表をデータベースの実績から作る")
    mlf.add_argument("--raw", default=None, help="書き出したCSVの場所（既定 var/ml/raw）")
    mlf.add_argument("--venue", default=None, help="表示する場（例 01）。省略すると作るだけ")
    mlf.add_argument("--category", default="一般", help="一般・SG・G1・女子・マスターズ・ルーキーズ・正月・お盆")
    mlf.add_argument("--show", action="store_true", help="作り直さず、前に作った表を表示するだけ")
    mlw = sub.add_parser("wind-table", help="場ごとの風の表を、データベースの過去の天気とレース結果から作る")
    mlw.add_argument("--raw", default=None, help="書き出したCSVの場所（既定 var/ml/raw）")
    mlw.add_argument("--venue", default=None, help="詳しく表示する場（例 01）")
    mlw.add_argument("--check", type=int, default=0, help="各場この数のレースで、風の向きを公式サイトの結果ページと照合（1秒1件）")
    mlw.add_argument("--show", action="store_true", help="作り直さず、前に作った表を表示するだけ")
    mlv = sub.add_parser("venue-check", help="場ごとの予想ルールを、データベースの実績で確かめる表を出す")
    mlv.add_argument("--venue", required=True, help="場（例 03）")
    mlv.add_argument("--raw", default=None)
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8000)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.cmd == "demo":
        from .demo import generate

        generate(days=args.days, venues=args.venues)
        print(f"demo data written to {store.DATA_DIR}")
    elif args.cmd in ("ml-train", "ml-synthetic", "ml-backfill", "ml-original"):
        from .ml import backfill, biyori, live, synthetic, train

        raw = Path(getattr(args, "raw", None) or live.ML_DIR / "raw")
        if args.cmd == "ml-original":
            n = biyori.run(raw, days=args.days, limit=args.limit)
            print(f"取り寄せ完了: オリジナル展示ありのレース {n} 件")
        elif args.cmd == "ml-backfill":
            n = backfill.run(raw, args.date_from, args.date_to, args.limit)
            print(f"取り寄せ完了: 展示ありのレース {n} 件")
        elif args.cmd == "ml-synthetic":
            synthetic.generate(raw)
            print(f"synthetic CSV written to {raw}")
        else:
            meta = train.run(raw, live.ML_DIR, test_days=args.test_days)
            print(train.summary_ja(meta))
    elif args.cmd == "ml-series":
        from .ml import live, series

        raw = Path(args.raw) if args.raw else live.ML_DIR / "raw"
        print(f"{series.run(raw)} days fetched")
    elif args.cmd == "formation":
        from .ml import formation_table, live

        raw = Path(args.raw) if args.raw else live.ML_DIR / "raw"
        if args.show:
            data = json.loads((live.ML_DIR / "formation.json").read_text(encoding="utf-8"))
        else:
            data = formation_table.build(raw, live.ML_DIR)
        print(f"{data['meta']['races']} races {data['meta']['data_range']} {data['meta'].get('by_category', '')}")
        if args.venue:
            print(formation_table.format_table(data, args.venue.zfill(2), args.category))
    elif args.cmd == "wind-table":
        from .ml import live, wind_table

        raw = Path(args.raw) if args.raw else live.ML_DIR / "raw"
        if args.check:  # 先に向きを確かめ、公式サイトと合わなければ表を作らない（作ると当日の予想に使われるため）
            from .fetcher import Fetcher

            rows = wind_table.check_icons(raw, Fetcher(), per_venue=args.check)
            print(wind_table.format_check(rows))
            if not wind_table.check_ok(rows):
                print("向きが公式サイトと合わないので、表は作りません")
                return
        if args.show:
            data = json.loads((live.ML_DIR / wind_table.OUT_NAME).read_text(encoding="utf-8"))
        else:
            data = wind_table.build(raw, live.ML_DIR)
        print(wind_table.report(data))
        if args.venue:
            print(wind_table.detail(data, args.venue.zfill(2)))
    elif args.cmd == "venue-check":
        from .ml import live, venue_check

        raw = Path(args.raw) if args.raw else live.ML_DIR / "raw"
        print(venue_check.build(raw, args.venue))
    elif args.cmd == "serve":
        root = Path(__file__).resolve().parent.parent / "web"
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
        print(f"http://localhost:{args.port}/")
        http.server.ThreadingHTTPServer(("0.0.0.0", args.port), handler).serve_forever()
    else:
        from .pipeline import Pipeline

        pipe = Pipeline(ai_enabled=not args.no_ai)
        if args.cmd == "run":
            pipe.run_forever(args.interval)
        else:
            date = args.date or store.now_jst().strftime("%Y%m%d")
            if args.cmd == "sync":
                pipe.sync_day(date)
            else:
                print(f"updated {pipe.tick(date)} races")


if __name__ == "__main__":
    main()
