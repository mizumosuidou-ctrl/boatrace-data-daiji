"""MINAMO コマンドライン。

  python -m minamo run              常駐（毎分：取得・予想・結果照合）
  python -m minamo sync [YYYYMMDD]  指定日の出走表を取得して事前予想
  python -m minamo tick [YYYYMMDD]  1回だけ直前情報・結果を更新
  python -m minamo demo [--days N]  架空データでサイトを確認
  python -m minamo serve [--port]   web/ をローカル配信
"""
from __future__ import annotations

import argparse
import functools
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
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8000)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.cmd == "demo":
        from .demo import generate

        generate(days=args.days, venues=args.venues)
        print(f"demo data written to {store.DATA_DIR}")
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
