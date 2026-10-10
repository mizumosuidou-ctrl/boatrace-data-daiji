"""MINAMO コマンドライン。

  python -m minamo run              常駐（毎分：取得・予想・結果照合）
  python -m minamo sync [YYYYMMDD]  指定日の出走表を取得して事前予想
  python -m minamo tick [YYYYMMDD]  1回だけ直前情報・結果を更新
  python -m minamo demo [--days N]  架空データでサイトを確認
  python -m minamo serve [--port]   web/ をローカル配信
  python -m minamo rebuild          日ごとの一覧と成績を作り直す（表示項目を増やしたとき）
  python -m minamo ml-train         LightGBMを学習（var/ml/raw のCSVから）
  python -m minamo ml-notify        学習の結果のひとこと（前回との比べ）を Discord に送る
  python -m minamo ml-softmax       レース内 softmax と今のやり方を同じ検証期間で比べる（モデルは保存しない）
  python -m minamo ml-softmax-roi   レース内 softmax と今のやり方を、買い目の回収率まで比べる（展示前・展示後）
  python -m minamo ml-calib         外れ方の分析（予想の確率と実際の1着率を、条件ごとに比べる）
  python -m minamo ml-ex-select     2連単の買い方を、偶然を差し引いて選ぶ（前半で選び、後半で確かめる。読むだけ）
  python -m minamo ml-live-compare  バックテストの買い方と実戦の記録を、同じレースで並べる（食い違いの切り分け。読むだけ）
  python -m minamo ml-upset         イン逃しの検証（①が負けるレースの見分け・勝ち艇の当て・①頭でない買い方。読むだけ）
  python -m minamo multi-check      買い方の多重比較の点検（補正しても100%を超えたと言えるか。読むだけ）
  python -m minamo audit            成績の自動点検（投資・当たり・払戻・合計のつじつま。読むだけ）
  python -m minamo ml-cv            期間をずらして何回か、特徴量を1つずつ外して効き方を比べる（予想は変えない）
  python -m minamo ml-refit         検証用に分けた直近の期間も学習に入れ直すと良くなるかを比べる（予想は変えない）
  python -m minamo ml-tune          LightGBM の設定を何通りか試して比べる（モデルは保存しない）
  python -m minamo ml-backfill      過去の展示データを公式サイトから取り寄せる
  python -m minamo ml-facts         データベースの実績が止まった日の次の日から、実績・展示・風を公式サイトで足す
  python -m minamo ml-original      過去のオリジナル展示をボートレース日和から取り寄せる
  python -m minamo ml-official      公式のダウンロードデータ（競走成績・番組表・ファン手帳）を取り込む
  python -m minamo ml-years         過去何年分を学習に使うと良くなるかを比べる（良くなったときだけ採用）
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
    mlu = sub.add_parser("ml-tune", help="LightGBM の設定（木の大きさ・学習率など）を何通りか試して比べる（モデルは保存しない）")
    mlu.add_argument("--raw", default=None, help="書き出したCSVの場所（既定 var/ml/raw）")
    mlu.add_argument("--test-days", type=int, default=90)
    mlq = sub.add_parser("ml-softmax", help="レース内 softmax で学習するやり方と今のやり方を同じ検証期間で比べる（モデルは保存しない）")
    mlq.add_argument("--raw", default=None, help="書き出したCSVの場所（既定 var/ml/raw）")
    mlq.add_argument("--test-days", type=int, default=90)
    mlr = sub.add_parser("ml-softmax-roi", help="レース内 softmax と今のやり方を、展示前・展示後とも作って、買い目の回収率（ev-check）まで比べる")
    mlr.add_argument("--raw", default=None, help="書き出したCSVの場所（既定 var/ml/raw）")
    mlr.add_argument("--test-days", type=int, default=90)
    mlc = sub.add_parser("ml-calib", help="外れ方の分析：予想の確率と実際の1着率を、条件（確率の帯・場・コース・風など）ごとに比べる（予想は変えない）")
    mlc.add_argument("--raw", default=None, help="書き出したCSVの場所（既定 var/ml/raw）")
    mlc.add_argument("--test-days", type=int, default=90)
    mlv = sub.add_parser("ml-cv", help="期間をずらして何回か、特徴量のまとまりを1つずつ外して、1着の対数損失がどれだけ悪くなるか（効き方）を比べる（モデルは書き換えない）")
    mlv.add_argument("--raw", default=None, help="書き出したCSVの場所（既定 var/ml/raw）")
    mlv.add_argument("--folds", type=int, default=4, help="検証の回数（新しい方から数える）")
    mlv.add_argument("--fold-days", type=int, default=60, help="検証1回の日数")
    mlv.add_argument("--valid-days", type=int, default=45, help="調整（止める位置を決める）の日数")
    mlv.add_argument("--post", action="store_true", help="展示後モデルを比べる（展示・風・オリジナル展示を外す。体重・部品交換を足す）")
    mlf = sub.add_parser("ml-refit", help="今の学習が入れていない直近の期間（調整45日＋検証）も入れ直すと、同じ検証期間で良くなるかを比べる（モデルは書き換えない）")
    mlf.add_argument("--raw", default=None, help="書き出したCSVの場所（既定 var/ml/raw）")
    mlf.add_argument("--folds", type=int, default=3, help="検証の回数（新しい方から数える）")
    mlf.add_argument("--fold-days", type=int, default=60, help="検証1回の日数")
    sub.add_parser("ml-synthetic", help="動作確認用の架空CSVを var/ml/raw に作る")
    mlb = sub.add_parser("ml-backfill", help="過去の展示データを公式サイトから取り寄せる（1秒1件）")
    mlb.add_argument("--from", dest="date_from", default="20250101")
    mlb.add_argument("--to", dest="date_to", default="20991231")
    mlb.add_argument("--limit", type=int, default=None)
    mlf = sub.add_parser("ml-facts", help="データベースの実績が止まった日の次の日から昨日まで、実績・展示・風を公式サイトで足す（1秒1件）")
    mlf.add_argument("--from", dest="date_from", default=None)
    mlf.add_argument("--to", dest="date_to", default=None)
    mlx = sub.add_parser("ml-official", help="公式のダウンロードデータ（競走成績K・番組表B・ファン手帳）を取り込む（1秒1件）。取り終えた日はとばす")
    mlx.add_argument("--raw", help="var/ml/raw の場所")
    mlx.add_argument("--from", dest="date_from", help="YYYYMMDD（省くと取り終えた最後の日の翌日、無ければ昨日）")
    mlx.add_argument("--to", dest="date_to", help="YYYYMMDD（省くと昨日）")
    mlx.add_argument("--fan", action="store_true", help="ファン手帳も（まだ取っていない期だけ）")
    mlx.add_argument("--peek", metavar="YYYYMMDD", help="1日分の中身と読み取れた数を見るだけ（書き出さない）")
    mln = sub.add_parser("ml-notify", help="学習の結果のひとこと（対数損失・前回との比べ）を Discord に送る。--failed を付けると失敗の知らせ")
    mln.add_argument("--failed", default=None, help="失敗の知らせの理由（例: daily status=1）")
    sub.add_parser("ml-original-live", help="本番で取ったオリジナル展示（var/state の orig）を学習の材料に書き出す（昨日まで、取り終えた日はとばす）")
    mly = sub.add_parser("ml-years", help="過去何年分を学習に使うと良くなるかを、同じ検証期間で比べる（良くなったときだけ採用）")
    mly.add_argument("--raw", default=None)
    mly.add_argument("--dry-run", action="store_true", help="比べるだけで、採用（train_window.json）は書かない")
    mlk = sub.add_parser("ml-kimarite", help="決まり手を公式サイトの結果一覧（1場1日で1ページ、1秒1件）から足す。取り終えた日はとばす")
    mlk.add_argument("--raw", help="var/ml/raw の場所")
    mlk.add_argument("--from", dest="date_from", help="YYYYMMDD（省くと facts.csv の最初の日）")
    mlk.add_argument("--to", dest="date_to", help="YYYYMMDD（省くと昨日）")
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
    mlq = sub.add_parser("odds-check", help="オッズの動き（締切15分前→5分前→確定）と結果を突き合わせる表を出す")
    mlq.add_argument("--raw", default=None)
    mle = sub.add_parser("ev-check", help="買い目の選び方（確率上位・期待値）を、学習に使っていない期間のオッズと結果で比べる")
    mle.add_argument("--raw", default=None)
    mle.add_argument("--split", action="store_true", help="22.（試し買いの組を live-check --breakdown と同じ分け方で）だけ出す")
    mau = sub.add_parser("audit", help="成績の自動点検：日ごとの一覧の投資・当たり・払戻・合計のつじつまを調べる（読むだけ）")
    mau.add_argument("--url", nargs="?", const="__site__", default=None, help="公開サイトを点検する（URL を省くと既定のサイト）。省くとサーバーの data フォルダ")
    mau.add_argument("--days", type=int, default=14, help="直近何日を点検するか")
    mxs = sub.add_parser("ml-ex-select", help="2連単の買い方を、前半で選び後半で確かめる。市場が正しいとしたときの最良の見かけを差し引く（読むだけ）")
    mxs.add_argument("--raw", default=None, help="書き出したCSVの場所（既定 var/ml/raw）")
    mxs.add_argument("--sims", type=int, default=300, help="市場が正しいとして結果を作り直す回数")
    mxs.add_argument("--split", type=float, default=0.6, help="前半（選ぶ側）の割合")
    mlc = sub.add_parser("ml-live-compare", help="バックテストの買い方と実戦の記録を、同じレースで並べる（読むだけ）")
    mlc.add_argument("--raw", default=None, help="書き出したCSVの場所（既定 var/ml/raw）")
    mlc.add_argument("--days", type=int, default=14, help="実戦の直近何日を使うか")
    mus = sub.add_parser("ml-upset", help="イン逃しの検証：①が負けるレースを見分けられるか、勝ち艇を当てられるか、①頭でない買い方は得か（読むだけ）")
    mus.add_argument("--raw", default=None, help="書き出したCSVの場所（既定 var/ml/raw）")
    mmc = sub.add_parser("multi-check", help="買い方の多重比較の点検：複数の買い方をまとめて判定して、補正しても100%を超えたと言えるか（読むだけ）")
    mmc.add_argument("--url", nargs="?", const="__site__", default=None, help="公開サイトを点検する（URL を省くと既定のサイト）。省くとサーバーの data フォルダ")
    mmc.add_argument("--days", type=int, default=60, help="直近何日を使うか")
    mmc.add_argument("--candidates", type=int, default=300, help="これまでに試した買い方の数の目安（バックテストの見方に使う）")
    mwk = sub.add_parser("weekly", help="週報：直近7日の実戦の成績と学習の様子（--send で Discord にも送る）")
    mwk.add_argument("--send", action="store_true")
    mwk.add_argument("--data", help="web/data の場所（既定は MINAMO_DATA_DIR）")
    mll = sub.add_parser("live-check", help="実戦の成績（試験中の買い目）を、過去の検証と同じ物差しで見る表を出す")
    mll.add_argument("--data", help="web/data の場所（既定は MINAMO_DATA_DIR）")
    mll.add_argument("--odds", action="store_true", help="実戦で決めたときのオッズを、データベースの5分前・1分前・確定オッズと比べる")
    mll.add_argument("--raw", default=None)
    mll.add_argument("--day", default=None, help="その日（YYYYMMDD）のレースを振り返る。--venue（場コード 例 07）でその場をレースごとに")
    mll.add_argument("--venue", default=None)
    mll.add_argument("--breakdown", action="store_true", help="試し買いを買った組ひとつずつで分けた成績（オッズ・期待値・確率・何点目・頭・イン逃げ指数）")
    mla = sub.add_parser("ability-check", help="自動発見のアビリティを買い目に使ったらどうだったかを、学習に使っていない期間で確かめる")
    mla.add_argument("--raw", default=None)
    mlr = sub.add_parser("rtm-compare", help="レースタイムモニターの予想（DEEP・NORMAL・裏の予想）とMINAMOを同じレースで比べる")
    mlr.add_argument("--raw", default=None)
    mlr.add_argument("--data", default=None, help="MINAMOのレースのJSONの場所（既定 web/data）")
    mrl = sub.add_parser("rtm-learn", help="レースタイムモニターの艇ごとの評価に、MINAMOに無い情報があるかを検証期間で確かめる")
    mrl.add_argument("--raw", default=None)
    sub.add_parser("rebuild", help="保存済みのレースから日ごとの一覧と成績を作り直す")
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8000)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.cmd == "demo":
        from .demo import generate

        generate(days=args.days, venues=args.venues)
        print(f"demo data written to {store.DATA_DIR}")
    elif args.cmd in ("ml-train", "ml-synthetic", "ml-backfill", "ml-original", "ml-facts"):
        from .ml import backfill, biyori, facts_backfill, live, synthetic, train

        raw = Path(getattr(args, "raw", None) or live.ML_DIR / "raw")
        if args.cmd == "ml-facts":
            n = facts_backfill.run(raw, args.date_from, args.date_to)
            print(f"取り寄せ完了: 実績を足した日 {n} 日")
        elif args.cmd == "ml-original":
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
            # 画面が読む「データの見方」（レースタイムの順位ごとの過去の成績など）
            store.write_json(store.DATA_DIR / "insights.json", {"racetime": meta.get("racetime_eval") or {},
                                                                 "updated_at": meta.get("trained_at")})
            print(train.summary_ja(meta))
    elif args.cmd == "ml-tune":
        from .ml import live, train

        raw = Path(args.raw or live.ML_DIR / "raw")
        print(train.tune_params(raw, live.ML_DIR, test_days=args.test_days))
    elif args.cmd == "ml-softmax":
        from .ml import live, train

        raw = Path(args.raw or live.ML_DIR / "raw")
        print(train.softmax_experiment(raw, live.ML_DIR, test_days=args.test_days))
    elif args.cmd == "ml-softmax-roi":
        from .ml import live, softmax_roi

        raw = Path(args.raw or live.ML_DIR / "raw")
        print(softmax_roi.compare(raw, live.ML_DIR, test_days=args.test_days))
    elif args.cmd == "ml-calib":
        from .ml import calib_report, live

        raw = Path(args.raw or live.ML_DIR / "raw")
        print(calib_report.build(raw, live.ML_DIR, test_days=args.test_days))
    elif args.cmd == "ml-cv":
        from .ml import cv_ablation, live

        raw = Path(args.raw or live.ML_DIR / "raw")
        print(cv_ablation.build(raw, live.ML_DIR, folds=args.folds, fold_days=args.fold_days, valid_days=args.valid_days, post=args.post))
    elif args.cmd == "ml-refit":
        from .ml import live, refit_check

        raw = Path(args.raw or live.ML_DIR / "raw")
        print(refit_check.build(raw, live.ML_DIR, folds=args.folds, fold_days=args.fold_days))
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
    elif args.cmd == "odds-check":
        from .ml import live, odds_history

        print(odds_history.build(Path(args.raw) if args.raw else live.ML_DIR / "raw"))
    elif args.cmd == "ev-check":
        from .ml import ev_check, live

        raw = Path(args.raw) if args.raw else live.ML_DIR / "raw"
        print(ev_check.build_split(live.ML_DIR, raw) if args.split else ev_check.build(live.ML_DIR, raw))
    elif args.cmd == "ml-kimarite":
        from .ml import facts_backfill, live

        raw = Path(args.raw) if args.raw else live.ML_DIR / "raw"
        print(f"{facts_backfill.fill_kimarite(raw, args.date_from, args.date_to)} days fetched")
    elif args.cmd == "ml-ex-select":
        from .ml import ex_select, live

        raw = Path(args.raw or live.ML_DIR / "raw")
        print(ex_select.run(live.ML_DIR, raw, sims=args.sims, split=args.split))
    elif args.cmd == "ml-live-compare":
        from . import audit
        from .ml import live, live_compare

        days, _ = audit.load_dir(store.DATA_DIR, args.days)
        print(live_compare.run(live.ML_DIR, Path(args.raw or live.ML_DIR / "raw"), days))
    elif args.cmd == "ml-upset":
        from .ml import live, upset

        print(upset.run(live.ML_DIR, Path(args.raw or live.ML_DIR / "raw")))
    elif args.cmd == "multi-check":
        from . import audit, multi, notify

        if args.url:
            days, _ = audit.load_url(notify.SITE_URL if args.url == "__site__" else args.url, args.days)
        else:
            days, _ = audit.load_dir(store.DATA_DIR, args.days)
        print(multi.build(days, n_candidates=args.candidates))
    elif args.cmd == "audit":
        from . import audit, notify

        if args.url:
            days, record = audit.load_url(notify.SITE_URL if args.url == "__site__" else args.url, args.days)
        else:
            days, record = audit.load_dir(store.DATA_DIR, args.days)
        print(audit.build(days, record))
    elif args.cmd == "weekly":
        from . import weekly

        text = weekly.build(Path(args.data) if args.data else store.DATA_DIR)
        print(text)
        if args.send:
            print(f"（Discord に {weekly.send(text)} 通送りました）")
    elif args.cmd == "live-check":
        from . import live_check

        data = Path(args.data) if args.data else store.DATA_DIR
        if args.day:
            print(live_check.day_review(data, args.day, args.venue))
        elif args.breakdown:
            print(live_check.breakdown(data))
        elif args.odds:
            from .ml import live

            print(live_check.odds_compare(data, Path(args.raw) if args.raw else live.ML_DIR / "raw"))
        else:
            print(live_check.build(data))
    elif args.cmd == "ability-check":
        from .ml import ability_check, live

        print(ability_check.build(live.ML_DIR, Path(args.raw) if args.raw else live.ML_DIR / "raw"))
    elif args.cmd == "rtm-compare":
        from .ml import live, rtm_compare

        raw = Path(args.raw) if args.raw else live.ML_DIR / "raw"
        print(rtm_compare.build(raw, Path(args.data) if args.data else store.DATA_DIR))
    elif args.cmd == "ml-official":
        from .ml import live, official

        raw = Path(args.raw) if args.raw else live.ML_DIR / "raw"
        if args.peek:
            print(official.peek(args.peek))
        else:
            n = official.run(raw, args.date_from, args.date_to)
            f = official.run_fan(raw) if args.fan else 0
            print(f"取り込み完了: 競走成績・番組表 {n} 日" + (f"、ファン手帳 {f} 期" if args.fan else ""))
    elif args.cmd == "ml-notify":
        from . import notify
        from .ml import live, train

        text = train.notify_text(train.history_load(live.ML_DIR), failed=args.failed)
        print(text or "（送る内容がありません）")
        if text:
            print("送信しました" if notify.send(text) else "送信しませんでした（Webhook が未設定、または失敗）")
    elif args.cmd == "ml-original-live":
        from .ml import live, original_live

        print(f"本番のオリジナル展示を書き出しました: {original_live.harvest(live.ML_DIR / 'raw')} 艇")
    elif args.cmd == "ml-years":
        from .ml import live, train

        raw = Path(args.raw) if args.raw else live.ML_DIR / "raw"
        print(train.years_check(raw, live.ML_DIR, write=not args.dry_run))
    elif args.cmd == "rtm-learn":
        from .ml import live, rtm_learn

        print(rtm_learn.build(live.ML_DIR, Path(args.raw) if args.raw else live.ML_DIR / "raw", store.DATA_DIR))
    elif args.cmd == "rebuild":
        print(f"作り直した日数: {store.rebuild_days()}")
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
