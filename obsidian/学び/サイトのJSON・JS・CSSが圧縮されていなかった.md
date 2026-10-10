---
tags: [学び]
日付: 2026-10-10
分類: サイト
---
# サイトの JSON・JS・CSS が圧縮されずに送られていた（nginx の設定）

## 分かったこと（公開サイトを外から測った）
- 圧縮（gzip）されていたのは HTML だけ。JSON・JS・CSS は圧縮なしで送られていた。原因は nginx の `gzip_types` が、初期設定のまま `#` でコメントアウトされていたこと（`gzip on;` だけでは HTML しか圧縮されない）。
- 最初の読み込み：約 754 KB（`app.js` 160 KB・`style.css` 61 KB・今日の `day.json` 531 KB ほか）。
- 「的中一覧」「成績」を開くと、30日分の `day.json`（1日 400〜530 KB）を全部読む＝約 15 MB。
- すべての通信が `no-cache`（毎回「変わっていないか」を問い合わせる）。ホーム画面は開いている間30秒ごとに一覧を取り直す。

## 直したこと（10/10 15:36、許可をもらって実行）
- `/etc/nginx/snippets/minamo.conf` の `/minamo/` に `gzip on; gzip_vary on; gzip_comp_level 5; gzip_min_length 1024; gzip_types application/json text/css application/javascript text/javascript image/svg+xml;` を追加（影響は /minamo/ だけ）。反映前に `nginx -t` で検査、`reload`。元のファイルは `minamo.conf.bak-20261010-153610`。
- 結果：`day.json` 531→76 KB、`app.js` 160→47 KB、`style.css` 61→13.5 KB。最初の読み込みは約 754→138 KB（82%減）。展開すると元の内容。

## 戻し方
`sudo cp -p /etc/nginx/snippets/minamo.conf.bak-20261010-153610 /etc/nginx/snippets/minamo.conf && sudo nginx -t && sudo systemctl reload nginx`

## 次にやること（画面のコードに触る。本番サイト改良の会話と かぶらないよう、先に作業ログを見る）
1. **「的中一覧」「成績」用の小さな要約ファイル**（日ごとの買ったレース・的中・払戻だけ）：30日分の読み込みを、約2 MB（圧縮後）から数十 KB へ。
2. **`app.js`・`style.css` に版番号**（`app.js?v=…`）を付けて長く保存：再訪問で毎回の問い合わせが不要になる（今の `no-cache` は、古い画面が残る不具合への対策）。
3. ホーム画面の30秒ごとの更新を、一覧全体ではなく「変わったレースだけ」にする（効果は小さい）。

## 学び
- nginx は `gzip on;` だけでは HTML しか圧縮しない。`gzip_types` に JSON・JS・CSS を足すのが定番の最初の一手。
- 速さの改善は、まず外から測る（大きさ・圧縮・キャッシュ）。コードを変えずに直せるものから。
