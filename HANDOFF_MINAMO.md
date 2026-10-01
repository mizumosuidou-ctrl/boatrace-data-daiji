# MINAMO 引き継ぎメモ（新しい会話用）

最終更新: 2026-10-01（前の会話で作成）

## ユーザーについて（必ず守ること）

- **日本語で話す。** ユーザーはプログラミングの初心者（本人いわく「ど素人」）。
- **指示は1つずつ。** 一度にたくさんの手順を出さない。1手順ごとに「できた」を待つ。
- コマンドは**コピーしてそのまま貼れる形**で出す。Mac のターミナル（`nozakitoshiaki@…%`）とサーバー（`ubuntu@v133-18-146-150-vir:~$`）の違いを、毎回はっきり伝える。
- スクリーンショットに IP・パスワードが写る場合は隠してもらう。
- **ANTHROPIC_API_KEY は GitHub（公開リポジトリ）に絶対に置かない。** サーバーの `/opt/minamo/deploy/.env` にだけ置く。
- SSH 鍵 `~/Downloads/ログイン用認証キー_20260826171914.key` は共有・アップロードしない。
- ユーザーのデータベース（Postgres `rtmonitor`、`site_archive.records`）は**読むだけ**。書き込まない。

## 全体像

- サイト: https://archive.mizu2017boat.com/minamo/ （KAGOYA VPS、既存 nginx の `/minamo/` → `/opt/minamo/web/`）
- サーバー: `ssh -i ~/Downloads/ログイン用認証キー_20260826171914.key ubuntu@133.18.146.150`
- コード: このリポジトリの `minamo/`（取得・予想・LightGBM）、`web/`（画面）、`deploy/`（docker compose）
- サーバーでの反映: `cd /opt/minamo && git pull` → `cd /opt/minamo/deploy && sudo docker compose up -d --build worker`
- 学習し直し: `sudo bash /opt/minamo/deploy/ml_refresh.sh`（DB から CSV を書き出して `python -m minamo ml-train`）

## いま動いているもの（2026-10-01 夜に開始）

| コンテナ名 | 内容 | 件数 | 終わる目安 |
|---|---|---|---|
| `minamo-biyori` | ボートレース日和から直近6か月のオリジナル展示（`var/ml/raw/original.csv`） | 28,340 R | 約1日半 |
| `minamo-backfill` | 公式サイトから 2025/1〜2026/3/10 の展示（`var/ml/raw/exhibition_backfill.csv`、2秒に1件） | 65,832 R | 約2日 |

進み具合: `sudo docker logs --tail 5 minamo-biyori` / `sudo docker logs --tail 5 minamo-backfill`
終わったか: ログに `finished` が出る。`sudo docker ps -a` で `Exited (0)`。

## GitHub とサーバーの状態

- GitHub の main には「修正3」（オリジナル展示）まで入っていて、サーバーにも反映済み。
- **このあとの修正（修正4〜6 ＋画面の小さな修正）は、ブランチ `claude/horse-racing-prediction-site-65nzxv` にある。**
  前の会話では GitHub に push できず（403）、ユーザーの Mac に `minamo-fix6.zip` として渡してある（ダウンロードフォルダ）。
  push できるようになったら、zip ではなく PR／マージで main に入れる方が簡単。
- 修正4〜6の中身:
  - 当地成績・直近90日の調子・モーター実績（コース補正）
  - 節間レースタイム（2日目以降、前日までの走りのみ。6人中の順位・節の全選手中の順位・上位15位以内）
  - モーター貢献P（ボートレース日和と同じ考え方：乗った走りの勝率点 − 節開始前1年の選手勝率。今のモーターの全期間）
  - モーター交換日で区切る（2連率がいっせいに0の日を検出＋調べた交換日一覧。学習後の交換は出走表から検出）
  - 3連単の2着・3着の平坦化（PL decay）を調整用期間で合わせる
  - 新しい特徴量は「修正3までより良いときだけ」採用
  - 画面: 節ﾀｲﾑ順・節ﾍﾞｽﾄ・貢献P の列、要因に当地・調子・ﾚｰｽﾀｲﾑ、細い確率帯の数字切れ修正

## 次にやること（順番）

1. 修正4〜6 を main に入れて、サーバーに `git pull` → `docker compose up -d --build worker`
2. 2つの取り寄せが終わったら `sudo bash /opt/minamo/deploy/ml_refresh.sh` で学習し直し、結果の表をユーザーに説明する
   （修正3まで／修正4／オリジナル展示の比較、モーター交換日の一覧が出る。交換日がユーザーの知識と合っているか確認してもらう）
3. 本番サイトを自分で見て改良（ネットワーク許可に archive.mizu2017boat.com を追加済み）
4. 任意: Claude API キーを `deploy/.env` に設定（見出しが「1号艇〇〇、…」の仮文章から、Claude の文章になる。有料）
5. 任意: 週1回の自動学習し直し（cron）

## ユーザーの考え（予想の方針）

- 予想の軸は**スタート順位**（ST そのものではなく）と、その差から読む**展開**。
- 2日目以降は節間レースタイム上位（6人中上位・節の上位15人くらい）を有利に。
- 1コースならイン逃げを強く、それ以外はスタート順位差・実力（コース別1着率）・展示（展示タイム・周回）が良ければ1着も買う。
- モーターはボートレース日和の「モーター貢献P」を重視。モーターは年1回交換されるので、交換前のデータを混ぜない。
