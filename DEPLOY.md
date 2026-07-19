# Streamlit Community Cloud 公開手順

## 1. GitHubへ配置

このフォルダの中身をGitHubリポジトリのルートへ置きます。

必須ファイル:

- `quick_app.py` — 公開する入口
- `requirements.txt` — Python依存関係
- `.streamlit/config.toml` — 画面設定
- `data/multi_zodiac_boat.db` — 初期選手マスター

## 2. Streamlit Community Cloudで公開

1. Streamlit Community CloudへGitHubアカウントでログイン
2. `Deploy an app` を選択
3. リポジトリ、ブランチを指定
4. Main file path に `quick_app.py` を指定
5. Advanced settings で Python 3.12 を選択
6. Deploy

## 3. 公開後の確認

アプリ内の「接続・保存状態を確認」を開き、次を確認します。

- BOAT RACE公式選手検索へ接続できる
- SQLiteへ書き込める
- 選手名または登録番号から診断を表示できる

確認例:

- `峰竜太`
- `峰`
- `4320`

## 4. 注意

- 公式サイトのHTML構造が変わると検索・プロフィール解析を修正する必要があります。
- 出生時刻が公開されていないため、ASC・ハウス・時柱は使用しません。
- 六星占術は未確認値を自動断定しません。
- 命理診断は心理傾向の仮説であり、着順を断定するものではありません。
- 公開環境でSQLiteの永続保存を重要用途に使う場合は、後で外部データベースへ移行します。選手検索自体は未登録選手を公式プロフィールから再取得できます。
