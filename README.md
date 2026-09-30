# MINAMO — Claude Race Intelligence

全国24場のボートレースを **公式サイトから自分でデータ取得** し、統計モデルと **Claude** が1レースずつ予想するリアルタイム予想サイトです（レースタイムモニター型）。

- **Monitor**：全場×12Rを一覧。締切カウントダウン、本命・推奨買い目、結果・払戻・的中を自動更新
- **Race**：Claudeの見解（見出し・展開・◎○▲・買い目と配分）、1周1マークの展開シミュレーション、各艇の1着/2連対/3連対確率とその根拠、出走表・直前情報
- **Record**：的中率・回収率・本命1着率を日別に公開。確信度の帯ごとの成績で予想の当てになり具合を検証

## しくみ

| 段階 | タイミング | 内容 |
| --- | --- | --- |
| 取得 | 毎朝 | 開催場（`/race/index`）と全レースの出走表（`/race/racelist`） |
| 直前 | 締切30分前〜 | 展示タイム・進入・スタート展示・気象（`beforeinfo`）、3連単オッズ（`odds3t`）を4分おき |
| 予想 | 取得のたび | 場別コース1着率×（勝率・当地・モーター・ボート・平均ST・F・展示・風）→ 1着確率 → 3連単120通り |
| 見解 | 展示が出たら1回 | 数字とデータをClaudeに渡し、見出し・展開・印・買い目をJSONで受け取る |
| 照合 | 締切6分後〜 | 結果（`raceresult`）を取得して的中・回収を記録 |

公式サイトへのアクセスは1秒1回以下。`ANTHROPIC_API_KEY` が無い場合は統計モデルの定型文で動きます。

## ローカルで見る

```bash
pip install -r requirements-minamo.txt
python -m minamo demo          # 架空データ（DEMO表示）を生成
python -m minamo serve         # http://localhost:8000
```

実データ：

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m minamo sync          # 今日の出走表を取得して事前予想
python -m minamo run           # 常駐：毎分 直前情報・見解・結果を更新
```

## サーバーで公開する（Docker）

1GB程度のVPS（さくら・ConoHa・Lightsailなど）で動きます。

```bash
git clone <this repo> && cd boatrace-data-daiji/deploy
cp .env.example .env   # DOMAIN と ANTHROPIC_API_KEY を記入
docker compose up -d   # worker（取得・予想）と Caddy（HTTPS配信）が起動
```

- DNSのAレコードをサーバーIPに向けると、Caddyが証明書を自動取得します
- データは `web/data/`、取得状態は `var/state/` に保存されます
- Claudeの費用目安：1日約150〜200レース×1回。Opus 5.5・effort mediumで1レース5〜10円前後（1日1,000〜2,000円程度）。`MINAMO_MODEL=claude-sonnet-5-5` で約半分

## テスト

```bash
pytest tests/minamo_tests -q
```

---

# (旧) MULTI-ZODIAC BOAT v36 — 全選手・名前入力診断 公開候補版

最優先の入口は `quick_app.py` です。

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run quick_app.py
```

## 最初に実現する目的

ボートレーサーの選手名または登録番号を入力すると、BOAT RACE公式プロフィールから生年月日・血液型などを確認し、MULTI-ZODIAC基礎診断を表示します。

- 登録済み選手は即表示
- 未登録選手は公式検索・プロフィール取得後に自動登録
- 部分一致で複数候補がある場合だけ候補選択
- 候補が1人なら検索から診断まで1回で完了

## 表示内容

- 選手名・登録番号
- 生年月日・血液型・支部・級別
- 太陽星座・四元素・三区分・支配星
- 血液型＋星座タイプ
- 四柱推命の年柱・月柱・日柱・日主
- 五行構成
- 指定日の流年・流月・流日
- 基礎的な攻勢・慎重傾向
- コピー用診断文

出生時刻が不明なため、ASC・ハウス・時柱は使用しません。六星占術は確認済み値のみ扱い、未確認値を自動断定しません。

## 公開

Streamlit Community Cloudでの公開手順は [DEPLOY.md](DEPLOY.md) を参照してください。公開時のエントリーファイルは `quick_app.py` です。

## 多機能版

レース情報、展示、F状態、勝負掛け、コメント、検証履歴などを含む多機能版は次で起動します。

```bash
streamlit run app.py
```

## テスト

```bash
pip install pytest
pytest -q
```

GitHub Actionsでもpush／pull request時にテストを実行します。
