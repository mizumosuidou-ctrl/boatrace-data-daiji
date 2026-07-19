# MULTI-ZODIAC BOAT v36 — 全選手・名前入力診断 公開候補版

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
