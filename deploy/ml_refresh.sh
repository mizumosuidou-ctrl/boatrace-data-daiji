#!/usr/bin/env bash
# あなたのデータベース（boatrace-postgres）からCSVを書き出し、LightGBMを学習し直す。
# データベースは読むだけで、書き換えない。
#   使い方:  sudo bash /opt/minamo/deploy/ml_refresh.sh
set -euo pipefail
cd "$(dirname "$0")"
bash ./db_export.sh
echo "[train] LightGBM ..."
sudo docker compose run --rm worker python -m minamo ml-train
echo "[series] 過去の開催（大会名・グレード）..."
sudo docker compose run --rm worker python -m minamo ml-series
echo "[formation] スタート隊形トゥエルブの表 ..."
sudo docker compose run --rm worker python -m minamo formation
echo "[wind] 場ごとの風の表 ..."
sudo docker compose run --rm worker python -m minamo wind-table || echo "        （風の表は作れませんでした。前の表のまま続けます）"
echo "[ev] 買い目の選び方の比べ（確率上位・期待値）..."
sudo docker compose run --rm worker python -m minamo ev-check || echo "        （比べられませんでした。続けます）"
