#!/usr/bin/env bash
# あなたのデータベース（boatrace-postgres）からCSVを書き出し、LightGBMを学習し直す。
# データベースは読むだけで、書き換えない。
#   使い方:  sudo bash /opt/minamo/deploy/ml_refresh.sh
set -euo pipefail
cd "$(dirname "$0")"
RAW=../var/ml/raw
mkdir -p "$RAW"
DB_CONTAINER=${DB_CONTAINER:-boatrace-postgres}
DB_NAME=${DB_NAME:-rtmonitor}
for name in facts exhibition motors; do
  echo "[export] $name ..."
  sudo docker exec -i "$DB_CONTAINER" sh -c "psql -q -U \"\$POSTGRES_USER\" -d $DB_NAME -v ON_ERROR_STOP=1" \
    < ../minamo/ml/sql/$name.sql > "$RAW/$name.csv.tmp"
  mv "$RAW/$name.csv.tmp" "$RAW/$name.csv"
  echo "        $(wc -l < "$RAW/$name.csv") 行"
done
echo "[train] LightGBM ..."
sudo docker compose run --rm worker python -m minamo ml-train
