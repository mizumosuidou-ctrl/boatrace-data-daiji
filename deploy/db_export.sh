#!/usr/bin/env bash
# あなたのデータベース（boatrace-postgres）から、学習用のCSVを var/ml/raw に書き出す。
# データベースは読むだけで、書き換えない。
#   使い方:  sudo bash /opt/minamo/deploy/db_export.sh [名前 ...]
#   名前は minamo/ml/sql/ のファイル名（.sql を除く）。省略すると全部。
#   facts・exhibition・motors は学習に必須なので、失敗したら止まる。ほかは失敗しても前回のファイルのまま続ける。
set -euo pipefail
cd "$(dirname "$0")"
RAW=../var/ml/raw
mkdir -p "$RAW"
DB_CONTAINER=${DB_CONTAINER:-boatrace-postgres}
DB_NAME=${DB_NAME:-rtmonitor}
REQUIRED="facts exhibition motors"
OPTIONAL="weather original_db f_state odds_hist odds_results rtm_preds rtm_shadow kimarite"

export_csv() {
  local name=$1
  echo "[export] $name ..."
  if sudo docker exec -i "$DB_CONTAINER" sh -c "psql -q -U \"\$POSTGRES_USER\" -d $DB_NAME -v ON_ERROR_STOP=1" \
      < "../minamo/ml/sql/$name.sql" > "$RAW/$name.csv.tmp"; then
    mv "$RAW/$name.csv.tmp" "$RAW/$name.csv"
    echo "        $(wc -l < "$RAW/$name.csv") 行"
  else
    rm -f "$RAW/$name.csv.tmp"
    return 1
  fi
}

if [ "$#" -gt 0 ]; then
  names="$*"
else
  names="$REQUIRED $OPTIONAL"
fi
for name in $names; do
  case " $REQUIRED " in
    *" $name "*) export_csv "$name" ;;
    *) export_csv "$name" || echo "        （$name は書き出せませんでした。前回のファイルのまま続けます）" ;;
  esac
done
