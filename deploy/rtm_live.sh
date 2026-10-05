#!/usr/bin/env bash
# レースタイムモニターの今日の予想（1番手の艇）を var/state/rtm_live.csv に書き出す。データベースは読むだけ。
# install_cron.sh が1分ごと（8時〜21時台）に動かす。成功したときは何も出さない
set -uo pipefail
cd "$(dirname "$0")"
OUT=../var/state/rtm_live.csv
mkdir -p ../var/state
exec 9>/tmp/minamo-rtm-live.lock
flock -n 9 || exit 0
DB_CONTAINER=${DB_CONTAINER:-boatrace-postgres}
DB_NAME=${DB_NAME:-rtmonitor}
# 問い合わせはデータベースの側でも40秒で止める（呼び出し側だけ止めると、問い合わせが残り続ける）
if timeout 50 docker exec -i -e PGOPTIONS="-c statement_timeout=40s" "$DB_CONTAINER" sh -c "psql -q -U \"\$POSTGRES_USER\" -d $DB_NAME -v ON_ERROR_STOP=1" \
    < ../minamo/ml/sql/rtm_live.sql > "$OUT.tmp"; then
  mv -f "$OUT.tmp" "$OUT"
else
  rm -f "$OUT.tmp"
  echo "$(date '+%F %T') rtm_live: 書き出せませんでした"
fi
