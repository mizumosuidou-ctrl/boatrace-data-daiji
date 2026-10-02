#!/usr/bin/env bash
# あなたのデータベースに、どんな種類のデータがあるかを一覧にする（読むだけ。書き換えない）。
#   使い方:  bash /opt/minamo/deploy/db_tables.sh
set -euo pipefail
DB_CONTAINER=${DB_CONTAINER:-boatrace-postgres}
DB_NAME=${DB_NAME:-rtmonitor}
sudo docker exec -i "$DB_CONTAINER" sh -c "psql -q -P pager=off -U \"\$POSTGRES_USER\" -d $DB_NAME -v ON_ERROR_STOP=1" <<'SQL'
\echo '== データの種類ごとの件数と期間 =='
SELECT source_table AS 種類, count(*) AS 件数, min(race_date) AS 最初, max(race_date) AS 最後
FROM site_archive.records
GROUP BY 1
ORDER BY 2 DESC;
\echo '== 種類ごとの項目名（各20件から） =='
WITH t AS (SELECT DISTINCT source_table FROM site_archive.records)
SELECT t.source_table AS 種類,
       (SELECT string_agg(DISTINCT k, ', ' ORDER BY k)
          FROM (SELECT r.payload FROM site_archive.records r
                 WHERE r.source_table = t.source_table LIMIT 20) p
          CROSS JOIN LATERAL jsonb_object_keys(p.payload::jsonb) AS k) AS 項目
FROM t
ORDER BY 1;
SQL
