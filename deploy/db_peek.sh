#!/usr/bin/env bash
# データベースの中身の「形」を、数件ずつ見る（読むだけ。書き換えない）。
#   使い方:  bash /opt/minamo/deploy/db_peek.sh
set -euo pipefail
DB_CONTAINER=${DB_CONTAINER:-boatrace-postgres}
DB_NAME=${DB_NAME:-rtmonitor}
sudo docker exec -i "$DB_CONTAINER" sh -c "psql -q -P pager=off -U \"\$POSTGRES_USER\" -d $DB_NAME -v ON_ERROR_STOP=1" <<'SQL'
\echo '== 1. 天気（race_archive_snapshots）：種類ごとの件数と、天気の見本 =='
SELECT payload->>'snapshot_kind' AS kind, count(*) FROM site_archive.records
 WHERE source_table = 'race_archive_snapshots' GROUP BY 1 ORDER BY 2 DESC LIMIT 10;
SELECT payload->>'race_date' AS d, payload->>'venue_code' AS v, payload->>'race_no' AS r,
       payload->>'grade' AS grade, payload->>'audience' AS audience,
       left((payload->'weather_json')::text, 300) AS weather
  FROM site_archive.records
 WHERE source_table = 'race_archive_snapshots' AND payload->>'weather_json' IS NOT NULL
 ORDER BY payload->>'race_date' DESC LIMIT 3;
SELECT count(*) AS 天気あり FROM site_archive.records
 WHERE source_table = 'race_archive_snapshots' AND coalesce(payload->>'weather_json', '') NOT IN ('', 'null', '{}');

\echo '== 2. オリジナル展示（exhibition_snapshots.original_metrics_json）=='
SELECT count(*) AS あり FROM site_archive.records
 WHERE source_table = 'exhibition_snapshots' AND coalesce(payload->>'original_metrics_json', '') NOT IN ('', 'null', '{}', '[]');
SELECT payload->>'race_date' AS d, payload->>'venue_code' AS v, payload->>'race_no' AS r, payload->>'lane' AS lane,
       left((payload->'original_metrics_json')::text, 300) AS original
  FROM site_archive.records
 WHERE source_table = 'exhibition_snapshots' AND coalesce(payload->>'original_metrics_json', '') NOT IN ('', 'null', '{}', '[]')
 ORDER BY payload->>'race_date' DESC LIMIT 3;

\echo '== 3. レースの区分（race_divisions.division）と開催の種類（series.series_kind）=='
SELECT payload->>'division' AS division, count(*) FROM site_archive.records
 WHERE source_table = 'race_divisions' GROUP BY 1 ORDER BY 2 DESC LIMIT 15;
SELECT payload->>'series_kind' AS series_kind, count(*) FROM site_archive.records
 WHERE source_table = 'series' GROUP BY 1 ORDER BY 2 DESC LIMIT 15;

\echo '== 4. 隊形の記録（race_entry_outcomes）=='
SELECT payload->>'formation_code' AS code, payload->>'formation_direction' AS dir, count(*) FROM site_archive.records
 WHERE source_table = 'race_entry_outcomes' GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 14;
SELECT left(payload::text, 600) AS 見本 FROM site_archive.records
 WHERE source_table = 'race_entry_outcomes' ORDER BY payload->>'race_date' DESC LIMIT 1;

\echo '== 5. オッズ（odds_snapshots / odds_race_analyses）=='
SELECT payload->>'target_label' AS label, count(*) FROM site_archive.records
 WHERE source_table = 'odds_snapshots' GROUP BY 1 ORDER BY 2 DESC LIMIT 10;
SELECT left((payload->'odds_json')::text, 300) AS odds FROM site_archive.records
 WHERE source_table = 'odds_snapshots' ORDER BY payload->>'race_date' DESC LIMIT 1;
SELECT left(payload::text, 700) AS 見本 FROM site_archive.records
 WHERE source_table = 'odds_race_analyses' ORDER BY payload->>'race_date' DESC LIMIT 1;

\echo '== 6. F の状態・コース別スタート・決まり手 =='
SELECT left(payload::text, 300) AS f_state FROM site_archive.records
 WHERE source_table = 'racer_f_state_daily' AND coalesce(payload->>'f_count', '0') NOT IN ('0', '') ORDER BY payload->>'race_date' DESC LIMIT 1;
SELECT left(payload::text, 500) AS start_daily FROM site_archive.records
 WHERE source_table = 'racer_course_start_daily' ORDER BY payload->>'race_date' DESC LIMIT 1;
SELECT payload->>'winning_method' AS 決まり手, count(*) FROM site_archive.records
 WHERE source_table = 'race_summaries' GROUP BY 1 ORDER BY 2 DESC LIMIT 10;
SQL
