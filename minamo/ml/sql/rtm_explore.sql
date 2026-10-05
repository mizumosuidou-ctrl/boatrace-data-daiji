-- レースタイムモニターの記録の中身を調べる（読むだけ。書き出しには使わない）。
--   sudo docker exec -i boatrace-postgres sh -c "psql -U \$POSTGRES_USER -d rtmonitor" < minamo/ml/sql/rtm_explore.sql
\pset pager off
\echo '== 1. 記録の種類（source_table）と件数・期間'
SELECT source_table, count(*) AS n,
       min(payload->>'race_date') AS first_date, max(payload->>'race_date') AS last_date
FROM site_archive.records GROUP BY 1 ORDER BY 2 DESC LIMIT 40;

\echo '== 2. 予想（prediction_mode_runs）の payload の項目'
SELECT k, count(*) FROM site_archive.records, jsonb_object_keys(payload) k
WHERE source_table = 'prediction_mode_runs' GROUP BY 1 ORDER BY 2 DESC LIMIT 60;

\echo '== 3. 予想の中身（prediction_json）の項目（直近3日）'
SELECT k, count(*) FROM site_archive.records
CROSS JOIN LATERAL (SELECT CASE WHEN left(payload->>'prediction_json', 1) = '{' THEN (payload->>'prediction_json')::jsonb END AS p) x
CROSS JOIN LATERAL jsonb_object_keys(p) k
WHERE source_table = 'prediction_mode_runs' AND p IS NOT NULL
  AND replace(payload->>'race_date', '-', '') >= to_char(current_date - 3, 'YYYYMMDD')
GROUP BY 1 ORDER BY 2 DESC LIMIT 60;

\echo '== 4. 予想の方式（mode・method_id）ごとの件数（直近14日）'
SELECT payload->>'mode' AS mode, payload->>'method_id' AS method_id, count(*)
FROM site_archive.records
WHERE source_table = 'prediction_mode_runs'
  AND replace(payload->>'race_date', '-', '') >= to_char(current_date - 14, 'YYYYMMDD')
GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 30;

\echo '== 5. 予想の中身の見本（直近のDEEP 1件、先頭4000文字）'
SELECT payload->>'race_date' AS d, payload->>'venue_code' AS v, payload->>'race_no' AS r, payload->>'method_id' AS m,
       left(payload->>'prediction_json', 4000) AS prediction_json
FROM site_archive.records
WHERE source_table = 'prediction_mode_runs' AND payload->>'mode' = 'DEEP'
ORDER BY payload->>'created_at' DESC NULLS LAST LIMIT 1;
