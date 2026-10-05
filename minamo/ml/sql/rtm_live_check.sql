-- rtm_live.sql が遅い・失敗するときの調べ（読むだけ）。
--   sudo docker exec -i boatrace-postgres sh -c "psql -U \$POSTGRES_USER -d rtmonitor" < minamo/ml/sql/rtm_live_check.sql
\pset pager off
\timing on
\echo '== 1. 予想の元の表（site_archive 以外に prediction_mode_runs があるか）'
SELECT table_schema, table_name FROM information_schema.tables
WHERE table_name ILIKE '%prediction_mode%' OR table_name ILIKE '%prediction_run%' ORDER BY 1, 2;
\echo '== 2. 元の表の列'
SELECT table_schema, column_name, data_type FROM information_schema.columns
WHERE table_name = 'prediction_mode_runs' AND table_schema <> 'site_archive' ORDER BY table_schema, ordinal_position;
\echo '== 3. site_archive.records の索引'
SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'site_archive' AND tablename = 'records';
\echo '== 4. 今日の予想の数と、一番新しい予想の時刻（site_archive に入るまでの遅れを見る）'
SELECT count(*), max(payload->>'created_at') AS newest, now() AT TIME ZONE 'Asia/Tokyo' AS now_jst
FROM site_archive.records
WHERE source_table = 'prediction_mode_runs'
  AND replace(payload->>'race_date', '-', '') = to_char((now() AT TIME ZONE 'Asia/Tokyo')::date, 'YYYYMMDD');
