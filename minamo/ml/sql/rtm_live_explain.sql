-- rtm_live.sql がなぜ遅いかを調べる（読むだけ）。
\pset pager off
\timing on
\echo '== 1. site_archive.records の列の型'
SELECT column_name, data_type FROM information_schema.columns
WHERE table_schema = 'site_archive' AND table_name = 'records' ORDER BY ordinal_position;
\echo '== 2. 予想の行の source_updated_at・race_date の見本（新しい順に3件）'
SELECT source_key, source_updated_at, race_date, venue_code, race_no FROM site_archive.records
WHERE source_table = 'prediction_mode_runs' ORDER BY source_updated_at DESC LIMIT 3;
\echo '== 3. 予想の行のうち、きのう以降に更新された数（全部の数）'
SELECT count(*) FILTER (WHERE source_updated_at >= to_char((now() AT TIME ZONE 'UTC') - interval '1 day', 'YYYY-MM-DD')) AS recent,
       count(*) AS total
FROM site_archive.records WHERE source_table = 'prediction_mode_runs';
\echo '== 4. 問い合わせの中身（どの索引を使い、どこに時間がかかるか）'
SET enable_seqscan = off;
EXPLAIN (ANALYZE, BUFFERS)
SELECT count(*) FROM site_archive.records
WHERE source_table = 'prediction_mode_runs'
  AND source_updated_at >= to_char((now() AT TIME ZONE 'UTC') - interval '1 day', 'YYYY-MM-DD');
