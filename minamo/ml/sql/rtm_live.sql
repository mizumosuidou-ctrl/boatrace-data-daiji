-- レースタイムモニターの今日の予想（prediction_mode_runs：DEEP・NORMAL）の1番手の艇。読むだけ。
-- deploy/rtm_live.sh が1分ごとに var/state/rtm_live.csv へ書き出し、MINAMOの試し買い「一致」が締切前に読む。
-- 速くするため：
--   - 索引のある source_updated_at（文字の時刻 '2026-10-05T…'）と、表の列の race_date（'2026-10-05'）で今日の分に絞る
--   - payload は jsonb_to_record で1回だけ開く（payload->>'…' を何回も書くと、そのたびに大きな payload を開き直す）
--   - prediction_json（大きな文字）は JSON として読まず、先頭の "ranking":[{"lane":N の N だけを文字で探す（JSON として読むと30秒かかる）
COPY (
  SELECT
    replace(r.race_date, '-', '')   AS race_date,
    lpad(r.venue_code, 2, '0')      AS venue,
    r.race_no                       AS race_no,
    j.mode, j.method_id, j.revision, j.created_at, j.capture_mode,
    substring(j.prediction_json FROM '"ranking"\s*:\s*\[\s*\{\s*"lane"\s*:\s*"?([1-6])') AS top_lane
  FROM site_archive.records r
  CROSS JOIN LATERAL jsonb_to_record(r.payload) AS j(mode text, method_id text, revision text, created_at text,
                                                      capture_mode text, prediction_json text)
  WHERE r.source_table = 'prediction_mode_runs'
    AND r.source_updated_at >= to_char((now() AT TIME ZONE 'UTC') - interval '1 day', 'YYYY-MM-DD')
    AND r.race_date = to_char((now() AT TIME ZONE 'Asia/Tokyo')::date, 'YYYY-MM-DD')
    AND j.prediction_json LIKE '{%'
) TO STDOUT WITH CSV HEADER;
