-- レースタイムモニターの今日の予想（prediction_mode_runs：DEEP・NORMAL）の1番手の艇。読むだけ。
-- deploy/rtm_live.sh が1分ごとに var/state/rtm_live.csv へ書き出し、MINAMOの試し買い「一致」が締切前に読む
COPY (
  -- 先に索引のある source_updated_at で直近20時間に絞る（payload を全部開くと30秒以上かかる）
  WITH t AS MATERIALIZED (
    SELECT payload FROM site_archive.records
    WHERE source_table = 'prediction_mode_runs'
      AND source_updated_at >= now() - interval '20 hours'
  )
  SELECT
    replace(payload->>'race_date', '-', '') AS race_date,
    lpad(payload->>'venue_code', 2, '0')    AS venue,
    payload->>'race_no'                     AS race_no,
    payload->>'mode'                        AS mode,
    payload->>'method_id'                   AS method_id,
    payload->>'revision'                    AS revision,
    payload->>'created_at'                  AS created_at,
    payload->>'capture_mode'                AS capture_mode,
    p->'ranking'->0->>'lane'                AS top_lane
  FROM t
  CROSS JOIN LATERAL (
    SELECT CASE WHEN left(payload->>'prediction_json', 1) = '{' THEN (payload->>'prediction_json')::jsonb END AS p
  ) AS x
  WHERE p IS NOT NULL AND jsonb_typeof(p->'ranking') = 'array'
    AND replace(payload->>'race_date', '-', '') = to_char((now() AT TIME ZONE 'Asia/Tokyo')::date, 'YYYYMMDD')
) TO STDOUT WITH CSV HEADER;
