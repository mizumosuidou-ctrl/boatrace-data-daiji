-- オリジナル展示（一周・まわり足・直線）。ボートレース日和から取り寄せた original.csv と同じ列
COPY (
  SELECT
    COALESCE(payload->>'race_date', race_date)                AS race_date,
    lpad(COALESCE(payload->>'venue_code', venue_code), 2, '0') AS venue,
    COALESCE((payload->>'race_no')::int, race_no)             AS race_no,
    payload->>'lane'              AS lane,
    o->>'lap'                     AS lap_time,
    o->>'turn'                    AS turn_time,
    o->>'straight'                AS straight_time,
    COALESCE(payload->>'captured_at', source_updated_at) AS captured_at
  FROM site_archive.records
  CROSS JOIN LATERAL (
    SELECT CASE WHEN left(payload->>'original_metrics_json', 1) = '{' THEN (payload->>'original_metrics_json')::jsonb END AS o
  ) AS x
  WHERE source_table = 'exhibition_snapshots' AND o IS NOT NULL
) TO STDOUT WITH CSV HEADER;
