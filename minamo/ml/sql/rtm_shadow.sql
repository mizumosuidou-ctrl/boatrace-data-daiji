-- レースタイムモニターの裏の予想（shadow_prediction_runs）の本線5点・12点と、その結果（shadow_prediction_outcomes）。読むだけ。
COPY (
  WITH o AS (
    SELECT payload->>'prediction_id' AS pid, payload->>'trifecta' AS trifecta, payload->>'trifecta_payout' AS payout
    FROM site_archive.records
    WHERE source_table = 'shadow_prediction_outcomes'
  )
  SELECT
    r.payload->>'race_date'                 AS race_date,
    lpad(r.payload->>'venue_code', 2, '0')  AS venue,
    r.payload->>'race_no'                   AS race_no,
    r.payload->>'algorithm_version'         AS version,
    r.payload->>'ready_at'                  AS ready_at,
    r.payload->>'capture_mode'              AS capture_mode,
    p->>'bet_decision'                      AS bet_decision,
    (SELECT string_agg(c #>> '{}', ' ') FROM jsonb_path_query(p, 'lax $.main_5[*].combination') c)      AS main5,
    (SELECT string_agg(c #>> '{}', ' ') FROM jsonb_path_query(p, 'lax $.trifecta_12[*].combination') c) AS twelve,
    o.trifecta                              AS result,
    o.payout                                AS payout
  FROM site_archive.records r
  CROSS JOIN LATERAL (
    SELECT CASE WHEN left(r.payload->>'prediction_json', 1) = '{' THEN (r.payload->>'prediction_json')::jsonb END AS p
  ) AS x
  LEFT JOIN o ON o.pid = r.payload->>'id'
  WHERE r.source_table = 'shadow_prediction_runs' AND r.payload->>'status' = 'READY' AND p IS NOT NULL
) TO STDOUT WITH CSV HEADER;
