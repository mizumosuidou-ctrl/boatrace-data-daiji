-- レースタイムモニターの予想（prediction_mode_runs：DEEP・NORMAL）の買い目。読むだけ。
-- 同じレース・同じ方式で何回も保存されている（revision）ので、比べるときは最後の版を使う
COPY (
  SELECT
    payload->>'race_date'                  AS race_date,
    lpad(payload->>'venue_code', 2, '0')   AS venue,
    payload->>'race_no'                    AS race_no,
    payload->>'mode'                       AS mode,
    payload->>'method_id'                  AS method_id,
    payload->>'revision'                   AS revision,
    payload->>'created_at'                 AS created_at,
    payload->>'capture_mode'               AS capture_mode,
    p->'bets'->>'kind'                     AS kind,
    (SELECT string_agg(c #>> '{}', ' ') FROM jsonb_path_query(p, 'lax $.bets.main[*].combination') c)           AS main,
    (SELECT string_agg(c #>> '{}', ' ') FROM jsonb_path_query(p, 'lax $.bets.cover.**.combination') c)          AS cover,
    (SELECT string_agg(c #>> '{}', ' ') FROM jsonb_path_query(p, 'lax $.bets.longshot_addon.**.combination') c) AS longshot
  FROM site_archive.records
  CROSS JOIN LATERAL (
    SELECT CASE WHEN left(payload->>'prediction_json', 1) = '{' THEN (payload->>'prediction_json')::jsonb END AS p
  ) AS x
  WHERE source_table = 'prediction_mode_runs' AND p IS NOT NULL
) TO STDOUT WITH CSV HEADER;
