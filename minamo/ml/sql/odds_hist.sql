-- 3連単オッズの履歴（締切15分前・10分前・5分前・1分前・確定）。1行＝1レース×1時点、オッズは「1-2-3:12.5 1-2-4:…」の文字
COPY (
  SELECT
    COALESCE(payload->>'race_date', race_date)                AS race_date,
    lpad(COALESCE(payload->>'venue_code', venue_code), 2, '0') AS venue,
    COALESCE((payload->>'race_no')::int, race_no)             AS race_no,
    payload->>'target_label'      AS label,
    payload->>'captured_at'       AS captured_at,
    (SELECT string_agg((e->>'combination') || ':' || (e->>'odds'), ' ')
       FROM jsonb_array_elements(CASE WHEN jsonb_typeof(o->'trifecta') = 'array' THEN o->'trifecta' ELSE '[]'::jsonb END) AS e
    ) AS trifecta
  FROM site_archive.records
  CROSS JOIN LATERAL (
    SELECT CASE WHEN left(payload->>'odds_json', 1) = '{' THEN (payload->>'odds_json')::jsonb END AS o
  ) AS x
  WHERE source_table = 'odds_snapshots' AND payload->>'target_label' IN ('T15', 'T10', 'T5', 'T1', 'FINAL')
) TO STDOUT WITH CSV HEADER;
