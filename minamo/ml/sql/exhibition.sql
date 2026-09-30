-- 展示（展示タイム・展示ST・展示進入・チルト）
COPY (
  SELECT
    COALESCE(payload->>'race_date', race_date)                AS race_date,
    lpad(COALESCE(payload->>'venue_code', venue_code), 2, '0') AS venue,
    COALESCE((payload->>'race_no')::int, race_no)             AS race_no,
    payload->>'lane'              AS lane,
    payload->>'exhibition_time'   AS exhibition_time,
    payload->>'exhibition_rank'   AS exhibition_rank,
    payload->>'start_timing'      AS ex_st,
    payload->>'exhibition_course' AS ex_course,
    payload->>'tilt'              AS tilt,
    payload->>'weight'            AS weight,
    payload->>'parts_exchange'    AS parts_exchange,
    COALESCE(payload->>'captured_at', source_updated_at) AS captured_at
  FROM site_archive.records
  WHERE source_table = 'exhibition_snapshots'
) TO STDOUT WITH CSV HEADER;
