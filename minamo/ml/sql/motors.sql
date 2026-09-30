-- レースごとのモーター成績
COPY (
  SELECT
    COALESCE(payload->>'race_date', race_date)                AS race_date,
    lpad(COALESCE(payload->>'venue_code', venue_code), 2, '0') AS venue,
    COALESCE((payload->>'race_no')::int, race_no)             AS race_no,
    payload->>'lane'        AS lane,
    payload->>'motor_no'    AS motor_no,
    payload->>'top_2_rate'  AS motor_2,
    payload->>'win_rate'    AS motor_win,
    payload->>'motor_rank'  AS motor_rank,
    COALESCE(payload->>'captured_at', source_updated_at) AS captured_at
  FROM site_archive.records
  WHERE source_table = 'race_motor_snapshots'
) TO STDOUT WITH CSV HEADER;
