-- レースごとの決まり手（1着の艇の決まり手。逃げ・差し・まくり・まくり差し・抜き・恵まれ）
COPY (
  SELECT
    COALESCE(payload->>'race_date', race_date)                AS race_date,
    lpad(COALESCE(payload->>'venue_code', venue_code), 2, '0') AS venue,
    COALESCE((payload->>'race_no')::int, race_no)             AS race_no,
    payload->>'winning_method'    AS winning_method,
    COALESCE(payload->>'updated_at', source_updated_at) AS updated_at
  FROM site_archive.records
  WHERE source_table = 'race_summaries'
) TO STDOUT WITH CSV HEADER;
