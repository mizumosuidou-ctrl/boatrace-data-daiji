-- レースごとの天気・風・波（レース時点）と、グレード・女子戦などの区分
-- weather_json は {"condition","windDirection"（風が吹いてくる方角。例 北西）,"windSpeedMps","waveHeightCm"} の文字
COPY (
  SELECT
    COALESCE(payload->>'race_date', race_date)                AS race_date,
    lpad(COALESCE(payload->>'venue_code', venue_code), 2, '0') AS venue,
    COALESCE((payload->>'race_no')::int, race_no)             AS race_no,
    payload->>'grade'             AS grade,
    payload->>'audience'          AS audience,
    payload->>'series_title'      AS series_title,
    payload->>'race_name'         AS race_name,
    w->>'condition'               AS weather,
    w->>'windDirection'           AS wind_from,
    w->>'windSpeedMps'            AS wind_speed,
    w->>'waveHeightCm'            AS wave_cm,
    COALESCE(payload->>'updated_at', source_updated_at) AS updated_at
  FROM site_archive.records
  CROSS JOIN LATERAL (
    SELECT CASE WHEN left(payload->>'weather_json', 1) = '{' THEN (payload->>'weather_json')::jsonb END AS w
  ) AS x
  WHERE source_table = 'race_archive_snapshots'
) TO STDOUT WITH CSV HEADER;
