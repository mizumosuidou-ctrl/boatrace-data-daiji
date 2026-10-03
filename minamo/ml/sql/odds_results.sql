-- オッズ履歴のあるレースの結果（3連単・2連単・決まり手）
COPY (
  SELECT
    COALESCE(payload->>'race_date', race_date)                AS race_date,
    lpad(COALESCE(payload->>'venue_code', venue_code), 2, '0') AS venue,
    COALESCE((payload->>'race_no')::int, race_no)             AS race_no,
    payload->>'trifecta'          AS trifecta,
    payload->>'exacta'            AS exacta,
    payload->>'winning_method'    AS winning_method
  FROM site_archive.records
  WHERE source_table = 'odds_race_analyses'
) TO STDOUT WITH CSV HEADER;
