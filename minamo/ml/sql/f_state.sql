-- 選手ごと・日ごとの F・L の数（F持ちかどうか）
COPY (
  SELECT
    COALESCE(payload->>'race_date', race_date) AS race_date,
    payload->>'registration_no'   AS toban,
    payload->>'f_count'           AS f_count,
    payload->>'l_count'           AS l_count
  FROM site_archive.records
  WHERE source_table = 'racer_f_state_daily'
) TO STDOUT WITH CSV HEADER;
