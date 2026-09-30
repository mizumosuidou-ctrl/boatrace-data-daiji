-- 1艇×1レースの実績（コース・ST・スタート順位・着順）
COPY (
  SELECT
    COALESCE(payload->>'race_date', race_date)                AS race_date,
    lpad(COALESCE(payload->>'venue_code', venue_code), 2, '0') AS venue,
    COALESCE((payload->>'race_no')::int, race_no)             AS race_no,
    payload->>'lane'              AS lane,
    payload->>'course'            AS course,
    payload->>'registration_no'   AS toban,
    payload->>'class_name'        AS grade,
    payload->>'start_rank'        AS start_rank,
    payload->>'start_timing'      AS st,
    payload->>'start_hundredths'  AS st_hundredths,
    payload->>'finish_order'      AS finish,
    payload->>'race_f'            AS race_f,
    payload->>'race_l'            AS race_l,
    payload->>'motor_no'          AS motor_no,
    payload->>'result_status'     AS result_status,
    payload->>'source_type'       AS source_type,
    COALESCE(payload->>'updated_at', source_updated_at) AS updated_at
  FROM site_archive.records
  WHERE source_table = 'racer_course_race_facts'
) TO STDOUT WITH CSV HEADER;
