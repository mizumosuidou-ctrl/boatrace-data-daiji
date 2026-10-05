-- レースタイムモニターの予想（prediction_mode_runs：DEEP・NORMAL）の、艇ごとの評価（ranking）。読むだけ。
-- 1行＝1つの予想の1艇。pos は ranking の並び（1＝一番上）。逃げ指数・着順予想・根拠の数は1艇目の行にだけ入れる（文字のまま）。
-- 同じレース・同じ方式で何回も保存されている（revision）ので、使うときは最後の版を使う
COPY (
  SELECT
    r.payload->>'race_date'                 AS race_date,
    lpad(r.payload->>'venue_code', 2, '0')  AS venue,
    r.payload->>'race_no'                   AS race_no,
    r.payload->>'mode'                      AS mode,
    r.payload->>'method_id'                 AS method_id,
    r.payload->>'revision'                  AS revision,
    r.payload->>'created_at'                AS created_at,
    r.payload->>'capture_mode'              AS capture_mode,
    e.pos                                   AS pos,
    e.b->>'lane'                            AS lane,
    e.b->>'assumed_course'                  AS course,
    e.b->>'class_name'                      AS class_name,
    e.b->>'score'                           AS score,
    e.b->>'st_score'                        AS st_score,
    e.b->>'rt_score'                        AS rt_score,
    e.b->>'rt_six_boat_rank'                AS rt_rank,
    e.b->>'motor_upset_score'               AS motor_score,
    e.b->>'finish_score'                    AS finish_score,
    e.b->>'average_st_rank'                 AS st_rank,
    e.b->>'start_rank_comparison_rank'      AS st_cmp_rank,
    e.b->>'effective_course_win_rate'       AS course_win,
    CASE WHEN e.pos = 1 THEN left((p->'inside_escape_index')::text, 600) END AS escape_index,
    CASE WHEN e.pos = 1 THEN left((p->'order')::text, 300) END              AS pred_order,
    CASE WHEN e.pos = 1 THEN (p->'evidence_heads')::text END                AS evidence
  FROM site_archive.records r
  CROSS JOIN LATERAL (
    SELECT CASE WHEN left(r.payload->>'prediction_json', 1) = '{' THEN (r.payload->>'prediction_json')::jsonb END AS p
  ) AS x
  CROSS JOIN LATERAL jsonb_array_elements(CASE WHEN jsonb_typeof(p->'ranking') = 'array' THEN p->'ranking' ELSE '[]'::jsonb END)
    WITH ORDINALITY AS e(b, pos)
  WHERE r.source_table = 'prediction_mode_runs' AND p IS NOT NULL
) TO STDOUT WITH CSV HEADER;
