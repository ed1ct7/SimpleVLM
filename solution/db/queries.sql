-- VK-VLM — примеры аналитических запросов к базе результатов оценки.
--   sqlite3 solution/db/vk_vlm.sqlite < solution/db/queries.sql

.headers on
.mode column

-- 1) Сводная таблица метрик (join факт-таблицы со справочником моделей, пивот по бенчмарку)
SELECT m.label AS model,
       ROUND(g.accuracy_extracted*100, 2) AS gqa_extracted,
       ROUND(g.accuracy*100, 2)           AS gqa_exact,
       ROUND(b.accuracy*100, 2)           AS mmbench
FROM model m
LEFT JOIN eval_run g ON g.model_key = m.model_key AND g.benchmark = 'gqa'
LEFT JOIN eval_run b ON b.model_key = m.model_key AND b.benchmark = 'mmbench'
ORDER BY m.is_ours, mmbench;

-- 2) MMBench: точность по категориям-навыкам для нашей 8B (GROUP BY + агрегат)
SELECT category,
       COUNT(*)                              AS n,
       ROUND(AVG(correct)*100, 1)            AS accuracy
FROM mmbench_prediction
WHERE model_key = 'mine-8b'
GROUP BY category
ORDER BY accuracy DESC;

-- 3) GQA: сколько ответов «спасает» извлечение короткого ответа (exact=0, extracted=1) по моделям
SELECT m.label AS model,
       SUM(CASE WHEN correct = 0 AND correct_extracted = 1 THEN 1 ELSE 0 END) AS recovered,
       SUM(correct)            AS exact_ok,
       SUM(correct_extracted)  AS extracted_ok
FROM gqa_prediction p JOIN model m ON m.model_key = p.model_key
GROUP BY m.label
ORDER BY recovered DESC;

-- 4) Примеры ошибок нашей 8B на GQA (где даже extracted не сошёлся)
SELECT question, gold, pred
FROM gqa_prediction
WHERE model_key = 'mine-8b' AND correct_extracted = 0
LIMIT 5;

-- 5) Где наша 8B уступает эталону больше всего (join по категории MMBench)
SELECT a.category,
       ROUND(AVG(a.correct)*100, 1) AS mine_8b,
       ROUND(AVG(b.correct)*100, 1) AS ref_saiga_8b,
       ROUND(AVG(b.correct)*100 - AVG(a.correct)*100, 1) AS gap
FROM mmbench_prediction a
JOIN mmbench_prediction b ON a.idx = b.idx
WHERE a.model_key = 'mine-8b' AND b.model_key = 'ref-saiga-8b'
GROUP BY a.category
ORDER BY gap DESC
LIMIT 8;
