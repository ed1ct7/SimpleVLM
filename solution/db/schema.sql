-- VK-VLM — реляционная схема для хранения и анализа результатов оценки.
-- СУБД: SQLite 3 (встроена в Python, файл solution/db/vk_vlm.sqlite).
-- Источник данных: solution/results/raw/*.meta.json (сводки прогонов) и *.jsonl (предсказания).
-- Нормализация до 3НФ: справочники моделей/бенчмарков вынесены, предсказания ссылаются по ключам.

PRAGMA foreign_keys = ON;

-- Справочник моделей (свои 8B/2B + эталоны deepvk)
CREATE TABLE IF NOT EXISTS model (
    model_key TEXT PRIMARY KEY,          -- mine-8b, mine-2b, ref-saiga-8b, ref-gemma-2b
    label     TEXT NOT NULL,             -- человекочитаемое имя
    is_ours   INTEGER NOT NULL           -- 1 = наша модель, 0 = эталон
);

-- Справочник бенчмарков
CREATE TABLE IF NOT EXISTS benchmark (
    benchmark TEXT PRIMARY KEY,          -- gqa, mmbench
    title     TEXT NOT NULL,
    n_total   INTEGER                    -- размер прогона (вопросов)
);

-- Сводка прогона: модель × бенчмарк → метрики (факт-таблица)
CREATE TABLE IF NOT EXISTS eval_run (
    model_key          TEXT NOT NULL REFERENCES model(model_key),
    benchmark          TEXT NOT NULL REFERENCES benchmark(benchmark),
    n                  INTEGER NOT NULL,
    accuracy           REAL,             -- exact (GQA) / по букве (MMBench)
    accuracy_extracted REAL,             -- GQA: короткий ответ из многословного
    accuracy_lenient   REAL,             -- GQA: gold словом где угодно
    letter_rate        REAL,             -- MMBench: доля распарсенных букв
    seconds            REAL,
    PRIMARY KEY (model_key, benchmark)
);

-- Предсказания GQA (открытые вопросы)
CREATE TABLE IF NOT EXISTS gqa_prediction (
    model_key         TEXT NOT NULL REFERENCES model(model_key),
    question_id       TEXT NOT NULL,
    image_id          TEXT,             -- ключ сшивки с картинкой COCO
    question          TEXT,
    gold              TEXT,
    pred              TEXT,
    correct           INTEGER,          -- строгий exact
    correct_extracted INTEGER,          -- после извлечения короткого ответа
    PRIMARY KEY (model_key, question_id)
);

-- Предсказания MMBench (выбор A/B/C/D), есть категория-навык
CREATE TABLE IF NOT EXISTS mmbench_prediction (
    model_key   TEXT NOT NULL REFERENCES model(model_key),
    idx         INTEGER NOT NULL,
    category    TEXT,                   -- навык (20 категорий)
    question    TEXT,
    gold        TEXT,
    pred_letter TEXT,
    correct     INTEGER,
    PRIMARY KEY (model_key, idx)
);

CREATE INDEX IF NOT EXISTS ix_gqa_model    ON gqa_prediction(model_key);
CREATE INDEX IF NOT EXISTS ix_mmb_model    ON mmbench_prediction(model_key);
CREATE INDEX IF NOT EXISTS ix_mmb_category ON mmbench_prediction(category);

-- Представления (VIEW) — инкапсулируют частые отчётные запросы
DROP VIEW IF EXISTS v_leaderboard;
CREATE VIEW v_leaderboard AS
SELECT m.model_key, m.label, m.is_ours,
       ROUND(g.accuracy_extracted*100, 2) AS gqa_extracted,
       ROUND(g.accuracy*100, 2)           AS gqa_exact,
       ROUND(b.accuracy*100, 2)           AS mmbench
FROM model m
LEFT JOIN eval_run g ON g.model_key = m.model_key AND g.benchmark = 'gqa'
LEFT JOIN eval_run b ON b.model_key = m.model_key AND b.benchmark = 'mmbench';

DROP VIEW IF EXISTS v_skill_accuracy;
CREATE VIEW v_skill_accuracy AS
SELECT model_key, category,
       COUNT(*)                   AS n,
       ROUND(AVG(correct)*100, 1) AS accuracy
FROM mmbench_prediction
GROUP BY model_key, category;
