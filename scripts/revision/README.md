# Набор скриптов для ревизии статьи PIERE-2026 (Mr-Flyguy/ri-yolo)

Данный каталог содержит полный, самодостаточный и идемпотентный набор скриптов для проведения всех
дополнительных экспериментов, запрошенных рецензентами конференции PIERE (R1.1, R1.2, R1.3, R1.4).

---

## 1. Быстрый старт (Запуск на GPU-машине)

Все этапы выполняются **одной командой**:

```bash
# Запуск всех этапов ревизии (R1.1 - R1.4) на GPU 0
./scripts/revision/run_all.sh --stage all --device 0

# Либо через python напрямую:
python scripts/revision/run_all.py --stage all --device 0
```

### Проверка плана перед запуском (Dry Run, 0 секунд):
```bash
python scripts/revision/run_all.py --stage all --dry-run
```

### Запуск этапов по отдельности:
```bash
# 1. Паспорт экспериментов и метаданные (R1.4)
python scripts/revision/run_all.py --stage info

# 2. Дообучение расписаний 40 и 60 эпох (R1.2) - 8 прогонов
python scripts/revision/run_all.py --stage train --device 0

# 2a. Включение приоритета 2 (исследование мозаики close_mosaic=0 и 50, +4 прогона)
python scripts/revision/run_all.py --stage train --with-mosaic --device 0

# 3. Единая валидация всех чекпойнтов на val и test (R1.1, R1.4)
python scripts/revision/run_all.py --stage eval --device 0

# 4. Пересчет корреляции карты L с физической яркостью (R1.3)
python scripts/revision/run_all.py --stage corr --device 0

# 5. Проверка карты L на синтетическом освещении (R1.3, только инференс)
python scripts/revision/run_all.py --stage illum --device 0

# 6. Статистический анализ (критерий Уэлча, Манн-Уитни, TOST, bootstrap)
python scripts/revision/run_all.py --stage stats

# 7. Генерация сводного SUMMARY.md, перерисовка Рис. 1 и Рис. 2, упаковка zip-архива
python scripts/revision/run_all.py --stage report
```

---

## 2. Оценка времени работы (на GPU RTX 3070 Ti)

| Этап | Содержание | Оценка времени |
|---|---|---|
| **info** | Сбор паспорта, архитектуры, гиперпараметров, разбиений | < 1 мин |
| **train** (обязательный) | 8 прогонов (40 и 60 эпох, baseline и postsppf, сиды 1, 2) | ~2 ч 45 мин |
| **train** (+мозаика) | 4 прогона (100 эпох, close_mosaic=0 и 50, сиды 1, 2) | ~2 ч 45 мин |
| **eval** | Валидация всех моделей на `split=val` и `split=test` | ~25 мин |
| **corr** | Диагностика карты L на 300 кадрах и всем val | ~8 мин |
| **illum** | Инференс 300 кадров с 6 синтетическими полями $E(x,y)$ | ~10 мин |
| **stats** | Welch $t$, TOST, 10 000 bootstrap-ресэмплов | ~2 мин |
| **report** | Построение графиков 300 dpi, формирование `SUMMARY.md` и zip | < 1 мин |
| **ИТОГО** | Обязательная программа (без мозаики) | **~3 ч 30 мин** |

---

## 3. Выходные артефакты

Все результаты сохраняются в каталог `results/revision/`:
- `results/revision/SUMMARY.md` — итоговый отчет в формате Markdown со всеми таблицами и ответами рецензентам.
- `results/revision/revision_results.zip` — готовый zip-архив со всеми данными для отправки.
- `results/revision/figures/fig1_mechanism_capacity.png` — Рис. 1: вклад механизма и емкости (300 dpi).
- `results/revision/figures/fig2_schedule_duration.png` — Рис. 2: точность от длины расписания (300 dpi).
- `results/revision/metrics_all.csv` — единая сводная таблица метрик всех моделей на val и test.
- `results/revision/stats_val.csv` и `stats_test.csv` — полные таблицы статистических тестов.
- `results/revision/corr_all.csv` — результаты корреляционного анализа карты освещенности.
- `results/revision/illum.csv` — результаты синтетического эксперимента с полями освещенности.
- `results/revision/info/` — паспорт окружения, архитектуры и гиперпараметров.
- `results/revision/logs/` — построчные логи каждого запущенного этапа.

---

## 4. Свойства скрипта

1. **Идемпотентность**: Если чекпойнт или таблица уже существуют и совпадают по MD5/числу эпох, этап не перезапускается впустую.
2. **Флаг `--force`**: Принудительно перезапустить все вычисления заново.
3. **Флаг `--smoke`**: Быстрый прогон на CPU (1 эпоха, крошечный батч) для проверки работоспособности конвейера. Результаты пишутся в изолированную папку `results/revision/smoke/`.
