# RI-YOLO: Experiment Journal & Project Board
> **Single Source of Truth (SSOT)** для инженерных решений, логов калибровки, прохождения Control Points и статуса 23 запусков.  
> Соответствует: `RI_YOLO_Technical_Roadmap.md` (Ревизия 2.0).

---

## 📊 Dashboard текущего статуса

- **Текущая фаза:** `Фаза 0: Аудит и калибровка`
- **Прогресс по GPU-запускам:** `0 / 23`
- **Блокирующие проверки (Sanity Checks):** `2 / 6 пройдены` (CP-0.4: Identity Check, CP-0.7: FLOPs Invariance)
- **Текущая рабочая ветка:** `main` (или worktree `feature/phase0-audit`)
- **Последнее обновление:** 2026-09-13

---

## 🧭 Реестр откалиброванных констант (Global Constants Registry)

| Параметр | Описание | Ожидаемое / Default | Утверждённое значение | Источник / Скрипт | Дата фиксации |
|---|---|---|---|---|---|
| $C$ (`nwd_c`) | Константа масштаба NWD (P4-anchored) | `12.8` | *TBD* | `scripts/obj_sizes.py` | |
| $\tau_{size}$ (`size_tau`) | Температурный порог размера для sizegate | `8.0` | *TBD* | `scripts/obj_sizes.py` | |
| $\lambda_{tv}$ (`lambda_tv`) | Вес Total Variation регуляризатора | `0.001` | *TBD* | `E6' sweep` (Фаза 2) | |
| $\alpha$ (`nwd_alpha`) | Балансировочный вес NWD vs CIoU | `0.2` | *TBD* | `E7' sweep` (Фаза 2) | |
| $\rho(L, Y)$ | Корреляция Пирсона карты L и яркости Y | $> 0.5$ | *TBD* | `scripts/corr_LY.py` | |
| $\gamma_{init}$ | Значение гейта в существующих чекпойнтах | $0.01 - 0.1$ | *TBD* | `scripts/dump_gamma.py` | |

---

## 🎯 Журнал архитектурных решений (ADR / Decision Log)

| ID | Дата | Контекст / Проблема | Принятое решение | Влияние на статьи |
|---|---|---|---|---|
| **ADR-00** | 2026-09-13 | Интеграция дорожной карты | Утверждена «Ревизия 2.0» и 23 контролируемых запуска | Разделение C1 (РИНЦ), C2 (РИНЦ), C3 (Scopus/Q1) |
| **ADR-01** | *Pending* | Сравнимость прогонов 20 vs 100 эпох (E0) | *Заполняется по итогам args_diff.txt и plot_curves.py* | C3 (Раздел «Парадокс сходимости») |
| **ADR-02** | *Pending* | Калибровка константы $C$ для NWD | *Заполняется по итогам obj_sizes.py* | C2 (Раздел «Метрика рамок») |
| **ADR-03** | *Pending* | Семантический статус карты $L$ | *Заполняется по итогам corr_LY.py* | Все статьи (сохранение или замена термина Retinex) |

---

## 🛠️ ФАЗА 0: Аудит, диагностика и подготовка инфраструктуры

### 1. Обязательные патчи (Code Patches P0-1 — P0-7)
- [x] **P0-1:** Динамический расчет сдвига индексов при переносе весов (`tasks.py`)
- [x] **P0-2:** Строгий `assert frac > 0.80` на долю перенесенных тензоров (`tasks.py`)
- [x] **P0-3:** FP32 вычисление TV Loss и очистка буфера `_illumination_map` (`loss.py`)
- [x] **P0-4:** Рефакторинг `bbox_nwd`: удаление `clamp`, подъем $\epsilon$, поддержка `scaleinv` (`metrics.py`)
- [x] **P0-5:** Режим размерного гейтирования `sizegate` в `BboxLoss` (`loss.py`)
- [x] **P0-6:** Добавление контрольного модуля `RFDBlockNoGate` (`block.py`, `tasks.py`)
- [x] **P0-7:** Проброс аргументов гиперпараметров в `default.yaml` (`use_rsl`, `lambda_tv`, `use_nwd`, `nwd_c`, `nwd_mode`, `size_tau`)

### 2. Конфигурационные файлы моделей (`cfg/models/v8/`)
- [x] `yolov8s-rfd-postsppf.yaml` (базовая модель, вставка после SPPF, слой 10)
- [x] `yolov8s-rfd-presppf.yaml` (вставка перед SPPF, слой 9)
- [x] `yolov8s-rfd-p4.yaml` (вставка после P4, слой 7)
- [x] `yolov8s-rfd-p3.yaml` (вставка после P3, слой 5)
- [x] `yolov8s-capctrl.yaml` (контроль емкости `RFDBlockNoGate`)

### 3. Диагностические контроли (Control Points Ф0.1 — Ф0.7)
- [ ] **CP-0.1 (E0 / Дифф конфигов):** `diff args.yaml` между 20ep и 100ep.  
  *Результат:* `artifacts/phase0/args_diff.txt`
- [ ] **CP-0.2 (E0 / Кривые обучения):** Построение кривых best/last и loss.  
  *Результат:* `artifacts/phase0/curves_20_vs_100.png`, CSV.
- [ ] **CP-0.3 (E1 / Анализ $\gamma$):** Извлечение $\gamma$ из имеющихся `best.pt`.  
  *Результат:* $|\gamma| = $ ______ (Ожидание: $0.01 - 0.1$).
- [x] **CP-0.4 (E1 / Тест тождественности):** Запуск `tests/test_identity.py`.  
  *Результат:* `5 / 5 passed`, $|d| = 0.00\text{e+}00$ (355/369 тензоров перенесено, 96.2%).
- [ ] **CP-0.5 (E1 / Семантика $L$):** Расчет корреляции Пирсона/Спирмена $\rho(L, Y)$.  
  *Результат:* $\rho = $ ______, $std(L) = $ ______.
- [ ] **CP-0.6 (Калибровка $C$):** Расчет распределения размеров рамок на ExDark.  
  *Результат:* $C_{rec} = $ ______, $\tau_{rec} = $ ______.
- [x] **CP-0.7 (Бенчмарк сложности):** Проверка инвариантности FLOPs через `bench.py`.  
  *Результат:* Все 5 вариантов RFD $= 30.91 - 30.92$ GFLOPs ($\Delta \le 0.01$). Артефакт: `artifacts/phase0/bench_init.csv`.

---

## 🔬 ФАЗА 1: Структурная аблация и надежность (Статья С1 / РИНЦ)

> **Цель:** Изолированное доказательство превосходства точки интеграции post-SPPF и статистическая значимость прироста на 5 сидах.

### E3′: Исследование глубины встраивания (5 запусков, seed=0, 100 эпох)
| ID | Run Name | Конфигурация | Точка интеграции | mAP@50 (last) | mAP@50 (max) | mAP@50-95 | Статус |
|---|---|---|---|---|---|---|---|
| E3′-1 | `e3p__pos-p3__s0` | `yolov8s-rfd-p3.yaml` | После P3 (C2f-4) | | | | ⬜ Pending |
| E3′-2 | `e3p__pos-p4__s0` | `yolov8s-rfd-p4.yaml` | После P4 (C2f-6) | | | | ⬜ Pending |
| E3′-3 | `e3p__pos-presppf__s0` | `yolov8s-rfd-presppf.yaml` | Перед SPPF | | | | ⬜ Pending |
| E3′-4 | `e3p__pos-postsppf__s0` | `yolov8s-rfd-postsppf.yaml` | После SPPF (base) | | | | ⬜ Pending |
| E3′-5 | `e3p__capctrl__s0` | `yolov8s-capctrl.yaml` | Контроль емкости | | | | ⬜ Pending |

### E2′: Мультисид-валидация для С1 (5 запусков)
| ID | Run Name | Модель | Seed | mAP@50 | mAP@50-95 | Статус |
|---|---|---|---|---|---|---|
| E2′-B1 | `e2p__baseline__s1` | `yolov8s.yaml` | 1 | | | ⬜ Pending |
| E2′-B2 | `e2p__baseline__s2` | `yolov8s.yaml` | 2 | | | ⬜ Pending |
| E2′-R1 | `e2p__postsppf__s1` | `yolov8s-rfd-postsppf.yaml` | 1 | | | ⬜ Pending |
| E2′-R2 | `e2p__postsppf__s2` | `yolov8s-rfd-postsppf.yaml` | 2 | | | ⬜ Pending |
| E2′-R3 | `e2p__postsppf__s3` | `yolov8s-rfd-postsppf.yaml` | 3 | | | ⬜ Pending |

**Контрольная точка С1:** Двусторонний t-тест / U-тест Манна-Уитни ($p < 0.05$).

---

## 🧪 ФАЗА 2: Декомпозиция Retinex и метрика рамок (Статья С2 / РИНЦ)

> **Цель:** Анализ влияния регуляризации Total Variation ($\lambda_{tv}$) и модификаций NWD.

### E6′: Сетка коэффициента регуляризации $\lambda_{tv}$ (4 запуска, seed=0)
| ID | Run Name | $\lambda_{tv}$ | $std(L)$ | mAP@50 | mAP@50-95 | Статус |
|---|---|---|---|---|---|---|
| E6′-1 | `e6p__lam-1e-4__s0` | $10^{-4}$ | | | | ⬜ Pending |
| E6′-2 | `e6p__lam-3e-3__s0` | $3 \times 10^{-3}$ | | | | ⬜ Pending |
| E6′-3 | `e6p__lam-1e-2__s0` | $10^{-2}$ | | | | ⬜ Pending |
| E6′-4 | `e6p__lam-3e-2__s0` | $3 \times 10^{-2}$ | | | | ⬜ Pending |

### E7′: Варианты формулировки метрики рамок (3 запуска, seed=0)
| ID | Run Name | Режим NWD | Константа $C$ / $\tau$ | mAP@50 | mAP@50 (small) | Статус |
|---|---|---|---|---|---|---|
| E7′-1 | `e7p__nwd-calibC__s0` | `abs` | $C_{calib}$ | | | ⬜ Pending |
| E7′-2 | `e7p__nwd-scaleinv__s0`| `scaleinv` | $C_{calib}$ | | | ⬜ Pending |
| E7′-3 | `e7p__nwd-sizegate__s0`| `sizegate` | $\tau_{calib}$ | | | ⬜ Pending |

---

## 🚀 ФАЗА 3: Системная синергия и разрешение аномалий (Статья С3 / Scopus Q1)

### E4′: Влияние мозаичной аугментации (2 запуска, seed=0, 100 эпох)
| ID | Run Name | Модель | `close_mosaic` | mAP@50 | mAP@50-95 | Статус |
|---|---|---|---|---|---|---|
| E4′-1 | `e4p__mosaic-close30__s0` | `yolov8s-rfd-postsppf.yaml` | 30 | | | ⬜ Pending |
| E4′-2 | `e4p__mosaic-close50__s0` | `yolov8s-rfd-postsppf.yaml` | 50 | | | ⬜ Pending |

### E5′: Длительность обучения и динамика сходимости (4 запуска, seed=0)
| ID | Run Name | Модель | Эпохи | `close_mosaic` | mAP@50 | Статус |
|---|---|---|---|---|---|---|
| E5′-1 | `e5p__base-ep40__s0` | `yolov8s.yaml` | 40 | 10 | | ⬜ Pending |
| E5′-2 | `e5p__postsppf-ep40__s0` | `yolov8s-rfd-postsppf.yaml` | 40 | 10 | | ⬜ Pending |
| E5′-3 | `e5p__base-ep60__s0` | `yolov8s.yaml` | 60 | 10 | | ⬜ Pending |
| E5′-4 | `e5p__postsppf-ep60__s0` | `yolov8s-rfd-postsppf.yaml` | 60 | 10 | | ⬜ Pending |

### E11: Пропускная способность и задержка (Inference Benchmark)
- [ ] Финальный прогон `scripts/bench.py` на зафиксированном GPU кластера:
  * FP32 Batch=1 / Batch=16
  * FP16 (AMP) Batch=1 / Batch=16
  * Формирование Таблицы 2 для статьи С3.
