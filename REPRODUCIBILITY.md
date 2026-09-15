# Регламент воспроизводимости (Reproducibility Guide)

В данном документе зафиксированы требования к программно-аппаратному окружению, структуре данных, стартовым чекпойнтам и воспроизведению всех стадий экспериментальной программы проекта RI-YOLO.

---

## 1. Аппаратное и программное окружение

### 1.1. Аппаратный базис
- **GPU:** NVIDIA GeForce RTX 3070 Ti (8 ГБ VRAM) или функциональный эквивалент с поддержкой CUDA Compute Capability >= 8.6.
- **CPU:** 8+ физических ядер.
- **ОЗУ:** >= 32 ГБ.
- **Хранилище:** SSD NVMe с резервом свободного места не менее 50 ГБ.

### 1.2. Программный стек
- **ОС:** Windows 10/11 64-bit или Linux (Ubuntu 22.04 LTS).
- **Python:** 3.10.x или 3.11.x (использовался 3.11.9).
- **CUDA Toolkit:** 12.1+.
- **PyTorch:** >= 2.1.0 с поддержкой CUDA.
- **Основные библиотеки:** `torchvision`, `scipy`, `pandas`, `numpy`, `matplotlib`, `pytest`.

### 1.3. Установка окружения

```bash
# Клонирование репозитория и переход в рабочую ветку
git checkout recovery-v2

# Создание виртуального окружения
python -m venv .venv
# Активация окружения (Windows PowerShell)
.venv\Scripts\Activate.ps1
# Или Linux: source .venv/bin/activate

# Установка пакета в режиме разработки
pip install -e .
pip install pytest scipy pandas matplotlib
```

---

## 2. Подготовка и верификация набора данных

### 2.1. Официальное разбиение ExDark
Датасет ExDark (Exclusively Dark, 12 классов) организован строго в соответствии с официальным разбиением:

```
datasets/
└── ExDark/
    ├── images/
    │   ├── train/     # 3000 изображений
    │   ├── val/       # 1800 изображений
    │   └── test/      # 2563 изображения (отложенная тестовая выборка)
    └── labels/
        ├── train/     # 3000 файлов меток (.txt, формат YOLO: class cx cy w h)
        ├── val/       # 1800 файлов меток
        └── test/      # 2563 файла меток
```

### 2.2. Проверка целостности данных

Запуск скрипта аудита разбиения:

```bash
python scripts/check_split.py
```

Критерии корректности:
1. `train images: 3000`, `val images: 1800`.
2. Пересечение между списками `train` и `val` по именам файлов строго равно 0.
3. Совпадения по содержимому (хэш MD5): ровно 3 пары дубликатов ракурсов (0,16 % валидационной выборки):
   - `val: 2015_03820 == train: 2015_03059`
   - `val: 2015_04079 == train: 2015_07094`
   - `val: 2015_06946 == train: 2015_03912`
4. Отсутствие пустых файлов меток или изображений без меток в выборках.

---

## 3. Стартовые веса и детерминизм

### 3.1. Единый путь и верификация весов
Все обучающие прогоны используют единый стартовый чекпойнт:
- Путь: `weights/yolov8s.pt`
- Контрольная сумма MD5 фиксируется и проверяется перед запуском каждого эксперимента.
- В логе каждого запуска печатается строка формата:
  `[INIT] weights=weights/yolov8s.pt md5=<хэш>`

### 3.2. Детерминизм обучения
Для обеспечения воспроизводимости:
- Все раннеры (`train_phase1.py`, `train_phase2.py`, `train_phase3.py`) устанавливают флаг `deterministic=True`.
- Фиксируются генераторы случайных чисел PyTorch, NumPy и Python `random` в соответствии с заданным параметром `seed` (0, 1, 2 или 3).

---

## 4. Проверка корректности кода (Unit-тесты)

Перед проведением любых вычислительных экспериментов выполняется проверка тождественного отображения модулей при инициализации:

```bash
pytest tests/test_identity.py -v
```

Все 5 конфигураций (Baseline, RFD pos-p3, pos-p4, pos-presppf, pos-postsppf, capctrl) должны успешно пройти проверку (5/5 passed).

---

## 5. Воспроизведение обучающих серий

### 5.1. Фаза 1 — Аблация позиций встраивания
Выполняется 12 прогонов (100 эпох, batch=16, imgsz=640, close_mosaic=10):

```bash
# Базовая модель на 4 сидах
python scripts/train_phase1.py --run e2p__baseline__s0 --data exdark.yaml --device 0 --epochs 100
python scripts/train_phase1.py --run e2p__baseline__s1 --data exdark.yaml --device 0 --epochs 100
python scripts/train_phase1.py --run e2p__baseline__s2 --data exdark.yaml --device 0 --epochs 100
python scripts/train_phase1.py --run e2p__baseline__s3 --data exdark.yaml --device 0 --epochs 100

# Позиции встраивания RFDBlock и контроль емкости (seed 0)
python scripts/train_phase1.py --run e3p__pos-p3__s0 --data exdark.yaml --device 0 --epochs 100
python scripts/train_phase1.py --run e3p__pos-p4__s0 --data exdark.yaml --device 0 --epochs 100
python scripts/train_phase1.py --run e3p__pos-presppf__s0 --data exdark.yaml --device 0 --epochs 100
python scripts/train_phase1.py --run e3p__pos-postsppf__s0 --data exdark.yaml --device 0 --epochs 100
python scripts/train_phase1.py --run e3p__capctrl__s0 --data exdark.yaml --device 0 --epochs 100
```

После каждого прогона выполняется аудит завершения:
```bash
python scripts/postrun_check.py --run runs/<run_name>
```

### 5.2. Фаза 2 — Аблация функций потерь и регуляризации
Выполняется серия из 7 прогонов (100 эпох, seed 0, позиция post-SPPF):

```bash
# Сетка коэффициента RSL lambda_tv
python scripts/train_phase2.py --run e6p__lam-1e-4__s0 --data exdark.yaml --device 0 --epochs 100
python scripts/train_phase2.py --run e6p__lam-1e-3__s0 --data exdark.yaml --device 0 --epochs 100
python scripts/train_phase2.py --run e6p__lam-1e-2__s0 --data exdark.yaml --device 0 --epochs 100
python scripts/train_phase2.py --run e6p__lam-1e-1__s0 --data exdark.yaml --device 0 --epochs 100

# Варианты комбинированной метрики NWD
python scripts/train_phase2.py --run e7p__nwd-calib__s0 --data exdark.yaml --device 0 --epochs 100
python scripts/train_phase2.py --run e7p__nwd-scaleinv__s0 --data exdark.yaml --device 0 --epochs 100
python scripts/train_phase2.py --run e7p__nwd-sizegate__s0 --data exdark.yaml --device 0 --epochs 100
```

### 5.3. Фаза 3 — Исследование длительности обучения
Серия E5' на 40 и 60 эпохах (seed 0, close_mosaic=10):

```bash
python scripts/train_phase3.py --run e5p__baseline__e40_s0 --data exdark.yaml --device 0 --epochs 40
python scripts/train_phase3.py --run e5p__baseline__e60_s0 --data exdark.yaml --device 0 --epochs 60
python scripts/train_phase3.py --run e5p__postsppf__e40_s0 --data exdark.yaml --device 0 --epochs 40
python scripts/train_phase3.py --run e5p__postsppf__e60_s0 --data exdark.yaml --device 0 --epochs 60
```

---

## 6. Сбор таблиц и расчёт статистики

1. **Сводная таблица статьи C1:**
   ```bash
   python scripts/collect_c1.py --out tables/C1_table1.csv
   ```
2. **Статистические критерии C1 (Манна — Уитни, Уэлч, межсидовые оценки):**
   ```bash
   python scripts/stats_c1.py --in tables/C1_table1.csv --out tables/C1_stats.csv
   ```
3. **Диагностика гладкости карты и корреляции с освещённостью (C2):**
   ```bash
   python scripts/collect_c2_diag.py
   ```
4. **Оценка точности по категориям размеров объектов:**
   ```bash
   python scripts/eval_by_size.py --out tables/C2_table2_bysize.csv
   ```
5. **Оценка обобщающей способности на отложенной тестовой выборке (C3):**
   ```bash
   python scripts/eval_test_split.py --data exdark.yaml --split test --out tables/C3_test_generalization.csv
   ```
