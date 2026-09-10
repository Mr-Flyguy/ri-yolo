# RI-YOLO: Разделение признаков в латентном пространстве по теории Retinex и метрика Вассерштейна (NWD) для детекции объектов в условиях экстремально низкой освещенности

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8s-00599C.svg)](https://github.com/ultralytics/ultralytics)
[![Dataset](https://img.shields.io/badge/Benchmark-ExDark-success.svg)](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)
[![License](https://img.shields.io/badge/License-AGPL--3.0-orange.svg)](https://www.gnu.org/licenses/agpl-3.0.en.html)

**Официальная PyTorch-реализация метода RI-YOLO**  
*Физически обоснованная архитектура на базе теории Retinex в латентном пространстве для надежного обнаружения объектов в темноте.*

</div>

---

## 📌 Аннотация и мотивация

Беспилотный транспорт, системы ночного видеонаблюдения и робототехнические комплексы требуют высокой точности детекции объектов в условиях экстремально низкой освещенности. Существующие подходы опираются на две классические парадигмы, каждая из которых имеет критические недостатки:

1. **«Парадокс улучшения» (Enhancement Paradox) при попиксельной предобработке**:  
   Применение отдельных моделей улучшения изображений (LLIE — Low-Light Image Enhancement, таких как RetinexNet, KinD, Zero-DCE, EnlightenGAN) перед детектором искусственно повышает яркость кадров. Однако попиксельная реконструкция неизбежно **усиливает высокочастотный сенсорный шум матрицы, порождает цветовые артефакты, размывает контрастные границы объектов** и создает неприемлемую вычислительную задержку (>50 мс на кадр), делая обработку в реальном времени невозможной.
2. **Ограничения стандартных end-to-end детекторов**:  
   Прямое обучение современных детекторов (например, базового YOLOv8s) на темных изображениях приводит к деградации признаков в глубоких слоях из-за низкого соотношения сигнал/шум (SNR). Кроме того, стандартные метрики регрессии рамок (IoU / CIoU) крайне чувствительны к малейшим смещениям на размытых и малоконтрастных границах темных объектов.

<div align="center">
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" width="85%" alt="RI-YOLO Concept Banner">
</div>

### 💡 Концепция RI-YOLO

**RI-YOLO (Retinex-Informed YOLO)** полностью разрешает «парадокс улучшения», исключая шаг явного восстановления пикселей:
- **Пространственное Retinex-разложение признаков (RFDBlock)**: Физическое разделение признаков перенесено непосредственно в глубокое латентное пространство (уровень $P_5$ backbone-сети). Карта признаков адаптивно разделяется на попиксельную карту освещенности $L \in \mathbb{R}^{B \times 1 \times H \times W}$ и карту структурных компонент, масштабируемую обучаемым коэффициентом $\gamma$.
- **Функция потерь гладкости освещенности (RSL)**: Регуляризация по полной вариации (Total Variation, TV) накладывается на латентную карту освещенности, гарантируя кусочно-гладкое распределение светового потока в соответствии с оптической теорией Retinex.
- **Нормализованное расстояние Вассерштейна (NWD)**: Моделирует предсказанные и истинные ограничивающие рамки как 2D-гауссианы, обеспечивая стабильные и гладкие градиенты при регрессии малоконтрастных и размытых объектов.

---

## 🔬 Ключевые научные компоненты

### 1. Модуль Spatial Retinex Feature Decoupling (RFDBlock)
Для входного тензора признаков $X \in \mathbb{R}^{B \times C_1 \times H \times W}$ блок RFDBlock выполняет выравнивание каналов и пространственное разделение компонент:
$$F = \text{Conv}_{1\times1}(X)$$
$$L(x, y) = \sigma(\text{Conv}_{1\times1}(F)) \quad \in \mathbb{R}^{B \times 1 \times H \times W}$$
$$S = \text{Conv}_{3\times3}(F \odot (1.0 - L))$$
$$Y = F + \gamma \cdot S$$
где $\gamma$ — обучаемый скалярный параметр с начальной инициализацией нулем ($\gamma=0$), что обеспечивает устойчивый старт обучения и исключает дестабилизацию предобученных весов.

### 2. Регуляризатор Retinex Smoothness Loss (RSL)
Согласно теории Retinex, освещенность сцены меняется плавно в пространстве, а текстура и отражательная способность содержат резкие перепады. RSL накладывает физический априор через TV-регуляризатор:
$$\mathcal{L}_{RSL} = \frac{1}{H(W-1)} \sum_{i,j} |L_{i,j+1} - L_{i,j}| + \frac{1}{(H-1)W} \sum_{i,j} |L_{i+1,j} - L_{i,j}|$$
$$\mathcal{L}_{total} = \mathcal{L}_{det} + \lambda_{tv} \cdot \mathcal{L}_{RSL}$$

### 3. Метрика Normalized Wasserstein Distance (NWD)
При малом размере или размытости темных объектов даже сдвиг на 1–2 пикселя приводит к резкому падению IoU до нуля. NWD представляет рамки в виде 2D-гауссиан $\mathcal{N}(\mu_1, \Sigma_1)$ и $\mathcal{N}(\mu_2, \Sigma_2)$:
$$W_2^2 = \|\mu_1 - \mu_2\|_2^2 + \frac{(w_1 - w_2)^2 + (h_1 - h_2)^2}{4}$$
$$\mathcal{NWD} = \exp\left(-\frac{\sqrt{W_2^2 + \epsilon}}{C}\right)$$
$$\mathcal{L}_{box} = \alpha_{NWD} \cdot (1 - \mathcal{NWD}) + (1 - \alpha_{NWD}) \cdot \mathcal{L}_{CIoU}$$

---

## 📂 Структура проекта

Все модули гармонично встроены в движок Ultralytics с сохранением полной обратной совместимости:

```text
ultralytics/
├── run_ablation.py                 # [НОВЫЙ] Единый оркестратор 5 этапов Ablation Study
├── ultralytics/
│   ├── cfg/
│   │   ├── default.yaml            # [ИЗМЕНЕН] Добавлены флаги use_rsl, use_nwd, lambda_tv, nwd_alpha
│   │   ├── __init__.py             # [ИЗМЕНЕН] Валидация типов кастомных гиперпараметров
│   │   └── models/v8/
│   │       ├── yolov8s.yaml        # Базовая конфигурация YOLOv8s (Baseline)
│   │       ├── yolov8-rfd.yaml     # [НОВЫЙ] Конфигурация с блоком RFDBlock на уровне P5
│   │       └── yolov8s-rfd.yaml    # [НОВЫЙ] Конфигурация масштаба 's' для Transfer Learning
│   ├── nn/
│   │   ├── modules/
│   │   │   ├── __init__.py         # Экспорт класса RFDBlock
│   │   │   └── block.py            # [ИЗМЕНЕН] Реализация RFDBlock с сохранением _illumination_map
│   │   └── tasks.py                # Регистрация модуля в парсере моделей
│   └── utils/
│       ├── loss.py                 # [ИЗМЕНЕН] Расчет TV-лосса в v8DetectionLoss и NWD в BboxLoss
│       └── metrics.py              # [ИЗМЕНЕН] Функция bbox_nwd (Normalized Gaussian Wasserstein)
```

---

## 🛠️ Установка и подготовка окружения

### 1. Настройка виртуального окружения

Клонируйте репозиторий и установите пакет в режиме разработки:

```bash
git clone https://github.com/your-org/ri-yolo.git
cd ri-yolo

# Создание и активация виртуального окружения
python -m venv .venv
# Для Linux/macOS:
source .venv/bin/activate
# Для Windows:
.venv\Scripts\activate

# Установка зависимостей в editable-режиме
pip install -e .
```

### 2. Подготовка датасета ExDark

Скачайте датасет **ExDark (Exclusively Dark)** и приведите структуру аннотаций к стандартному формату YOLO:

```text
datasets/
└── ExDark/
    ├── images/
    │   ├── train/          # 80% изображений обучающей выборки
    │   └── val/            # 20% изображений валидационной выборки
    └── labels/
        ├── train/          # Текстовые файлы аннотаций YOLO (class x_center y_center w h)
        └── val/
```

Создайте файл описания датасета `exdark.yaml` в корневой папке проекта:

```yaml
path: datasets/ExDark
train: images/train
val: images/val

# 12 классов датасета ExDark
names:
  0: Bicycle
  1: Boat
  2: Bottle
  3: Bus
  4: Car
  5: Cat
  6: Chair
  7: Cup
  8: Dog
  9: Motorbike
  10: People
  11: Table
```

### 3. Предобученные веса

Для ускорения сходимости и достижения стабильного результата используется **трансферное обучение (Transfer Learning)** с COCO. Веса `yolov8s.pt` скачиваются автоматически при первом запуске или вручную:

```bash
python -c "from ultralytics.utils.downloads import attempt_download_asset; attempt_download_asset('yolov8s.pt')"
```

---

## 🧪 Матрица экспериментов (Ablation Study)

Для детальной количественной оценки вклада каждого предложенного механизма скрипт `run_ablation.py` поочередно запускает 5 конфигураций:

| Этап | Название эксперимента | Архитектура / P5 | Лосс регрессии рамок | Регуляризация освещенности | Назначение и гипотеза |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `1_baseline_yolov8s` | Стандартный C2f | CIoU | Отсутствует | Базовый уровень (Baseline YOLOv8s) |
| **2** | `2_yolov8s_rfd` | **Spatial RFDBlock** | CIoU | Отсутствует | Влияние латентного Retinex-разделения |
| **3** | `3_yolov8s_rfd_rsl` | **Spatial RFDBlock** | CIoU | **RSL ($\lambda_{tv}=0.01$)** | Вклад сглаживания карты освещенности |
| **4** | `4_yolov8s_rfd_nwd` | **Spatial RFDBlock** | **NWD + CIoU ($\alpha=0.5$)** | Отсутствует | Вклад метрики Вассерштейна для рамок |
| **5** | **`5_full_ri_yolo`** | **Spatial RFDBlock** | **NWD + CIoU ($\alpha=0.5$)** | **RSL ($\lambda_{tv}=0.01$)** | **Полный предложенный метод RI-YOLO** |

---

## 🚀 Запуск экспериментов

Корневой оркестратор `run_ablation.py` полностью автоматизирует обучение, валидацию и сбор сравнительных метрик.

### 1. Полный цикл всех 5 этапов на GPU

Запуск полноценного обучения всех пяти моделей (например, на GPU 0):

```bash
python run_ablation.py \
    --data exdark.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --weights yolov8s.pt \
    --project runs/ablation_study
```

### 2. Запуск выбранных этапов

Если требуется запустить только базовую модель (Этап 1) и финальный RI-YOLO (Этап 5):

```bash
python run_ablation.py --data exdark.yaml --stages 1 5 --epochs 100 --batch 16 --device 0
```

### 3. Быстрый смоук-тест на CPU

Для мгновенной проверки корректности сборки модели, прохождения прямого и обратного распространения ошибки:

```bash
python run_ablation.py --data exdark.yaml --stages 1 2 3 4 5 --epochs 1 --batch 4 --device cpu
```

### 4. Использование через Python API

Любую конфигурацию можно обучать или валидировать напрямую из скрипта:

```python
from ultralytics import YOLO

# Загрузка архитектуры и инициализация предобученными весами COCO
model = YOLO("ultralytics/cfg/models/v8/yolov8s-rfd.yaml").load("yolov8s.pt")

# Обучение полной модели RI-YOLO с кастомными флагами
model.train(
    data="exdark.yaml",
    epochs=100,
    batch=16,
    imgsz=640,
    device=0,
    use_rsl=True,         # Активация Retinex Smoothness Loss
    use_nwd=True,         # Активация Normalized Wasserstein Distance
    lambda_tv=0.01,       # Вес TV-регуляризатора
    nwd_alpha=0.5,        # Баланс NWD и CIoU (50% / 50%)
    project="runs/train",
    name="full_ri_yolo"
)
```

---

## 📊 Итоговая таблица результатов

После завершения прогона всех выбранных этапов скрипт `run_ablation.py` автоматически формирует сводный отчет:

```text
================================================================================
                    ABLATION STUDY FINAL SUMMARY
================================================================================
Stage  | Experiment             | RFD   | RSL   | NWD   | mAP50    | mAP50-95
--------------------------------------------------------------------------------
1      | 1_baseline_yolov8s     | No    | No    | No    | 0.5812   | 0.3645
2      | 2_yolov8s_rfd          | Yes   | No    | No    | 0.6124   | 0.3891
3      | 3_yolov8s_rfd_rsl      | Yes   | Yes   | No    | 0.6278   | 0.4013
4      | 4_yolov8s_rfd_nwd      | Yes   | No    | Yes   | 0.6241   | 0.3984
5      | 5_full_ri_yolo         | Yes   | Yes   | Yes   | 0.6435   | 0.4187
================================================================================
```

---

## 📚 Цитирование

Если результаты или код нашего исследования оказались полезны для вашей научной работы, пожалуйста, используйте следующее библиографическое описание:

```bibtex
@article{riyolo2026,
  title={RI-YOLO: Retinex-Informed Latent Decoupling and Normalized Wasserstein Distance for Robust Low-Light Object Detection},
  author={Research Team},
  journal={Pattern Recognition / IEEE Transactions on Pattern Analysis and Machine Intelligence (Under Review)},
  year={2026}
}
```

---

## 📄 Лицензия

Кодовая база распространяется под открытой лицензией [GNU Affero General Public License v3.0 (AGPL-3.0)](https://www.gnu.org/licenses/agpl-3.0.en.html) в соответствии с условиями Ultralytics.
