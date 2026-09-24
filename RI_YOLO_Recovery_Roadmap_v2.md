---
title: "RI-YOLO: дорожная карта восстановления, версия 2.0"
subtitle: "Полный план проверок, исправлений кода, восстановления утраченной диагностики и дозапусков. Отменяет версию 1.0 в части программы экспериментов. Каждая контрольная точка требует отчёта перед переходом к следующему блоку"
author: "Технический регламент проекта (версия 2.0, составлен по результатам научного аудита)"
date: "15 сентября 2026"
lang: ru
documentclass: article
papersize: a4
fontsize: 11pt
geometry: "top=25mm,bottom=25mm,left=22mm,right=18mm"
numbersections: true
toc: true
toc-depth: 2
colorlinks: true
linkcolor: "RoyalBlue"
urlcolor: "RoyalBlue"
---

# Как пользоваться документом

## Типы задач

| Метка     | Что означает                                                              | Кто выполняет |
| --------- | ------------------------------------------------------------------------- | ------------- |
| **FIX**   | Правка кода. Обязательна до соответствующих запусков                      | агент-кодер   |
| **CHECK** | Проверка без вычислений. Результат влияет на решения                      | агент-кодер   |
| **RUN**   | Обучающий прогон на GPU                                                   | сервер        |
| **CALC**  | Расчёт по существующим чекпойнтам, обучение не требуется                  | агент-кодер   |
| **КТ-n**  | **Контрольная точка. Остановиться, прислать результат, дождаться ответа** | вы            |

## Правило контрольных точек

Контрольная точка — это не формальность. За каждой стоит развилка, на которой возможны два и более сценария, и выбор неверного означает потерю от одного дня до нескольких недель. **Переход к следующему блоку без прохождения контрольной точки запрещён.**

Формат отчёта по контрольной точке: перечисленные в ней файлы и вывод команд целиком, без сокращений и без пересказа. Пересказанный вывод бесполезен — в нём теряются именно те детали, ради которых проверка затевалась.

## Сводка объёма работ

| Блок | Содержание                        | GPU-часов | Календарно   |
| ---- | --------------------------------- | --------: | ------------ |
| 0    | Проверки без вычислений           |         0 | 1 вечер      |
| 1    | Исправления кода                  |         0 | 2–3 часа     |
| 2    | Восстановление диагностики Фазы 2 |         0 | 1–2 часа     |
| 3    | Дозапуски для статьи C1           |         4 | 1 день       |
| 4    | Дозапуски для статьи C2           |       6–8 | 1,5 дня      |
| 5    | Завершение и достройка Фазы 3     |         8 | 2 дня        |
| 6    | Пересборка артефактов             |         0 | 2 часа       |
|      | **Итого**                         | **18–20** | **5–6 дней** |

Блоки 0–2 выполняются немедленно и параллельно с тем, что уже считается на сервере. Останавливать текущие прогоны Фазы 3 не нужно.

---

# Блок 0. Проверки без вычислений

Выполняется первым. Результаты этого блока определяют, нужны ли вообще некоторые пункты последующих.

## CHECK-0.1. Режим функции потерь в Фазе 1

**Зачем.** `train_phase1.py` не передаёт `use_rsl` и `use_nwd` в `train_kwargs`, поэтому применялись умолчания из `default.yaml`. Если умолчание оказалось `True`, вся аблация позиций выполнялась с включённым регуляризатором, и тогда статья C1 описывает не то, что заявлено, а граница между C1 и C2 исчезает.

```bash
cd $PRJ
echo "=== default.yaml ==="
grep -nE "^(use_rsl|use_nwd|lambda_tv|nwd_alpha|nwd_c|nwd_mode|size_tau|close_mosaic):" \
  ultralytics/cfg/default.yaml

echo "=== фактические аргументы всех прогонов Фазы 1 ==="
for d in runs/e3p__* runs/e2p__*; do
  echo "--- $d"
  grep -E "^(use_rsl|use_nwd|lambda_tv|nwd_mode|nwd_c|close_mosaic|epochs|batch|imgsz|seed|deterministic):" "$d/args.yaml"
done
```

**Ожидается:** `use_rsl: false`, `use_nwd: false`, `close_mosaic: 10` во всех десяти прогонах.

## CHECK-0.2. Идентичность стартовых весов

**Зачем.** Фаза 1 грузит `yolov8s.pt`, Фазы 2 и 3 — `weights/yolov8s.pt`. Если это разные файлы, фазы несопоставимы.

```bash
md5sum yolov8s.pt weights/yolov8s.pt 2>&1
ls -la yolov8s.pt weights/yolov8s.pt 2>&1
```

## CHECK-0.3. Фактический вес регуляризатора в серии E7′

**Зачем.** Дефект A2: в лог пишется исходный `lambda_tv`, а не применённый. По `C2_table2.csv` во всех трёх прогонах стоит 0,001, тогда как автоподбор должен был подставить 0,01. Пока не известно, какое значение применялось, «рекорд проекта» описывает неизвестную конфигурацию.

```bash
for r in nwd-calib nwd-scaleinv nwd-sizegate; do
  echo "=== e7p__${r}__s0"
  grep -E "^(use_rsl|lambda_tv|use_nwd|nwd_alpha|nwd_mode|nwd_c|size_tau):" \
    runs/e7p__${r}__s0/args.yaml
done
echo "=== то же для E6' (контроль корректности логирования) ==="
for l in 1e-4 1e-3 1e-2 1e-1; do
  echo "--- e6p__lam-${l}__s0"
  grep -E "^(use_rsl|lambda_tv|use_nwd):" runs/e6p__lam-${l}__s0/args.yaml
done
```

## CHECK-0.4. Подтверждение отсутствия логов диагностики

**Зачем.** Дефект A1: в `train_phase2.py` колбэк вызван, а не зарегистрирован.

```bash
ls -la runs/e6p__*/rfd_log.csv runs/e7p__*/rfd_log.csv 2>&1
echo "--- для сравнения, Фаза 1 (там регистрация корректна) ---"
ls -la runs/e3p__*/rfd_log.csv runs/e2p__*/rfd_log.csv 2>&1
head -3 runs/e3p__pos-postsppf__s0/rfd_log.csv 2>&1
```

## CHECK-0.5. Формы тензоров в режиме размерного гейтирования

**Зачем.** Гипотеза о причине коллапса `sizegate` до 0,5911. Если коэффициент имеет форму $[N]$, а метрики — $[N,1]$, произведение даёт $[N,N]$, лосс завышается примерно в $N$ раз, градиенты локализации взрываются.

Временно добавить в `BboxLoss.forward` сразу после вычисления `alpha`:

```python
if not hasattr(self, "_shape_dbg"):
    self._shape_dbg = True
    print(
        "[SHAPE] iou",
        tuple(iou.shape),
        "| nwd",
        tuple(nwd.shape),
        "| alpha",
        tuple(alpha.shape) if torch.is_tensor(alpha) else type(alpha).__name__,
        "| weight",
        tuple(weight.shape),
    )
    _tmp = alpha * (1.0 - nwd) + (1.0 - alpha) * (1.0 - iou)
    print("[SHAPE] loss term before weight:", tuple(_tmp.shape), "<- ожидается [N, 1]")
```

Запустить обучение на **двух эпохах** в режиме `sizegate`, дождаться печати, прервать:

```bash
python scripts/train_phase2.py --run e7p__nwd-sizegate__s0 \
  --data exdark.yaml --device 0 --epochs 2 --project runs_debug 2>&1 | grep -A2 "\[SHAPE\]"
```

Каталог `runs_debug` обязателен: нельзя перезаписать существующий прогон.

## CHECK-0.6. Полный листинг патчей

Прислать исходный код целиком, а не фрагменты:

```bash
sed -n '/class RFDBlock/,/^class /p' ultralytics/nn/modules/block.py
sed -n '/class RFDBlockNoGate/,/^class /p' ultralytics/nn/modules/block.py
sed -n '/def bbox_nwd/,/^def /p' ultralytics/utils/metrics.py
sed -n '/class BboxLoss/,/^class /p' ultralytics/utils/loss.py
sed -n '/def get_tv_loss/,/^    def /p' ultralytics/utils/loss.py
cat scripts/rfd_callbacks.py
```

## CHECK-0.7. Списки разбиения набора данных

```bash
wc -l splits/train.txt splits/val.txt
md5sum splits/train.txt splits/val.txt
comm -12 <(sort splits/train.txt) <(sort splits/val.txt) | wc -l
grep -E "^(train|val|nc|names):" data/exdark.yaml
```

**Ожидается:** 5896 и 1467, пересечение равно нулю.

> ## КТ-1. ПРИСЛАТЬ МНЕ
>
> Вывод команд CHECK-0.1 … CHECK-0.7 целиком, каждый под своим заголовком. Дополнительно файлом: `scripts/rfd_callbacks.py`, фрагменты `block.py`, `metrics.py`, `loss.py`.
>
> **Что решается на этой точке:**
>
> 1. Нужна ли правка текста статьи C1 (зависит от CHECK-0.1).
> 2. Является ли коллапс `sizegate` научным результатом или программным дефектом (CHECK-0.5). От этого зависит, остаётся ли в C2 целый раздел.
> 3. Какую конфигурацию в действительности описывает «рекорд проекта» (CHECK-0.3).
> 4. Сопоставимы ли фазы между собой (CHECK-0.2).
>
> **До ответа не начинать блок 3 и далее.** Блоки 1 и 2 можно выполнять параллельно.

---

# Блок 1. Исправления кода

Все правки — до дозапусков. После начала блока 3 код замораживается: любое изменение потребует перезапуска затронутых прогонов.

## FIX-1.1. Регистрация колбэка в Фазе 2

Файл `scripts/train_phase2.py`, функция `run_experiment`:

```diff
     model = YOLO(exp_info["cfg"])
-    add_rfd_logging(model)
+    model.add_callback("on_fit_epoch_end", add_rfd_logging)
```

Последствие дефекта: для всех семи прогонов Фазы 2 файл `rfd_log.csv` не создан, колонка `L_std_final` равна `NA`. Правка не восстанавливает утраченные данные — она нужна для дозапусков блока 4.

## FIX-1.2. Логирование фактических гиперпараметров

Тот же файл. Сейчас перед формированием строки отчёта переменная `overrides` перечитывается из исходного словаря, из-за чего в лог попадает значение до автоподбора:

```diff
-    overrides = exp_info.get("hyp_overrides", {})
     row = {
         "run": run_name,
         ...
-        "lambda_tv": overrides.get("lambda_tv", "NA"),
-        "use_nwd": overrides.get("use_nwd", False),
-        "nwd_mode": overrides.get("nwd_mode", "NA"),
-        "nwd_c": overrides.get("nwd_c", "NA"),
-        "size_tau": overrides.get("size_tau", "NA"),
+        "lambda_tv": train_kwargs.get("lambda_tv", "NA"),
+        "use_rsl": train_kwargs.get("use_rsl", False),
+        "use_nwd": train_kwargs.get("use_nwd", False),
+        "nwd_mode": train_kwargs.get("nwd_mode", "NA"),
+        "nwd_c": train_kwargs.get("nwd_c", "NA"),
+        "size_tau": train_kwargs.get("size_tau", "NA"),
```

Источником истины должен быть словарь, фактически переданный в `model.train()`. Заодно добавлена колонка `use_rsl`, которой не было.

## FIX-1.3. Словарь рецептивных полей в сборщике C1

Файл `scripts/collect_c1.py`. Контроль ёмкости стоит слоем 10, после SPPF, что подтверждено конфигурацией `yolov8s-capctrl.yaml`:

```diff
 TRF = {
     "baseline": (None, None, 0.0),
     "pos-p3": (95, 14.8, 0.164),
     "pos-p4": (239, 37.3, 0.657),
     "pos-presppf": (399, 62.3, 2.624),
     "pos-postsppf": (783, 122.3, 2.624),
-    "capctrl": (399, 62.3, 2.623),
+    "capctrl": (783, 122.3, 2.623),
 }
```

## FIX-1.4. Форма коэффициента в режиме размерного гейтирования

**Выполнять только при подтверждении дефекта на КТ-1.** Файл `ultralytics/utils/loss.py`, `BboxLoss.forward`:

```diff
     if self.nwd_mode == "sizegate":
         tgt = target_bboxes[fg_mask]
         size = torch.sqrt((tgt[:, 2] - tgt[:, 0]).clamp(min=1e-4)
                           * (tgt[:, 3] - tgt[:, 1]).clamp(min=1e-4))
-        alpha = self.alpha * torch.exp(-size / self.size_tau)
+        alpha = (self.alpha * torch.exp(-size / self.size_tau)).unsqueeze(-1)
     else:
         alpha = self.alpha
+    assert (not torch.is_tensor(alpha)) or alpha.shape == iou.shape, \
+        f"alpha {tuple(alpha.shape)} != iou {tuple(iou.shape)}"
```

Проверочное утверждение оставить в коде постоянно: оно стоит ноль времени и предотвращает повторение дефекта в других режимах.

## FIX-1.5. Защита от подмены стартовых весов

Файлы всех трёх раннеров. Привести значение по умолчанию к единому виду и проверять контрольную сумму:

```python
# в parse_args() всех трёх файлов
parser.add_argument("--weights", type=str, default="weights/yolov8s.pt", ...)

# в начале train_single_run / run_experiment
import hashlib
h = hashlib.md5(open(args.weights, "rb").read()).hexdigest()
print(f"[INIT] weights={args.weights} md5={h}")
```

Контрольная сумма попадёт в лог каждого прогона, и вопрос о сопоставимости фаз больше не возникнет.

## FIX-1.6. Файл некорректного бенчмарка

```bash
git mv artifacts/phase0/bench_init.csv artifacts/phase0/bench_init_INVALID_no_warmup.csv
```

Значения 497–700 мс и 1,4–2,0 кадра/с получены без прогрева. Действительный источник — `bench_gpu_rtx3070ti.csv`.

## FIX-1.7. Приведение README в соответствие

| Что                              | Сейчас                                     | Должно быть                                                                                         |
| -------------------------------- | ------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| Строка `capctrl`, колонка TRF    | 399 / 62,3 %                               | **783 / 122,3 %**                                                                                   |
| Подпись к $\lambda_{tv}=10^{-2}$ | «Оптимальный баланс (Пик точности)»        | «Значение в области нечувствительности; различия по сетке не превышают 1,7 стандартного отклонения» |
| Подпись к $\lambda_{tv}=10^{-1}$ | «Пересглаживание информативных контуров»   | «Намечается деградация, уровня значимости не достигает»                                             |
| Подпись к $\lambda_{tv}=10^{-4}$ | «Недостаточная регуляризация шума матрицы» | Убрать: карта $\mathbf{L}$ с яркостью не коррелирует и шум матрицы подавлять не может               |
| Заголовок и аннотация            | «Разделение признаков по теории Retinex»   | «Модуляция признаков, мотивированная теорией Retinex»                                               |

> ## КТ-2. ПРИСЛАТЬ МНЕ
>
> Вывод `git diff --stat` и полный `git diff` по всем изменённым файлам. Плюс подтверждение, что `pytest tests/test_identity.py` проходит после правок (5 конфигураций из 5).
>
> **Что решается:** корректность патчей до того, как на них будут потрачены GPU-часы. Ошибка в FIX-1.4 обесценит весь блок 4.

---

# Блок 2. Восстановление диагностики Фазы 2

Поэпохные траектории карты $\mathbf{L}$ утрачены безвозвратно — их можно получить только перезапуском, который не оправдан. Но финальное состояние восстанавливается по сохранённым чекпойнтам без обучения, и для таблицы статьи C2 этого достаточно.

## CALC-2.1. Статистика карты и корреляция с яркостью по семи чекпойнтам

```bash
mkdir -p artifacts/phase2/diag
for r in e6p__lam-1e-4__s0 e6p__lam-1e-3__s0 e6p__lam-1e-2__s0 e6p__lam-1e-1__s0 \
  e7p__nwd-calib__s0 e7p__nwd-scaleinv__s0 e7p__nwd-sizegate__s0; do
  echo "=== $r"
  python scripts/corr_LY.py \
    --ckpt runs/$r/weights/best.pt \
    --images splits/val.txt --n 300 \
    --out-csv artifacts/phase2/diag/rho_$r.csv \
    --out-fig artifacts/phase2/diag/Lmaps_$r.png
done
```

Добавить опорную точку без регуляризатора — прогон Фазы 1:

```bash
python scripts/corr_LY.py --ckpt runs/e3p__pos-postsppf__s0/weights/best.pt \
  --images splits/val.txt --n 300 \
  --out-csv artifacts/phase2/diag/rho_lam0.csv \
  --out-fig artifacts/phase2/diag/Lmaps_lam0.png
```

## CALC-2.2. Сводная таблица диагностики

```python
# scripts/collect_c2_diag.py
import glob
import os
import re

import pandas as pd
import torch

rows = []
for f in sorted(glob.glob("artifacts/phase2/diag/rho_*.csv")):
    tag = os.path.basename(f)[4:-4]
    d = pd.read_csv(f).iloc[0].to_dict()
    lam = 0.0 if tag == "lam0" else None
    m = re.search(r"lam-([\d.e-]+)", tag)
    if m:
        lam = float(m.group(1))
    ck = "runs/e3p__pos-postsppf__s0/weights/best.pt" if tag == "lam0" else f"runs/{tag}/weights/best.pt"
    sd = torch.load(ck, map_location="cpu", weights_only=False)["model"].state_dict()
    g = [float(v.reshape(-1)[0]) for k, v in sd.items() if k.endswith("gamma")]
    rows.append(
        {
            "run": tag,
            "lambda_tv": lam,
            "pearson": round(d["pearson"], 4),
            "spearman": round(d["spearman"], 4),
            "L_mean": round(d["L_mean"], 4),
            "L_std": round(d["L_std"], 4),
            "gamma_final": round(g[0], 6) if g else None,
        }
    )
df = pd.DataFrame(rows).sort_values(["lambda_tv", "run"], na_position="last")
df.to_csv("tables/C2_diag.csv", index=False)
print(df.to_string(index=False))
```

## CALC-2.3. Аналитический рисунок градиентов

Не требует ни обучения, ни чекпойнтов:

```bash
python scripts/fig_gradients.py --C 12.8 --C-calib 10.24 --out figs/C2_fig1_gradients.png
```

## CALC-2.4. Разбивка точности по размерам объектов

```bash
python scripts/eval_by_size.py \
  --ckpts runs/e3p__pos-postsppf__s0/weights/best.pt \
  runs/e6p__lam-1e-2__s0/weights/best.pt \
  runs/e7p__nwd-calib__s0/weights/best.pt \
  runs/e7p__nwd-scaleinv__s0/weights/best.pt \
  runs/e7p__nwd-sizegate__s0/weights/best.pt \
  --out tables/C2_table2_bysize.csv
```

Границы категорий взять из `C_calib.txt`: мелкие — менее 32 пикселей (3,1 % объектов), средние — от 32 до 96 (34,7 %), крупные — от 96 (62,2 %). **Доля мелких объектов в наборе составляет 3,1 %**, поэтому величина $\mathrm{AP}_S$ будет крайне шумной, и это надо будет оговорить в статье.

> ## КТ-3. ПРИСЛАТЬ МНЕ
>
> Файлы: `tables/C2_diag.csv`, `tables/C2_table2_bysize.csv`, изображения `artifacts/phase2/diag/Lmaps_*.png` (все восемь), `figs/C2_fig1_gradients.png`.
>
> **Что решается:**
>
> 1. Вырождается ли карта $\mathbf{L}$ при росте веса регуляризатора. Если $L_{std}$ падает с ростом $\lambda_{tv}$ — это несущий результат C2, и он получен без единого GPU-часа.
> 2. Меняется ли корреляция с яркостью под действием регуляризатора. Если при $\lambda_{tv}=10^{-1}$ корреляция вырастет — формулировка о «мотивированности теорией Retinex» получает эмпирическую опору.
> 3. Подтверждается ли аналитический прогноз о разнонаправленном влиянии метрики на объекты разного размера.
>
> Это самая выгодная контрольная точка всего плана: три возможных результата для статьи C2 ценой нуля машино-часов.

---

# Блок 3. Дозапуски для статьи C1

Выполняется после КТ-1. Два прогона, четыре часа.

## RUN-3.1 и RUN-3.2. Базовая модель на сидах 0 и 3

```bash
python scripts/train_phase1.py --run e2p__baseline__s0 \
  --data exdark.yaml --device 0 --epochs 100
python scripts/train_phase1.py --run e2p__baseline__s3 \
  --data exdark.yaml --device 0 --epochs 100
```

Потребуется добавить две записи в словарь `PHASE1_EXPERIMENTS`:

```python
"e2p__baseline__s0": {"group": "E2", "cfg": "yolov8s.yaml", "seed": 0,
                      "desc": "Standard YOLOv8s baseline (seed 0)"},
"e2p__baseline__s3": {"group": "E2", "cfg": "yolov8s.yaml", "seed": 3,
                      "desc": "Standard YOLOv8s baseline (seed 3)"},
```

**Что дают эти два прогона — три разные вещи одновременно:**

1. **Парное сравнение.** Все пять конфигураций серии E3′ выполнены на сиде 0, а базовая модель — только на сидах 1 и 2. Общего сида у базы и позиций нет ни одного, поэтому сейчас сравнение межсидовое. Прогон на сиде 0 даёт опорную точку для всех пяти позиций сразу.
2. **Сбалансированная схема 4 против 4.** Минимально достижимое двустороннее значение критерия Манна — Уитни становится равным $2/\binom{8}{4} = 0{,}0286$, и непараметрический критерий делается применимым.
3. **Недостающая точка кривой длительности.** В серии E5′ базовая ветвь на 100 эпохах сейчас усредняется по сидам 1 и 2, тогда как точки 40 и 60 эпох выполнены на сиде 0. Прогон RUN-3.1 закрывает этот разрыв.

## CHECK-3.3. Контроль корректности прогонов

```bash
for r in e2p__baseline__s0 e2p__baseline__s3; do
  echo "=== $r"
  grep -E "transferred|weights=" logs/$r.log | head -3
  python scripts/postrun_check.py --run runs/$r
done
```

## CALC-3.4. Пересборка таблицы C1 и статистики

```bash
python scripts/collect_c1.py --out tables/C1_table1.csv
python scripts/stats_c1.py --summary artifacts/phase1/summary.csv --out tables/C1_stats.csv
```

Скрипт статистики, если его ещё нет:

```python
# scripts/stats_c1.py
import argparse
from math import comb

import numpy as np
import pandas as pd
from scipy import stats

p = argparse.ArgumentParser()
p.add_argument("--summary")
p.add_argument("--out")
a = p.parse_args()
d = pd.read_csv(a.summary)
d.columns = [c.strip() for c in d.columns]
base = d[d.run.str.contains("baseline")]
post = d[d.run.str.contains("postsppf")]
rows = []
for col in ["mAP50", "mAP50_95"]:
    x, y = base[col].values, post[col].values
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    dfw = (vy / len(y) + vx / len(x)) ** 2 / ((vy / len(y)) ** 2 / (len(y) - 1) + (vx / len(x)) ** 2 / (len(x) - 1))
    t, pv = stats.ttest_ind(y, x, equal_var=False)
    u, pu = stats.mannwhitneyu(y, x, alternative="two-sided")
    se = np.sqrt(vx / len(x) + vy / len(y))
    tc = stats.t.ppf(0.975, dfw)
    diff = y.mean() - x.mean()
    sp = np.sqrt(((len(x) - 1) * vx + (len(y) - 1) * vy) / (len(x) + len(y) - 2))
    rows.append(
        {
            "metric": col,
            "n_base": len(x),
            "n_rfd": len(y),
            "base_mean": round(x.mean(), 4),
            "base_sd": round(x.std(ddof=1), 4),
            "rfd_mean": round(y.mean(), 4),
            "rfd_sd": round(y.std(ddof=1), 4),
            "delta_pp": round(100 * diff, 3),
            "welch_t": round(float(t), 3),
            "welch_p": round(float(pv), 4),
            "df": round(dfw, 2),
            "ci95_lo_pp": round(100 * (diff - tc * se), 3),
            "ci95_hi_pp": round(100 * (diff + tc * se), 3),
            "mw_U": float(u),
            "mw_p": round(float(pu), 4),
            "mw_p_min_possible": round(2 / comb(len(x) + len(y), len(x)), 4),
            "cohen_d": round(diff / sp, 2),
        }
    )
out = pd.DataFrame(rows)
out.to_csv(a.out, index=False)
print(out.to_string(index=False))
```

> ## КТ-4. ПРИСЛАТЬ МНЕ
>
> Файлы: `artifacts/phase1/summary.csv` (обновлённый, 12 строк), `tables/C1_table1.csv`, `tables/C1_stats.csv`, вывод `postrun_check.py` по обоим новым прогонам.
>
> **Что решается:**
>
> 1. Сохраняется ли значимость прироста по mAP@50-95 при базе из четырёх сидов. Возможен исход, при котором значимость исчезнет, — тогда несущий тезис C1 переносится целиком на стоимостную часть.
> 2. Становится ли значимым прирост по mAP@50.
> 3. Законен ли непараметрический критерий и какое значение он даёт.
> 4. Меняется ли парето-фронт после уточнения базы.
>
> **После этой точки статья C1 закрывается окончательно, и команда получает финальные числа.**

---

# Блок 4. Дозапуски для статьи C2

Выполняется после КТ-1 и КТ-3. От шести до восьми часов.

## Обоснование: почему сетка веса регуляризатора не переделывается

Наблюдаемый размах по сетке $\lambda_{tv}$ составляет 0,44 п.п. при межзапусковом стандартном отклонении 0,26 п.п., то есть 1,7 $\sigma$. Для доказательства различий такого размера потребовалось бы от пяти до тринадцати прогонов на точку, то есть от двадцати до пятидесяти двух прогонов на всю сетку. Это несопоставимо с ценностью результата. **Правильное решение — не добирать прогоны, а переформулировать вывод:** в исследованном диапазоне качество к весу регуляризатора нечувствительно. Такой вывод полноценен и, в отличие от «найден оптимум», устойчив к проверке.

Отсюда же следует, что выбор $\lambda_{tv}$ для серии E7′ не был выбором оптимума — это фиксация значения внутри области нечувствительности. Формулировка в статье меняется, эксперименты — нет.

## RUN-4.1. Изоляция вклада метрики

**Главный недостающий прогон статьи C2.** Все три конфигурации серии E7′ выполнены с включённым регуляризатором, поэтому вклад метрики рамок не отделён от вклада регуляризатора. В аблации отсутствует ячейка «модуль плюс метрика без регуляризатора».

Добавить в `PHASE2_EXPERIMENTS`:

```python
"e7p__nwd-calib-norsl__s0": {
    "group": "E7",
    "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
    "seed": 0,
    "desc": "Calibrated NWD without smoothness regularization (isolation run)",
    "hyp_overrides": {"use_rsl": False, "use_nwd": True,
                      "nwd_alpha": 0.2, "nwd_mode": "abs", "nwd_c": 10.24},
},
```

```bash
python scripts/train_phase2.py --run e7p__nwd-calib-norsl__s0 \
  --data exdark.yaml --device 0 --epochs 100
```

Важно: для этого прогона автоподбор веса регуляризатора не должен срабатывать — группа `E7` активирует его в коде. Либо передать `--lambda_tv 0.0` явно, либо изменить условие автоподбора на проверку `overrides.get("use_rsl")`.

## RUN-4.2 и RUN-4.3. Повторы «рекордной» конфигурации

«Рекорд проекта» 0,6976 измерен одним прогоном и отстоит от точки отсчёта на $0{,}6\sigma$. Два дополнительных сида превращают его либо в результат, либо в честно закрытый вопрос.

```bash
python scripts/train_phase2.py --run e7p__nwd-calib__s1 --data exdark.yaml --device 0 --epochs 100
python scripts/train_phase2.py --run e7p__nwd-calib__s2 --data exdark.yaml --device 0 --epochs 100
```

Записи в словарь добавляются с `"seed": 1` и `"seed": 2`, а `hyp_overrides` **копируется с фактически применённых значений**, установленных на КТ-1. Если окажется, что исходный прогон шёл при $\lambda_{tv}=0{,}01$, повторы должны идти при том же значении, иначе сравнение недействительно.

## RUN-4.4. Повтор размерного гейтирования

**Выполнять только при подтверждении дефекта на КТ-1** и после FIX-1.4.

```bash
python scripts/train_phase2.py --run e7p__nwd-sizegate__s0 \
  --data exdark.yaml --device 0 --epochs 100 --project runs_fixed
```

Каталог `runs_fixed` обязателен: исходный прогон сохраняется как свидетельство дефекта, он понадобится при описании в статье.

Если дефект не подтвердится, прогон не нужен, а отрицательный результат остаётся в силе — но тогда требуется иное объяснение, и для него нужна диагностика доли положительных анкоров по эпохам.

> ## КТ-5. ПРИСЛАТЬ МНЕ
>
> Файлы: `artifacts/phase2/summary.csv` (обновлённый), `tables/C2_table1.csv`, `tables/C2_table2.csv`, `tables/C2_diag.csv` с добавленными строками новых прогонов, `runs_fixed/e7p__nwd-sizegate__s0/results.csv` при наличии.
>
> **Что решается:**
>
> 1. Отделим ли вклад метрики от вклада регуляризатора (RUN-4.1).
> 2. Воспроизводится ли «рекорд» на других сидах (RUN-4.2, RUN-4.3).
> 3. Был ли коллапс размерного гейтирования дефектом или свойством метода (RUN-4.4).
>
> **После этой точки собирается каркас статьи C2.**

---

# Блок 5. Завершение и достройка Фазы 3

Шесть текущих прогонов останавливать не нужно: они выполняются корректно, колбэк в `train_phase3.py` зарегистрирован правильно, значения `hyp_overrides` пусты, то есть режим обучения исследуется в чистом виде.

## Два дефекта дизайна, требующие достройки

**В серии E4′ отсутствует базовая ветвь.** Оба прогона мозаичной серии выполняются на конфигурации с модулем, поэтому измеряется чувствительность конкретной модификации, а не тёмных данных как таковых. Вывод «мозаичная аугментация вредна при низкой освещённости» из этих данных не следует.

**В серии E5′ смешаны сиды** — закрывается прогоном RUN-3.1 из блока 3.

## RUN-5.1 и RUN-5.2. Базовая ветвь мозаичной серии

```python
"e4p__baseline-close0__s0": {
    "group": "E4", "cfg": "yolov8s.yaml", "seed": 0,
    "epochs": 100, "close_mosaic": 0,
    "desc": "Baseline YOLOv8s, mosaic across all 100 epochs",
    "hyp_overrides": {},
},
"e4p__baseline-close50__s0": {
    "group": "E4", "cfg": "yolov8s.yaml", "seed": 0,
    "epochs": 100, "close_mosaic": 50,
    "desc": "Baseline YOLOv8s, mosaic disabled for last 50 epochs",
    "hyp_overrides": {},
},
```

```bash
python scripts/train_phase3.py --run e4p__baseline-close0__s0 --data exdark.yaml --device 0
python scripts/train_phase3.py --run e4p__baseline-close50__s0 --data exdark.yaml --device 0
```

Вместе с прогоном RUN-3.1 (базовая модель, сид 0, `close_mosaic=10`) получается полная схема два на три: две архитектуры на три режима аугментации. Только такая схема позволяет утверждать, что эффект относится к данным, а не к модели.

## RUN-5.3 и RUN-5.4. Повторы контроля ёмкости

Контроль ёмкости измерен одним прогоном и при этом **опережает полный модуль по mAP@50-95** — 0,4370 против 0,4348. Это строка, определяющая, работает ли механизм модуляции вообще, и держать её на одном прогоне нельзя.

```bash
python scripts/train_phase1.py --run e3p__capctrl__s1 --data exdark.yaml --device 0 --epochs 100
python scripts/train_phase1.py --run e3p__capctrl__s2 --data exdark.yaml --device 0 --epochs 100
```

## CHECK-5.5. Контроль режима аугментации

```bash
grep -c "Closing dataloader mosaic" logs/e4p__mosaic-close0__s0.log    # ожидается 0
grep -n "Closing dataloader mosaic" logs/e4p__mosaic-close50__s0.log   # ожидается эпоха 51
grep -n "Closing dataloader mosaic" logs/e4p__baseline-close50__s0.log # ожидается эпоха 51
```

> ## КТ-6. ПРИСЛАТЬ МНЕ
>
> Файлы: `artifacts/phase3/summary.csv` со всеми прогонами Фазы 3, обновлённый `artifacts/phase1/summary.csv` с тремя строками контроля ёмкости, `tables/C3_regime.csv`, `figs/C3_fig3_regime.png`, вывод CHECK-5.5.
>
> **Что решается:**
>
> 1. Относится ли эффект мозаичной аугментации к тёмным данным или только к модифицированной модели.
> 2. Отличается ли контроль ёмкости от полного модуля статистически при трёх прогонах. **Это определяет главный тезис статьи C3.**
> 3. Форма кривой «качество — длительность» для обеих архитектур на согласованном сиде.

---

# Блок 6. Пересборка артефактов

## CALC-6.1. Полная пересборка

```bash
python scripts/collect_c1.py --out tables/C1_table1.csv
python scripts/stats_c1.py --summary artifacts/phase1/summary.csv --out tables/C1_stats.csv
python scripts/collect_c2.py
python scripts/collect_c2_diag.py
python scripts/collect_c3.py
python scripts/bench.py --models yolov8s.yaml yolov8s-rfd-p3.yaml yolov8s-rfd-p4.yaml \
  yolov8s-rfd-presppf.yaml yolov8s-rfd-postsppf.yaml yolov8s-capctrl.yaml \
  --batches 1 16 --half 0 1 --out artifacts/phase0/bench_gpu_rtx3070ti.csv
```

## CALC-6.2. Контроль целостности

```python
# scripts/final_check.py
import os
import sys

import pandas as pd

need = [
    "tables/C1_table1.csv",
    "tables/C1_stats.csv",
    "tables/C2_table1.csv",
    "tables/C2_table2.csv",
    "tables/C2_diag.csv",
    "tables/C3_table1.csv",
    "tables/C3_table2.csv",
    "tables/C3_regime.csv",
    "artifacts/phase0/bench_gpu_rtx3070ti.csv",
    "artifacts/phase0/env.txt",
]
missing = [f for f in need if not os.path.exists(f)]
print("MISSING:" if missing else "ALL PRESENT")
for f in missing:
    print("  ", f)

# согласованность: ГФЛОП должны совпадать у всех позиций
b = pd.read_csv("artifacts/phase0/bench_gpu_rtx3070ti.csv")
g = b[b.model.str.contains("rfd|capctrl")].gflops.unique()
print("GFLOPs across positions:", g, "-> spread", round(float(g.max() - g.min()), 3))
assert g.max() - g.min() < 0.05, "ГФЛОП расходятся: ошибка в конфигурации"

# согласованность: TRF контроля ёмкости
t = pd.read_csv("tables/C1_table1.csv")
cc = t[t.config == "capctrl"]
assert int(cc.TRF_px.iloc[0]) == 783, "TRF контроля ёмкости не исправлен (FIX-1.3)"
print("TRF capctrl:", int(cc.TRF_px.iloc[0]), "OK")

# число прогонов
print("Прогонов в Фазе 1:", len(pd.read_csv("artifacts/phase1/summary.csv")))
sys.exit(1 if missing else 0)
```

## CALC-6.3. Архив для статей

```bash
tar czf artifacts_v2.tar.gz tables/ figs/ artifacts/ \
  $(for d in runs/*/; do
    echo "$d/results.csv" "$d/args.yaml"
    [ -f "$d/rfd_log.csv" ] && echo "$d/rfd_log.csv"
  done)
sha256sum artifacts_v2.tar.gz > artifacts_v2.sha256
```

> ## КТ-7. ПРИСЛАТЬ МНЕ
>
> Вывод `final_check.py` и архив `artifacts_v2.tar.gz` либо, если он велик, каталоги `tables/` и `artifacts/` целиком.
>
> **Что решается:** финальная сверка всех чисел перед написанием текстов C2 и C3.

---

# Сводные таблицы

## Все дозапуски

|   № | Прогон                      | Конфигурация                         | Блок | Для статьи | Часов |
| --: | --------------------------- | ------------------------------------ | ---- | ---------- | ----: |
|   1 | `e2p__baseline__s0`         | `yolov8s.yaml`, сид 0, 100 эпох      | 3    | C1, C3     |     2 |
|   2 | `e2p__baseline__s3`         | `yolov8s.yaml`, сид 3, 100 эпох      | 3    | C1         |     2 |
|   3 | `e7p__nwd-calib-norsl__s0`  | модуль + метрика, без регуляризатора | 4    | C2         |     2 |
|   4 | `e7p__nwd-calib__s1`        | повтор «рекорда», сид 1              | 4    | C2         |     2 |
|   5 | `e7p__nwd-calib__s2`        | повтор «рекорда», сид 2              | 4    | C2         |     2 |
|   6 | `e7p__nwd-sizegate__s0`     | повтор после исправления формы       | 4    | C2         |     2 |
|   7 | `e4p__baseline-close0__s0`  | база, мозаика все 100 эпох           | 5    | C3         |     2 |
|   8 | `e4p__baseline-close50__s0` | база, мозаика отключена на 50        | 5    | C3         |     2 |
|   9 | `e3p__capctrl__s1`          | контроль ёмкости, сид 1              | 5    | C1, C3     |     2 |
|  10 | `e3p__capctrl__s2`          | контроль ёмкости, сид 2              | 5    | C1, C3     |     2 |

Прогон 6 условный — только при подтверждении дефекта. Итого от восемнадцати до двадцати часов.

## Что НЕ запускать

| Предложение                                                        | Почему нет                                                                                                  |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| Дополнительные сиды для позиций P3, P4, до SPPF                    | Вывод «позиции неразличимы» уже поддержан; для доказательства различий нужно от 5 до 13 прогонов на позицию |
| Расширение сетки $\lambda_{tv}$                                    | Различия внутри сетки — 1,7 $\sigma$; вместо прогонов требуется переформулировка вывода                     |
| Новые режимы метрики рамок                                         | Три режима уже покрывают аналитический прогноз; четвёртый ничего не добавит                                 |
| Кроссдатасетные эксперименты                                       | Исключены ревизией 2 стратегии по форматным ограничениям                                                    |
| Перезапуск сетки $\lambda_{tv}$ ради траекторий карты $\mathbf{L}$ | Финальное состояние восстанавливается в блоке 2 без обучения                                                |
| Перезапуск Фазы 1                                                  | Дефекты A1 и A2 находятся в `train_phase2.py` и Фазы 1 не касаются                                          |

## Матрица влияния дефектов

| Дефект                         | Затронутые прогоны              | Нужен перезапуск                                 | Статья |
| ------------------------------ | ------------------------------- | ------------------------------------------------ | ------ |
| A1 — колбэк не зарегистрирован | 7 прогонов Фазы 2               | Нет, диагностика восстанавливается по чекпойнтам | C2     |
| A2 — неверное логирование веса | 3 прогона E7′                   | Нет, истина в `args.yaml`                        | C2     |
| A3 — оптимум не доказан        | Не дефект данных, дефект вывода | Нет, переформулировка                            | C2     |
| A4 — «рекорд» на одном сиде    | `e7p__nwd-calib__s0`            | Да, два повтора                                  | C2     |
| A5 — возможный дефект формы    | `e7p__nwd-sizegate__s0`         | Условно, по КТ-1                                 | C2     |
| TRF контроля ёмкости           | Ни одного, ошибка в сборщике    | Нет, пересборка таблицы                          | C1     |
| База из двух сидов             | Ни одного                       | Да, два новых прогона                            | C1, C3 |
| E4′ без базовой ветви          | Ни одного                       | Да, два новых прогона                            | C3     |
| E5′ смешение сидов             | Ни одного                       | Да, покрывается прогоном 1                       | C3     |

**Ключевой вывод матрицы: ни один из выполненных прогонов не испорчен.** Все дефекты касаются либо логирования, либо полноты схемы, либо интерпретации. Двадцать три прогона остаются в силе полностью.

## Контрольные точки

| Точка | Блок | Что прислать                                              | Что решается                                                   |
| ----- | ---- | --------------------------------------------------------- | -------------------------------------------------------------- |
| КТ-1  | 0    | Вывод семи проверок, листинги патчей                      | Режим Фазы 1, природа коллапса, фактический вес регуляризатора |
| КТ-2  | 1    | `git diff` всех правок, результат теста тождественности   | Корректность патчей до траты GPU-часов                         |
| КТ-3  | 2    | `C2_diag.csv`, разбивка по размерам, карты $\mathbf{L}$   | Три возможных результата C2 ценой нуля машино-часов            |
| КТ-4  | 3    | Обновлённый `summary.csv` Фазы 1, таблица и статистика C1 | **Окончательное закрытие статьи C1**                           |
| КТ-5  | 4    | Обновлённый `summary.csv` Фазы 2, таблицы C2              | Изоляция вклада метрики, воспроизводимость «рекорда»           |
| КТ-6  | 5    | `summary.csv` Фазы 3, кривые режима, контроль ёмкости     | **Главный тезис статьи C3**                                    |
| КТ-7  | 6    | Вывод контроля целостности, архив артефактов              | Финальная сверка перед написанием текстов                      |

---

# Что делать прямо сейчас

1. Запустить блок 0 целиком — это один вечер и ноль машино-часов.
2. Параллельно выполнить блок 2: диагностика по семи чекпойнтам даёт материал для C2 немедленно.
3. Прислать КТ-1 и КТ-3. После них я скажу, нужна ли правка текста C1 и остаётся ли в C2 раздел о размерном гейтировании.
4. Текущие прогоны Фазы 3 не трогать.
5. Команде по статье C1 отдать `C1_Thesis_Materials.md` версии 2.0 — структура финальна, числа уточнятся на КТ-4.
