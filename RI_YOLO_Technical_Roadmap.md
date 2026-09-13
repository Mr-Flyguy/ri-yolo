# RI-YOLO — Technical Roadmap

Пошаговый runbook для выполнения программы из 23 обучающих запусков.
Версия 1.0 · соответствует «Стратегии публикаций, ревизия 2.0».

---

## 0. Соглашения и подготовка окружения

### 0.1. Легенда идентификаторов

| ID | Что | Запусков | Фаза | Для статьи |
|---|---|---:|---|---|
| E0 | Аудит существующих логов | 0 | 0 | все |
| E1 | Диагностика на существующих чекпойнтах | 0 | 0 / 2 | С2, С3 |
| E3′ | Глубина встраивания | 5 | 1 | С1 |
| E2′ | Повторы с разными сидами | 5 | 1 / 3 | С1, С3 |
| E6′ | Сетка веса регуляризатора | 4 | 2 | С2 |
| E7′ | Варианты метрики рамок | 3 | 2 | С2 |
| E4′ | Мозаичная аугментация | 2 | 3 | С3 |
| E5′ | Длительность обучения | 4 | 3 | С3 |
| E11 | Замер пропускной способности | 0 | 3 | С3 |

Итого **23** обучающих запуска.

### 0.2. Структура каталогов

Создаётся один раз, до начала работ:

```bash
export PRJ=/work/ri-yolo
mkdir -p $PRJ/{runs,artifacts,logs,scripts,splits,figs,tables}
mkdir -p $PRJ/artifacts/{phase0,phase1,phase2,phase3}
cd $PRJ
```

Правило именования прогонов — **строго** такое, иначе скрипт сборки метрик не найдёт данные:

```
runs/<EXP_ID>__<CONFIG>__s<SEED>/
      ^          ^            ^
      e3p        pos-postsppf  0

примеры:
runs/e3p__pos-p3__s0
runs/e2p__baseline__s1
runs/e6p__lam-1e-2__s0
runs/e7p__nwd-scaleinv__s0
runs/e4p__mosaic-close50__s0
runs/e5p__postsppf-ep40__s0
```

### 0.3. Фиксация окружения

```bash
cd $PRJ
python -c "import torch, ultralytics, platform; \
print('torch', torch.__version__); \
print('cuda', torch.version.cuda); \
print('cudnn', torch.backends.cudnn.version()); \
print('ultralytics', ultralytics.__version__); \
print('gpu', torch.cuda.get_device_name(0)); \
print('python', platform.python_version())" | tee artifacts/phase0/env.txt

pip freeze > artifacts/phase0/requirements.lock
nvidia-smi -q > artifacts/phase0/nvidia-smi.txt
git -C /path/to/ri-yolo rev-parse HEAD > artifacts/phase0/commit.txt
```

Файл `artifacts/phase0/env.txt` целиком копируется в раздел Implementation details каждой из трёх статей.

### 0.4. Фиксация разбиения набора данных

Разбиение по конвенции MAET: 5896 обучающих / 1467 тестовых.

```bash
# splits/train.txt и splits/val.txt — абсолютные пути к изображениям
wc -l splits/train.txt splits/val.txt     # ожидается 5896 и 1467
md5sum splits/*.txt > artifacts/phase0/splits.md5
```

`data/exdark.yaml`:

```yaml
path: /work/ri-yolo/datasets/ExDark
train: /work/ri-yolo/splits/train.txt
val:   /work/ri-yolo/splits/val.txt
nc: 12
names: [Bicycle, Boat, Bottle, Bus, Car, Cat, Chair, Cup, Dog, Motorbike, People, Table]
```

Проверка отсутствия пересечения (обязательна, попадает в раздел воспроизводимости):

```bash
comm -12 <(sort splits/train.txt) <(sort splits/val.txt) | wc -l   # должно быть 0
```

### 0.5. Базовая команда обучения

Все запуски выполняются через один скрипт `scripts/train.py`, чтобы гиперпараметры нигде не расходились:

```python
# scripts/train.py
import argparse, os, torch, random, numpy as np
from ultralytics import YOLO
from rfd_callbacks import add_rfd_logging     # см. §0.7

p = argparse.ArgumentParser()
p.add_argument("--model",   required=True)             # путь к .yaml
p.add_argument("--name",    required=True)             # <EXP_ID>__<CONFIG>__s<SEED>
p.add_argument("--seed",    type=int, default=0)
p.add_argument("--epochs",  type=int, default=100)
p.add_argument("--close_mosaic", type=int, default=10)
p.add_argument("--use_rsl",  action="store_true")
p.add_argument("--lambda_tv", type=float, default=0.001)
p.add_argument("--use_nwd",  action="store_true")
p.add_argument("--nwd_alpha", type=float, default=0.2)
p.add_argument("--nwd_c",     type=float, default=12.8)
p.add_argument("--nwd_mode",  default="abs", choices=["abs", "scaleinv", "sizegate"])
a = p.parse_args()

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

model = YOLO(a.model).load("weights/yolov8s.pt")
model.add_callback("on_fit_epoch_end", add_rfd_logging)
model.train(
    data="data/exdark.yaml",
    project="runs", name=a.name, exist_ok=False,
    epochs=a.epochs, batch=16, imgsz=640, workers=8, device=0,
    seed=a.seed, deterministic=True, amp=True,
    optimizer="SGD", lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
    warmup_epochs=3.0, close_mosaic=a.close_mosaic,
    use_rsl=a.use_rsl, lambda_tv=a.lambda_tv,
    use_nwd=a.use_nwd, nwd_alpha=a.nwd_alpha,
    nwd_c=a.nwd_c, nwd_mode=a.nwd_mode,
    plots=True, save_period=-1, val=True,
)
```

**Единственное, что меняется между запусками, — аргументы командной строки.** Ничего не правится в `default.yaml` вручную.

### 0.6. Обязательные правки кода (выполняются ДО первого запуска)

Семь патчей. Без них часть прогонов даст молча неверный результат.

**P0-1. Динамическое определение позиции вставки при переносе весов.**
Текущее условие в `BaseModel.load()` привязано к строке `"10.structure_conv"` и **не сработает** для позиций P3, P4 и preSPPF — веса загрузятся со сдвигом, обучение не упадёт, результат будет испорчен.

```python
# ultralytics/nn/tasks.py, BaseModel.load()
rfd_idx = [i for i, m in enumerate(self.model)
           if m.__class__.__name__ in ("RFDBlock", "RFDBlockNoGate")]
ckpt_has_rfd = any("structure_conv" in k for k in csd)
if rfd_idx and not ckpt_has_rfd:
    shift_from, n_shift = min(rfd_idx), len(rfd_idx)
    remapped = {}
    for k, v in csd.items():
        parts = k.split(".")
        if len(parts) > 2 and parts[0] == "model" and parts[1].isdigit():
            idx = int(parts[1])
            new_idx = idx + n_shift if idx >= shift_from else idx
            remapped[f"model.{new_idx}." + ".".join(parts[2:])] = v
        else:
            remapped[k] = v
    csd = remapped
    LOGGER.info(f"[RFD] weight remap: shift +{n_shift} for idx >= {shift_from}")
```

**P0-2. Assert на долю перенесённых весов.**

```python
# там же, сразу после загрузки csd в state_dict
sd = self.model.state_dict() if hasattr(self, "model") else self.state_dict()
matched = sum(1 for k in csd if k in sd and csd[k].shape == sd[k].shape)
frac = matched / len(sd)
LOGGER.info(f"[RFD] transferred {matched}/{len(sd)} tensors ({frac:.3f})")
assert frac > 0.80, f"weight remapping failed: only {frac:.3f} transferred"
```

**P0-3. Регуляризатор: fp32, накопление по микробатчам, очистка буфера.**

```python
# v8DetectionLoss.get_tv_loss(): считать в float32
ill = m._illumination_map.float()

# v8DetectionLoss.__call__(): после использования обнулить буферы
from ultralytics.utils.torch_utils import de_parallel
for m in de_parallel(self.model).modules():
    if hasattr(m, "_illumination_map"):
        m._illumination_map = None
```

**P0-4. Метрика рамок: убрать clamp, поднять eps, добавить режимы.**

```python
def bbox_nwd(box1, box2, xywh=False, constant=12.8, mode="abs", eps=1e-4):
    # ... вычисление cx, cy, w, h как раньше ...
    w2_sq = (cx1-cx2).pow(2) + (cy1-cy2).pow(2) + ((w1-w2).pow(2) + (h1-h2).pow(2))/4.0
    w2 = torch.sqrt(w2_sq + eps*eps)
    if mode == "abs":                      # исходный вариант
        return torch.exp(-w2 / constant)
    if mode == "scaleinv":                 # масштабно-инвариантный
        s = torch.sqrt(w2_box * h2_box).clamp(min=eps)   # размер цели
        return torch.exp(-(w2 / s) / constant)
    raise ValueError(mode)
```

`clamp(min=0, max=1)` удалён: он избыточен и зануляет градиент на границе.

**P0-5. Размерно-зависимое взвешивание** (режим `sizegate`) — в `BboxLoss.forward()`:

```python
if self.nwd_mode == "sizegate":
    tgt = target_bboxes[fg_mask]
    size = torch.sqrt((tgt[:,2]-tgt[:,0]).clamp(min=1e-4) * (tgt[:,3]-tgt[:,1]).clamp(min=1e-4))
    alpha = self.alpha * torch.exp(-size / self.size_tau)   # size_tau из §1.6
else:
    alpha = self.alpha
loss_iou = (alpha*(1.0-nwd) + (1.0-alpha)*(1.0-iou)) * weight
```

**P0-6. Контрольный блок равной ёмкости.**
Точный контроль ёмкости — это тот же блок с отключённым гейтом (разница 512 параметров, 0,02 %). Использовать `C2f` нельзя: у него другое число параметров.

```python
class RFDBlockNoGate(nn.Module):
    """Контроль ёмкости: та же топология, гейт заменён константой."""
    def __init__(self, c1, c2):
        super().__init__()
        assert c1 == c2
        self.cv1 = Conv(c1, c2, 1, 1)
        self.structure_conv = Conv(c2, c2, 3, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        feat = self.cv1(x)
        return x + self.gamma * self.structure_conv(feat * 0.5)
```

Зарегистрировать в `parse_model()` рядом с `RFDBlock`. Добавить `assert c1 == c2` и в сам `RFDBlock`.

**P0-7. Проброс аргументов.** В `ultralytics/cfg/default.yaml` добавить `use_rsl: False`, `lambda_tv: 0.001`, `use_nwd: False`, `nwd_alpha: 0.2`, `nwd_c: 12.8`, `nwd_mode: abs`, `size_tau: 8.0`. Проверить, что они доезжают до `v8DetectionLoss` через `self.hyp`.

### 0.7. Колбэк логирования состояния модуля

```python
# scripts/rfd_callbacks.py
import csv, os, torch

def add_rfd_logging(trainer):
    path = os.path.join(trainer.save_dir, "rfd_log.csv")
    new = not os.path.exists(path)
    rows = []
    for name, m in trainer.model.named_modules():
        if m.__class__.__name__ not in ("RFDBlock", "RFDBlockNoGate"):
            continue
        r = {"epoch": trainer.epoch, "module": name,
             "gamma": float(m.gamma.detach().cpu())}
        ill = getattr(m, "_illumination_map", None)
        if ill is not None:
            ill = ill.detach().float()
            r.update({
                "L_mean": float(ill.mean()),
                "L_std_spatial": float(ill.std(dim=(2, 3)).mean()),
                "L_min": float(ill.min()), "L_max": float(ill.max()),
                "L_sat_frac": float(((ill < 0.02) | (ill > 0.98)).float().mean()),
            })
        rows.append(r)
    if not rows:
        return
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
```

**Важно:** буфер `_illumination_map` обнуляется после вычисления лосса (P0-3), поэтому колбэк может получить `None`. Чтобы этого не происходило, в P0-3 обнуление выполнять **не в `__call__` лосса, а в `on_train_batch_end`**, либо сохранять в блоке отдельную detached-копию `self._ill_stat = illumination_mask.detach()` специально для логирования.

### 0.8. Конфигурации моделей

Пять `.yaml` в `ultralytics/cfg/models/v8/`. Отличаются индексом вставки и, соответственно, ссылками `Concat` в шее. **Индексы в шее пересчитаны — переписывать «на глаз» нельзя.**

`yolov8s-rfd-postsppf.yaml` (базовая конфигурация проекта, вставка после SPPF):

```yaml
nc: 80
scales: {s: [0.33, 0.50, 1024]}
backbone:
  - [-1, 1, Conv, [64, 3, 2]]        # 0
  - [-1, 1, Conv, [128, 3, 2]]       # 1
  - [-1, 3, C2f, [128, True]]        # 2
  - [-1, 1, Conv, [256, 3, 2]]       # 3
  - [-1, 6, C2f, [256, True]]        # 4  P3
  - [-1, 1, Conv, [512, 3, 2]]       # 5
  - [-1, 6, C2f, [512, True]]        # 6  P4
  - [-1, 1, Conv, [1024, 3, 2]]      # 7
  - [-1, 3, C2f, [1024, True]]       # 8
  - [-1, 1, SPPF, [1024, 5]]         # 9
  - [-1, 1, RFDBlock, [1024]]        # 10 <<< ВСТАВКА
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 11
  - [[-1, 6], 1, Concat, [1]]                    # 12
  - [-1, 3, C2f, [512]]                          # 13
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 14
  - [[-1, 4], 1, Concat, [1]]                    # 15
  - [-1, 3, C2f, [256]]                          # 16  P3 out
  - [-1, 1, Conv, [256, 3, 2]]                   # 17
  - [[-1, 13], 1, Concat, [1]]                   # 18
  - [-1, 3, C2f, [512]]                          # 19  P4 out
  - [-1, 1, Conv, [512, 3, 2]]                   # 20
  - [[-1, 10], 1, Concat, [1]]                   # 21  <- от RFDBlock
  - [-1, 3, C2f, [1024]]                         # 22  P5 out
  - [[16, 19, 22], 1, Detect, [nc]]              # 23
```

`yolov8s-rfd-presppf.yaml` — вставка **перед** SPPF (тот же размер блока, TRF 399 px):

```yaml
# backbone 0..8 идентичен, далее:
  - [-1, 1, RFDBlock, [1024]]        # 9  <<< ВСТАВКА
  - [-1, 1, SPPF, [1024, 5]]         # 10
head:
  # 11..20 идентичны postsppf
  - [[-1, 10], 1, Concat, [1]]       # 21  <- от SPPF
  - [-1, 3, C2f, [1024]]             # 22
  - [[16, 19, 22], 1, Detect, [nc]]  # 23
```

`yolov8s-rfd-p4.yaml` — вставка после P4 (блок 256 каналов):

```yaml
backbone:
  # 0..6 как в базовой
  - [-1, 1, RFDBlock, [512]]         # 7  <<< ВСТАВКА (512*0.5 = 256 каналов)
  - [-1, 1, Conv, [1024, 3, 2]]      # 8
  - [-1, 3, C2f, [1024, True]]       # 9
  - [-1, 1, SPPF, [1024, 5]]         # 10
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 11
  - [[-1, 7], 1, Concat, [1]]                    # 12  <- от RFDBlock (P4)
  - [-1, 3, C2f, [512]]                          # 13
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 14
  - [[-1, 4], 1, Concat, [1]]                    # 15
  - [-1, 3, C2f, [256]]                          # 16
  - [-1, 1, Conv, [256, 3, 2]]                   # 17
  - [[-1, 13], 1, Concat, [1]]                   # 18
  - [-1, 3, C2f, [512]]                          # 19
  - [-1, 1, Conv, [512, 3, 2]]                   # 20
  - [[-1, 10], 1, Concat, [1]]                   # 21
  - [-1, 3, C2f, [1024]]                         # 22
  - [[16, 19, 22], 1, Detect, [nc]]              # 23
```

`yolov8s-rfd-p3.yaml` — вставка после P3 (блок 128 каналов):

```yaml
backbone:
  # 0..4 как в базовой
  - [-1, 1, RFDBlock, [256]]         # 5  <<< ВСТАВКА (256*0.5 = 128 каналов)
  - [-1, 1, Conv, [512, 3, 2]]       # 6
  - [-1, 6, C2f, [512, True]]        # 7   P4
  - [-1, 1, Conv, [1024, 3, 2]]      # 8
  - [-1, 3, C2f, [1024, True]]       # 9
  - [-1, 1, SPPF, [1024, 5]]         # 10
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 11
  - [[-1, 7], 1, Concat, [1]]                    # 12
  - [-1, 3, C2f, [512]]                          # 13
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 14
  - [[-1, 5], 1, Concat, [1]]                    # 15  <- от RFDBlock (P3)
  - [-1, 3, C2f, [256]]                          # 16
  - [-1, 1, Conv, [256, 3, 2]]                   # 17
  - [[-1, 13], 1, Concat, [1]]                   # 18
  - [-1, 3, C2f, [512]]                          # 19
  - [-1, 1, Conv, [512, 3, 2]]                   # 20
  - [[-1, 10], 1, Concat, [1]]                   # 21
  - [-1, 3, C2f, [1024]]                         # 22
  - [[16, 19, 22], 1, Detect, [nc]]              # 23
```

`yolov8s-capctrl.yaml` — идентичен `postsppf`, строка 10 заменена на `[-1, 1, RFDBlockNoGate, [1024]]`.

Проверка всех конфигураций одной командой (до обучения):

```bash
python - << 'EOF'
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops
for cfg in ["yolov8s.yaml", "yolov8s-rfd-p3.yaml", "yolov8s-rfd-p4.yaml",
            "yolov8s-rfd-presppf.yaml", "yolov8s-rfd-postsppf.yaml",
            "yolov8s-capctrl.yaml"]:
    m = YOLO(cfg).model
    n = sum(p.numel() for p in m.parameters())
    print(f"{cfg:30s} params={n/1e6:7.3f}M  GFLOPs={get_flops(m, imgsz=640):6.2f}")
EOF
```

Ожидаемые значения (nc=80; для nc=12 параметров на 26 316 меньше):

| Конфигурация | Параметры, М | ГFLOPs |
|---|---:|---:|
| yolov8s (база) | 11,17 | 28,6 |
| rfd-p3 | 11,33 | 30,7 |
| rfd-p4 | 11,82 | 30,7 |
| rfd-presppf | 13,79 | 30,7 |
| rfd-postsppf | 13,79 | 30,7 |
| capctrl | 13,79 | 30,7 |

Если ГFLOPs у p3/p4/p5 различаются — ошибка в конфигурации: стоимость блока инвариантна относительно уровня.

---

## ФАЗА 0 — аудит и подготовка (0 GPU-часов, 2–3 дня)

Выполняется **полностью** до первого нового запуска. Результаты могут изменить формулировки всех трёх статей.

### Ф0.1 — E0: сравнимость существующих прогонов

**Цель:** установить, сравнимы ли имеющиеся прогоны на 20 и 100 эпох.
**Артефакт:** `artifacts/phase0/args_diff.txt` → раздел «Материалы и методы» С3.

```bash
cd $PRJ
diff runs/exp_20ep/args.yaml runs/exp_100ep/args.yaml | tee artifacts/phase0/args_diff.txt
```

**Точка контроля:** в выводе не должно быть ничего, кроме `epochs`, `name`, `save_dir`.
Если есть другие отличия — прогоны несравнимы, аномалия «20 против 100» объясняется конфигурацией, и старые числа в статьи не идут. Зафиксировать факт в `artifacts/phase0/NOTES.md` и опираться только на новые запуски Фазы 3.

### Ф0.2 — E0: кривые обучения и вопрос best/last

**Цель:** определить, из какого чекпойнта брались отчётные метрики.
**Артефакт:** `artifacts/phase0/curves_20_vs_100.png` → **С3, Рисунок 3** (левая панель).

```bash
python scripts/plot_curves.py \
  --runs runs/exp_20ep runs/exp_100ep \
  --cols metrics/mAP50\(B\) metrics/mAP50-95\(B\) train/box_loss val/box_loss \
  --out artifacts/phase0/curves_20_vs_100.png \
  --csv artifacts/phase0/curves_20_vs_100.csv
```

```python
# scripts/plot_curves.py
import argparse, pandas as pd, matplotlib.pyplot as plt, os
p = argparse.ArgumentParser()
p.add_argument("--runs", nargs="+"); p.add_argument("--cols", nargs="+")
p.add_argument("--out"); p.add_argument("--csv")
a = p.parse_args()
fig, axes = plt.subplots(1, len(a.cols), figsize=(5*len(a.cols), 4))
frames = []
for r in a.runs:
    df = pd.read_csv(os.path.join(r, "results.csv"))
    df.columns = [c.strip() for c in df.columns]
    df["run"] = os.path.basename(r); frames.append(df)
    for ax, c in zip(axes, a.cols):
        if c in df: ax.plot(df["epoch"], df[c], label=os.path.basename(r))
for ax, c in zip(axes, a.cols):
    ax.set_title(c); ax.set_xlabel("epoch"); ax.grid(alpha=.3); ax.legend()
plt.tight_layout(); plt.savefig(a.out, dpi=200)
pd.concat(frames).to_csv(a.csv, index=False)
for r in a.runs:
    df = pd.read_csv(os.path.join(r, "results.csv")); df.columns=[c.strip() for c in df.columns]
    m = "metrics/mAP50(B)"
    print(f"{r}: max={df[m].max():.4f} @epoch {int(df[m].idxmax())+1}; last={df[m].iloc[-1]:.4f}")
```

**Точка контроля / дерево решений:**

| Наблюдение | Вывод | Действие |
|---|---|---|
| `max` у 100-эпочного прогона ≥ 0,72 | метрики брались из `last.pt` | пересобрать все старые таблицы по `max`; «парадокс» снимается |
| `max` < 0,72 и `val/box_loss` растёт при падающем `train/box_loss` | переобучение | сохранить как гипотезу, проверить в Ф3.2 |
| `max` < 0,72, обе кривые падают | причина в режиме обучения | основная гипотеза — доля эпох без мозаики, проверяется в Ф3.1 |

Записать вывод в `artifacts/phase0/NOTES.md` одной строкой.

### Ф0.3 — E1: извлечение γ из существующих чекпойнтов

**Цель:** ответить, работает ли блок вообще. **Блокирующая проверка для всего проекта.**
**Артефакт:** `artifacts/phase0/gamma_existing.csv`.

```bash
python scripts/dump_gamma.py --ckpt runs/*/weights/best.pt --out artifacts/phase0/gamma_existing.csv
```

```python
# scripts/dump_gamma.py
import argparse, glob, torch, csv
p = argparse.ArgumentParser(); p.add_argument("--ckpt", nargs="+"); p.add_argument("--out")
a = p.parse_args()
rows = []
for pat in a.ckpt:
    for f in sorted(glob.glob(pat)):
        ck = torch.load(f, map_location="cpu", weights_only=False)
        sd = (ck.get("model") or ck).state_dict() if hasattr(ck.get("model", None), "state_dict") else ck["model"]
        for k, v in sd.items():
            if k.endswith("gamma"):
                rows.append({"ckpt": f, "key": k, "gamma": float(v.reshape(-1)[0])})
                print(f"{f}\t{k}\t{float(v.reshape(-1)[0]):+.6f}")
with open(a.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["ckpt", "key", "gamma"]); w.writeheader(); w.writerows(rows)
```

**Точка контроля:**

| |γ| | Вывод | Действие |
|---|---|---|
| < 0,01 | блок практически инертен | **остановиться**; включить в план вариант `gamma0=0.01` и отдельную группу параметров с `lr×10`; все прежние приросты считать шумом |
| 0,01 – 0,1 | умеренный вклад | продолжать по плану, зафиксировать значение для С2 |
| > 0,1 | существенный вклад | продолжать по плану |
| < 0 | ветвь работает на вычитание | продолжать, отдельно отметить в С2 |

### Ф0.4 — E1: тест тождественности при γ = 0

**Цель:** доказать корректность переноса весов и нейтральность блока при инициализации.
**Артефакт:** `tests/test_identity.py` в репозитории + строка в разделе воспроизводимости всех трёх статей.

```python
# tests/test_identity.py
import torch, pytest
from ultralytics import YOLO

CFGS = ["yolov8s-rfd-p3.yaml", "yolov8s-rfd-p4.yaml",
        "yolov8s-rfd-presppf.yaml", "yolov8s-rfd-postsppf.yaml",
        "yolov8s-capctrl.yaml"]

@pytest.mark.parametrize("cfg", CFGS)
def test_identity_at_init(cfg):
    base = YOLO("weights/yolov8s.pt").model.eval()
    mod  = YOLO(cfg).load("weights/yolov8s.pt").model.eval()
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        a = base(x)[0]; b = mod(x)[0]
    assert a.shape == b.shape
    assert torch.allclose(a, b, atol=1e-4), f"{cfg}: identity broken, max|d|={(a-b).abs().max():.2e}"
```

```bash
pytest tests/test_identity.py -v | tee artifacts/phase0/identity_test.log
```

**Точка контроля:** пять тестов из пяти зелёные. Красный тест означает ошибку в индексах шеи или в ремаппинге весов — **дальше двигаться нельзя**, любой прогон даст мусор.

### Ф0.5 — E1: статистика поля освещённости и корреляция с яркостью

**Цель:** установить, что именно выучила карта L. Определяет терминологию всех трёх статей.
**Артефакт:** `artifacts/phase0/rho_LY.csv`, `artifacts/phase0/L_maps.png` → **С3, Рисунок 2**.

```bash
python scripts/corr_LY.py \
  --ckpt runs/exp_100ep/weights/best.pt \
  --images splits/val.txt --n 300 \
  --out-csv artifacts/phase0/rho_LY.csv \
  --out-fig artifacts/phase0/L_maps.png
```

```python
# scripts/corr_LY.py
import argparse, cv2, glob, numpy as np, torch, csv
from scipy.stats import pearsonr, spearmanr
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--ckpt"); p.add_argument("--images"); p.add_argument("--n", type=int, default=300)
p.add_argument("--out-csv"); p.add_argument("--out-fig")
a = p.parse_args()

model = YOLO(a.ckpt).model.eval().cuda()
rfd = [m for m in model.modules() if m.__class__.__name__ == "RFDBlock"][0]
rfd.train()                        # чтобы заполнялся _illumination_map
buf = {}
rfd.register_forward_hook(lambda m, i, o: buf.__setitem__("L", m._illumination_map))

paths = [l.strip() for l in open(a.images)][: a.n]
L_all, Y_all, examples = [], [], []
for path in paths:
    img = cv2.resize(cv2.imread(path), (640, 640))
    x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().cuda() / 255
    with torch.no_grad():
        model(x)
    L = buf["L"][0, 0].float().cpu().numpy()
    Y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    Yd = cv2.resize(Y, (L.shape[1], L.shape[0]), interpolation=cv2.INTER_AREA)
    L_all.append(L.ravel()); Y_all.append(Yd.ravel())
    if len(examples) < 3: examples.append((img, L))

L_all, Y_all = np.concatenate(L_all), np.concatenate(Y_all)
res = {"pearson": pearsonr(L_all, Y_all)[0], "spearman": spearmanr(L_all, Y_all)[0],
       "L_std": float(L_all.std()), "L_mean": float(L_all.mean()), "n_images": len(paths)}
print(res)
with open(a.out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(res)); w.writeheader(); w.writerow(res)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 4, figsize=(16, 8))
for i, (img, L) in enumerate(examples):
    ax[0, i].imshow(img[:, :, ::-1]); ax[0, i].set_title("input"); ax[0, i].axis("off")
    ax[1, i].imshow(L, cmap="jet", vmin=0, vmax=1); ax[1, i].set_title("L"); ax[1, i].axis("off")
idx = np.random.choice(len(L_all), 20000, replace=False)
ax[0, 3].scatter(Y_all[idx], L_all[idx], s=1, alpha=.15)
ax[0, 3].set_xlabel("luminance Y"); ax[0, 3].set_ylabel("L")
ax[0, 3].set_title(f"rho={res['pearson']:.3f}")
ax[1, 3].axis("off")
plt.tight_layout(); plt.savefig(a.out_fig, dpi=200)
```

**Точка контроля:**

| ρ (Пирсон) | Вывод | Что меняется в текстах |
|---|---|---|
| > 0,5 | карта действительно связана с освещённостью | термин «иллюминационная модуляция» обоснован, ρ выносится в аннотацию С3 |
| 0,2 – 0,5 | частичная связь | формулировка «коррелирует с освещённостью», без утверждений о декомпозиции |
| < 0,2 | карта — маска релевантности признаков | **термин «Retinex» убирается из всех трёх статей**, С3 переформулируется вокруг этого результата |

| std(L) | Вывод |
|---|---|
| < 0,02 | вырождение регуляризатора подтверждено, блок эквивалентен свёртке |
| ≥ 0,02 | пространственная структура сохраняется |

### Ф0.6 — B.4: статистика размеров объектов и калибровка C

**Цель:** получить значение C для Фазы 2 и границы разбиения по размерам.
**Артефакт:** `artifacts/phase0/obj_sizes.csv`, `artifacts/phase0/C_calib.txt` → **С2, текст раздела «Материалы и методы»**.

```bash
python scripts/obj_sizes.py --labels datasets/ExDark/labels/train --imgsz 640 \
  --out artifacts/phase0/obj_sizes.csv | tee artifacts/phase0/C_calib.txt
```

```python
# scripts/obj_sizes.py
import argparse, glob, numpy as np, os, csv
p = argparse.ArgumentParser(); p.add_argument("--labels"); p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--out"); a = p.parse_args()
sizes = []
for f in glob.glob(os.path.join(a.labels, "*.txt")):
    for line in open(f):
        v = line.split()
        if len(v) >= 5:
            w, h = float(v[3]), float(v[4])
            sizes.append(np.sqrt(w*h) * a.imgsz)
s = np.array(sizes)
print(f"objects: {len(s)}")
print(f"mean={s.mean():.1f}px  median={np.median(s):.1f}px  p10/p50/p90={np.percentile(s,[10,50,90]).round(1)}")
print(f"small(<32px)={100*(s<32).mean():.1f}%  medium(32-96)={100*((s>=32)&(s<96)).mean():.1f}%  large(>=96)={100*(s>=96).mean():.1f}%")
for stride, lvl in [(8,'P3'),(16,'P4'),(32,'P5')]:
    print(f"{lvl}: mean size = {s.mean()/stride:.2f} grid units  -> C_{lvl} = {s.mean()/stride:.2f}")
print(f"RECOMMENDED nwd_c (P4-anchored) = {s.mean()/16:.2f}   (current default 12.8)")
print(f"RECOMMENDED size_tau = {s.mean()/16/2:.2f}")
with open(a.out, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["size_px"]); w.writerows([[x] for x in s])
```

**Выход этого скрипта — вход для Фазы 2.** Записать полученное `nwd_c` в `artifacts/phase0/C_calib.txt` и далее подставлять его в запуски E7′.

Дополнительно распечатать фактический масштаб координат внутри лосса (проверка предположения о нормировке страйдом):

```python
# временная вставка в v8DetectionLoss.__call__, один прогон на 20 итераций
if self.step % 20 == 0 and fg_mask.sum() > 0:
    tb = target_bboxes[fg_mask]
    print(f"[SCALE] target wh mean={(tb[:,2:]-tb[:,:2]).mean():.2f} units, max={(tb[:,2:]-tb[:,:2]).max():.2f}")
```

### Ф0.7 — B.1: базовый замер сложности и пропускной способности

**Цель:** таблица эффективности. Делается сразу, чтобы не возвращаться.
**Артефакт:** `artifacts/phase0/bench.csv` → **С3, Таблица 2**; колонки параметров → **С1, Таблица 1**.

```bash
python scripts/bench.py --models yolov8s.yaml yolov8s-rfd-p3.yaml yolov8s-rfd-p4.yaml \
    yolov8s-rfd-presppf.yaml yolov8s-rfd-postsppf.yaml yolov8s-capctrl.yaml \
    --batches 1 16 --half 0 1 --out artifacts/phase0/bench.csv
```

```python
# scripts/bench.py
import argparse, time, torch, numpy as np, csv
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops
p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+"); p.add_argument("--batches", nargs="+", type=int, default=[1])
p.add_argument("--half", nargs="+", type=int, default=[0]); p.add_argument("--out")
p.add_argument("--warmup", type=int, default=50); p.add_argument("--iters", type=int, default=300)
a = p.parse_args()
rows = []
for cfg in a.models:
    base = YOLO(cfg).model
    npar = sum(q.numel() for q in base.parameters()); gflops = get_flops(base, imgsz=640)
    for bs in a.batches:
        for hf in a.half:
            m = YOLO(cfg).model.eval().cuda()
            if hf: m = m.half()
            x = torch.randn(bs, 3, 640, 640, device="cuda", dtype=torch.half if hf else torch.float)
            with torch.no_grad():
                for _ in range(a.warmup): m(x)
                torch.cuda.synchronize(); t = []
                for _ in range(a.iters):
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    m(x); torch.cuda.synchronize(); t.append(time.perf_counter()-t0)
            t = np.array(t)*1000
            row = {"model": cfg, "params_M": round(npar/1e6,3), "gflops": round(gflops,2),
                   "batch": bs, "half": hf, "lat_med_ms": round(float(np.median(t)),2),
                   "lat_p95_ms": round(float(np.percentile(t,95)),2),
                   "fps": round(1000*bs/float(np.median(t)),1)}
            print(row); rows.append(row)
            del m; torch.cuda.empty_cache()
with open(a.out,"w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
```

**Точка контроля:** ГFLOPs у `p3`, `p4`, `presppf`, `postsppf` должны совпасть с точностью до 0,05. Различие означает ошибку в конфигурации.

### Ф0.8 — приведение репозитория в порядок

Не вычислительная, но блокирующая задача.

```bash
# 1. заменить таблицу результатов в README на фактические данные либо пометить как пример формата
# 2. синхронизировать гиперпараметры в README: lambda_tv=0.001, nwd_alpha=0.2
# 3. исправить формулу выхода блока: y = x + gamma*S (а не F + gamma*S)
# 4. убрать блок цитирования со статусом "Under Review"
# 5. исправить плейсхолдер your-org в команде клонирования
# 6. заменить баннер Ultralytics на собственную схему
git add README.md && git commit -m "docs: align README with actual experimental results"
```

### Чек-лист выхода из Фазы 0

- [ ] `artifacts/phase0/env.txt`, `requirements.lock`, `commit.txt`, `splits.md5`
- [ ] `args_diff.txt` — прогоны признаны сравнимыми или несравнимыми, вывод записан
- [ ] `curves_20_vs_100.png` + вывод по best/last
- [ ] `gamma_existing.csv` — блок признан активным или инертным
- [ ] `identity_test.log` — 5/5 зелёных
- [ ] `rho_LY.csv` + `L_maps.png` — принято решение по терминологии
- [ ] `obj_sizes.csv` + `C_calib.txt` — получено значение `nwd_c` и `size_tau`
- [ ] `bench.csv` — ГFLOPs совпадают между позициями
- [ ] README приведён в соответствие
- [ ] Семь патчей P0-1…P0-7 внесены и закоммичены

---

## ФАЗА 1 — данные для статьи С1 (10 запусков, ≈ 20 GPU-часов)

Собирает **Таблицу 1 и Рисунок 1 тезисов**. После Фазы 1 текст С1 пишется без возврата к вычислениям.

### Ф1.1 — E3′ + E2′: матрица запусков

Все запуски — 100 эпох, штатная функция потерь (`use_rsl` и `use_nwd` выключены). **Варьируется только конфигурация модели и сид.**

| № | EXP_ID | CONFIG | Модель | Сид | Назначение |
|---:|---|---|---|---:|---|
| 1 | e2p | baseline | yolov8s.yaml | 0 | база, строка 1 Таблицы 1 |
| 2 | e2p | baseline | yolov8s.yaml | 1 | оценка дисперсии |
| 3 | e2p | baseline | yolov8s.yaml | 2 | оценка дисперсии |
| 4 | e3p | pos-p3 | yolov8s-rfd-p3.yaml | 0 | TRF 95 px |
| 5 | e3p | pos-p4 | yolov8s-rfd-p4.yaml | 0 | TRF 239 px |
| 6 | e3p | pos-presppf | yolov8s-rfd-presppf.yaml | 0 | TRF 399 px |
| 7 | e3p | pos-postsppf | yolov8s-rfd-postsppf.yaml | 0 | TRF 783 px |
| 8 | e2p | pos-postsppf | yolov8s-rfd-postsppf.yaml | 1 | дисперсия ключевой конфигурации |
| 9 | e2p | pos-postsppf | yolov8s-rfd-postsppf.yaml | 2 | дисперсия ключевой конфигурации |
| 10 | e3p | cap-ctrl | yolov8s-capctrl.yaml | 0 | контроль ёмкости |

### Ф1.2 — скрипт запуска

```bash
# scripts/run_phase1.sh
set -e
cd $PRJ

run () {  # $1=EXP_ID  $2=CONFIG  $3=MODEL  $4=SEED
  NAME="$1__$2__s$4"
  if [ -d "runs/$NAME" ]; then echo "SKIP $NAME"; return; fi
  echo "=== $NAME ==="
  python scripts/train.py --model "$3" --name "$NAME" --seed "$4" \
      --epochs 100 --close_mosaic 10 2>&1 | tee "logs/$NAME.log"
  python scripts/postrun_check.py --run "runs/$NAME"
}

run e2p baseline      yolov8s.yaml                 0
run e2p baseline      yolov8s.yaml                 1
run e2p baseline      yolov8s.yaml                 2
run e3p pos-p3        yolov8s-rfd-p3.yaml          0
run e3p pos-p4        yolov8s-rfd-p4.yaml          0
run e3p pos-presppf   yolov8s-rfd-presppf.yaml     0
run e3p pos-postsppf  yolov8s-rfd-postsppf.yaml    0
run e2p pos-postsppf  yolov8s-rfd-postsppf.yaml    1
run e2p pos-postsppf  yolov8s-rfd-postsppf.yaml    2
run e3p cap-ctrl      yolov8s-capctrl.yaml         0
```

```bash
nohup bash scripts/run_phase1.sh > logs/phase1.out 2>&1 &
tail -f logs/phase1.out
```

### Ф1.3 — Sanity checks

**Перед запуском серии** (один раз):

```bash
pytest tests/test_identity.py -v          # 5/5 зелёных
grep -c "" splits/train.txt               # 5896
nvidia-smi --query-gpu=memory.free --format=csv   # достаточно памяти под batch=16
```

**В первые 3 минуты каждого прогона** — проверить в `logs/$NAME.log`:

```bash
grep "transferred" logs/$NAME.log     # ожидается доля > 0.90
grep "weight remap" logs/$NAME.log    # для RFD-конфигураций: shift +1 for idx >= N
```

Соответствие `shift_from` конфигурации:

| CONFIG | Ожидаемая строка |
|---|---|
| pos-p3 | `shift +1 for idx >= 5` |
| pos-p4 | `shift +1 for idx >= 7` |
| pos-presppf | `shift +1 for idx >= 9` |
| pos-postsppf / cap-ctrl | `shift +1 for idx >= 10` |
| baseline | строки нет |

Несовпадение — **немедленно прервать прогон**, ошибка в P0-1.

**На эпохе 5** — автоматическая проверка вменяемости:

```bash
python - << 'EOF'
import pandas as pd, sys, glob
for f in glob.glob("runs/*/results.csv"):
    d = pd.read_csv(f); d.columns=[c.strip() for c in d.columns]
    if len(d) >= 5:
        m = d["metrics/mAP50(B)"].iloc[4]
        flag = "OK" if m > 0.30 else "!! CHECK"
        print(f"{flag} {f}: mAP50@ep5 = {m:.4f}")
EOF
```

mAP50 ниже 0,30 на пятой эпохе при инициализации весами COCO означает, что перенос весов не сработал.

**После каждого прогона** — `scripts/postrun_check.py`:

```python
# scripts/postrun_check.py
import argparse, os, pandas as pd, torch, json, sys
p = argparse.ArgumentParser(); p.add_argument("--run"); a = p.parse_args()
r = a.run
d = pd.read_csv(os.path.join(r, "results.csv")); d.columns = [c.strip() for c in d.columns]
m50, m5095 = "metrics/mAP50(B)", "metrics/mAP50-95(B)"
best_i = d[m5095].idxmax()
out = {
    "run": os.path.basename(r), "epochs": int(d["epoch"].max()),
    "best_epoch": int(d["epoch"].iloc[best_i]),
    "best_mAP50": float(d[m50].iloc[best_i]),
    "best_mAP50_95": float(d[m5095].iloc[best_i]),
    "last_mAP50": float(d[m50].iloc[-1]),
    "last_mAP50_95": float(d[m5095].iloc[-1]),
    "train_box_first": float(d["train/box_loss"].iloc[0]),
    "train_box_last": float(d["train/box_loss"].iloc[-1]),
    "val_box_min": float(d["val/box_loss"].min()),
    "val_box_last": float(d["val/box_loss"].iloc[-1]),
}
ck = os.path.join(r, "weights", "best.pt")
if os.path.exists(ck):
    sd = torch.load(ck, map_location="cpu", weights_only=False)["model"].state_dict()
    g = [float(v.reshape(-1)[0]) for k, v in sd.items() if k.endswith("gamma")]
    out["gamma"] = g[0] if g else None
rl = os.path.join(r, "rfd_log.csv")
if os.path.exists(rl):
    rd = pd.read_csv(rl)
    out["L_std_final"] = float(rd["L_std_spatial"].iloc[-1]) if "L_std_spatial" in rd else None
    out["L_sat_final"] = float(rd["L_sat_frac"].iloc[-1]) if "L_sat_frac" in rd else None
assert out["train_box_last"] < out["train_box_first"], "train loss did not decrease"
assert out["best_mAP50"] > 0.50, f"suspiciously low mAP50: {out['best_mAP50']}"
json.dump(out, open(os.path.join(r, "summary.json"), "w"), indent=2, ensure_ascii=False)
print(json.dumps(out, indent=2, ensure_ascii=False))
```

### Ф1.4 — сбор данных под Таблицу 1 статьи С1

```bash
python scripts/collect_c1.py --out tables/C1_table1.csv | tee tables/C1_table1.txt
```

```python
# scripts/collect_c1.py
import glob, json, os, argparse, numpy as np, pandas as pd
TRF = {"baseline": (None, None, 0.0),
       "pos-p3":       (95,  14.8, 0.164),
       "pos-p4":       (239, 37.3, 0.657),
       "pos-presppf":  (399, 62.3, 2.624),
       "pos-postsppf": (783, 122.3, 2.624),
       "cap-ctrl":     (399, 62.3, 2.623)}
ORDER = ["baseline", "pos-p3", "pos-p4", "pos-presppf", "pos-postsppf", "cap-ctrl"]
p = argparse.ArgumentParser(); p.add_argument("--out"); a = p.parse_args()
rows = []
for cfg in ORDER:
    js = [json.load(open(f)) for f in glob.glob(f"runs/*__{cfg}__s*/summary.json")]
    if not js: continue
    m50 = np.array([j["best_mAP50"] for j in js]); m95 = np.array([j["best_mAP50_95"] for j in js])
    g   = [j.get("gamma") for j in js if j.get("gamma") is not None]
    trf, cov, dpar = TRF[cfg]
    rows.append({"config": cfg, "n_runs": len(js),
        "TRF_px": trf, "coverage_pct": cov, "delta_params_M": dpar,
        "mAP50_mean": round(m50.mean(), 4), "mAP50_std": round(m50.std(ddof=1), 4) if len(js) > 1 else None,
        "mAP50_95_mean": round(m95.mean(), 4), "mAP50_95_std": round(m95.std(ddof=1), 4) if len(js) > 1 else None,
        "gamma_mean": round(float(np.mean(g)), 4) if g else None})
df = pd.DataFrame(rows); df.to_csv(a.out, index=False); print(df.to_string(index=False))
base = df[df.config == "baseline"]
if len(base):
    b = base.mAP50_mean.iloc[0]
    print("\nΔ к базе, п.п.:")
    for _, r in df.iterrows():
        print(f"  {r.config:15s} {100*(r.mAP50_mean-b):+.2f}")
```

**Важное правило отчётности:** разброс измерен только для `baseline` и `pos-postsppf` (по 3 сида). Для остальных позиций в статье указывается одно значение, а в тексте пишется, что разброс оценён по двум конфигурациям и составляет столько-то. Выдавать одиночные прогоны за среднее нельзя.

### Ф1.5 — Рисунок 1 статьи С1

Схема backbone с четырьмя точками-кандидатами и значениями TRF. Векторная графика, готовится вручную (draw.io / Inkscape / TikZ), данные берутся из `tables/C1_table1.csv`. Рисунок **не должен** содержать членов функции потерь — иначе он совпадёт с Рисунком 1 статьи С3.

Экспорт: `figs/C1_fig1_architecture.svg` и `.png` (300 dpi, ширина ≥ 1800 px).

### Ф1.6 — B.3: измерение ERF (резерв, не публикуется)

```bash
python scripts/measure_erf.py --ckpt runs/e3p__pos-postsppf__s0/weights/best.pt \
   --layers 4 6 8 9 10 --out artifacts/phase1/erf.png --csv artifacts/phase1/erf.csv
```

```python
# scripts/measure_erf.py
import argparse, torch, numpy as np, matplotlib.pyplot as plt, csv
from ultralytics import YOLO
p = argparse.ArgumentParser()
p.add_argument("--ckpt"); p.add_argument("--layers", nargs="+", type=int)
p.add_argument("--imgsz", type=int, default=640); p.add_argument("--out"); p.add_argument("--csv")
a = p.parse_args()
model = YOLO(a.ckpt).model.eval()
rows = []
fig, axes = plt.subplots(1, len(a.layers), figsize=(4*len(a.layers), 4))
for ax, li in zip(np.atleast_1d(axes), a.layers):
    acts = {}
    h = model.model[li].register_forward_hook(lambda m, i, o: acts.__setitem__("o", o))
    x = torch.randn(1, 3, a.imgsz, a.imgsz, requires_grad=True)
    model(x); o = acts["o"]
    o[0, o.shape[1]//2, o.shape[2]//2, o.shape[3]//2].backward()
    erf = x.grad.abs().sum(1)[0].numpy(); h.remove()
    nz = np.argwhere(erf > 0.01*erf.max())
    hgt = nz[:,0].max()-nz[:,0].min()+1; wid = nz[:,1].max()-nz[:,1].min()+1
    rows.append({"layer": li, "erf_h_px": int(hgt), "erf_w_px": int(wid)})
    ax.imshow(erf, cmap="inferno"); ax.set_title(f"layer {li}: {wid}x{hgt}px"); ax.axis("off")
    model.zero_grad(set_to_none=True)
plt.tight_layout(); plt.savefig(a.out, dpi=200)
with open(a.csv,"w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=["layer","erf_h_px","erf_w_px"]); w.writeheader(); w.writerows(rows)
print(rows)
```

Результат кладётся в `artifacts/phase1/` и в статью **не включается** — используется только при ответе рецензенту на замечание «TRF — грубая оценка».

### Чек-лист выхода из Фазы 1

- [ ] 10 каталогов в `runs/`, в каждом `summary.json`
- [ ] `tables/C1_table1.csv` заполнена, шесть строк
- [ ] Проверено: ГFLOPs одинаковы у всех позиций (из `bench.csv`)
- [ ] `figs/C1_fig1_architecture.svg` готов
- [ ] `artifacts/phase1/erf.csv` — резерв
- [ ] Записан вывод: подтверждает ли `cap-ctrl` вклад позиции или объясняет эффект ёмкостью
- [ ] **Можно писать текст С1**

---

## ФАЗА 2 — данные для статьи С2 (7 запусков, ≈ 14 GPU-часов)

Собирает **Таблицы 1–2 и Рисунки 1–4 статьи РИНЦ**.

### Ф2.1 — E6′: сетка веса регуляризатора

Все запуски: модель `yolov8s-rfd-postsppf.yaml`, 100 эпох, сид 0, `use_rsl` включён, `use_nwd` выключен.

| № | CONFIG | lambda_tv |
|---:|---|---|
| 11 | lam-1e-4 | 0.0001 |
| 12 | lam-1e-3 | 0.001 |
| 13 | lam-1e-2 | 0.01 |
| 14 | lam-1e-1 | 0.1 |

```bash
# scripts/run_phase2a.sh
set -e; cd $PRJ
for LAM in 0.0001 0.001 0.01 0.1; do
  TAG=$(python -c "print(f'lam-{$LAM:.0e}'.replace('e-0','e-'))")
  NAME="e6p__${TAG}__s0"
  [ -d "runs/$NAME" ] && { echo "SKIP $NAME"; continue; }
  python scripts/train.py --model yolov8s-rfd-postsppf.yaml --name "$NAME" --seed 0 \
      --epochs 100 --close_mosaic 10 --use_rsl --lambda_tv $LAM 2>&1 | tee "logs/$NAME.log"
  python scripts/postrun_check.py --run "runs/$NAME"
done
```

**Опорная точка:** запуск `pos-postsppf s0` из Фазы 1 — это тот же прогон при `lambda_tv = 0` (регуляризатор выключен). Он добавляется в таблицу пятой строкой без нового обучения.

**Sanity check во время прогона** — соотношение членов лосса. Временно включить в `v8DetectionLoss.__call__`:

```python
if self.step % 100 == 0:
    print(f"[LOSS] box={loss[0].item():.4f} cls={loss[1].item():.4f} dfl={loss[2].item():.4f} "
          f"tv={tv.item():.6f} lam*tv={(self.hyp.lambda_tv*tv).item():.8f}")
```

Вывод сохранить в `artifacts/phase2/loss_ratio.txt` — это прямое доказательство для текста С2 того, каково реальное соотношение вкладов. Ожидаемый порядок при `lambda_tv=0.001`: `lam*tv` ≈ 1e-5…1e-4 против `box` ≈ 1.

### Ф2.2 — E7′: варианты метрики рамок

Все запуски: `yolov8s-rfd-postsppf.yaml`, 100 эпох, сид 0, `use_rsl` включён с лучшим `lambda_tv` из Ф2.1, `use_nwd` включён.

| № | CONFIG | Параметры | Что проверяет |
|---:|---|---|---|
| 15 | nwd-calib | `--nwd_mode abs --nwd_c <C из Ф0.6>` | корректная калибровка константы |
| 16 | nwd-scaleinv | `--nwd_mode scaleinv --nwd_c 0.3` | масштабно-инвариантная нормировка |
| 17 | nwd-sizegate | `--nwd_mode sizegate --nwd_c <C> --size_tau <τ из Ф0.6>` | размерно-зависимое взвешивание |

```bash
# scripts/run_phase2b.sh
set -e; cd $PRJ
C=$(grep "RECOMMENDED nwd_c" artifacts/phase0/C_calib.txt | awk '{print $(NF-3)}')
TAU=$(grep "RECOMMENDED size_tau" artifacts/phase0/C_calib.txt | awk '{print $NF}')
LAM=0.001   # подставить победителя Ф2.1
echo "using C=$C tau=$TAU lambda_tv=$LAM"

python scripts/train.py --model yolov8s-rfd-postsppf.yaml --name e7p__nwd-calib__s0 --seed 0 \
  --epochs 100 --use_rsl --lambda_tv $LAM --use_nwd --nwd_alpha 0.2 --nwd_mode abs --nwd_c $C \
  2>&1 | tee logs/e7p__nwd-calib__s0.log
python scripts/postrun_check.py --run runs/e7p__nwd-calib__s0

python scripts/train.py --model yolov8s-rfd-postsppf.yaml --name e7p__nwd-scaleinv__s0 --seed 0 \
  --epochs 100 --use_rsl --lambda_tv $LAM --use_nwd --nwd_alpha 0.2 --nwd_mode scaleinv --nwd_c 0.3 \
  2>&1 | tee logs/e7p__nwd-scaleinv__s0.log
python scripts/postrun_check.py --run runs/e7p__nwd-scaleinv__s0

python scripts/train.py --model yolov8s-rfd-postsppf.yaml --name e7p__nwd-sizegate__s0 --seed 0 \
  --epochs 100 --use_rsl --lambda_tv $LAM --use_nwd --nwd_alpha 0.2 --nwd_mode sizegate \
  --nwd_c $C --size_tau $TAU 2>&1 | tee logs/e7p__nwd-sizegate__s0.log
python scripts/postrun_check.py --run runs/e7p__nwd-sizegate__s0
```

**Опорная строка:** исходный вариант с `nwd_c=12.8` — это имеющийся прогон «стадия 4/5» из старой серии. Если он выполнялся до внесения патчей P0-3…P0-5, его **необходимо перезапустить**, иначе сравнение некорректно. В этом случае добавляется запуск №18 (`e7p__nwd-orig__s0`), и Фаза 2 становится 8 запусками.

### Ф2.3 — переоценка по размерам объектов (0 запусков)

Выполняется на готовых чекпойнтах, обучение не требуется.

```bash
python scripts/eval_by_size.py \
  --ckpts runs/e6p__lam-1e-3__s0/weights/best.pt \
          runs/e7p__nwd-calib__s0/weights/best.pt \
          runs/e7p__nwd-scaleinv__s0/weights/best.pt \
          runs/e7p__nwd-sizegate__s0/weights/best.pt \
  --out tables/C2_table2.csv
```

```python
# scripts/eval_by_size.py
import argparse, os, json, csv
from ultralytics import YOLO
p = argparse.ArgumentParser(); p.add_argument("--ckpts", nargs="+"); p.add_argument("--out")
a = p.parse_args()
rows = []
for ck in a.ckpts:
    m = YOLO(ck)
    r = m.val(data="data/exdark.yaml", imgsz=640, batch=16, device=0, split="val", plots=False)
    d = r.results_dict
    rows.append({
        "ckpt": os.path.basename(os.path.dirname(os.path.dirname(ck))),
        "mAP50": round(float(d.get("metrics/mAP50(B)", 0)), 4),
        "mAP50_95": round(float(d.get("metrics/mAP50-95(B)", 0)), 4),
        # разбивка по размерам берётся из COCO-eval статистик валидатора
        "AP_small": round(float(getattr(r.box, "aps", [None]*3)[0] or 0), 4),
        "AP_medium": round(float(getattr(r.box, "aps", [None]*3)[1] or 0), 4),
        "AP_large": round(float(getattr(r.box, "aps", [None]*3)[2] or 0), 4),
    })
    print(rows[-1])
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
```

Если используемая версия валидатора не отдаёт `AP_small/medium/large` напрямую, разбивка считается через `pycocotools`: экспортировать предсказания в COCO-json (`model.val(save_json=True)`), затем `COCOeval` с порогами площади 32² и 96² — границы подтверждены статистикой из Ф0.6.

**Это несущий результат С2.** Прогноз, сделанный до эксперимента: вклад метрики положителен на `AP_small` и отрицателен на `AP_large`. Результат фиксируется независимо от того, подтвердится прогноз или нет.

### Ф2.4 — Рисунки статьи С2

**Рисунок 1 — аналитические кривые градиента. Обучение не требуется.**

```bash
python scripts/fig_gradients.py --C 12.8 --C-calib $C --out figs/C2_fig1_gradients.png
```

```python
# scripts/fig_gradients.py
import argparse, numpy as np, matplotlib.pyplot as plt
p = argparse.ArgumentParser(); p.add_argument("--C", type=float, default=12.8)
p.add_argument("--C-calib", type=float); p.add_argument("--out"); a = p.parse_args()
A = np.linspace(0.5, 40, 400); d = 0.1*A
g_iou = 2*A/((A+d)**2)
plt.figure(figsize=(6,4))
plt.plot(A, g_iou, label="|d(IoU)/dδ|")
for C, st in [(a.C, "--"), (a.C_calib, "-.")]:
    plt.plot(A, (1/C)*np.exp(-d/C), st, label=f"|d(NWD)/dδ|, C={C:.1f}")
plt.xlabel("object size, grid units"); plt.ylabel("localisation gradient magnitude")
plt.yscale("log"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
plt.savefig(a.out, dpi=300)
```

**Рисунок 2 — траектории γ и std(L).**

```bash
python scripts/fig_traj.py --runs runs/e6p__lam-*__s0 --out figs/C2_fig2_trajectories.png
```

```python
# scripts/fig_traj.py
import argparse, glob, os, pandas as pd, matplotlib.pyplot as plt
p = argparse.ArgumentParser(); p.add_argument("--runs", nargs="+"); p.add_argument("--out")
a = p.parse_args()
runs = sorted(sum([glob.glob(r) for r in a.runs], []))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for r in runs:
    f = os.path.join(r, "rfd_log.csv")
    if not os.path.exists(f): continue
    d = pd.read_csv(f); lab = os.path.basename(r).split("__")[1]
    ax[0].plot(d["epoch"], d["gamma"].abs(), label=lab)
    if "L_std_spatial" in d: ax[1].plot(d["epoch"], d["L_std_spatial"], label=lab)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("|gamma|"); ax[0].set_yscale("log")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("spatial std of L")
ax[1].axhline(0.02, ls=":", c="r")      # порог вырождения
for x in ax: x.grid(alpha=.3); x.legend()
plt.tight_layout(); plt.savefig(a.out, dpi=300)
```

**Рисунок 3 — карты L при слабой и сильной регуляризации.**

```bash
for R in e6p__lam-1e-4__s0 e6p__lam-1e-1__s0; do
  python scripts/corr_LY.py --ckpt runs/$R/weights/best.pt --images splits/val.txt --n 100 \
     --out-csv artifacts/phase2/rho_$R.csv --out-fig artifacts/phase2/Lmaps_$R.png
done
# собрать две панели в один рисунок 2x3
python scripts/merge_panels.py artifacts/phase2/Lmaps_*.png figs/C2_fig3_Lmaps.png
```

**Рисунок 4 — чувствительность качества к весу регуляризатора.**

```bash
python scripts/fig_sensitivity.py --pattern "runs/e6p__lam-*__s0" \
   --baseline runs/e3p__pos-postsppf__s0 --out figs/C2_fig4_sensitivity.png
```

### Ф2.5 — сбор Таблицы 1 статьи С2

```bash
python scripts/collect_c2.py --out tables/C2_table1.csv | tee tables/C2_table1.txt
```

```python
# scripts/collect_c2.py
import glob, json, os, re, argparse, pandas as pd
p = argparse.ArgumentParser(); p.add_argument("--out"); a = p.parse_args()
rows = []
runs = sorted(glob.glob("runs/e6p__lam-*__s0")) + ["runs/e3p__pos-postsppf__s0"]
for r in runs:
    sj = os.path.join(r, "summary.json")
    if not os.path.exists(sj): continue
    s = json.load(open(sj))
    lam = 0.0 if "pos-postsppf" in r else float(re.search(r"lam-([\d.e-]+)", r).group(1))
    rho = None
    rf = f"artifacts/phase2/rho_{os.path.basename(r)}.csv"
    if os.path.exists(rf): rho = float(pd.read_csv(rf)["pearson"].iloc[0])
    rows.append({"lambda_tv": lam, "gamma": s.get("gamma"),
                 "L_std_final": s.get("L_std_final"), "L_sat_final": s.get("L_sat_final"),
                 "rho_LY": rho, "mAP50": s["best_mAP50"], "mAP50_95": s["best_mAP50_95"]})
df = pd.DataFrame(rows).sort_values("lambda_tv")
df.to_csv(a.out, index=False); print(df.to_string(index=False))
```

Для строк таблицы, где `rho_LY` пуст, прогнать `corr_LY.py` по соответствующему чекпойнту — это 0 GPU-часов.

### Чек-лист выхода из Фазы 2

- [ ] 7 (или 8) новых каталогов в `runs/` с `summary.json`
- [ ] `tables/C2_table1.csv` — пять строк по `lambda_tv`, все колонки заполнены
- [ ] `tables/C2_table2.csv` — четыре строки вариантов метрики с разбивкой по размерам
- [ ] `figs/C2_fig1_gradients.png` … `C2_fig4_sensitivity.png`
- [ ] `artifacts/phase2/loss_ratio.txt` — зафиксировано соотношение членов лосса
- [ ] Записан вывод: подтвердился ли прогноз о разнонаправленном влиянии метрики по размерам
- [ ] Записан вывод: наступает ли вырождение и при каком `lambda_tv`
- [ ] **Можно писать текст С2**

---

## ФАЗА 3 — данные для статьи С3 (6 запусков, ≈ 10 GPU-часов)

Собирает **Таблицы 1–2 и Рисунки 1–3 статьи IEEE**. Часть данных уже получена в Фазах 0–2 и повторно не считается.

### Ф3.1 — E4′: мозаичная аугментация

Модель `yolov8s-rfd-postsppf.yaml`, 100 эпох, сид 0, штатная функция потерь.

| № | CONFIG | close_mosaic | Доля эпох без мозаики |
|---:|---|---:|---:|
| 18 | mosaic-close0 | 0 | 0 % |
| 19 | mosaic-close50 | 50 | 50 % |
| — | (есть из Фазы 1) | 10 | 10 % |

```bash
set -e; cd $PRJ
for CM in 0 50; do
  NAME="e4p__mosaic-close${CM}__s0"
  [ -d "runs/$NAME" ] && { echo "SKIP $NAME"; continue; }
  python scripts/train.py --model yolov8s-rfd-postsppf.yaml --name "$NAME" --seed 0 \
      --epochs 100 --close_mosaic $CM 2>&1 | tee "logs/$NAME.log"
  python scripts/postrun_check.py --run "runs/$NAME"
done
```

**Sanity check:** в логе прогона с `close_mosaic=50` на эпохе 51 должна появиться строка `Closing dataloader mosaic`. Для `close_mosaic=0` этой строки быть не должно.

```bash
grep -n "Closing dataloader mosaic" logs/e4p__mosaic-close50__s0.log
grep -c "Closing dataloader mosaic" logs/e4p__mosaic-close0__s0.log   # ожидается 0
```

### Ф3.2 — E5′: длительность обучения

Две конфигурации × две длительности. `close_mosaic=10` во всех.

| № | EXP_ID | CONFIG | Модель | Эпох |
|---:|---|---|---|---:|
| 20 | e5p | baseline-ep40 | yolov8s.yaml | 40 |
| 21 | e5p | baseline-ep60 | yolov8s.yaml | 60 |
| 22 | e5p | postsppf-ep40 | yolov8s-rfd-postsppf.yaml | 40 |
| 23 | e5p | postsppf-ep60 | yolov8s-rfd-postsppf.yaml | 60 |

```bash
set -e; cd $PRJ
run5 () {  # $1=CONFIG $2=MODEL $3=EPOCHS
  NAME="e5p__$1__s0"
  [ -d "runs/$NAME" ] && { echo "SKIP $NAME"; return; }
  python scripts/train.py --model "$2" --name "$NAME" --seed 0 \
      --epochs "$3" --close_mosaic 10 2>&1 | tee "logs/$NAME.log"
  python scripts/postrun_check.py --run "runs/$NAME"
}
run5 baseline-ep40  yolov8s.yaml              40
run5 baseline-ep60  yolov8s.yaml              60
run5 postsppf-ep40  yolov8s-rfd-postsppf.yaml 40
run5 postsppf-ep60  yolov8s-rfd-postsppf.yaml 60
```

Точки на 20 и 100 эпохах берутся из старых прогонов (если Ф0.1 признала их сравнимыми) либо из Фазы 1 (точка 100). Точка 20 при несравнимости старых прогонов требует двух дополнительных запусков — заложить их как резерв.

**Отдельно зафиксировать долю эпох без мозаики для каждой точки:** при `close_mosaic=10` это 50 %, 25 %, 17 % и 10 % для 20, 40, 60 и 100 эпох соответственно. Это ключ к интерпретации Рисунка 3 — длительность и доля чистых эпох в штатной конфигурации связаны, и разделяет их именно серия Ф3.1.

### Ф3.3 — Рисунок 3 статьи С3

Две панели: (а) качество как функция длительности обучения; (б) качество как функция доли эпох без мозаики при фиксированной длительности 100 эпох.

```bash
python scripts/fig_regime.py \
  --epochs-runs "runs/e5p__*__s0" "runs/e2p__baseline__s0" "runs/e3p__pos-postsppf__s0" \
  --mosaic-runs "runs/e4p__mosaic-*__s0" "runs/e3p__pos-postsppf__s0" \
  --out figs/C3_fig3_regime.png --csv tables/C3_regime.csv
```

```python
# scripts/fig_regime.py
import argparse, glob, json, os, re, pandas as pd, matplotlib.pyplot as plt
p = argparse.ArgumentParser()
p.add_argument("--epochs-runs", nargs="+"); p.add_argument("--mosaic-runs", nargs="+")
p.add_argument("--out"); p.add_argument("--csv"); a = p.parse_args()

def load(pats):
    rows = []
    for pat in pats:
        for r in glob.glob(pat):
            sj = os.path.join(r, "summary.json")
            if os.path.exists(sj):
                s = json.load(open(sj)); s["run"] = os.path.basename(r); rows.append(s)
    return pd.DataFrame(rows)

ep = load(a.epochs_runs); mo = load(a.mosaic_runs)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for fam, mark in [("baseline", "o"), ("postsppf", "s")]:
    d = ep[ep.run.str.contains(fam)].sort_values("epochs")
    ax[0].plot(d.epochs, d.best_mAP50, marker=mark, label=fam)
ax[0].set_xlabel("training epochs"); ax[0].set_ylabel("mAP@0.5"); ax[0].grid(alpha=.3); ax[0].legend()

def frac(run, total=100):
    m = re.search(r"close(\d+)", run)
    return 100.0*int(m.group(1))/total if m else 10.0
mo["clean_pct"] = mo.run.map(frac)
mo = mo.sort_values("clean_pct")
ax[1].plot(mo.clean_pct, mo.best_mAP50, marker="^", color="tab:green")
ax[1].set_xlabel("epochs without mosaic, % of schedule"); ax[1].set_ylabel("mAP@0.5"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(a.out, dpi=300)
pd.concat([ep.assign(panel="epochs"), mo.assign(panel="mosaic")]).to_csv(a.csv, index=False)
```

### Ф3.4 — Таблица 1 статьи С3: аблация компонентов

Строится из уже имеющихся прогонов, новых вычислений не требует.

| Строка таблицы | Источник данных |
|---|---|
| Базовая модель | `runs/e2p__baseline__s{0,1,2}` → среднее ± СКО |
| + модуль модуляции | `runs/e3p__pos-postsppf__s0` + `runs/e2p__pos-postsppf__s{1,2}` → среднее ± СКО |
| + регуляризатор | `runs/e6p__lam-<лучший>__s0` |
| + метрика рамок | `runs/e7p__nwd-<лучший>__s0` |
| Контроль ёмкости | `runs/e3p__cap-ctrl__s0` |

```bash
python scripts/collect_c3.py --out tables/C3_table1.csv | tee tables/C3_table1.txt
```

```python
# scripts/collect_c3.py
import glob, json, os, argparse, numpy as np, pandas as pd
SPEC = [
    ("Baseline",              ["runs/e2p__baseline__s*"]),
    ("+ modulation module",   ["runs/e3p__pos-postsppf__s0", "runs/e2p__pos-postsppf__s*"]),
    ("+ smoothness reg.",     ["runs/e6p__lam-1e-3__s0"]),
    ("+ box metric",          ["runs/e7p__nwd-sizegate__s0"]),
    ("Capacity control",      ["runs/e3p__cap-ctrl__s0"]),
]
p = argparse.ArgumentParser(); p.add_argument("--out"); a = p.parse_args()
rows = []
for label, pats in SPEC:
    files = sorted({f for pat in pats for f in glob.glob(os.path.join(pat, "summary.json"))})
    js = [json.load(open(f)) for f in files]
    if not js: print(f"MISSING: {label}"); continue
    m50 = np.array([j["best_mAP50"] for j in js]); m95 = np.array([j["best_mAP50_95"] for j in js])
    ci = 1.96*m50.std(ddof=1)/np.sqrt(len(js)) if len(js) > 2 else None
    rows.append({"config": label, "n": len(js),
                 "mAP50": round(m50.mean(),4), "mAP50_sd": round(m50.std(ddof=1),4) if len(js)>1 else None,
                 "mAP50_ci95": round(ci,4) if ci else None,
                 "mAP50_95": round(m95.mean(),4),
                 "gamma": round(float(np.mean([j["gamma"] for j in js if j.get("gamma")])),4) if any(j.get("gamma") for j in js) else None})
df = pd.DataFrame(rows); df.to_csv(a.out, index=False); print(df.to_string(index=False))
b = df.mAP50.iloc[0]
print("\nΔ к базе, п.п.:")
for _, r in df.iterrows(): print(f"  {r.config:22s} {100*(r.mAP50-b):+.2f}")
```

**Правило интерпретации для текста С3:** если доверительный интервал строки накрывает значение базовой модели, в таблице ставится пометка «нз» (незначимо), а в тексте пишется «сопоставимо с базовой моделью». Это не слабость работы, а один из её заявленных результатов.

### Ф3.5 — Таблица 2 статьи С3: эффективность

Данные уже собраны в Ф0.7 (`artifacts/phase0/bench.csv`). Требуется только форматирование:

```bash
python - << 'EOF'
import pandas as pd
d = pd.read_csv("artifacts/phase0/bench.csv")
d = d[(d.batch==1) & (d.half==0)][["model","params_M","gflops","lat_med_ms","fps"]]
d = d[d.model.isin(["yolov8s.yaml","yolov8s-rfd-postsppf.yaml","yolov8s-capctrl.yaml"])]
d.to_csv("tables/C3_table2.csv", index=False); print(d.to_string(index=False))
EOF
```

Строки сопоставления с опубликованными методами добавляются вручную, **обязательно** с колонками «базовый детектор», «разрешение», «разбиение». Собственные значения берутся только из `bench.csv`.

### Ф3.6 — Рисунок 2 статьи С3

Готов из Ф0.5 (`artifacts/phase0/L_maps.png`). Пересчитать на финальном чекпойнте лучшей конфигурации:

```bash
python scripts/corr_LY.py --ckpt runs/e6p__lam-1e-3__s0/weights/best.pt \
  --images splits/val.txt --n 500 \
  --out-csv tables/C3_rho.csv --out-fig figs/C3_fig2_illumination.png
```

Значение ρ выносится в аннотацию С3 и в раздел с результатами.

### Ф3.7 — Рисунок 1 статьи С3

Общая схема системы: тракт детектора, одна точка встраивания, члены функции потерь, путь сохранения карты освещённости. **Перерисовывается заново**, не переиспользуется из С1 — без точек-кандидатов и без значений TRF.

Экспорт: `figs/C3_fig1_system.svg` / `.png`.

### Чек-лист выхода из Фазы 3

- [ ] 6 новых каталогов в `runs/` с `summary.json`
- [ ] `tables/C3_table1.csv` — пять строк с доверительными интервалами
- [ ] `tables/C3_table2.csv` — эффективность
- [ ] `tables/C3_regime.csv` + `figs/C3_fig3_regime.png`
- [ ] `figs/C3_fig1_system.svg`, `figs/C3_fig2_illumination.png`
- [ ] `tables/C3_rho.csv` — значение ρ для аннотации
- [ ] Записан вывод: что сильнее влияет на качество — модификация архитектуры, режим обучения или разброс между сидами
- [ ] **Можно писать текст С3**

---

## 4. Сводка: 23 запуска и распределение по артефактам

| № | Прогон | Эпох | Фаза | GPU-ч | Куда идёт |
|---:|---|---:|---|---:|---|
| 1–3 | `e2p__baseline__s{0,1,2}` | 100 | 1 | 6 | С1 Т1, С3 Т1 |
| 4 | `e3p__pos-p3__s0` | 100 | 1 | 2 | С1 Т1 |
| 5 | `e3p__pos-p4__s0` | 100 | 1 | 2 | С1 Т1 |
| 6 | `e3p__pos-presppf__s0` | 100 | 1 | 2 | С1 Т1 |
| 7 | `e3p__pos-postsppf__s0` | 100 | 1 | 2 | С1 Т1, С2 (λ=0), С3 Т1 |
| 8–9 | `e2p__pos-postsppf__s{1,2}` | 100 | 1 | 4 | С1 Т1, С3 Т1 |
| 10 | `e3p__cap-ctrl__s0` | 100 | 1 | 2 | С1 Т1, С3 Т1 |
| 11–14 | `e6p__lam-{1e-4,1e-3,1e-2,1e-1}__s0` | 100 | 2 | 8 | С2 Т1, Р2–Р4 |
| 15 | `e7p__nwd-calib__s0` | 100 | 2 | 2 | С2 Т2 |
| 16 | `e7p__nwd-scaleinv__s0` | 100 | 2 | 2 | С2 Т2 |
| 17 | `e7p__nwd-sizegate__s0` | 100 | 2 | 2 | С2 Т2, С3 Т1 |
| 18 | `e4p__mosaic-close0__s0` | 100 | 3 | 2 | С3 Р3 |
| 19 | `e4p__mosaic-close50__s0` | 100 | 3 | 2 | С3 Р3 |
| 20 | `e5p__baseline-ep40__s0` | 40 | 3 | 0,8 | С3 Р3 |
| 21 | `e5p__baseline-ep60__s0` | 60 | 3 | 1,2 | С3 Р3 |
| 22 | `e5p__postsppf-ep40__s0` | 40 | 3 | 0,8 | С3 Р3 |
| 23 | `e5p__postsppf-ep60__s0` | 60 | 3 | 1,2 | С3 Р3 |
| | **Итого** | | | **≈ 44** | |

Оценка исходит из 2 GPU-часов на 100 эпох. При иной производительности пересчитать пропорционально.

Данные без обучения (0 GPU-часов): E0, E1, разбивка по размерам, ERF, корреляция с яркостью, замер пропускной способности, аналитические кривые градиента.

---

## 5. Финальная сборка и контроль

### 5.1. Единая команда сборки всех таблиц

```bash
# scripts/build_all.sh
set -e; cd $PRJ
for d in runs/*/; do
  [ -f "$d/summary.json" ] || python scripts/postrun_check.py --run "${d%/}" || true
done
python scripts/collect_c1.py --out tables/C1_table1.csv
python scripts/collect_c2.py --out tables/C2_table1.csv
python scripts/collect_c3.py --out tables/C3_table1.csv
python scripts/fig_gradients.py  --C 12.8 --C-calib "$(grep 'RECOMMENDED nwd_c' artifacts/phase0/C_calib.txt | awk '{print $(NF-3)}')" --out figs/C2_fig1_gradients.png
python scripts/fig_traj.py       --runs "runs/e6p__lam-*__s0" --out figs/C2_fig2_trajectories.png
python scripts/fig_regime.py     --epochs-runs "runs/e5p__*__s0" "runs/e2p__baseline__s0" "runs/e3p__pos-postsppf__s0" \
                                 --mosaic-runs "runs/e4p__mosaic-*__s0" "runs/e3p__pos-postsppf__s0" \
                                 --out figs/C3_fig3_regime.png --csv tables/C3_regime.csv
echo "=== BUILD OK ==="
ls -la tables/ figs/
```

### 5.2. Архив артефактов для статей

```bash
cd $PRJ
tar czf artifacts_for_papers.tar.gz \
    tables/ figs/ artifacts/ \
    $(for d in runs/*/; do echo "$d/results.csv" "$d/args.yaml" "$d/summary.json"; \
      [ -f "$d/rfd_log.csv" ] && echo "$d/rfd_log.csv"; done)
sha256sum artifacts_for_papers.tar.gz > artifacts_for_papers.sha256
```

Веса (`best.pt`) в архив не входят из-за размера; они выкладываются отдельно вместе с репозиторием.

### 5.3. Матрица «артефакт → место в статье»

| Файл | С1 | С2 | С3 |
|---|---|---|---|
| `tables/C1_table1.csv` | **Таблица 1** | — | 1 строка текстом |
| `figs/C1_fig1_architecture.svg` | **Рисунок 1** | — | — |
| `tables/C2_table1.csv` | — | **Таблица 1** | — |
| `tables/C2_table2.csv` | — | **Таблица 2** | — |
| `figs/C2_fig1_gradients.png` | — | **Рисунок 1** | — |
| `figs/C2_fig2_trajectories.png` | — | **Рисунок 2** | — |
| `figs/C2_fig3_Lmaps.png` | — | **Рисунок 3** | — |
| `figs/C2_fig4_sensitivity.png` | — | **Рисунок 4** | — |
| `tables/C3_table1.csv` | — | — | **Таблица 1** |
| `tables/C3_table2.csv` | колонка параметров | — | **Таблица 2** |
| `figs/C3_fig1_system.svg` | — | — | **Рисунок 1** |
| `figs/C3_fig2_illumination.png` | — | 1 предложение | **Рисунок 2** |
| `figs/C3_fig3_regime.png` | — | — | **Рисунок 3** |
| `artifacts/phase0/env.txt` | Материалы и методы | Материалы и методы | Setup |
| `artifacts/phase1/erf.csv` | резерв рецензенту | — | — |
| `artifacts/phase2/loss_ratio.txt` | — | текст результатов | — |

### 5.4. Сводный контроль перед написанием текстов

```bash
python - << 'EOF'
import os, glob, json, sys
need_runs = 23
runs = [d for d in glob.glob("runs/*/") if os.path.exists(os.path.join(d,"summary.json"))]
need_files = ["tables/C1_table1.csv","tables/C2_table1.csv","tables/C2_table2.csv",
              "tables/C3_table1.csv","tables/C3_table2.csv","tables/C3_regime.csv",
              "figs/C2_fig1_gradients.png","figs/C2_fig2_trajectories.png",
              "figs/C2_fig3_Lmaps.png","figs/C2_fig4_sensitivity.png",
              "figs/C3_fig2_illumination.png","figs/C3_fig3_regime.png",
              "artifacts/phase0/env.txt","artifacts/phase0/C_calib.txt",
              "artifacts/phase0/bench.csv","artifacts/phase0/rho_LY.csv",
              "artifacts/phase0/identity_test.log"]
print(f"runs with summary.json: {len(runs)}/{need_runs}")
missing = [f for f in need_files if not os.path.exists(f)]
print("MISSING:" if missing else "ALL ARTIFACTS PRESENT")
for f in missing: print("  ", f)
sys.exit(1 if missing else 0)
EOF
```

### 5.5. Типовые отказы и действия

| Симптом | Причина | Действие |
|---|---|---|
| `assert frac > 0.80` падает | неверный `shift_from` в P0-1 или ошибка индексов в `.yaml` | сверить `Concat` в шее с §0.8; прогнать `test_identity.py` |
| `test_identity` красный на одной конфигурации | ссылка `Concat` указывает не на тот слой | исправить `.yaml`, перезапустить тест |
| mAP50 на 5-й эпохе < 0,30 | веса не перенеслись | проверить строку `transferred` в логе |
| В `rfd_log.csv` пусто | буфер обнуляется раньше колбэка | перенести очистку в `on_train_batch_end` (см. §0.7) |
| `gamma` в `summary.json` равна 0.0000 | градиентный затвор | добавить группу параметров для `gamma` с `lr×10` либо `gamma0=0.01`; перезапустить затронутые прогоны |
| `L_std_spatial` < 0,02 с первых эпох | вырождение регуляризатора | это результат, а не ошибка; зафиксировать для С2 |
| ГFLOPs различаются между позициями | ошибка в масштабировании каналов в `.yaml` | проверить, что для P3 указано `[256]`, для P4 `[512]`, для P5 `[1024]` |
| Разброс между сидами > 1,0 п.п. | нестабильность обучения | увеличить число сидов до 5 для базовой модели, пересчитать интервалы |
| OOM при `batch=16` | нехватка памяти | **не менять `batch`** (это изменит эффективный вес регуляризатора); уменьшить `workers` или использовать другую карту |

### 5.6. Жёсткие правила серии

1. `batch=16` и `nbs=64` фиксированы во всех 23 запусках. Изменение `batch` меняет эффективный вес регуляризатора и делает прогоны несравнимыми.
2. Отчётная метрика — из `best.pt`, критерий отбора `fitness = 0.1·mAP50 + 0.9·mAP50-95`. Указывается в каждой статье явно.
3. Ни один прогон не перезапускается «потому что результат не понравился». Все выполненные прогоны попадают в `runs/` и в архив.
4. Любая правка кода после начала Фазы 1 требует перезапуска всех затронутых прогонов. Патчи P0-1…P0-7 вносятся до Фазы 1 и далее код заморожен.
5. `artifacts/phase0/NOTES.md` ведётся как журнал решений: каждая развилка из Фазы 0 фиксируется одной строкой с датой.
