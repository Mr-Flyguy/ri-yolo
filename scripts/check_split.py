# scripts/check_split.py
from pathlib import Path
import collections
import hashlib

ROOT = Path("datasets/ExDark")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}

def stems(d):
    p = Path(d)
    if not p.exists():
        return {}
    return {x.stem: x for x in p.iterdir() if x.suffix in IMG_EXT}

tr = stems(ROOT / "images/train")
va = stems(ROOT / "images/val")
te = stems(ROOT / "images/test")

total_found = len(tr) + len(va) + len(te)
print(f"train images: {len(tr)}   val images: {len(va)}   test images: {len(te)}   sum: {total_found}")
print(f"ExDark total should be 7363 -> diff: {total_found - 7363}")

overlap = set(tr) & set(va)
print(f"\n*** ПЕРЕСЕЧЕНИЕ train/val: {len(overlap)} изображений ***")
for s in sorted(overlap)[:20]:
    print("   ", s, "|", tr[s].name, "|", va[s].name)

def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()

print("\nВычисление MD5 и поиск дубликатов...")
tr_h = {}
for s, p in tr.items():
    tr_h.setdefault(md5(p), []).append(s)

dup = [(md5(p), s, tr_h[md5(p)]) for s, p in va.items() if md5(p) in tr_h]
print(f"*** СОВПАДЕНИЯ ПО СОДЕРЖИМОМУ (val внутри train): {len(dup)} ***")
for h, va_stem, tr_stems in dup:
    print(f"   val: {va_stem} == train: {tr_stems} (md5: {h})")

for split in ["train", "val", "test"]:
    lab = ROOT / f"labels/{split}"
    if not lab.exists():
        continue
    files = list(lab.glob("*.txt"))
    img_dict = tr if split == "train" else (va if split == "val" else te)
    n_img = len(img_dict)

    n_obj = 0
    empty_cnt = 0
    c = collections.Counter()

    for f in files:
        try:
            content = f.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            continue
        if not content:
            empty_cnt += 1
            continue
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        if not lines:
            empty_cnt += 1
        else:
            n_obj += len(lines)
            for l in lines:
                parts = l.split()
                if parts:
                    try:
                        c[int(parts[0])] += 1
                    except ValueError:
                        pass

    print(f"\n{split}: файлов меток {len(files)}, изображений {n_img}, "
          f"объектов {n_obj}, плотность {n_obj/max(n_img,1):.2f} об/кадр")
    print(f"   пустых файлов меток: {empty_cnt}")
    missing = n_img - len(files)
    print(f"   изображений без файла меток: {missing}")
    print(f"{split} по классам:", dict(sorted(c.items())))
