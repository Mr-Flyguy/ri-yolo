# scripts/check_split.py
from pathlib import Path
import collections
import hashlib

ROOT = Path("datasets/ExDark")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}

def stems(d):
    return {p.stem: p for p in Path(d).iterdir() if p.suffix in IMG_EXT}

tr = stems(ROOT / "images/train")
va = stems(ROOT / "images/val")
print(f"train images: {len(tr)}   val images: {len(va)}   sum: {len(tr)+len(va)}")
print(f"ExDark total should be 7363 -> excess: {len(tr)+len(va)-7363}")

overlap = set(tr) & set(va)
print(f"\n*** ПЕРЕСЕЧЕНИЕ train/val: {len(overlap)} изображений ***")
for s in sorted(overlap)[:20]:
    print("   ", s, "|", tr[s].name, "|", va[s].name)

def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()
tr_h = {}
for s, p in tr.items():
    tr_h.setdefault(md5(p), []).append(s)
dup = [(md5(p), s) for s, p in va.items() if md5(p) in tr_h]
print(f"\n*** СОВПАДЕНИЯ ПО СОДЕРЖИМОМУ (val внутри train): {len(dup)} ***")

for split in ["train", "val"]:
    lab = ROOT / f"labels/{split}"
    files = list(lab.glob("*.txt"))
    n_obj = sum(sum(1 for line in f.open() if line.strip()) for f in files)
    n_img = len(tr if split == "train" else va)
    print(f"\n{split}: файлов меток {len(files)}, изображений {n_img}, "
          f"объектов {n_obj}, плотность {n_obj/max(n_img,1):.2f} об/кадр")
    empty = [f.name for f in files if f.stat().st_size == 0]
    print(f"   пустых файлов меток: {len(empty)}")
    missing = n_img - len(files)
    print(f"   изображений без файла меток: {missing}")

for split in ["train", "val"]:
    c = collections.Counter()
    for f in (ROOT / f"labels/{split}").glob("*.txt"):
        for line in f.open():
            if line.strip():
                c[int(line.split()[0])] += 1
    print(f"\n{split} по классам:", dict(sorted(c.items())))
