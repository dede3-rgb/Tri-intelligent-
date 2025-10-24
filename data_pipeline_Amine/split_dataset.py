# Organise un dataset source en 5 classes cibles + split train/val/test.
import os, shutil, random, json, hashlib
from PIL import Image
from pathlib import Path
from collections import defaultdict

random.seed(42)
RAW = Path(r"C:\Users\Amine\.cache\kagglehub\datasets\farzadnekouei\trash-type-image-dataset\versions\1")
OUT = Path("data")  # produira data/train|val|test/<classe>/*.jpg

SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}
IMG_EXT = {".jpg",".jpeg",".png",".bmp",".webp"}

# 2) mapping NOM_ORIGINE -> 5 bacs cibles
CANON = {
    "paper":"papier_emballage", "cardboard":"papier_emballage", "paper_cardboard":"papier_emballage",
    "glass":"verre", "bottle_glass":"verre",
    "plastic":"plastique", "bottle_plastic":"plastique",
    "metal":"metal", "can":"metal", "aluminum":"metal",
    "trash":"non_recyclable", "other":"non_recyclable", "organic":"non_recyclable",
}

def hash_file(p: Path)->str:
    h = hashlib.md5()
    with open(p,'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()

def gather_images(root: Path):
    files_by_class = defaultdict(list)
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            try:
                Image.open(p).verify()  # Vérifie que c'est bien une image
            except Exception:
                continue
            except Exception:
                continue
            src_cls = p.parent.name.lower().strip()
            tgt_cls = CANON.get(src_cls)  # remap vers 5 classes
            if tgt_cls:
                files_by_class[tgt_cls].append(p)
    return files_by_class

def main():
    files_by_class = gather_images(RAW)
    OUT.mkdir(exist_ok=True)
    for sp in SPLIT: (OUT/sp).mkdir(exist_ok=True)

    # dédoublonnage global par hash
    seen = set()
    stats = {}
    for cls, files in files_by_class.items():
        uniq = []
        for f in files:
            try:
                h = hash_file(f)
            except Exception:
                continue
            if h not in seen:
                seen.add(h)
                uniq.append(f)

        random.shuffle(uniq)
        n = len(uniq); n_tr = int(n*SPLIT["train"]); n_va = int(n*SPLIT["val"])
        buckets = {
            "train": uniq[:n_tr],
            "val"  : uniq[n_tr:n_tr+n_va],
            "test" : uniq[n_tr+n_va:],
        }
        for sp, lst in buckets.items():
            d = OUT/sp/cls; d.mkdir(parents=True, exist_ok=True)
            for src in lst:
                shutil.copy2(src, d/src.name)
        stats[cls] = {k: len(v) for k,v in buckets.items()}

    with open("data_stats.json","w",encoding="utf-8") as f:
        json.dump(stats,f,indent=2,ensure_ascii=False)

    # mapping index→classe pour l’entraînement
    classes = sorted(stats.keys())
    with open("class_mapping.json","w",encoding="utf-8") as f:
        json.dump({i:c for i,c in enumerate(classes)}, f, indent=2, ensure_ascii=False)

    print("OK → data/{train,val,test} + data_stats.json + class_mapping.json ✅")

if __name__ == "__main__":
    main()
