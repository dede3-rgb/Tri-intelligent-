import shutil
from pathlib import Path
from PIL import Image, ImageDraw
import torch

ROOT = Path("data")
classes = ["papier_emballage", "verre", "plastique", "metal", "non_recyclable"]
for split in ["train","val","test"]:
    for cls in classes:
        (ROOT/split/cls).mkdir(parents=True, exist_ok=True)

def make_img(path: Path, color, text):
    img = Image.new("RGB", (96,96), color)
    d = ImageDraw.Draw(img)
    d.text((10,40), text, fill=(255,255,255))
    img.save(path)

palette = {
    "papier_emballage": (70,120,200),
    "verre": (60,180,75),
    "plastique": (255,165,0),
    "metal": (180,180,180),
    "non_recyclable": (120,120,120),
}
idx = 0
for split in ["train","val","test"]:
    n = 6 if split=="train" else 2
    for cls in classes:
        for k in range(n):
            make_img(ROOT/split/cls/f"img_{idx}.png", palette[cls], f"{cls[:3]}")
            idx += 1

# 3) class_mapping minimal (index -> classe)
mapping = {i: c for i,c in enumerate(classes)}
import json
with open("class_mapping.json","w",encoding="utf-8") as f:
    json.dump(mapping, f, indent=2, ensure_ascii=False)

# 4) Test get_dataloaders
from datasets import get_dataloaders
train_loader, val_loader, test_loader, found_classes = get_dataloaders(
    data_dir="data", img_size=64, batch_size=4, num_workers=0, aug=False
)

print("Classes détectées:", found_classes)
batch = next(iter(train_loader))
x, y = batch
print("Batch shape:", x.shape, "| labels:", y.shape)

assert x.shape[1] == 3, "Les images doivent être RGB (C=3)."
assert len(found_classes) == 5, "On attend 5 classes."
assert x.shape[0] == 4, "Batch size attendu = 4."
print("Smoke test OK ✅")
