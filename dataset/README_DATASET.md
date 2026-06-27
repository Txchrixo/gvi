# GVI training dataset

Task: YOLO segmentation for Godot scene reconstruction.

Classes: platform, wall, floor, ladder, spike, hazard, door, exit, player_spawn, enemy, pickup, coin, button, lever, water, background, foreground_decoration, light

Recommended workflow:

```bash
gvi dataset init --type platformer ./dataset
gvi autolabel ./dataset/raw --dataset ./dataset --backend heuristic --classes platform ladder spike door enemy pickup
gvi dataset stats ./dataset
gvi review ./dataset
gvi train ./dataset --model yolo11n-seg.pt --epochs 80 --imgsz 640
```

Folders:

- `raw/`: unlabelled screenshots, PSD renders or mockups.
- `images/{train,val,test}`: training images.
- `labels/{train,val,test}`: YOLO segmentation labels.
- `annotations/gvi`: rich GVI JSON labels.
- `autolabel/`: teacher outputs before human correction.
- `review/`: active learning review files.
- `overlays/`: visual QA overlays.
- `models/`: trained weights.
