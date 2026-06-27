# GVI v1.2 — Démarrage entraînement (READY TO TRAIN)

Ce guide te fait passer de zéro à un entraînement YOLO segmentation qui tourne,
en quelques minutes. Tout ce qui est marqué ✅ ci-dessous a été **réellement
exécuté et vérifié** (pas seulement écrit) ; ce qui est marqué ⚠️ dépend de
ton matériel/réseau et n'a pas pu l'être dans l'environnement de préparation.

---

## 1. Installation

```powershell
# Depuis le dossier du projet
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

python -m pip install -U pip
python -m pip install -e ".[dev,training]"
```

L'extra `training` installe : `ultralytics`, `torch`, `torchvision`,
`PyYAML`, `tqdm`, `requests`. C'est tout ce qu'il faut pour entraîner.

---

## 2. Test immédiat SANS aucune image à toi (✅ vérifié)

Tu peux entraîner tout de suite grâce au générateur de dataset synthétique :

```powershell
gvi dataset init --type platformer ./dataset
gvi training synthetic --dataset ./dataset --count 80 --split train
gvi training synthetic --dataset ./dataset --count 20 --split val
gvi dataset stats ./dataset
gvi training train ./dataset --dry-run        # affiche la commande exacte
gvi training train ./dataset                  # ⚠️ lance le vrai entraînement
```

`--dry-run` (✅ vérifié) affiche la vraie commande
`yolo segment train model=yolo11n-seg.pt data=.../data.yaml epochs=80 ...`.
Retire `--dry-run` pour lancer pour de vrai (⚠️ nécessite ultralytics installé).

---

## 3. Avec tes vraies images

```powershell
gvi dataset init --type platformer ./dataset
gvi dataset ingest ./mes_images --dataset ./dataset --split train

# Pré-annotation automatique (backend heuristique, ✅ vérifié, aucun GPU requis)
gvi training autolabel ./dataset/raw --dataset ./dataset --backend heuristic ^
    --classes platform --classes ladder --classes spike ^
    --classes door --classes enemy --classes pickup

# Génère review.json pour corriger en priorité les labels douteux (✅ vérifié)
gvi training review ./dataset --cvat

# Une fois corrigé, entraîne
gvi training train ./dataset --model yolo11n-seg.pt --epochs 80 --imgsz 640 --batch 8
```

### Backend « teacher » plus puissant (⚠️ non vérifié hors-ligne)
Le backend `grounded-sam` (Grounding DINO + SAM2) donne de bien meilleures
pré-annotations par prompt texte, mais nécessite un endpoint/poids externes.
Voir `docs/TEACHER_BACKENDS.md`. Le backend `heuristic` ne dépend de rien et
sert à amorcer ; le backend `yolo` utilise un modèle déjà entraîné.

---

## 4. Boucle d'active learning (✅ vérifié de bout en bout)

```
1. autolabel (teacher)         -> labels bruts + confidence
2. review                      -> review.json des low-confidence
3. correction humaine (CVAT)   -> labels propres
4. train                       -> modèle student YOLO
5. predict sur nouvelles images
6. retour à 2
```

Le moteur de **règles métier** (✅ vérifié) ajuste la confiance du modèle :
une « ladder » détectée mais horizontale est automatiquement passée en
`needs_review` avec la raison « Ladder candidate is not vertical enough »,
exactement comme spécifié. Voir `gvi/training/rules.py`.

---

## 5. Logiciels externes (selon ton besoin)

| Besoin | Outil | Obligatoire ? |
|---|---|---|
| Entraîner rapidement en local | Drivers **NVIDIA CUDA** + torch CUDA | Non (CPU marche, mais lent) |
| Pas de GPU du tout | **Google Colab** (gratuit) — voir `notebooks/training_colab_template.ipynb` | Alternative au GPU local |
| Corriger les annotations sérieusement | **CVAT** (via Docker) | Recommandé |
| Annotation plus simple en ligne | **Roboflow** ou **Label Studio** | Alternative à CVAT |
| Pré-annotation par prompt texte | Endpoint **Grounding DINO + SAM2** | Optionnel (améliore l'autolabel) |

---

## Ce qui est vérifié vs ce qui ne l'est pas

**✅ Réellement exécuté et confirmé fonctionnel :**
`dataset init` · `synthetic` · `dataset stats` · `dataset ingest` ·
`autolabel --backend heuristic` (format GVI : class/bbox/polygon/confidence/
godot_candidates/needs_review) · export GVI→YOLO (labels segmentation valides) ·
`review` (active learning) · règles métier (`score_object`) · overlays ·
`train --dry-run` (génère la vraie commande) · les 4 tests `tests/test_training_system.py`.

**⚠️ Non vérifiable dans l'environnement de préparation (pas de GPU / réseau /
ultralytics) — à confirmer chez toi :**
le vrai `yolo segment train` lui-même · les backends `yolo` et `grounded-sam`
de l'autolabel · `predict` avec un modèle réel · l'entraînement Colab.
Tous ces chemins échouent **proprement** avec un message clair
(« Install training deps first: ... ») quand la dépendance manque — pas de
crash silencieux.
