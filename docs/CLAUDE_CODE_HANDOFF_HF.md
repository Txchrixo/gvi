# Prompt de transfert pour Claude Code

> Copie-colle **tout le bloc ci-dessous** dans Claude Code (dans le dossier du
> projet `gvi_v1_2_training_ready`). Il contient tout le contexte nécessaire :
> Claude Code n'a aucun accès à la conversation où ce backend a été conçu, donc
> ce fichier lui transmet l'objectif, l'architecture, et ce qui reste à vérifier.

---

## CONTEXTE

Tu travailles sur GVI, un outil qui convertit des images en scènes Godot 4 et
qui possède une couche d'entraînement ML (dossier `gvi/training/`). Un nouveau
backend "teacher" pour l'auto-labellisation vient d'être ajouté :
**`gvi/training/hf_teacher.py`** — il utilise **Grounding DINO + SAM 2** via
Hugging Face `transformers` pour pré-annoter des images à partir de prompts
texte définis dans la taxonomie.

Ce backend a été écrit et **partiellement vérifié hors-ligne** (sans GPU ni
`transformers` installés), via des moteurs DINO/SAM2 simulés. Ce qui a été
confirmé fonctionnel : l'enregistrement du backend, les imports paresseux,
l'échec propre sans dépendances, la construction du prompt depuis la taxonomie,
le mapping phrase→classe, la conversion masque→polygone, et le câblage complet
`label_image` → `AnnotationObject` → `score_object` → ligne YOLO.

**Ce qui n'a PAS pu être vérifié et que tu dois confirmer :** l'inférence réelle
de Grounding DINO et SAM 2 sur de vraies images, avec les vrais poids de modèles.

## ARCHITECTURE À RESPECTER (ne pas casser)

- Un backend teacher implémente l'interface dans `gvi/training/autolabel.py` :
  un attribut `id: str` + une méthode
  `label_image(image_path, taxonomy, selected_classes) -> AnnotationFile`.
- Chaque objet détecté est un `AnnotationObject` (voir
  `gvi/training/annotations.py`) avec : `id`, `class_name`, `bbox_xywh`
  (x, y, w, h en pixels), `polygon` optionnel, `confidence` (0..1), `source`,
  `godot_candidates`, `needs_review`, `review_reason`.
- `godot_candidates` / `needs_review` sont remplis par `score_object()`
  (`gvi/training/rules.py`), comme dans les backends `yolo` et `heuristic`.
- Le backend est enregistré dans `get_backend()` (dans `autolabel.py`) sous les
  alias : `huggingface`, `hf`, `grounding-dino`, `dino`.
- Les dépendances lourdes (`transformers`, `torch`) sont importées **à
  l'intérieur des méthodes** (lazy), jamais au niveau module. **Garde ça.**

## TA MISSION

1. **Installe les dépendances et vérifie l'environnement :**
   ```bash
   python -m pip install -e ".[hf]"
   python -c "import transformers, torch; print('transformers', transformers.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
   ```

2. **Lance la suite de tests existante** (elle inclut 4 tests HF qui doivent
   passer ; `test_hf_backend_fails_cleanly_without_transformers` sera
   automatiquement *skipped* une fois transformers installé) :
   ```bash
   python -m pytest tests/test_training_system.py -v
   ```

3. **Test d'inférence réelle** — le point clé que je n'ai pas pu faire. Prépare
   3 à 5 vraies images (screenshots de jeux 2D de préférence, ou utilise celles
   de `test_images/`), puis :
   ```bash
   gvi dataset init --type platformer ./dataset
   cp test_images/*.png ./dataset/raw/    # ou tes propres images
   gvi training autolabel ./dataset/raw \
       --dataset ./dataset \
       --backend huggingface \
       --classes platform --classes ladder --classes spike \
       --classes door --classes enemy --classes pickup \
       --conf 0.30
   ```
   Puis **inspecte visuellement** les overlays générés dans
   `./dataset/overlays/*.jpg` et les annotations dans
   `./dataset/annotations/gvi/*.json`. Juge la qualité réelle de détection.

4. **Corrige uniquement ce qui casse réellement à l'exécution.** Causes
   probables si ça plante (l'API `transformers` évolue selon la version) :
   - Les noms de classes d'auto-modèles : `AutoModelForZeroShotObjectDetection`
     et `AutoProcessor` pour Grounding DINO. Vérifie que ta version de
     transformers les expose ; sinon adapte aux classes spécifiques
     (`GroundingDinoForObjectDetection`, `GroundingDinoProcessor`).
   - SAM 2 : le code tente d'abord `Sam2Processor`/`Sam2Model` de transformers,
     puis se rabat sur le package standalone `sam2`
     (`SAM2ImagePredictor.from_pretrained`). Si aucune des deux API n'existe
     dans ta version, soit installe `sam2`, soit lance avec `use_sam2=False`
     (les polygones tomberont alors sur la box — l'entraînement reste possible).
   - La signature de `post_process_grounded_object_detection` (les noms
     `box_threshold`/`text_threshold` vs `threshold`) change selon les
     versions ; adapte si transformers lève une TypeError dessus.

5. **Si tu corriges quoi que ce soit dans `hf_teacher.py`**, garde l'interface
   intacte (mêmes entrées/sorties, mêmes imports lazy) et **relance les 2-3 et
   la suite de tests** pour confirmer la non-régression.

6. **Termine l'entraînement du student** une fois les labels jugés corrects :
   ```bash
   gvi training review ./dataset --cvat        # corrige les low-confidence
   gvi training train ./dataset --model yolo11n-seg.pt --epochs 80
   ```

## CRITÈRE DE RÉUSSITE

- `gvi training autolabel --backend huggingface` tourne sans crash sur de
  vraies images.
- Les overlays montrent des détections plausibles (pas parfaites — c'est un
  teacher, il sera corrigé).
- Les fichiers `annotations/gvi/*.json` contiennent des objets au bon format
  (class_name issu de la taxonomie, bbox, polygon si SAM2 actif, confidence,
  godot_candidates remplis).
- `python -m pytest tests/test_training_system.py` reste vert.

## NE FAIS PAS

- Ne mets pas `transformers`/`torch` en import top-level dans `hf_teacher.py`.
- Ne change pas le format `AnnotationObject` ni l'interface `get_backend`.
- Ne supprime pas les autres backends (`heuristic`, `yolo`, `grounded-sam`).
- N'ajoute pas `datasets` ni d'intégration Hub : hors scope pour l'instant.
