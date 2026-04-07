# Migration — automatic-darts-

## Version cible
- **Python 3.12** (actuellement Python 3.x, version exacte non spécifiée)

## Changelog de migration (2026-04-07)

### Shapely 2.0 compatibilité
- `dartdetection.py:143` et `darts.py:266` : `dart_polygon.exterior.coords` → `list(dart_polygon.exterior.coords)`
  - En Shapely 2.0, `.coords` retourne un `CoordinateSequence` au lieu d'une liste.
    `max()` peut échouer ou donner des résultats inattendus sans conversion explicite.

### Bugs corrigés
- `dartdetection.py` : import `numpy` dupliqué (lignes 1 et 4) → supprimé
- `dartdetection.py` : import `shapely` inutilisé + import `Polygon` manquant en haut de fichier → consolidé
- `dartdetection.py` : import `shapely.geometry` dupliqué (lignes 6 et 129) → supprimé le doublon
- `dartdetection.py` : `main()` appelé deux fois dans `__main__` → corrigé

### Fichiers créés
- `requirements.txt` : dépendances avec versions Python 3.12 (`numpy>=2.2`, `opencv-contrib-python>=4.11`, `shapely>=2.0`)
- `.gitignore` : ignore les fichiers générés (calibration `.npz`, `counter_data.txt`, `__pycache__/`)

---

## Etat actuel des dépendances

Aucun fichier de dépendances n'existe. Dépendances identifiées par analyse des imports :

| Dépendance | Version actuelle | Version cible (Python 3.12) | Breaking changes |
|------------|-----------------|---------------------------|------------------|
| opencv-python | Non spécifiée | **>=4.11** | Pas de breaking change majeur si >= 4.x. |
| numpy | Non spécifiée | **>=2.2** | Alias deprecated supprimés. |
| shapely | Non spécifiée | **>=2.0** | **API restructurée** dans Shapely 2.0. Voir ci-dessous. |
| tkinter | Built-in | Built-in | Inclus avec Python 3.12. |

## Modifications de code nécessaires

### 1. Shapely 1.x → 2.0 (ATTENTION)

Shapely 2.0 est une réécriture majeure basée sur pygeos :

```python
# Constructeurs — OK, API stable
from shapely.geometry import Point, LineString, Polygon  # Pas de changement

# Méthodes deprecated en 2.0
# Les objets ne sont plus mutables
# .coords retourne un CoordinateSequence (pas une liste)

# Performance : les opérations sont vectorisées par défaut
# Certains comportements changent subtilement :
point.distance(other)        # OK
polygon.contains(point)      # OK
linestring.intersection(...)  # OK — mais peut retourner des types différents

# Si le code utilise np.array(geom.coords) :
np.array(geom.coords)  # OK en 2.0, mais vérifier le dtype
```

**Fichiers impactés :** `darts.py`, `dartdetection.py`

### 2. NumPy 2.x

```python
np.float   → np.float64
np.int     → np.int_
np.bool    → np.bool_
```

**Fichiers impactés :** `darts.py`, `dartdetection.py`

### 3. OpenCV — vérifier les API utilisées

Le repo utilise :
- `cv2.goodFeaturesToTrack()` — API stable
- `cv2.fitLine()` — API stable
- `cv2.ximgproc.thinning()` — **module opencv-contrib-python nécessaire**
- Filtre Kalman — API stable

```python
# Si cv2.ximgproc est utilisé, installer opencv-contrib-python au lieu de opencv-python
pip install opencv-contrib-python>=4.11
```

### 4. Tkinter

Tkinter est inclus avec Python 3.12. Pas de changement nécessaire si les imports sont déjà en Python 3 (`from tkinter import *`).

## requirements.txt à créer

```
numpy>=2.2,<3
opencv-contrib-python>=4.11,<5
shapely>=2.0,<3
```

Note : `opencv-contrib-python` inclut `opencv-python` + les modules extra (`ximgproc` pour squelettisation).

## Réutilisabilité pour la fusion (hors GUI)

**Priorité de fork : 2 (HAUTE)** — module de détection pur, filtre de Kalman unique parmi les repos.

### Fichiers coeur à extraire

| Fichier | Rôle | Couplage GUI | Fonctions pures testables |
|---------|------|:------------:|--------------------------|
| `dartdetection.py` | Pipeline complet de détection : seuillage, corners Harris, filtrage géométrique, squelettisation, Kalman | **Aucun** | `getThreshold()`, `getCorners()`, `filterCornersLine()`, classe `KalmanFilter` |
| `game_501.py` | Logique de jeu 501 (bust detection) | **Aucun** | `update_score()`, `check_game_over()` |

### Fichiers à ignorer

| Fichier | Raison |
|---------|--------|
| `darts.py` | Tkinter GUI, calibration manuelle (4 points), boucle de jeu couplée à l'affichage |

### Valeur unique pour la fusion

- **Filtre de Kalman 4D** (position + vitesse) : aucun autre repo n'en a. Utile pour stabiliser la détection entre frames.
- **Vote majoritaire 3 caméras** : l'algorithme de consensus est dans `darts.py` (GUI) mais le principe est simple à réimplémenter.
- **Squelettisation** (`cv2.ximgproc.thinning`) : approche unique pour trouver l'axe du dart.

### Testabilité

**Facile** — `dartdetection.py` est un module pur avec des entrées/sorties NumPy.

**Tests unitaires à ajouter :**
- `KalmanFilter` : predict/update avec des séquences de mesures connues
- `getCorners` : image synthétique avec des coins connus → positions détectées
- `filterCornersLine` : ensemble de coins + contraintes → coins filtrés attendus
- `getThreshold` : image de diff → masque binaire → nombre d'objets détectés

## Ordre de migration recommandé

1. Créer un venv Python 3.12
2. Créer le `requirements.txt` (ci-dessus)
3. Installer les dépendances
4. Vérifier/corriger les alias NumPy deprecated
5. Tester l'API Shapely 2.0 (geometry operations)
6. Vérifier que `cv2.ximgproc.thinning()` fonctionne avec opencv-contrib
7. Ajouter les tests unitaires (Kalman, corners, filtering)
8. Tester le pipeline complet (3 caméras + vote majoritaire)
