# Observations des nuages d'eau surfondus avec EarthCARE au-dessus du Dôme C, Antarctique

**Auteur :** Mouhamadou DIAMANKA

---

## Contexte scientifique

Le satellite **EarthCARE** (Earth Cloud, Aerosol and Radiation Explorer), développé conjointement par l'ESA et la JAXA, est dédié à l'étude des nuages et des aérosols depuis l'orbite. Sa combinaison unique d'instruments actifs et passifs permet une caractérisation verticale fine des propriétés microphysiques des nuages.

Le **Dôme C** (Antarctica, −75.1°N / 123.35°E) abrite la **station Concordia**, une station de recherche franco-italienne en opération toute l'année. C'est l'un des rares sites au monde offrant des observations au sol continues en région polaire, ce qui en fait un site de validation privilégié pour les satellites.

Ce projet analyse les passages d'EarthCARE au-dessus de Concordia durant l'**été austral 2025–2026** pour évaluer la représentation des nuages d'eau surfondue — des gouttelettes liquides maintenues en suspension à des températures négatives — dont la détection et la quantification restent un défi majeur en télédétection.

---

## Objectifs

1. Évaluer la qualité des premières observations de nuages d'eau surfondue issues du produit **ACM-CLP-2B** d'EarthCARE.
2. Comparer ces observations avec les mesures au sol effectuées à la station **Concordia** (déc. 2025 – jan. 2026).
3. Comparer avec les réanalyses **ERA5** du ECMWF.

---

## Données utilisées

### EarthCARE (ESA + JAXA)

Produit utilisé : **ACM_CLP_2B** — produit nuages combiné multi-instruments, résolution 1 km.

| Instrument | Nom complet | Rôle |
|---|---|---|
| **ATLID** | Atmospheric Lidar | Lidar à 355 nm — profils d'extinction, type de particules |
| **CPR** | Cloud Profiling Radar | Radar à 94 GHz — contenu en glace, précipitations |
| **MSI** | Multi-Spectral Imager | Imageur multi-spectral — propriétés nuages de jour |
| **BBR** | Broad Band Radiometer | Radiomètre — flux radiatifs large bande |

Les orbites couvrent trois frames par passage : **F**, **G**, **H** (segments successifs le long de la trace au sol).

**Sources d'accès :**
- ESA via la bibliothèque Python `earthcarekit`
- JAXA via le portail [gportal.jaxa.jp](https://gportal.jaxa.jp/gpr/)

### Observations au sol — Station Concordia

Mesures de référence pour la validation in-situ durant l'été austral.

### Réanalyses ERA5

Données atmosphériques de réanalyse du Centre Européen pour les Prévisions Météorologiques à Moyen Terme (ECMWF).

---

## Structure du projet

```
earthcare_code/
│
├── config.py          # Constantes, paramètres, styles, chemins
├── data_io.py         # Téléchargement et lecture des fichiers HDF5
├── processing.py      # Calculs dérivés et préparation des orbites
├── gridding.py        # Accumulation spatiale sur grille lat×lon
├── plotting.py        # Visualisation (profils 2D, cartes polaires)
├── orbits_nc.py       # Stockage/lecture des orbites en NetCDF4
│
├── data/
│   ├── jaxa/          # Fichiers HDF5 téléchargés depuis le portail JAXA
│   └── cache/
│       ├── grid_cache.nc        # Cache de la grille accumulée
│       ├── grid_cache_orbites.nc
│       └── orbite_cache.nc      # Cache des orbites préparées
│
└── readme.md
```

---

## Paramètres analysés

### Variables physiques

| Variable | Clé | Description | Unité |
|---|---|---|---|
| Type de particule | `particle_type` | Classification nuage/précipitation (0–13) | — |
| Température | `temperature` | Température atmosphérique sur la grille | K |
| Contenu en glace | `iwc` | Ice Water Content (profil vertical) | kg m⁻³ |
| Contenu en eau liquide | `lwc` | Liquid Water Content (profil vertical) | kg m⁻³ |
| Chemin en glace | `iwp` | Ice Water Path (colonne intégrée) | kg m⁻² |
| Chemin en eau liquide | `lwp` | Liquid Water Path (colonne intégrée) | kg m⁻² |
| Latitude / Longitude | `lat`, `lon` | Géolocalisation | ° |
| Hauteur | `height` | Altitude des niveaux de la grille | m |
| Temps | `time` | Temps en secondes depuis le début de la frame | s |
| Élévation de surface | `surface_elevation` | Topographie sous l'orbite | m |

### Classification des types de particules (0–13)

| Code | Type | Couleur |
|---|---|---|
| 0 | Clear (ciel clair) | `#ffffff` |
| 1 | Warm water (eau chaude) | `#2196F3` |
| 2 | **Supercooled water** (eau surfondue) | `#0009b0` |
| 3 | 3D ice | `#02ab24` |
| 4 | 2D plate | `#9C27B0` |
| 5 | Mix 3D ice and 2D plate | `#7B1FA2` |
| 6 | Liquid drizzle | `#80DEEA` |
| 7 | Mixed phase drizzle | `#00ACC1` |
| 8 | Rain | `#FF9890` |
| 9 | Snow | `#B0BEC5` |
| 10 | Water + liquid drizzle | `#0288D1` |
| 11 | Water + rain | `#E65100` |
| 12 | Mixed phase | `#b53c00` |
| 13 | Unknown | `#310202` |

---

## Configuration (`config.py`)

Tous les paramètres du projet sont centralisés dans `config.py`.

### Paramètres principaux

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| `FILE_TYPE` | `"ACM_CLP_2B"` | Type de produit EarthCARE |
| `ORBIT_FRAME` | `"09039G"` | Numéro d'orbite + frame (ex. orbite 9039, frame G) |
| `DATE_START` | `"2025-12-01"` | Début de la plage temporelle d'analyse |
| `DATE_END` | `"2025-12-10"` | Fin de la plage temporelle d'analyse |
| `DOWNLOAD_PERIODS` | `[("2025-12-01", "2026-02-01")]` | Périodes de téléchargement multi-orbites |
| `T_MIN` / `T_MAX` | `440` / `560` s | Fenêtre temporelle d'analyse dans la frame |
| `LAT_REF` / `LON_REF` | `-75.1` / `123.35` | Coordonnées de la station Concordia (Dôme C) |
| `DEFAULT_DLAT` / `DEFAULT_DLON` | `1.0°` / `10.0°` | Résolution de la grille d'accumulation |
| `GRID_CACHE` | `./data/cache/grid_cache.nc` | Chemin du cache de grille |
| `ORBIT_CACHE` | `./data/cache/orbite_cache.nc` | Chemin du cache d'orbites |
| `FORCE_REBUILD` | `True` | Forcer la reconstruction des caches |

### Cercles de distance

| Rayon | Couleur | Usage |
|---|---|---|
| 500 km | `#00FFF2` (cyan) | Zone de proximité immédiate |
| 1000 km | `#F8F404` (jaune) | Zone étendue |

---

## Pipeline de traitement

```
Fichiers HDF5 (ESA/JAXA)
        │
        ▼
  data_io.load_orbit()
  data_io.load_multi_orbits()
  data_io.load_jaxa_orbits()
        │
        ▼
  processing.prepare_single_orbit()     ← calcul des variables dérivées
  processing.prepare_multi_orbits()
        │
        ├──► plotting.*                 ← visualisation d'une orbite
        │
        ▼
  gridding.GridAccumulator.accumulate() ← accumulation jour par jour
        │
        ▼
  GridAccumulator.save()                ← persistance NetCDF
        │
        ▼
  GridAccumulator.load()
  GridAccumulator.mean_range()
  GridAccumulator.std_range()
        │
        ▼
  plotting.plot_grid_*()                ← cartes et statistiques grillées
```

---

## Référence des modules

### `data_io.py` — Entrées/Sorties

| Fonction | Description |
|---|---|
| `download_product(orbit_and_frame, start_time, end_time)` | Télécharge un produit pour une orbite et frame spécifiques |
| `download_multi_period(periods, frames)` | Télécharge sur plusieurs périodes pour les frames F, G, H |
| `search_product(start_time, end_time, ...)` | Recherche les fichiers disponibles (par frame, orbite, ou toutes frames) |
| `load_orbit(filepath, extra_fields)` | Lit un fichier HDF5 et retourne un `dict` de tableaux NumPy |
| `load_multi_orbits(filepaths, extra_fields)` | Charge une liste de fichiers, trie par (orbite, frame, start_time) |
| `load_jaxa_orbits(directory, extra_fields)` | Charge tous les `.h5` d'un répertoire JAXA |
| `merge_orbit_sources(esa_orbits, jaxa_orbits)` | Fusionne et trie les orbites ESA + JAXA |
| `group_by_orbit(orbites)` | Regroupe les frames par numéro d'orbite → `{num: [frame_F, frame_G, frame_H]}` |
| `get_t0_utc(filepath)` | Extrait l'heure UTC de début depuis l'en-tête HDF5 |
| `get_frame_id(orbit_data)` | Retourne la frame (F, G ou H) d'une orbite chargée |

### `processing.py` — Traitement et calculs

| Fonction | Description |
|---|---|
| `haversine(lat1, lon1, lat2, lon2)` | Distance orthodromique entre deux points (km) |
| `distance_to_ref(lat, lon)` | Distance à la station de référence (Concordia) |
| `calculate_local_times(t0_utc, time_s, lon_arr)` | Temps solaire local le long de la trace (1° lon = 4 min) |
| `mask_negative(arr)` | Remplace les valeurs négatives par `NaN` |
| `mask_particle_type(particle_type)` | Masque les codes hors [0–13] et le ciel clair (code 0) |
| `build_time_grid(time_s, height)` | Construit la grille temps 2D pour `pcolormesh` |
| `prepare_single_orbit(orbit_data, t0_utc)` | Calcule toutes les variables dérivées d'une orbite (distance, temps locaux, masques, temp en °C…) |
| `prepare_multi_orbits(raw_orbits)` | Prépare une liste d'orbites brutes pour la grille (lwp, iwp, lat, lon) |
| `save_multi_orbits(orbites, filepath)` | Sauvegarde les orbites préparées en NetCDF |
| `load_multi_orbits_nc(filepath)` | Recharge les orbites depuis le NetCDF intermédiaire |

### `gridding.py` — Accumulation spatiale

La classe `GridAccumulator` gère une grille lat × lon accumulée **jour par jour**. Chaque cellule stocke `sum`, `sum²` et `count` pour permettre le calcul différé de la moyenne et de l'écart-type sur n'importe quelle plage de dates.

#### Instanciation

```python
from gridding import GridAccumulator

grid = GridAccumulator(dlat=1.0, dlon=10.0)
```

#### Méthodes principales

| Méthode | Description |
|---|---|
| `accumulate(orbit_data)` | Ajoute une orbite à la grille |
| `mean(day)` | Moyenne pour un jour donné (`"YYYY-MM-DD"`) |
| `std(day)` | Écart-type pour un jour donné |
| `count(day)` | Nombre d'observations par cellule pour un jour |
| `mean_range(d1, d2)` | Moyenne sur une plage de dates |
| `std_range(d1, d2)` | Écart-type sur une plage de dates |
| `count_range(d1, d2)` | Nombre d'obs. sur une plage de dates |
| `n_orbits_range(d1, d2)` | Nombre total d'orbites sur la plage |
| `save(filepath)` | Sauvegarde la grille en NetCDF4 |
| `GridAccumulator.load(filepath)` | Recharge une grille depuis le NetCDF |

#### Exemple d'utilisation

```python
from gridding import GridAccumulator
from config import GRID_CACHE

# Charger la grille existante
grid = GridAccumulator.load(GRID_CACHE)

# Statistiques sur une période
means = grid.mean_range("2025-12-01", "2026-01-31")
stds  = grid.std_range("2025-12-01", "2026-01-31")

# Accéder aux données LWP moyennes (tableau lat×lon)
lwp_mean = means["lwp"]
```

### `data_io.py` — Exemple de workflow complet

```python
from data_io import load_jaxa_orbits, merge_orbit_sources, load_multi_orbits
from processing import prepare_multi_orbits
from gridding import GridAccumulator
from config import JAXA_DATA_DIR, GRID_CACHE, DOWNLOAD_PERIODS

# 1. Téléchargement (ESA)
from data_io import download_multi_period
download_multi_period(DOWNLOAD_PERIODS, frames=["F", "G", "H"])

# 2. Chargement
esa_orbits  = load_multi_orbits(search_product(...))
jaxa_orbits = load_jaxa_orbits(JAXA_DATA_DIR)
all_orbits  = merge_orbit_sources(esa_orbits, jaxa_orbits)

# 3. Accumulation
grid = GridAccumulator()
for orb in all_orbits:
    grid.accumulate(orb)

# 4. Sauvegarde
grid.save(GRID_CACHE)
```

---

## Champs HDF5

Les champs lus dans les fichiers HDF5 sont définis dans `config.HDF5_FIELDS` :

| Clé | Chemin HDF5 |
|---|---|
| `particle_type` | `ScienceData/Data/cloud_particle_type_cpr_atlid_msi_1km` |
| `temperature` | `ScienceData/Data/GRID_temperature_1km` |
| `iwc` | `ScienceData/Data/cloud_ice_content_1km` |
| `iwp` | `ScienceData/Data/cloud_ice_water_path_1km` |
| `lwc` | `ScienceData/Data/cloud_water_content_1km` |
| `lwp` | `ScienceData/Data/cloud_water_path_1km` |
| `lat` | `ScienceData/Geo/latitude` |
| `lon` | `ScienceData/Geo/longitude` |
| `height` | `ScienceData/Geo/height` |
| `time` | `ScienceData/Geo/time` |
| `surface_elevation` | `ScienceData/Geo/surface_elevation` |

---

## Installation et dépendances

### Environnement Python

```bash
python -m venv .env
.env\Scripts\activate       # Windows
# ou
source .env/bin/activate    # Linux/macOS

pip install numpy xarray matplotlib cartopy pandas h5py scipy earthcarekit jupyter
```

### Dépendances principales

| Bibliothèque | Usage |
|---|---|
| `numpy` | Calculs numériques et tableaux |
| `xarray` | Tableaux multidimensionnels, lecture/écriture NetCDF |
| `h5py` | Lecture des fichiers HDF5 EarthCARE |
| `earthcarekit` | API de téléchargement et de recherche ESA |
| `matplotlib` | Visualisation 2D |
| `cartopy` | Cartes géographiques (projections polaires) |
| `scipy` | Calculs scientifiques complémentaires |
| `pandas` | Manipulation de données tabulaires |

---

## Backlog

- [ ] Cartographier l'élévation du continent Antarctique sur les figures polaires
- [ ] Quantifier les erreurs associées aux observations (LWP et IWP)
- [ ] Augmenter la résolution de la grille (dlat = 1°, dlon = 5°)
- [ ] Analyser l'évolution temporelle des nuages d'eau surfondue
- [ ] Intégrer les frames H et F dans l'analyse principale
- [ ] Ajouter la comparaison ERA5
- [ ] Intégrer les données de validation Concordia
