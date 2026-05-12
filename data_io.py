# ============================================================
# io_data.py
# Téléchargement EarthCARE et lecture des fichiers HDF5 (ESA).
# Gère toutes les frames (F, G, H) d'une même orbite.
# ============================================================

import h5py
import earthcarekit as eck
import numpy as np
from datetime import datetime
from pathlib import Path

from config import FILE_TYPE, HDF5_FIELDS

# Toutes les frames à traiter
ALL_FRAMES = ["F", "G", "H"]


# ============================================================
# TÉLÉCHARGEMENT
# ============================================================

def download_product(orbit_and_frame: str,
                     start_time: str,
                     end_time: str) -> None:
    """Télécharge un produit pour une orbite et une frame données."""
    eck.ecdownload(
        file_type=FILE_TYPE,
        orbit_and_frame=orbit_and_frame,
        start_time=start_time,
        end_time=end_time,
    )


def download_multi_period(periods: list[tuple[str, str]],
                           frames: list[str] | None = None) -> None:
    """Télécharge le produit sur plusieurs périodes pour toutes les frames.

    Parameters
    ----------
    periods : list of (start_time, end_time)
    frames  : list de frames à télécharger (défaut : F, G, H)
    """
    frames = frames or ALL_FRAMES
    for start, end in periods:
        for frame in frames:
            try:
                eck.ecdownload(
                    file_type=FILE_TYPE,
                    frame_id=frame,
                    start_time=start,
                    end_time=end,
                )
            except Exception as e:
                print(f"[download] Erreur frame={frame} {start}->{end} : {e}")


# ============================================================
# RECHERCHE
# ============================================================

def search_product(start_time: str,
                   end_time: str,
                   orbit_and_frame: str | None = None,
                   frame_id: str | None = None,
                   all_frames: bool = False):
    """Recherche les fichiers produit disponibles.

    Parameters
    ----------
    orbit_and_frame : str, optional
        Ex. "09039G" — cherche une orbite et frame spécifiques.
    frame_id : str, optional
        Ex. "G" — cherche une frame particulière.
    all_frames : bool, default False
        Si True, cherche toutes les frames F, G, H et concatène
        les résultats. Ignoré si orbit_and_frame ou frame_id est fourni.

    Returns
    -------
    Dataset earthcarekit avec attribut filepath.
    """
    kwargs = dict(file_type=FILE_TYPE, start_time=start_time, end_time=end_time)

    if orbit_and_frame:
        kwargs["orbit_and_frame"] = orbit_and_frame
        return eck.search_product(**kwargs)

    if frame_id:
        kwargs["frame_id"] = frame_id
        return eck.search_product(**kwargs)

    if all_frames:
        # Chercher chaque frame séparément et fusionner les filepaths
        all_fps = []
        for frame in ALL_FRAMES:
            try:
                ds = eck.search_product(**kwargs, frame_id=frame)
                if hasattr(ds, "filepath"):
                    fps = list(ds.filepath)
                    all_fps.extend(fps)
                    print(f"[search] Frame {frame} : {len(fps)} fichier(s)")
            except Exception as e:
                print(f"[search] Frame {frame} introuvable : {e}")
        print(f"[search] Total : {len(all_fps)} fichier(s) (frames F+G+H)")
        return all_fps   # liste de chemins

    # Par défaut : frame_id="*" pour tout récupérer d'un coup
    kwargs["frame_id"] = "*"
    return eck.search_product(**kwargs)


# ============================================================
# LECTURE HDF5
# ============================================================

def load_orbit(filepath: str,
               extra_fields: dict | None = None) -> dict:
    """Lit un fichier HDF5 EarthCARE et retourne un dict numpy.

    Parameters
    ----------
    filepath : str
    extra_fields : dict, optional
        Champs supplémentaires {nom: chemin_hdf5}.

    Returns
    -------
    dict {nom_variable: ndarray}
    """
    fields = {**HDF5_FIELDS, **(extra_fields or {})}
    data = {}
    with h5py.File(filepath, "r") as f:
        for name, path in fields.items():
            try:
                if name in ("frame_id", "start_time"):
                    data[name] = f[path][()]
                else:
                    data[name] = f[path][:]
            except KeyError:
                pass   # champ absent — ignoré silencieusement
    return data


def load_multi_orbits(filepaths,
                      extra_fields: dict | None = None) -> list[dict]:
    """Charge une liste de fichiers orbite avec gestion des erreurs.

    Parameters
    ----------
    filepaths : iterable de str ou Path
    extra_fields : dict, optional

    Returns
    -------
    list[dict]  orbites chargées avec succès, triées par orbite + start_time
    """
    orbites = []
    for fp in filepaths:
        try:
            data = load_orbit(str(fp), extra_fields=extra_fields)
            orbites.append(data)
            nom = data.get("nom_orbite", Path(fp).name)
            print(f"[load] {nom} — {Path(fp).name}")
        except Exception as e:
            print(f"[load] Erreur {fp} : {e}")

    # Tri par nom_orbite + frame_id + start_time
    orbites = _sort_orbits(orbites)
    print(f"[load] {len(orbites)} orbite(s) chargée(s) et triée(s)")
    return orbites


# ============================================================
# TRI DES ORBITES
# ============================================================

def _parse_start_time(val) -> datetime:
    """Convertit start_time (bytes ou str) en datetime."""
    if val is None:
        return datetime.min
    if isinstance(val, (bytes, np.bytes_)):
        val = val.decode("utf-8")
    val = str(val).strip().removeprefix("UTC=").rstrip("Z")
    try:
        return datetime.fromisoformat(val)
    except ValueError:
        return datetime.min


def _decode_frame(val) -> str:
    """Décode frame_id en str propre."""
    if val is None:
        return ""
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode().strip()
    s = str(val).strip()
    if s.startswith("b'") and s.endswith("'"):
        return s[2:-1]
    return s


def _decode_orbit_num(val) -> str:
    """Décode nom_orbite en str propre."""
    if val is None:
        return ""
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode().strip()
    return str(val).strip()


def _sort_key(orb: dict) -> tuple:
    """Clé de tri : (nom_orbite, frame_id, start_time).

    - nom_orbite  : ordre numérique des orbites
    - frame_id    : F < G < H (ordre de passage)
    - start_time  : secondaire, départage les doublons
    """
    orbit_num  = _decode_orbit_num(orb.get("nom_orbite", ""))
    frame      = _decode_frame(orb.get("frame_id", ""))
    start      = _parse_start_time(orb.get("start_time"))
    return (orbit_num, frame, start)


def _sort_orbits(orbites: list[dict]) -> list[dict]:
    """Trie une liste d'orbites par (nom_orbite, frame_id, start_time)."""
    return sorted(orbites, key=_sort_key)


# ============================================================
# UTILITAIRES
# ============================================================

def get_t0_utc(filepath: str, fallback=None) -> datetime:
    """Extrait l'heure UTC de début depuis l'en-tête HDF5."""
    try:
        with h5py.File(filepath, "r") as f:
            raw = f["HeaderData/VariableProductHeader/MainProductHeader/sensingStartTime"][()]
            s = raw.decode().replace("UTC=", "").replace("Z", "")
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    except Exception:
        return fallback or datetime(2025, 12, 30, 21, 50, 0)


def get_frame_id(orbit_data: dict) -> str:
    """Retourne la frame (F, G ou H) d'une orbite chargée."""
    return _decode_frame(orbit_data.get("frame_id"))


def group_by_orbit(orbites: list[dict]) -> dict[str, list[dict]]:
    """Regroupe les frames par numéro d'orbite.

    Retourne un dict {num_orbite: [frame_F, frame_G, frame_H]}
    utile pour traiter les trois frames d'une même orbite ensemble.

    Example
    -------
    >>> groups = group_by_orbit(orbites)
    >>> for num, frames in groups.items():
    ...     print(num, [get_frame_id(f) for f in frames])
    09039  ['F', 'G', 'H']
    09040  ['G', 'H']
    """
    groups: dict[str, list] = {}
    for orb in orbites:
        num = _decode_orbit_num(orb.get("nom_orbite", "unknown"))
        groups.setdefault(num, []).append(orb)
    return groups
    
#Integrer les donnees de la jaxa telechargées sur https://gportal.jaxa.jp/gpr/
def load_jaxa_orbits(directory: str,
                     extra_fields: dict | None = None) -> list[dict]:
    
    files = sorted(Path(directory).glob("*.h5"))
    if not files:
        print(f"Aucun fichier .h5 trouvé dans : {directory}")
        return []

    print(f"{len(files)} fichier(s) trouvé(s) dans {directory}")
    return load_multi_orbits(files, extra_fields=extra_fields)




def merge_orbit_sources(esa_orbits: list[dict],
                        jaxa_orbits: list[dict],
                        sort_by: str = "start_time") -> list[dict]:

    def _parse_start_time(val) -> datetime:
        if val is None:
            return datetime.min
        if isinstance(val, (bytes, np.bytes_)):
            val = val.decode("utf-8")
        val = val.removeprefix("UTC=")
        return datetime.fromisoformat(val)

    def _sort_key(orb):
        return _parse_start_time(orb.get(sort_by))

    merged = sorted(esa_orbits + jaxa_orbits, key=_sort_key)
    print(f"{len(esa_orbits)} orbites ESA  +  {len(jaxa_orbits)} orbites JAXA"
          f" {len(merged)} au total, triées par '{sort_by}'.")
    return merged