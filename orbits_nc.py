# ============================================================
# io_orbits_nc.py
# Sauvegarde et lecture des orbites brutes (toutes frames,
# ESA + JAXA) dans un fichier NetCDF4 intermédiaire.
#
# Workflow
# --------
#   1. Charger les HDF5 une seule fois
#   2. Sauvegarder dans orbites_raw.nc  (save_raw_orbits)
#   3. Pour le gridding, lire orbit par orbit (iter_raw_orbits)
#      sans tout charger en mémoire d'un coup
#
# Structure du .nc
# ----------------
# Dimensions : orbite (variable), point (taille max des traces)
# Une orbite = un groupe de variables indexées par orbite_id.
# Les traces de longueurs différentes sont paddées avec NaN.
# ============================================================

from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from config import RAW_PARAMS_1D, RAW_PARAMS_2D


# ============================================================
# SAUVEGARDE
# ============================================================

def save_raw_orbits(orbites: list[dict], filepath: str) -> None:
    """Sauvegarde toutes les orbites brutes dans un fichier NetCDF4.

    Chaque orbite est stockée avec ses paramètres 1D (lat, lon, lwp,
    iwp...) et 2D (temperature, iwc... avec dimension height).
    Les traces de longueurs différentes sont alignées par padding NaN.

    Parameters
    ----------
    orbites : list[dict]
        Sortie de load_multi_orbits() — dicts avec les clés HDF5_FIELDS
        + start_time, nom_orbite, frame_id, t0_utc.
    filepath : str
        Chemin du fichier .nc à créer.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Longueur max des traces (pour le padding)
    n_max  = max(len(orb["lat"]) for orb in orbites)
    n_orb  = len(orbites)

    # Hauteur max pour les paramètres 2D
    h_max = 0
    for orb in orbites:
        arr = orb.get("height")
        if arr is not None and arr.ndim == 2:
            h_max = max(h_max, arr.shape[1])

    print(f"[raw_nc] {n_orb} orbites | n_max={n_max} pts | h_max={h_max} niveaux")

    # --- Construire les arrays paddes ---
    orbit_ids  = []
    start_days = []
    frames     = []
    orbits_num = []

    data_1d = {p: np.full((n_orb, n_max), np.nan) for p in RAW_PARAMS_1D}
    data_2d = {p: np.full((n_orb, n_max, h_max), np.nan) for p in RAW_PARAMS_2D}

    for i, orb in enumerate(orbites):
        # Métadonnées
        orbit_ids.append(_orbit_id(orb))
        start_days.append(_start_day(orb))
        frames.append(_decode(orb.get("frame_id", b"?")))
        orbits_num.append(_decode(orb.get("nom_orbite", b"?")))

        n = len(orb["lat"])

        # Paramètres 1D
        for p in RAW_PARAMS_1D:
            arr = orb.get(p)
            if arr is not None:
                arr = np.asarray(arr, dtype=float).ravel()
                data_1d[p][i, :len(arr)] = arr

        # Paramètres 2D
        for p in RAW_PARAMS_2D:
            arr = orb.get(p)
            if arr is not None:
                arr = np.asarray(arr, dtype=float)
                if arr.ndim == 2:
                    nr, nc = arr.shape
                    data_2d[p][i, :nr, :nc] = arr
                elif arr.ndim == 1:
                    # height 1D (rare) → broadcast
                    data_2d[p][i, :len(arr), 0] = arr

    # --- Construire le Dataset ---
    ds = xr.Dataset(
        coords={
            "orbite":  ("orbite",  orbit_ids),
            "point":   ("point",   np.arange(n_max)),
            "level":   ("level",   np.arange(h_max)),
        },
        attrs={
            "title":       "EarthCARE ACM_CLP_2B raw orbits",
            "n_orbits":    n_orb,
            "date_start":  start_days[0]  if start_days else "",
            "date_end":    start_days[-1] if start_days else "",
            "conventions": "CF-1.8",
        },
    )

    # Variables de métadonnées
    ds["start_day"]  = ("orbite", start_days)
    ds["frame_id"]   = ("orbite", frames)
    ds["nom_orbite"] = ("orbite", orbits_num)

    # Variables 1D
    for p in RAW_PARAMS_1D:
        ds[p] = (["orbite", "point"], data_1d[p])

    # Variables 2D
    for p in RAW_PARAMS_2D:
        ds[p] = (["orbite", "point", "level"], data_2d[p])

    # Remove any existing file first — HDF5 can error when overwriting
    # a file that was left open/corrupted by a previous failed run.
    Path(filepath).unlink(missing_ok=True)
    ds.to_netcdf(filepath, mode="w")
    size_mb = Path(filepath).stat().st_size / 1e6
    print(f"[raw_nc] Sauvegardé → {filepath}  ({size_mb:.1f} MB)")
    if start_days:
        print(f"         Période : {start_days[0]} -> {start_days[-1]}")


# ============================================================
# LECTURE
# ============================================================

def load_raw_orbits(filepath: str) -> list[dict]:
    """Charge toutes les orbites depuis un fichier orbites_raw.nc.

    Retourne le même format que load_multi_orbits() — compatible
    avec GridAccumulator.accumulate() sans modification.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    list[dict]  avec les clés : orbit_id, nom_orbite, frame_id,
                start_time, t0_utc, lat, lon, lwp, iwp, iwc, lwc,
                temperature, particle_type, height, surface_elevation
    """
    ds = xr.open_dataset(filepath)
    n_orb = len(ds["orbite"])

    orbites = []
    for i in range(n_orb):
        orb_ds = ds.isel(orbite=i)

        orbit_id  = str(orb_ds["orbite"].values)
        start_day = str(orb_ds["start_day"].values)
        frame     = str(orb_ds["frame_id"].values)
        nom_orb   = str(orb_ds["nom_orbite"].values)

        # t0_utc depuis start_day
        t0_utc = _parse_datetime(start_day)

        # start_time comme bytes (compatibilité GridAccumulator)
        start_time_bytes = start_day.replace(" ", "T").encode()

        # Longueur réelle de la trace (premier NaN en lat)
        lat_full = orb_ds["lat"].values
        valid_mask = ~np.isnan(lat_full)
        n_valid = int(np.sum(valid_mask))

        orb = {
            "orbit_id":   orbit_id,
            "nom_orbite": nom_orb.encode(),
            "frame_id":   frame.encode(),
            "start_time": start_time_bytes,
            "t0_utc":     t0_utc,
        }

        # Paramètres 1D — tronquer au vrai n
        for p in RAW_PARAMS_1D:
            if p in orb_ds:
                arr = orb_ds[p].values[:n_valid]
                orb[p] = arr

        # Paramètres 2D — tronquer les lignes padding
        for p in RAW_PARAMS_2D:
            if p in orb_ds:
                arr = orb_ds[p].values[:n_valid, :]
                orb[p] = arr

        orbites.append(orb)

    ds.close()
    print(f"[raw_nc] Chargé ← {filepath}  ({len(orbites)} orbites)")
    if orbites:
        print(f"         {orbites[0]['start_time'].decode()} "
              f"-> {orbites[-1]['start_time'].decode()}")
    return orbites


def iter_raw_orbits(filepath: str):
    """Itère sur les orbites une par une sans tout charger en mémoire.

    Utile pour le gridding sur de nombreuses orbites.

    Yields
    ------
    dict  même format que load_raw_orbits() mais orbite par orbite
    """
    ds = xr.open_dataset(filepath)
    n_orb = len(ds["orbite"])

    for i in range(n_orb):
        orb_ds = ds.isel(orbite=i)

        start_day      = str(orb_ds["start_day"].values)
        frame          = str(orb_ds["frame_id"].values)
        nom_orb        = str(orb_ds["nom_orbite"].values)
        start_time_bytes = start_day.replace(" ", "T").encode()

        lat_full   = orb_ds["lat"].values
        valid_mask = ~np.isnan(lat_full)
        n_valid    = int(np.sum(valid_mask))

        orb = {
            "orbit_id":   str(orb_ds["orbite"].values),
            "nom_orbite": nom_orb.encode(),
            "frame_id":   frame.encode(),
            "start_time": start_time_bytes,
            "t0_utc":     _parse_datetime(start_day),
        }

        for p in RAW_PARAMS_1D:
            if p in orb_ds:
                orb[p] = orb_ds[p].values[:n_valid]

        for p in RAW_PARAMS_2D:
            if p in orb_ds:
                orb[p] = orb_ds[p].values[:n_valid, :]

        yield orb

    ds.close()


def describe_raw_nc(filepath: str) -> None:
    """Affiche un résumé du contenu d'un fichier orbites_raw.nc."""
    ds = xr.open_dataset(filepath)
    n    = int(ds.attrs.get("n_orbits", len(ds["orbite"])))
    d1   = ds.attrs.get("date_start", "?")
    d2   = ds.attrs.get("date_end",   "?")

    print(f"\n{'='*55}")
    print(f"  {Path(filepath).name}")
    print(f"  {n} orbites  |  {d1} -> {d2}")
    print(f"  Taille : {Path(filepath).stat().st_size/1e6:.1f} MB")
    print(f"{'='*55}")
    print("  Variables 1D  :", RAW_PARAMS_1D)
    print("  Variables 2D  :", RAW_PARAMS_2D)
    print(f"  Dimensions    : orbite={len(ds['orbite'])}, "
          f"point={len(ds['point'])}, level={len(ds['level'])}")

    # Frames présentes
    frames = list(ds["frame_id"].values)
    unique_frames = sorted(set(str(f) for f in frames))
    print(f"  Frames        : {unique_frames}")
    print(f"{'='*55}\n")
    ds.close()


# ============================================================
# HELPERS INTERNES
# ============================================================

def _decode(val) -> str:
    if val is None:
        return ""
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode().strip()
    s = str(val).strip()
    if s.startswith("b'") and s.endswith("'"):
        return s[2:-1]
    return s


def _start_day(orb: dict) -> str:
    """Extrait la date de début (YYYY-MM-DD HH:MM:SS) depuis start_time."""
    t0 = orb.get("start_time")
    if t0 is None:
        # Fallback : t0_utc
        utc = orb.get("t0_utc")
        return utc.strftime("%Y-%m-%d %H:%M:%S") if utc else "unknown"
    s = _decode(t0).replace("UTC=", "").rstrip("Z")
    # Normaliser : "2025-12-30T21:50:05" → "2025-12-30 21:50:05"
    return s.replace("T", " ")


def _orbit_id(orb: dict) -> str:
    """Construit un identifiant unique NUMFRAMEID_YYYYMMDD_HHMMSS."""
    num   = _decode(orb.get("nom_orbite", b"?"))
    frame = _decode(orb.get("frame_id",   b"?"))
    day   = _start_day(orb).replace("-", "").replace(" ", "_").replace(":", "")
    return f"{num}{frame}_{day}"


def _parse_datetime(s: str) -> datetime | None:
    """Convertit une chaîne date en datetime."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s[:len(fmt)], fmt)
        except ValueError:
            continue
    return None