from datetime import timedelta
import numpy as np
from config import LAT_REF, LON_REF
from pathlib import Path
import xarray as xr
from datetime import datetime

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))

def distance_to_ref(lat, lon):
    return haversine(lat, lon, LAT_REF, LON_REF)

def calculate_local_times(t0_utc, time_s, lon_arr):
    
    return [t0_utc + timedelta(seconds=float(s) + lon * 240)
        for s, lon in zip(time_s, lon_arr)]

def mask_negative(arr):
    return np.where(arr < 0, np.nan, arr)


def mask_particle_type(particle_type):
   
    data_masked = np.ma.masked_where(
        (particle_type < 0) | (particle_type > 13), particle_type
    )
    data_plot = np.ma.masked_where(data_masked == 0, data_masked)
    return data_masked, data_plot



def build_time_grid(time_s, height):
    t = time_s - time_s[0]
    T2D = np.tile(t[:, np.newaxis], (1, height.shape[1]))
    return t, T2D, height


def prepare_single_orbit(orbit_data: dict, t0_utc) -> dict:
    
    lat = orbit_data["lat"]
    lon = orbit_data["lon"]
    height = orbit_data["height"]
    time_s = orbit_data["time"]
    temperature =orbit_data["temperature"]
    particle_type= orbit_data["particle_type"]
    iwc= orbit_data["iwc"]
    lwc = orbit_data["lwc"]
    iwp= orbit_data["iwp"]
    lwp= orbit_data["lwp"]

    t, T2D, HGT = build_time_grid(time_s, height)
    _, data_plot = mask_particle_type(particle_type)

    local_times =calculate_local_times(t0_utc, t, lon)
    t_utc_start = (t0_utc + timedelta(seconds=float(t[0]))).strftime("%H:%M UTC")
    t_utc_end = (t0_utc + timedelta(seconds=float(t[-1]))).strftime("%H:%M UTC")

    present_type = sorted(
        [int(v) for v in np.unique(particle_type) if 1 <= int(v) <= 13]
    )
    return {
        **orbit_data,
        "t": t,
        "T2D": T2D,
        "HGT": HGT,
        "temp_c":temperature - 273.15,
        "data_plot": data_plot,
        "iwc_plot": np.ma.masked_where(iwc < 0, iwc),
        "lwc_plot":np.ma.masked_where(lwc < 0, lwc),
        "lwp_plot": mask_negative(lwp),
        "iwp_plot": mask_negative(iwp),
        "distance": distance_to_ref(lat, lon),
        "local_times":local_times,
        "t_utc_start": t_utc_start,
        "t_utc_end": t_utc_end,
        "present_type": present_type,
    }

def _orbit_id(data: dict) -> str:
    #Format : ORBITE_YYYYMMDD_HHMMSS
    num = data.get("nom_orbite", b"unknown")
    if isinstance(num, (bytes, np.bytes_)):
        num = num.decode()
    else:
        num = str(num).strip()

    fid = data.get("frame_id", b"G")
    if isinstance(fid, (bytes, np.bytes_)):
        fid = fid.decode()
    else:
        fid = str(fid).strip()
        if fid.startswith("b'") and fid.endswith("'"):
            fid = fid[2:-1]
            
    t0 = data.get("start_time")
    if t0 is not None:
        #date_str = t0.strftime("%Y%m%d_%H%M%S")
        date_str = t0.decode().replace("UTC=", "").replace("Z", ""),
    else:
        date_str = "unknowndate"

    return f"{num}{fid}_{date_str}"

def prepare_multi_orbits(raw_orbits: list[dict]) -> list[dict]:
    result = []
    for data in raw_orbits:
        t0   = data.get("start_time")
        elev = data.get("surface_elevation")

        def _arr(key):
            v = data.get(key)
            return np.asarray(v, dtype=float) if v is not None else None

        result.append({
            # Métadonnées — conservées en bytes pour compatibilité
            # avec save_raw_orbits() et GridAccumulator.accumulate()
            "orbit_id":   _orbit_id(data),
            "nom_orbite": data.get("nom_orbite", b"unknown"),
            "frame_id":   data.get("frame_id",   b"?"),
            "start_time": t0,
            "start_day":  t0.decode().replace("UTC=", "").replace("Z", "") if t0 else "",
            # Champs 1D
            "lat":               np.asarray(data["lat"], dtype=float),
            "lon":               np.asarray(data["lon"], dtype=float),
            "time":              _arr("time"),
            "lwp":               mask_negative(data["lwp"]),
            "iwp":               mask_negative(data["iwp"]),
            "surface_elevation": np.asarray(elev, dtype=float) if elev is not None else None,
            # Champs 2D
            "iwc":          _arr("iwc"),
            "lwc":          _arr("lwc"),
            "temperature":  _arr("temperature"),
            "particle_type":_arr("particle_type"),
            "height":       _arr("height"),
        })
    return result

def save_multi_orbits(orbites: list[dict], filepath: str) -> None:
   
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    datasets = []
    for orb in orbites:
        n = len(orb["lat"])
        elev = orb.get("surface_elevation")
        data_vars = {
            "lat": ("point", orb["lat"]),
            "lon": ("point", orb["lon"]),
            "lwp": ("point", orb["lwp"]),
            "iwp": ("point", orb["iwp"]),
        }
        if elev is not None:
            data_vars["surface_elevation"] = ("point", elev)
        ds = xr.Dataset(
            data_vars,
            coords={"point": np.arange(n)},
            attrs={
                "orbit_id":   orb["orbit_id"],
                "nom_orbite": orb["nom_orbite"],
                "start_day": orb["start_day"],
                "start_day": orb["start_day"],
            },
        )
        datasets.append(ds)

    combined = xr.concat(datasets, dim="orbite")
    combined["orbite"] = [orb["orbit_id"] for orb in orbites]

    #Metadonnees globales
    combined.attrs["n_orbites"] = len(orbites)
    combined.attrs["description"] = "Orbites EarthCARE ACM_CLP_2B preparees"
    combined.attrs["first_day"] = orbites[0]["start_day"] if orbites else ""
    combined.attrs["last_day"] = orbites[-1]["start_day"] if orbites else ""
    print(orb)
    combined["start_day"] = ("orbite", [orb["start_day"] for orb in orbites])
    combined["nom_orbite"] = ("orbite", [orb["nom_orbite"] for orb in orbites])

    combined.to_netcdf(filepath, mode="w")
    size_kb = Path(filepath).stat().st_size /1e3
    print(f"Sauvegarde dans {filepath}")
    print(f"{len(orbites)} orbites | {size_kb:.0f} KB")
    print(f"Periode : du {combined.attrs['first_day']}" f" au {combined.attrs['last_day']}")


def load_multi_orbits_nc(filepath: str) -> list[dict]:
    
    ds = xr.open_dataset(filepath)

    orbites = []
    for i in range(len(ds["orbite"])):
        orb_ds = ds.isel(orbite=i)

        orbit_id   = str(orb_ds["orbite"].values)
        nom_orbite = str(orb_ds["nom_orbite"].values)
        start_day = str(orb_ds["start_day"].values)
        start_time = str(orb_ds["start_time"].values)

        #Reconstruire t0_utc depuis start_day
        try:
            t0_utc = datetime.strptime(start_day, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            t0_utc = None

        elev = orb_ds["surface_elevation"].values.copy() if "surface_elevation" in orb_ds else None
        orbites.append({
            "orbit_id":   orbit_id,
            "nom_orbite": nom_orbite,
            "start_day": start_day,
            "t0_utc": t0_utc,
            "lat": orb_ds["lat"].values.copy(),
            "lon": orb_ds["lon"].values.copy(),
            "lwp": orb_ds["lwp"].values.copy(),
            "iwp": orb_ds["iwp"].values.copy(),
            "surface_elevation": elev,
        })

    ds.close()
    print(f"orbites Charge : {filepath}  ({len(orbites)} orbites)")
    if orbites:
        print(f"Periode : du  {orbites[0]['start_day']}"
              f" au {orbites[-1]['start_day']}")
    return orbites
