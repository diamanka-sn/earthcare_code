FILE_TYPE = "ACM_CLP_2B"
ORBIT_FRAME = "09039G"
DATE_START = "2025-12-01"
DATE_END = "2025-12-10"

DOWNLOAD_PERIODS = [
    ("2025-12-01", "2026-02-01"),
]

T_MIN=440
T_MAX=560

LAT_REF= -75.1
LON_REF= 123.35

TITLE_COLOR  = "#003399"
TITLE_WEIGHT = "bold"

CIRCLES_CONFIG = [
    {"radius": 500, "color": "#00FFF2", "label": "500 km",  "linestyle": "-"},
    {"radius": 1000, "color": "#F8F404", "label": "1000 km", "linestyle": "-"},
]

PARTICLE_LABEL = {
    0: "clear",
    1: "warm water",
    2: "Supercooled water",
    3: "3d ice",
    4: "2d plate",
    5: "Mix 3D ice and 2D plate",
    6: "liquid drizzle",
    7: "Mixed phase drizzle",
    8: "Rain",
    9: "snow",
    10: "water + liquid drizzle",
    11: "water + rain",
    12: "Mixed Phase",
    13: "unknown",
}

PARTICLE_COLORS = {
    0:"#ffffff",
    1:"#2196F3",
    2:"#0009b0",
    3:"#02ab24",
    4:"#9C27B0",
    5:"#7B1FA2",
    6:"#80DEEA",
    7:"#00ACC1",
    8:"#FF9890",
    9:"#B0BEC5",
    10:"#0288D1",
    11:"#E65100",
    12:"#b53c00",
    13:"#310202",
}

HDF5_FIELDS = {
    "particle_type": "ScienceData/Data/cloud_particle_type_cpr_atlid_msi_1km",
    "temperature": "ScienceData/Data/GRID_temperature_1km",
    "iwc": "ScienceData/Data/cloud_ice_content_1km",
    "iwp": "ScienceData/Data/cloud_ice_water_path_1km",
    "lwc": "ScienceData/Data/cloud_water_content_1km",
    "lwp": "ScienceData/Data/cloud_water_path_1km",
    "lat": "ScienceData/Geo/latitude",
    "lon": "ScienceData/Geo/longitude",
    "height": "ScienceData/Geo/height",
    "time": "ScienceData/Geo/time",
    "surface_elevation": "ScienceData/Geo/surface_elevation",}

HDF5_FIELDS_ORBIT_META = {
    "nom_orbite": "HeaderData/VariableProductHeader/MainProductHeader/orbitNumber",
    "frame_id": "HeaderData/VariableProductHeader/MainProductHeader/frameID",
    "start_time": "HeaderData/VariableProductHeader/MainProductHeader/sensingStartTime"}

JAXA_DATA_DIR = "./data/jaxa"
GRID_CACHE = "./data/cache/grid_cache.nc"
GRID_CACHE_orbite = "./data/cache/grid_cache_orbites.nc"
ORBIT_CACHE = "./data/cache/orbite_cache.nc"
FORCE_REBUILD: bool = True



DEFAULT_DLAT =  1.0  
DEFAULT_DLON = 10.0  

GRID_PARAMS_1D = [
    "iwp",         
    "lwp",          
    "surface_elevation",
]
GRID_PARAMS_2D = [
    "iwc",           
    "lwc",           
    "temperature",    
    "particle_type",
]
ALL_PARAMS = GRID_PARAMS_1D + GRID_PARAMS_2D

RAW_PARAMS_1D = [
    "lat", "lon", "time",
    "lwp", "iwp",
    "surface_elevation",
]
RAW_PARAMS_2D = [
    "iwc", "lwc",
    "temperature",
    "particle_type",
    "height",
]