from datetime import timedelta
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.cm as cm

import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.geodesic as cgeo
import pandas as pd
from gridding import GridAccumulator
from processing import haversine
from config import (TITLE_COLOR, TITLE_WEIGHT, PARTICLE_COLORS, PARTICLE_LABEL,
    CIRCLES_CONFIG, LON_REF, LAT_REF, ORBIT_FRAME, DATE_END, DATE_START)


CMAP_PART   = mcolors.ListedColormap([PARTICLE_COLORS[i] for i in range(14)])
BOUNDS_PART = np.arange(-0.5, 14.5, 1)
NORM_PART   = mcolors.BoundaryNorm(BOUNDS_PART, CMAP_PART.N)

CMAP_TEMP = plt.cm.rainbow
NORM_TEMP = mcolors.Normalize(vmin=-48, vmax=-12)


# ============================================================
# HELPERS INTERNES
# ============================================================

def _add_time_labels(ax, t, t0_utc, local_times, t_min, t_max,
                     y_utc=-0.18, y_local=-0.25):
    idx_start = np.argmin(np.abs(t - t_min))
    idx_end   = np.argmin(np.abs(t - t_max))
    dt_start  = t0_utc + timedelta(seconds=t_min)
    dt_end    = t0_utc + timedelta(seconds=t_max)
    lst_start = local_times[idx_start]
    lst_end   = local_times[idx_end]

    kw = dict(transform=ax.transAxes, fontsize=10,
              color=TITLE_COLOR, fontweight=TITLE_WEIGHT)
    for y, left_dt, right_dt, center_label in [
        (y_utc,   dt_start,  dt_end,  "Time / UTC"),
        (y_local, lst_start, lst_end, "Time / Local"),
    ]:
        ax.text(0.0, y, left_dt.strftime("%H:%M:%S"),  ha="left",   **kw)
        ax.text(1.0, y, right_dt.strftime("%H:%M:%S"), ha="right",  **kw)
        ax.text(0.5, y, center_label,                  ha="center", **kw)


def _add_particle_legend(ax_leg, present_types):
    patches = [
        mpatches.Patch(facecolor=PARTICLE_COLORS[i], edgecolor="grey",
                       linewidth=0.5, label=PARTICLE_LABEL[i])
        for i in present_types
    ]
    ax_leg.axis("off")
    ax_leg.legend(handles=patches,
                  loc="upper center", bbox_to_anchor=(0.5, 0.85),
                  ncol=min(5, len(present_types)), fontsize=10,
                  frameon=True, edgecolor="grey",
                  title="cloud particle type", title_fontsize=9,
                  borderaxespad=4)


def _make_polar_map(header_text):
    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    fig.patch.set_facecolor("white")
    fig.text(0.02, 0.98, header_text, fontsize=7, va="top")

    ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)

    theta  = np.linspace(0, 2 * np.pi, 100)
    verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * 0.5 + [0.5, 0.5])
    ax.set_boundary(circle, transform=ax.transAxes)
    return fig, ax


def _setup_std_polar_ax(ax):
    """Gridlines standard + marqueur Dome C + cercles de distance."""
    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False,
                      y_inline=False, alpha=0.3,
                      ylocs=np.arange(-90, -55, 10))
    gl.ylabel_style = {"size": 10}
    gl.xlabel_style = {"size": 10}
    gl.top_labels   = False
    gl.right_label  = False

    ax.plot(LON_REF, LAT_REF, color="#FFA500", marker=".", markersize=5,
            linestyle="none", label="Dome C", transform=ccrs.PlateCarree())
    _add_distance_circles(ax)


def _add_distance_circles(ax):
    gd = cgeo.Geodesic()
    for cfg in CIRCLES_CONFIG:
        cp = gd.circle(lon=LON_REF, lat=LAT_REF,
                       radius=cfg["radius"] * 1000,
                       n_samples=100, endpoint=True)
        ax.plot(cp[:, 0], cp[:, 1],
                color=cfg["color"], linestyle=cfg["linestyle"],
                linewidth=2, transform=ccrs.PlateCarree(),
                zorder=10, label=f"Radius: {cfg['label']}")


def _fig_header(t_utc_start, t_utc_end, orbit_id=None):
    orbit_id = orbit_id or ORBIT_FRAME
    return (
        f"ECA_JXBA_ACM_CLP_2B_20251230T215005Z_20251230T234304Z_{orbit_id}.h5\n"
        f"From : {t_utc_start} to {t_utc_end}\norbit: {orbit_id}")


def _fill_orbit_inset(ax_i, d, t_min, t_max):
    """Remplit un axes SouthPolarStereo avec la position de l'orbite."""
    ax_i.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
    ax_i.add_feature(cfeature.LAND,      facecolor="lightgray")
    ax_i.add_feature(cfeature.OCEAN,     facecolor="aliceblue")
    ax_i.add_feature(cfeature.COASTLINE, linewidth=0.4)
    ax_i.gridlines(alpha=0.25, linewidth=0.4)

    theta = np.linspace(0, 2 * np.pi, 100)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    ax_i.set_boundary(mpath.Path(verts * 0.5 + [0.5, 0.5]),
                      transform=ax_i.transAxes)

    mask = (d["t"] >= t_min) & (d["t"] <= t_max)
    ax_i.scatter(d["lon"][mask], d["lat"][mask],
                 c="red", s=2, transform=ccrs.PlateCarree(), zorder=5)

    _add_distance_circles(ax_i)
    ax_i.plot(LON_REF, LAT_REF, color="#FFA500", marker="*", markersize=5,
              linestyle="none", transform=ccrs.PlateCarree(), zorder=10)
    ax_i.set_title("Orbit position", fontsize=7, color=TITLE_COLOR)


def _resolve_grid_range(grid, day, d1, d2):
    """Retourne (means, stds, n_orbits, label) selon les arguments fournis."""
    if day is not None:
        return (grid.mean(day), grid.std(day),
                grid._days[day]["_n_orbits"], day)
    if d1 is not None and d2 is not None:
        return (grid.mean_range(d1, d2), grid.std_range(d1, d2),
                grid.n_orbits_range(d1, d2), f"{d1} to {d2}")
    d1, d2 = grid.dates[0], grid.dates[-1]
    return (grid.mean_range(d1, d2), grid.std_range(d1, d2),
            grid.n_orbits, f"{d1} to {d2}")


# ============================================================
# PLOTS PROFIL UNIQUE (orbite)
# ============================================================

def plot_cloud_classification(d: dict, t_min: int, t_max: int,
                              scatter: bool = False) -> None:
    if scatter:
        fig = plt.figure(figsize=(20, 7))
        gs  = fig.add_gridspec(2, 2, height_ratios=[5, 1],
                               width_ratios=[4, 1], hspace=0.15, wspace=0.35)
        ax1    = fig.add_subplot(gs[0, 0])
        ax_leg = fig.add_subplot(gs[1, 0])
        ax_map = fig.add_subplot(gs[:, 1], projection=ccrs.SouthPolarStereo())
        _fill_orbit_inset(ax_map, d, t_min, t_max)
    else:
        fig, (ax1, ax_leg) = plt.subplots(
            2, 1, figsize=(16, 7),
            gridspec_kw={"height_ratios": [5, 1], "hspace": 0.15})

    fig.patch.set_facecolor("white")
    fig.text(0.02, 0.98, _fig_header(d["t_utc_start"], d["t_utc_end"]),
             fontsize=7, va="top")

    ax1.pcolormesh(d["T2D"], d["HGT"], d["data_plot"],
                   cmap=CMAP_PART, norm=NORM_PART, shading="auto")
    ax1.plot(d["t"], d["surface_elevation"],
             color="saddlebrown", linewidth=1.5, label="surface elevation")
    ax1.set_ylabel("Altitude/ m", fontsize=10)
    ax1.set_xlabel("time/ s", fontsize=9)
    ax1.set_title("Cloud classification", fontsize=13,
                  color=TITLE_COLOR, fontweight=TITLE_WEIGHT)
    ax1.set_ylim(3000, 6000)
    ax1.set_xlim(t_min, t_max)
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.7)

    _add_time_labels(ax1, d["t"], d["t0_utc"], d["local_times"],
                     t_min, t_max, y_utc=-0.15, y_local=-0.23)
    _add_particle_legend(ax_leg, d["present_type"])
    plt.show()


def plot_temperature(d: dict, t_min: int, t_max: int,
                     scatter: bool = False) -> None:
    if scatter:
        fig = plt.figure(figsize=(20, 5))
        gs  = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.35)
        ax     = fig.add_subplot(gs[0, 0])
        ax_map = fig.add_subplot(gs[0, 1], projection=ccrs.SouthPolarStereo())
        _fill_orbit_inset(ax_map, d, t_min, t_max)
    else:
        fig, ax = plt.subplots(figsize=(16, 5))

    fig.patch.set_facecolor("white")

    c  = ax.pcolormesh(d["T2D"], d["HGT"], np.ma.masked_invalid(d["temp_c"]),
                       cmap=CMAP_TEMP, norm=NORM_TEMP, shading="auto")
    cb = plt.colorbar(c, ax=ax, pad=0.01, aspect=25, shrink=0.95)
    cb.set_label("Temperature en °C", fontsize=8)
    ax.set_ylabel("Altitude/m", fontsize=10)
    ax.set_xlabel("Time/s", fontsize=9)
    ax.set_title("Temperature", fontsize=13, color=TITLE_COLOR, fontweight=TITLE_WEIGHT)
    ax.set_ylim(3000, 6000)
    ax.set_xlim(t_min, t_max)
    plt.show()


def _add_cloud_contours_global(ax, T2D, HGT, particle_type_raw,
                                linewidth=0.8, alpha=0.9):
    cloud_mask = np.where(particle_type_raw >= 1, 1.0, 0.0)
    ax.contour(T2D, HGT, cloud_mask, levels=[0.5],
               colors="black", linewidths=linewidth, alpha=alpha)


def _add_cloud_contours(ax, T2D, HGT, particle_type_raw, present_types,
                        linewidth=0.8, alpha=0.9):
    for ptype in present_types:
        binary = np.where(particle_type_raw == ptype, 1.0, 0.0)
        ax.contour(T2D, HGT, binary, levels=[0.5],
                   colors=[PARTICLE_COLORS[ptype]],
                   linewidths=linewidth, alpha=alpha)


def plot_temperature_and_classification(d: dict, t_min: int, t_max: int,
                                        scatter: bool = False) -> None:
    if scatter:
        fig = plt.figure(figsize=(20, 7))
        gs  = fig.add_gridspec(2, 2, height_ratios=[5, 1],
                               width_ratios=[4, 1], hspace=0.154, wspace=0.35)
        ax     = fig.add_subplot(gs[0, 0])
        ax_leg = fig.add_subplot(gs[1, 0])
        ax_map = fig.add_subplot(gs[:, 1], projection=ccrs.SouthPolarStereo())
        _fill_orbit_inset(ax_map, d, t_min, t_max)
    else:
        fig, (ax, ax_leg) = plt.subplots(
            2, 1, figsize=(16, 7),
            gridspec_kw={"height_ratios": [5, 1], "hspace": 0.154})

    fig.patch.set_facecolor("white")

    c  = ax.pcolormesh(d["T2D"], d["HGT"], np.ma.masked_invalid(d["temp_c"]),
                       cmap=CMAP_TEMP, norm=NORM_TEMP, shading="auto")
    cb = plt.colorbar(c, ax=ax, pad=0.01, aspect=25, shrink=0.95,
                      ticks=np.arange(-48, -11, 4))
    cb.set_label("Temperature °C", fontsize=8)

    _add_cloud_contours_global(ax, d["T2D"], d["HGT"], d["particle_type"])

    ax.set_ylabel("Altitude / m", fontsize=10)
    ax.set_xlabel("Time / s", fontsize=9)
    ax.set_title("Temperature and cloud classification",
                 fontsize=12, color=TITLE_COLOR, fontweight=TITLE_WEIGHT)
    ax.set_ylim(3000, 6000)
    ax.set_xlim(t_min, t_max)

    _add_time_labels(ax, d["t"], d["t0_utc"], d["local_times"],
                     t_min, t_max, y_utc=-0.20, y_local=-0.25)
    plt.show()


def _plot_water_content_2d(d: dict, data_key: str, title: str, cbar_label: str,
                            t_min: int, t_max: int,
                            vmin=None, vmax=None, extra_twin=None,
                            scatter: bool = False) -> None:
    if scatter:
        fig = plt.figure(figsize=(20, 7))
        gs  = fig.add_gridspec(2, 2, height_ratios=[5, 1],
                               width_ratios=[4, 1], hspace=0.15, wspace=0.35)
        ax1    = fig.add_subplot(gs[0, 0])
        ax_leg = fig.add_subplot(gs[1, 0])
        ax_map = fig.add_subplot(gs[:, 1], projection=ccrs.SouthPolarStereo())
        _fill_orbit_inset(ax_map, d, t_min, t_max)
    else:
        fig, (ax1, ax_leg) = plt.subplots(
            2, 1, figsize=(16, 7),
            gridspec_kw={"height_ratios": [5, 1], "hspace": 0.15})

    fig.patch.set_facecolor("white")

    kwargs = dict(cmap="rainbow", shading="auto")
    if vmin is not None:
        kwargs["vmin"] = vmin
    if vmax is not None:
        kwargs["vmax"] = vmax

    c1 = ax1.pcolormesh(d["T2D"], d["HGT"], d[data_key], **kwargs)
    ax1.plot(d["t"], d["surface_elevation"],
             color="saddlebrown", linewidth=1.5, label="surface elevation", zorder=5)

    cloud_mask = np.where(d["particle_type"] >= 1, 1.0, 0.0)
    contour    = ax1.contour(d["T2D"], d["HGT"], cloud_mask,
                              levels=[0.5], colors="black",
                              linewidths=0.8, alpha=0.9)
    ax1.set_ylabel("Altitude/m", fontsize=10)
    ax1.set_xlabel("time / s", fontsize=9)
    ax1.set_title(title, fontsize=13, color=TITLE_COLOR, fontweight=TITLE_WEIGHT)
    ax1.set_ylim(3000, 6000)
    ax1.set_xlim(t_min, t_max)

    if extra_twin:
        extra_twin(ax1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, _       = contour.legend_elements()
    ax1.legend(lines1 + lines2, labels1 + ["Clouds"], loc="upper right", fontsize=8)

    ax_leg.axis("off")
    cb = fig.colorbar(c1, ax=ax_leg, orientation="horizontal", fraction=0.5, pad=0.1)
    cb.set_label(cbar_label, fontsize=9)
    plt.show()


def plot_ice_water_content(d: dict, t_min: int, t_max: int,
                           scatter: bool = False) -> None:
    _plot_water_content_2d(d, "iwc_plot", "Ice water content", "IWC /($g/m³$)",
                            t_min, t_max, vmin=0.0, vmax=0.20, scatter=scatter)


def plot_liquid_water_content(d: dict, t_min: int, t_max: int,
                               scatter: bool = False) -> None:
    _plot_water_content_2d(d, "lwc_plot", "Liquid water content", "LWC / ($g/m³$)",
                            t_min, t_max, scatter=scatter)


def plot_lat_lon(d: dict, t_min: int, t_max: int) -> None:
    fig, ax = plt.subplots(figsize=(16, 4))
    fig.patch.set_facecolor("white")
    ax_r = ax.twinx()

    ax.plot(d["t"], d["lon"], color=TITLE_COLOR,   linewidth=1.5, label="Longitude")
    ax.axhline(y=LON_REF,    color=TITLE_COLOR,   linestyle="--", linewidth=1, alpha=0.6)
    ax_r.plot(d["t"], d["lat"], color="#FFA500",   linewidth=1.5, label="Latitude")
    ax_r.axhline(y=LAT_REF,    color="#FFA500",   linestyle="--", linewidth=1, alpha=0.6)

    ax.set_ylabel("Longitude / deg",  fontsize=9, color=TITLE_COLOR)
    ax_r.set_ylabel("Latitude / deg", fontsize=9, color="#FFA500")
    ax_r.tick_params(axis="y", labelcolor="#FFA500")
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(50, 140)
    ax.set_xlabel("Time / s", fontsize=10)
    ax.set_title("Latitude et longitude", fontsize=13,
                 color=TITLE_COLOR, fontweight=TITLE_WEIGHT)

    _add_time_labels(ax, d["t"], d["t0_utc"], d["local_times"],
                     t_min, t_max, y_utc=-0.20, y_local=-0.30)
    plt.show()


def plot_distance(d: dict) -> None:
    fig, ax = plt.subplots(figsize=(16, 4))
    fig.patch.set_facecolor("white")
    ax.plot(d["t"], d["distance"], color=TITLE_COLOR, linewidth=1.5)
    ax.set_ylabel("Distance / km", fontsize=10)
    ax.set_xlabel("Time / s",      fontsize=10)
    ax.set_title("Distance to Concordia", fontsize=13,
                 color=TITLE_COLOR, fontweight=TITLE_WEIGHT)
    ax.set_xlim(460, 580)
    ax.set_ylim(700, 1200)
    plt.show()


def plot_water_paths(d: dict, t_min: int, t_max: int,
                     scatter: bool = False) -> None:
    if scatter:
        fig = plt.figure(figsize=(20, 5))
        gs  = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.35)
        ax_lwp = fig.add_subplot(gs[0, 0])
        ax_map = fig.add_subplot(gs[0, 1], projection=ccrs.SouthPolarStereo())
        _fill_orbit_inset(ax_map, d, t_min, t_max)
    else:
        fig, ax_lwp = plt.subplots(figsize=(16, 5))

    fig.patch.set_facecolor("white")

    line1, = ax_lwp.plot(d["t"], d["lwp_plot"], color="red",
                          linewidth=1.5, label="Liquid Water Path (LWP)")
    ax_lwp.set_ylabel("LWP / $g/m²$", color="red", fontsize=10, fontweight=TITLE_WEIGHT)
    ax_lwp.tick_params(axis="y", labelcolor="red")
    ax_lwp.set_ylim(0, 80)
    ax_lwp.set_xlim(t_min, t_max)

    ax_iwp = ax_lwp.twinx()
    line2, = ax_iwp.plot(d["t"], d["iwp_plot"], color="blue",
                          linewidth=1.5, label="Ice Water Path (IWP)")
    ax_iwp.set_ylabel("IWP / $g/m²$", fontsize=10, color="blue", fontweight=TITLE_WEIGHT)
    ax_iwp.tick_params(axis="y", labelcolor="blue")
    ax_iwp.set_ylim(0, 120)
    ax_iwp.set_xlim(t_min, t_max)

    ax_lwp.set_xlabel("Time / s", fontsize=9)
    ax_lwp.legend([line1, line2], [line1.get_label(), line2.get_label()],
                   loc="upper right", fontsize=8)
    plt.show()


# ============================================================
# PLOTS MULTI-ORBITES (cartes polaires scatter)
# ============================================================

_SCATTER_META = {
    "lwp": ("Liquid Water Path", "LWP ($g/m²$)",      0,  40, "rainbow"),
    "iwp": ("Ice Water Path",    "IWP ($g/m²$)",       0,  40, "rainbow"),
    "surface_elevation": ("Surface elevation", "Elevation (m)", -500, 4000, "terrain"),
}


def plot_multi_orbit_scatter(orbites: list[dict], param: str = "lwp",
                              n_orbites_label: str = "",
                              vmin: float | None = None,
                              vmax: float | None = None,
                              cmap: str | None = None) -> None:
    """Carte polaire scatter pour lwp, iwp ou surface_elevation."""
    if param not in _SCATTER_META:
        raise ValueError(f"param doit être l'un de {list(_SCATTER_META)}")
    title, cbar_label, _vmin, _vmax, _cmap = _SCATTER_META[param]
    vmin = vmin if vmin is not None else _vmin
    vmax = vmax if vmax is not None else _vmax
    cmap = cmap or _cmap

    header = (f"ACM_CLP_2B — {title}\n"
              f"Orbits: {n_orbites_label}\n"
              f"Start date: {DATE_START}   End date: {DATE_END}")
    fig, ax = _make_polar_map(header)
    _setup_std_polar_ax(ax)

    sc = None
    for orb in orbites:
        values = orb.get(param)
        if values is None:
            continue
        sc = ax.scatter(orb["lon"], orb["lat"], c=values,
                        cmap=cmap, s=5,
                        transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)

    if sc is not None:
        cbar = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.7, pad=0.05)
        cbar.set_label(cbar_label, fontsize=10)

    ax.set_title(title, fontsize=10, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)
    ax.legend(loc="upper right")
    plt.show()


# Raccourcis nommés pour la compatibilité avec le notebook existant
def plot_multi_orbit_lwp(orbites, n_orbites_label=""):
    plot_multi_orbit_scatter(orbites, "lwp", n_orbites_label)

def plot_multi_orbit_iwp(orbites, n_orbites_label=""):
    plot_multi_orbit_scatter(orbites, "iwp", n_orbites_label)

def plot_multi_orbit_elevation(orbites, n_orbites_label="", vmin=-500, vmax=4000):
    plot_multi_orbit_scatter(orbites, "surface_elevation", n_orbites_label,
                              vmin=vmin, vmax=vmax)


def plot_polar_scatter(lon_arr, lat_arr, data_scatter, title, cbar_label,
                       t_utc_start, t_utc_end, orbit_id=None,
                       vmin=0, vmax=50, gridlines_labels=False,
                       cmap="rainbow") -> None:
    fig, ax = _make_polar_map(
        _fig_header(t_utc_start, t_utc_end, orbit_id) + "\n")

    if gridlines_labels:
        ax.gridlines(draw_labels=True, dms=True, x_inline=False,
                     y_inline=True, alpha=0.5)
    else:
        ax.gridlines()

    _add_distance_circles(ax)

    sc = ax.scatter(lon_arr, lat_arr, c=data_scatter, cmap=cmap, s=10,
                    transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
    ax.plot(LON_REF, LAT_REF, color="#FFA500", marker=".", markersize=5,
            linestyle="none", label="Dome C", transform=ccrs.PlateCarree())

    cbar = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.7, pad=0.05)
    cbar.set_label(cbar_label, fontsize=10)
    ax.set_title(title, fontsize=10, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)
    ax.legend(loc="upper right")
    plt.show()




# ============================================================
# PLOTS GRILLE (GridAccumulator)
# ============================================================

def _polar_grid_map(lon_bins, lat_bins, data_2d, title, cbar_label,
                    vmin=None, vmax=None, cmap="rainbow",
                    n_orbits=None, label=None, save_path=None):
    LON2D, LAT2D = np.meshgrid(lon_bins, lat_bins)

    subtitle = f"{len(lat_bins)}×{len(lon_bins)} cells"
    if n_orbits is not None:
        subtitle += f"  | {n_orbits} orbits \n{label}"

    fig, ax = _make_polar_map(subtitle)
    _setup_std_polar_ax(ax)

    pc   = ax.pcolormesh(LON2D, LAT2D, data_2d, cmap=cmap, vmin=vmin, vmax=vmax,
                         transform=ccrs.PlateCarree(), shading="auto")
    cbar = plt.colorbar(pc, ax=ax, orientation="vertical", shrink=0.7, pad=0.05)
    cbar.set_label(cbar_label, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)
    ax.legend(loc="upper right")
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_grid_mean(grid, param: str = "lwp",
                   cbar_label: str | None = None,
                   vmin=None, vmax=None,
                   day=None, d1=None, d2=None) -> None:
    means, _, n_orbits, label = _resolve_grid_range(grid, day, d1, d2)
    if param not in means:
        raise KeyError(f"Paramètre '{param}' absent. Disponibles : {list(means)}")
    _polar_grid_map(grid.lon_bins, grid.lat_bins,
                    means[param],
                    title=f"{param.upper()} - Mean ({label})",
                    cbar_label=cbar_label or f"{param.upper()} ($g/m²$)",
                    vmin=vmin, vmax=vmax,
                    label=label, n_orbits=n_orbits)


def plot_grid_std(grid, param: str = "lwp",
                  cbar_label: str | None = None,
                  vmin=0, vmax=50,
                  day=None, d1=None, d2=None) -> None:
    _, stds, n_orbits, label = _resolve_grid_range(grid, day, d1, d2)
    if param not in stds:
        raise KeyError(f"Paramètre '{param}' absent. Disponibles : {list(stds)}")
    _polar_grid_map(grid.lon_bins, grid.lat_bins,
                    stds[param],
                    title=f"{param.upper()} - std dev ({label})",
                    cbar_label=cbar_label or f"std {param}",
                    cmap="rainbow", vmin=vmin, vmax=vmax,
                    label=label, n_orbits=n_orbits)


def plot_grid_standard_error(grid, param: str = "lwp",
                             cbar_label: str | None = None,
                             vmin: float = 0, vmax: float | None = None,
                             day: str | None = None,
                             d1: str | None = None,
                             d2: str | None = None) -> None:
    """Carte polaire de l'erreur standard (std / √n) par cellule de grille."""
    _, stds, n_orbits, label = _resolve_grid_range(grid, day, d1, d2)

    counts = _get_counts(grid, day, d1, d2)

    if param not in stds:
        raise KeyError(f"Paramètre '{param}' absent. Disponibles : {list(stds)}")

    n = counts[param].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        std_err = np.where(n > 0, stds[param] / np.sqrt(n), np.nan)

    _polar_grid_map(
        grid.lon_bins, grid.lat_bins, std_err,
        title=f"{param.upper()} — Erreur standard (std / √n)  |  {label}",
        cbar_label=cbar_label or f"Erreur standard {param.upper()} (g/m²)",
        cmap="rainbow", vmin=vmin, vmax=vmax,
        label=label, n_orbits=n_orbits,
    )


def plot_grid_results(grid: GridAccumulator, day: str | None = None,
                      d1: str | None = None, d2: str | None = None):
    means, stds, n_orbits, label = _resolve_grid_range(grid, day, d1, d2)

    class _FakeGrid:
        pass
    fg          = _FakeGrid()
    fg.lon_bins = grid.lon_bins
    fg.lat_bins = grid.lat_bins
    fg.dlat     = grid.dlat
    fg.dlon     = grid.dlon
    fg.n_orbits = n_orbits
    fg.label    = label
    fg._means   = means
    fg._stds    = stds
    fg.mean     = lambda: means
    fg.std      = lambda: stds
    return fg


def plot_grid_mean_std_se(grid, param: str = "lwp",
                          cbar_label: str | None = None,
                          vmin_mean=None, vmax_mean=None, bounds_mean=None, log_mean=False, twoslope_mean=None,
                          vmin_std=None,  vmax_std=None,  bounds_std=None,  log_std=False,  twoslope_std=None,
                          vmin_se=None,   vmax_se=None,   bounds_se=None,   log_se=False,   twoslope_se=None,
                          vmin_n=None,    vmax_n=None,    bounds_n=None,    log_n=False,    twoslope_n=None,
                          day: str | None = None,
                          d1: str | None = None,
                          d2: str | None = None) -> None:
    """Affiche côte à côte la moyenne, l'écart-type, l'erreur sur la moyenne et le nombre d'observations.

    Priorité d'échelle par panneau : log_* > twoslope_* > bounds_* > vmin_*/vmax_* (linéaire).
    log_*=True              → LogNorm(vmin_* or 0.1, vmax_*)
    twoslope_*=(v0, vc, v1) → TwoSlopeNorm : linéaire [v0→vc] puis linéaire [vc→v1]
    bounds_*=[...]          → BoundaryNorm avec bornes libres
    """
    means, stds, n_orbits, label = _resolve_grid_range(grid, day, d1, d2)
    counts = _get_counts(grid, day, d1, d2)

    if param not in means:
        raise KeyError(f"Paramètre '{param}' absent. Disponibles : {list(means)}")

    n = counts[param].astype(float)
    n_plot = np.where(n > 0, n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        std_err = np.where(n > 0, stds[param] / np.sqrt(n), np.nan)

    LON2D, LAT2D = np.meshgrid(grid.lon_bins, grid.lat_bins)
    proj = ccrs.SouthPolarStereo()

    fig = plt.figure(figsize=(28, 7))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"{param.upper()}  |  {n_orbits} orbites  |  "
        f"grille {grid.dlat}°×{grid.dlon}°\n{label}",
        fontsize=12, fontweight=TITLE_WEIGHT, color=TITLE_COLOR,
    )

    theta  = np.linspace(0, 2 * np.pi, 100)
    verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * 0.5 + [0.5, 0.5])

    base_label = cbar_label or f"{param.upper()} (g/m²)"
    panels = [
        (means[param], f"Mean ({base_label})",      vmin_mean, vmax_mean, bounds_mean, log_mean, twoslope_mean, f"{param.upper()} — Mean"),
        (stds[param],  f"Std dev ({base_label})",   vmin_std,  vmax_std,  bounds_std,  log_std,  twoslope_std,  f"{param.upper()} — Std dev"),
        (std_err,      f"Std error ({base_label})", vmin_se,   vmax_se,   bounds_se,   log_se,   twoslope_se,   f"{param.upper()} — Std error"),
        (n_plot,       "Number of observations",    vmin_n,    vmax_n,    bounds_n,    log_n,    twoslope_n,    f"{param.upper()} — Nb observations"),
    ]

    for col, (data, cb_label, vmin, vmax, bounds, use_log, twoslope, title) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 4, col, projection=proj)
        ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.set_boundary(circle, transform=ax.transAxes)
        ax.gridlines(alpha=0.3)

        cmap = plt.get_cmap("rainbow")
        if use_log:
            norm = mcolors.LogNorm(vmin=vmin or 0.1, vmax=vmax)
            pc = ax.pcolormesh(LON2D, LAT2D, data, cmap=cmap, norm=norm,
                               transform=ccrs.PlateCarree(), shading="auto")
            plt.colorbar(pc, ax=ax, orientation="horizontal",
                         shrink=0.8, pad=0.04, label=cb_label)
        elif twoslope is not None:
            v0, vc, v1 = twoslope
            norm = mcolors.TwoSlopeNorm(vmin=v0, vcenter=vc, vmax=v1)
            pc = ax.pcolormesh(LON2D, LAT2D, data, cmap=cmap, norm=norm,
                               transform=ccrs.PlateCarree(), shading="auto")
            cb = plt.colorbar(pc, ax=ax, orientation="horizontal",
                              shrink=0.8, pad=0.04, label=cb_label)
            cb.set_ticks([v0, vc, v1])
        elif bounds is not None:
            norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N, extend="max")
            pc = ax.pcolormesh(LON2D, LAT2D, data, cmap=cmap, norm=norm,
                               transform=ccrs.PlateCarree(), shading="auto")
            cb = plt.colorbar(pc, ax=ax, orientation="horizontal",
                              shrink=0.8, pad=0.04, label=cb_label)
            cb.set_ticks(bounds)
        else:
            pc = ax.pcolormesh(LON2D, LAT2D, data, cmap=cmap,
                               vmin=vmin, vmax=vmax,
                               transform=ccrs.PlateCarree(), shading="auto")
            plt.colorbar(pc, ax=ax, orientation="horizontal",
                         shrink=0.8, pad=0.04, label=cb_label)

        ax.plot(LON_REF, LAT_REF, color="#FFA500", marker=".", markersize=5,
                linestyle="none", transform=ccrs.PlateCarree())
        _add_distance_circles(ax)
        ax.set_title(title, fontsize=10, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)

    plt.tight_layout()
    plt.show()


def plot_grid_lwp_iwp(grid, day=None, d1=None, d2=None) -> None:
    means, stds, n_orbits, label = _resolve_grid_range(grid, day, d1, d2)
    LON2D, LAT2D = np.meshgrid(grid.lon_bins, grid.lat_bins)

    proj = ccrs.SouthPolarStereo()
    fig  = plt.figure(figsize=(20, 8))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"LWP and IWP  |  {n_orbits} orbits  |  "
        f"grid {grid.dlat}°×{grid.dlon}°\n{label}",
        fontsize=12, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)

    specs   = [("lwp", "LWP mean ($g/m²$)",  0,  20, "rainbow"),
               ("iwp", "IWP mean ($g/m²$)",  0, 100, "rainbow"),
               ("lwp", "LWP std ($g/m²$)",   0,  20, "rainbow"),
               ("iwp", "IWP std ($g/m²$)",   0,  50, "rainbow")]
    sources = [means["lwp"], means["iwp"], stds["lwp"], stds["iwp"]]
    labels  = ["LWP mean", "IWP mean", "LWP std", "IWP std"]

    theta  = np.linspace(0, 2 * np.pi, 100)
    verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * 0.5 + [0.5, 0.5])

    for col, (data, (_, cbar_label, vmin, vmax, cmap), lbl) in enumerate(
            zip(sources, specs, labels)):
        ax = fig.add_subplot(1, 4, col + 1, projection=proj)
        ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.set_boundary(circle, transform=ax.transAxes)
        ax.gridlines(alpha=0.3)

        pc = ax.pcolormesh(LON2D, LAT2D, data, cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(), shading="auto")
        ax.plot(LON_REF, LAT_REF, color="#FFA500", marker=".", markersize=5,
                linestyle="none", transform=ccrs.PlateCarree())
        _add_distance_circles(ax)

        plt.colorbar(pc, ax=ax, orientation="horizontal",
                     shrink=0.8, pad=0.04, label=cbar_label)
        ax.set_title(lbl, fontsize=10, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)

    plt.tight_layout()
    plt.show()


def _get_counts(grid, day, d1, d2) -> dict:
    """Retourne le dict de counts pour le jour ou la plage demandée."""
    if day is not None:
        return grid.count(day)
    if d1 is not None and d2 is not None:
        return grid.count_range(d1, d2)
    return grid.count_range(grid.dates[0], grid.dates[-1])


def plot_grid_count(grid_or_dict,
                    param: str = "lwp", day: str | None = None,
                    d1: str | None = None, d2: str | None = None,
                    save_path: str | None = None) -> None:
    if isinstance(grid_or_dict, dict):
        counts   = grid_or_dict
        lon_bins = lat_bins = n_orbits = None
        label    = param
    else:
        grid     = grid_or_dict
        lon_bins = grid.lon_bins
        lat_bins = grid.lat_bins
        _, _, n_orbits, label = _resolve_grid_range(grid, day, d1, d2)
        counts = _get_counts(grid, day, d1, d2)

    if param not in counts:
        raise KeyError(f"Paramètre '{param}' absent des counts.")

    data = counts[param].astype(float)
    data = np.where(data == 0, np.nan, data)

    _polar_grid_map(lon_bins, lat_bins, data,
                    title=f"{param.upper()} - Number of observations per bin  ({label})",
                    cbar_label="Number of observations",
                    cmap="rainbow", n_orbits=n_orbits, label=label,
                    save_path=save_path)


def plot_grid_count_histogram(grid_or_dict, param: str = "lwp",
                               day: str | None = None,
                               d1: str | None = None,
                               d2: str | None = None) -> None:
    if isinstance(grid_or_dict, dict):
        counts = grid_or_dict
        label  = param
    else:
        grid = grid_or_dict
        _, _, _, label = _resolve_grid_range(grid, day, d1, d2)
        counts = _get_counts(grid, day, d1, d2)

    if param not in counts:
        raise KeyError(f"Paramètre '{param}' absent des counts.")

    data = counts[param].ravel()
    data = data[data > 0]

    if len(data) == 0:
        print("[plot] Aucune cellule avec des données.")
        return

    fig, ax1 = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"{param.upper()} - number of observations per bins  ({label})",
                 fontsize=12, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)

    ax1.hist(data, bins=40, color=TITLE_COLOR, edgecolor="white",
             linewidth=0.3, alpha=0.85)
    ax1.set_xlabel("Number of observations", fontsize=10)
    ax1.set_ylabel("number of cells",        fontsize=10)
    ax1.set_title("Echelle", fontsize=10)

    stats = pd.Series(data)
    txt   = (f"n cells = {len(data)}\n"
             f"min  = {int(stats.min())}\n"
             f"max  = {int(stats.max())}\n"
             f"mean = {stats.mean():.1f}\n"
             f"med  = {stats.median():.1f}")
    ax1.text(0.97, 0.97, txt, transform=ax1.transAxes,
             va="top", ha="right", fontsize=8,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    plt.tight_layout()
    plt.show()


def orbites_above_threshold(orbites: list[dict], param: str = "lwp",
                             threshold: float = 100.0, plot: bool = True) -> list[dict]:
    series = {}
    for orb in orbites:
        values = orb.get(param)
        if values is None:
            continue
        col_id      = orb.get("orbit_id") or orb.get("nom_orbite", "unknown")
        series[col_id] = pd.Series(np.asarray(values, dtype=float))

    if not series:
        print(f"Aucune donnée pour {param}")
        return []

    df         = pd.DataFrame(series)
    cols_above = df.columns[(df > threshold).any()].tolist()

    print(f"{param.upper()} > {threshold} g/m²")
    print(f"  {len(cols_above)}/{len(orbites)} orbites concernées :")
    for col in cols_above:
        n_above  = int((df[col] > threshold).sum())
        max_val  = float(df[col].max())
        orb_info = next((o for o in orbites
                         if (o.get("orbit_id") or o.get("nom_orbite")) == col), {})
        print(f"  {col}  date: {orb_info.get('start_day', '')}  "
              f"max={max_val:.1f}  n>{threshold}: {n_above}")

    if not cols_above:
        print("Aucune orbite ne dépasse ce seuil.")
        return []

    filtered = [orb for orb in orbites
                if (orb.get("orbit_id") or orb.get("nom_orbite")) in cols_above]

    if plot:
        LABELS    = {"lwp": ("Liquid Water Path", "LWP ($g/m^2$)"),
                     "iwp": ("Ice Water Path",    "IWP ($g/m^2$)")}
        long_name, cbar_label = LABELS.get(param, (param.upper(), param))
        vmax_plot = float(df[cols_above].max().max())

        header = (f"ACM_CLP\n{param.upper()} > {threshold} g/m²  "
                  f"({len(filtered)}/{len(orbites)} orbites)")
        fig, ax = _make_polar_map(header)
        _setup_std_polar_ax(ax)

        for orb in orbites:
            ax.scatter(orb["lon"], orb["lat"],
                       color="lightgrey", s=2, alpha=0.4,
                       transform=ccrs.PlateCarree(), zorder=1)

        sc = None
        for orb in filtered:
            values = np.asarray(orb.get(param, []), dtype=float)
            values = np.where(values < 0,         np.nan, values)
            values = np.where(values <= threshold, np.nan, values)
            mask   = ~np.isnan(values)
            sc     = ax.scatter(orb["lon"][mask], orb["lat"][mask],
                                c=values[mask], cmap="rainbow", s=6,
                                vmin=threshold, vmax=vmax_plot,
                                transform=ccrs.PlateCarree(), zorder=5)

        if sc is not None:
            cbar = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.7, pad=0.05)
            cbar.set_label(cbar_label, fontsize=10)
            cbar.ax.axhline(y=threshold, color="red", linewidth=1.5, linestyle="--")

        ax.set_title(f"{long_name} — orbites avec {param.upper()} > {threshold} g/m²",
                     fontsize=10, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)
        ax.legend(loc="upper right")
        plt.show()

    return filtered


# ============================================================
# ÉVOLUTION TEMPORELLE LWP / IWP PAR RAYON (grille journalière)
# ============================================================

_RADIUS_STYLE = [
    {"radius": 500,  "color": "#1565C0", "linestyle": "-",  "label": "R = 500 km"},
    {"radius": 1000, "color": "#C62828", "linestyle": "--", "label": "R = 1000 km"},
]


def _grid_distance_mask(grid, radius_km: float) -> np.ndarray:
    """Masque booléen (n_lat × n_lon) des cellules à ≤ radius_km du Dôme C."""
    LAT2D, LON2D = np.meshgrid(grid.lat_bins, grid.lon_bins, indexing="ij")
    return haversine(LAT2D, LON2D, LAT_REF, LON_REF) <= radius_km


def plot_temporal_lwp_iwp(grid, d1: str | None = None, d2: str | None = None,
                          mode: str = "jour") -> None:
    """
    Évolution temporelle du LWP et IWP moyennés spatialement dans les cercles
    de 500 km et 1000 km autour du Dôme C.

    Paramètres
    ----------
    grid : GridAccumulator  grille chargée avec GridAccumulator.load()
    d1   : str | None       début de la période (YYYY-MM-DD), défaut = premier jour
    d2   : str | None       fin   de la période (YYYY-MM-DD), défaut = dernier jour
    mode : str              "jour"  → un tick par jour, labels "DD Mon YYYY"
                            "heure" → ticks toutes les 6 h, labels "DD Mon HHh"
    """
    if mode not in ("jour", "heure"):
        raise ValueError(f"mode doit être 'jour' ou 'heure', reçu : '{mode}'")
    d1 = d1 or grid.dates[0]
    d2 = d2 or grid.dates[-1]
    days = [d for d in grid.dates if d1 <= d <= d2]
    if not days:
        raise ValueError(f"Aucun jour entre {d1} et {d2}. Disponibles : {grid.dates}")

    masks = {cfg["radius"]: _grid_distance_mask(grid, cfg["radius"])
             for cfg in _RADIUS_STYLE}

    # Séries temporelles : {radius: {param: [valeur par jour]}}
    series: dict[int, dict[str, list]] = {
        cfg["radius"]: {"lwp": [], "iwp": []} for cfg in _RADIUS_STYLE
    }
    dates_dt = [pd.Timestamp(d) for d in days]

    for day in days:
        means = grid.mean(day)
        for cfg in _RADIUS_STYLE:
            r = cfg["radius"]
            mask = masks[r]
            for param in ("lwp", "iwp"):
                vals = means[param][mask]
                series[r][param].append(np.nanmean(vals) if np.any(~np.isnan(vals)) else np.nan)

    # Conversion g/m² (données brutes en kg/m², facteur ×1000)
    # — vérification : si les valeurs sont déjà en g/m² on ne multiplie pas
    sample_lwp = np.nanmax([v for cfg in _RADIUS_STYLE
                            for v in series[cfg["radius"]]["lwp"] if not np.isnan(v)] or [0])
    scale = 1000.0 if sample_lwp < 1.0 else 1.0
    unit  = "g/m²"

    # Figure
    fig, (ax_lwp, ax_iwp) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"hspace": 0.06}
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"EarthCARE ACM_CLP_2B — LWP & IWP moyens autour du Dôme C\n"
        f"Période : {d1} → {d2}  |  grille {grid.dlat}°×{grid.dlon}°",
        fontsize=12, fontweight=TITLE_WEIGHT, color=TITLE_COLOR
    )

    for cfg in _RADIUS_STYLE:
        r   = cfg["radius"]
        kw  = dict(color=cfg["color"], linestyle=cfg["linestyle"],
                   linewidth=1.8, marker="o", markersize=4, label=cfg["label"])
        lwp_vals = np.array(series[r]["lwp"]) * scale
        iwp_vals = np.array(series[r]["iwp"]) * scale
        ax_lwp.plot(dates_dt, lwp_vals, **kw)
        ax_iwp.plot(dates_dt, iwp_vals, **kw)

    for ax, ylabel, title in [
        (ax_lwp, f"LWP moyen ({unit})", "Liquid Water Path"),
        (ax_iwp, f"IWP moyen ({unit})", "Ice Water Path"),
    ]:
        ax.set_ylabel(ylabel, fontsize=10, color=TITLE_COLOR, fontweight=TITLE_WEIGHT)
        ax.set_title(title, fontsize=11, color=TITLE_COLOR, fontweight=TITLE_WEIGHT, loc="left")
        ax.legend(fontsize=9, framealpha=0.8)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.set_ylim(bottom=0)
        ax.axhline(0, color="grey", linewidth=0.5)

    # Axe temporel selon le mode choisi
    n_days = len(days)
    if mode == "heure":
        interval_h = max(6, (n_days * 24) // 20 // 6 * 6)   # ~20 labels max, multiple de 6
        ax_iwp.xaxis.set_major_locator(mdates.HourLocator(interval=interval_h))
        ax_iwp.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%Hh UTC"))
        ax_iwp.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
        ax_iwp.set_xlabel("Heure (UTC)", fontsize=10)
    else:  # "jour"
        interval_d = max(1, n_days // 12)
        ax_iwp.xaxis.set_major_locator(mdates.DayLocator(interval=interval_d))
        ax_iwp.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%Y"))
        ax_iwp.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
        ax_iwp.set_xlabel("Date (UTC)", fontsize=10)
    plt.setp(ax_iwp.xaxis.get_majorticklabels(), ha="center", fontsize=9)

    plt.tight_layout()
    plt.show()


# ============================================================
# Série temporelle LWP dans un cercle autour d'un point de référence
# ============================================================

def plot_lwp_timeseries_orbites(orbites: list,
                                 radius_km: float = 500.0,
                                 vmax: float | None = None) -> None:
    """LWP vs temps depuis la liste orbites déjà en mémoire.

    Parameters
    ----------
    orbites   : list[dict]  sortie de prepare_multi_orbits()
    radius_km : float       rayon de sélection en km autour du Dôme C
    vmax      : float       limite haute de l'axe Y (g/m²), None = auto
    """
    from datetime import timedelta as _td, datetime as _dt
    from processing import haversine as _hav

    _EPOCH = _dt(1970, 1, 1)

    def _parse_t0(orb):
        # t0_utc présent (load_raw_orbits)
        t0 = orb.get("t0_utc")
        if isinstance(t0, _dt):
            return t0
        # Extraire la chaîne depuis start_day ou start_time
        s = orb.get("start_day") or ""
        if not s:
            st = orb.get("start_time")
            if st is not None:
                s = st.decode() if isinstance(st, bytes) else str(st)
        s = s.replace("UTC=", "").replace("Z", "").strip()[:19]
        # Accepter "2025-12-05T11:13:38" et "2025-12-05 11:13:38"
        s = s.replace("T", " ")
        try:
            return _dt.strptime(s, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None

    all_times, all_lwp, all_dist = [], [], []

    for orb in orbites:
        t0 = _parse_t0(orb)
        if t0 is None:
            continue
        lat  = np.asarray(orb["lat"],  dtype=np.float64)
        lon  = np.asarray(orb["lon"],  dtype=np.float64)
        lwp  = np.asarray(orb["lwp"],  dtype=np.float64)
        time_rel = np.asarray(orb.get("time", np.zeros(len(lat))), dtype=np.float64)

        dist = _hav(lat, lon, LAT_REF, LON_REF)
        mask = dist <= radius_km
        if not np.any(mask):
            continue

        t0_unix = (t0 - _EPOCH).total_seconds()
        times_abs = [_EPOCH + _td(seconds=float(t0_unix + tr)) for tr in time_rel[mask]]

        all_times.extend(times_abs)
        all_lwp.append(np.where(lwp[mask] < 0, np.nan, lwp[mask]))
        all_dist.append(dist[mask])

    if not all_times:
        print(f"Aucun point dans le rayon {radius_km} km.")
        return

    all_lwp  = np.concatenate(all_lwp)
    all_dist = np.concatenate(all_dist)
    order    = np.argsort([t.timestamp() for t in all_times])
    all_times = [all_times[i] for i in order]
    all_lwp   = all_lwp[order]
    all_dist  = all_dist[order]
    n_pts = len(all_times)

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor("white")

    sc = ax.scatter(all_times, all_lwp, s=6, c=all_dist, cmap="viridis_r",
                    vmin=0, vmax=radius_km, linewidths=0, alpha=0.8)
    fig.colorbar(sc, ax=ax, pad=0.01, label="Distance au Dôme C (km)").ax.tick_params(labelsize=8)

    ax.set_ylabel("LWP (g/m²)", fontsize=10)
    ax.set_xlabel("Date (UTC)", fontsize=10)
    if vmax is not None:
        ax.set_ylim(0, vmax)
    ax.grid(alpha=0.3)

    period = f"{all_times[0]:%Y-%m-%d} → {all_times[-1]:%Y-%m-%d}"
    ax.set_title(
        f"LWP — rayon {radius_km:.0f} km autour du Dôme C  |  "
        f"{n_pts} profils  |  {period}",
        fontsize=11, fontweight=TITLE_WEIGHT, color=TITLE_COLOR,
    )

    n_days = (all_times[-1] - all_times[0]).days + 1
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, n_days // 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%Y"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 12]))
    plt.tight_layout()
    plt.show()


def plot_lwp_timeseries(orbit_cache: str,
                        radius_km: float = 500.0,
                        d1: str | None = None,
                        d2: str | None = None,
                        vmax: float | None = None) -> None:
    """LWP vs temps pour tous les profils dans un cercle autour du Dôme C.

    Parameters
    ----------
    orbit_cache : str   chemin vers orbites_raw.nc
    radius_km   : float rayon de sélection en km
    d1, d2      : filtres de date "YYYY-MM-DD"
    vmax        : limite haute de l'axe Y (g/m²), None = auto
    """
    from processing import extract_circle_timeseries

    ts = extract_circle_timeseries(orbit_cache, radius_km=radius_km, d1=d1, d2=d2)

    times    = ts["time_dt"]
    lwp      = ts["lwp"]
    dist_km  = ts["dist_km"]
    n_pts    = len(times)

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor("white")

    sc = ax.scatter(times, lwp, s=6, c=dist_km, cmap="viridis_r",
                    vmin=0, vmax=radius_km, linewidths=0, alpha=0.8)
    cb = fig.colorbar(sc, ax=ax, pad=0.01, label="Distance au Dôme C (km)")
    cb.ax.tick_params(labelsize=8)

    ax.set_ylabel("LWP (g/m²)", fontsize=10)
    ax.set_xlabel("Date (UTC)", fontsize=10)
    if vmax is not None:
        ax.set_ylim(0, vmax)
    ax.grid(alpha=0.3)

    period = f"{times[0]:%Y-%m-%d} → {times[-1]:%Y-%m-%d}"
    ax.set_title(
        f"LWP — rayon {radius_km:.0f} km autour du Dôme C  |  "
        f"{n_pts} profils  |  {period}",
        fontsize=11, fontweight=TITLE_WEIGHT, color=TITLE_COLOR,
    )

    n_days = (times[-1] - times[0]).days + 1
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, n_days // 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%Y"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 12]))

    plt.tight_layout()
    plt.show()


# ============================================================
# Série temporelle dans un cercle autour d'un point de référence
# ============================================================

def plot_circle_timeseries(ts: dict,
                            lwp_bounds=None,   lwp_log=False,
                            iwp_bounds=None,   iwp_log=False,
                            iwc_bounds=None,   iwc_log=False,
                            lwc_bounds=None,   lwc_log=False,
                            temp_vmin=-50,     temp_vmax=-10) -> None:
    """Figure 5 panneaux : LWP, IWP (1D) + curtains IWC, LWC, Température.

    Parameters
    ----------
    ts : dict   sortie de extract_circle_timeseries()
    *_bounds    : liste de bornes pour BoundaryNorm (ex. [0,1,5,50])
    *_log       : True → LogNorm
    temp_vmin/vmax : plage linéaire de la température (°C)
    """
    from matplotlib.dates import date2num

    times = ts["time_dt"]
    t_num = np.array(date2num(times))

    lwp  = ts["lwp"]
    iwp  = ts["iwp"]
    iwc  = ts["iwc"]          # (n_pts × n_levels)
    lwc  = ts["lwc"]
    temp = ts["temperature"] - 273.15    # °C
    hgt  = ts["height"] / 1000.0        # km

    hgt_med = np.nanmedian(hgt, axis=0)  # altitude médiane par niveau

    radius_km = ts["radius_km"]
    n_pts     = len(times)

    fig, axes = plt.subplots(5, 1, figsize=(16, 18),
                             gridspec_kw={"height_ratios": [1, 1, 2, 2, 2]})
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Série temporelle — rayon {radius_km:.0f} km autour de "
        f"({ts['lat_ref']:.1f}°N, {ts['lon_ref']:.1f}°E)  |  {n_pts} profils\n"
        f"{times[0]:%Y-%m-%d} → {times[-1]:%Y-%m-%d}",
        fontsize=12, fontweight=TITLE_WEIGHT, color=TITLE_COLOR,
    )

    # ── panneau 1 : LWP ──────────────────────────────────────
    ax = axes[0]
    ax.scatter(times, lwp, s=4, c=ts["dist_km"], cmap="viridis_r",
               vmin=0, vmax=radius_km, linewidths=0, alpha=0.7)
    ax.set_ylabel("LWP (g/m²)", fontsize=9)
    ax.set_title("Liquid Water Path", fontsize=9, color=TITLE_COLOR)
    ax.grid(alpha=0.3)

    # ── panneau 2 : IWP ──────────────────────────────────────
    ax = axes[1]
    sc = ax.scatter(times, iwp, s=4, c=ts["dist_km"], cmap="viridis_r",
                    vmin=0, vmax=radius_km, linewidths=0, alpha=0.7)
    ax.set_ylabel("IWP (g/m²)", fontsize=9)
    ax.set_title("Ice Water Path", fontsize=9, color=TITLE_COLOR)
    ax.grid(alpha=0.3)
    cb = fig.colorbar(sc, ax=axes[:2], orientation="vertical",
                      pad=0.01, shrink=0.8, label="Distance au Dôme C (km)")
    cb.ax.tick_params(labelsize=8)

    # helper : norme pour curtain
    def _make_norm(data, bounds, use_log):
        pos = data[data > 0]
        vmin = float(np.nanpercentile(pos, 5))  if pos.size else 0.01
        vmax = float(np.nanpercentile(pos, 99)) if pos.size else 1.0
        if use_log:
            return mcolors.LogNorm(vmin=max(vmin, 1e-6), vmax=vmax)
        if bounds is not None:
            return mcolors.BoundaryNorm(boundaries=bounds, ncolors=256, extend="max")
        return mcolors.Normalize(vmin=vmin, vmax=vmax)

    # ── panneaux 3-5 : curtains ──────────────────────────────
    curtains = [
        (iwc,  "IWC (g/m³)",       iwc_bounds,  iwc_log,  "rainbow", 2),
        (lwc,  "LWC (g/m³)",       lwc_bounds,  lwc_log,  "rainbow", 3),
        (temp, "Température (°C)", None,         False,    "RdBu_r",  4),
    ]
    for data, ylabel, bounds, use_log, cmap_name, idx in curtains:
        ax = axes[idx]
        if idx == 4:
            norm = mcolors.Normalize(vmin=temp_vmin, vmax=temp_vmax)
        else:
            norm = _make_norm(data, bounds, use_log)

        T_mesh = np.tile(t_num[:, np.newaxis], (1, hgt_med.shape[0]))
        H_mesh = np.tile(hgt_med[np.newaxis, :], (len(t_num), 1))

        pc = ax.pcolormesh(T_mesh, H_mesh, data,
                           cmap=cmap_name, norm=norm,
                           shading="nearest", rasterized=True)
        ax.set_ylabel("Altitude (km)", fontsize=9)
        ax.set_title(ylabel, fontsize=9, color=TITLE_COLOR)
        cb2 = fig.colorbar(pc, ax=ax, orientation="vertical",
                           pad=0.01, shrink=0.95, label=ylabel)
        cb2.ax.tick_params(labelsize=8)
        if bounds is not None and not use_log:
            cb2.set_ticks(bounds)
        ax.xaxis_date()
        ax.grid(alpha=0.2)

    # ── formatage axe X ──────────────────────────────────────
    n_days = (times[-1] - times[0]).days + 1
    interval_d = max(1, n_days // 10)
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval_d))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%Y"))
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 12]))
        if ax is not axes[-1]:
            plt.setp(ax.get_xticklabels(), visible=False)
    axes[-1].set_xlabel("Date (UTC)", fontsize=10)

    plt.tight_layout()
    plt.show()


from datetime import datetime, timedelta
import matplotlib.dates as mdates
from processing import haversine

RADIUS_KM = 500
EPOCH_2000 = datetime(2000, 1, 1, 0, 0, 0)   # "seconds since 2000-1-1 00:00:00"

times_all, lwp_all, dist_all = [], [], []
for orb in orbites:
    lat  = np.asarray(orb["lat"],  dtype=float)
    lon  = np.asarray(orb["lon"],  dtype=float)
    lwp  = np.asarray(orb["lwp"],  dtype=float)
    time = np.asarray(orb["time"], dtype=np.float64)   # secondes depuis 2000-01-01

    dist = haversine(lat, lon, -75.1, 123.35)
    mask = dist <= RADIUS_KM
    if not np.any(mask):
        continue

    times_abs = [EPOCH_2000 + timedelta(seconds=float(t)) for t in time[mask]]
    times_all.extend(times_abs)
    lwp_all.append(np.where(lwp[mask] < 0, np.nan, lwp[mask]))
    dist_all.append(dist[mask])

lwp_all  = np.concatenate(lwp_all)
dist_all = np.concatenate(dist_all)

fig, ax = plt.subplots(figsize=(16, 5))
sc = ax.scatter(times_all, lwp_all, s=6, c=dist_all, cmap="viridis_r",
                vmin=0, vmax=RADIUS_KM, linewidths=0, alpha=0.8)
fig.colorbar(sc, ax=ax, label="Distance au Dôme C (km)")
ax.set_ylabel("LWP (g/m²)")
ax.set_xlabel("Date (UTC)")
ax.set_title(f"LWP vs temps — rayon {RADIUS_KM} km | {len(times_all)} profils")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%Y"))
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

df = pd.DataFrame({
    "time": times_all,
    "lwp":  lwp_all,          # LWP=0 inclus, NaN=invalide exclu
}).dropna(subset=["lwp"])     # retire uniquement les NaN (valeurs invalides)

df["hour_bin"] = df["time"].dt.floor("h")   # arrondi à l'heure inférieure

# ── Statistiques par bin horaire ──
stats = df.groupby("hour_bin")["lwp"].agg(
    mean="mean",
    std="std",
    n="count",
).reset_index()
stats["se"] = stats["std"] / np.sqrt(stats["n"])   # erreur sur la moyenne

# ── Figure ──
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
fig.patch.set_facecolor("white")
fig.suptitle(
    f"LWP horaire — rayon {RADIUS_KM} km autour du Dôme C (LWP=0 inclus)",
    fontsize=12, fontweight="bold", color="#003399"
)

labels = ["Moyenne (g/m²)", "Écart-type (g/m²)", "Erreur sur la moyenne (g/m²)"]
cols   = ["mean", "std", "se"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for ax, col, label, color in zip(axes, cols, labels, colors):
    ax.bar(stats["hour_bin"], stats[col], width=1/24, color=color,
           alpha=0.8, align="edge")
    ax.set_ylabel(label, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

n_days = (stats["hour_bin"].iloc[-1] - stats["hour_bin"].iloc[0]).days + 1
axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=max(1, n_days // 10)))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%Y"))
axes[-1].set_xlabel("Date (UTC)", fontsize=10)

plt.tight_layout()
plt.show()

print(f"Bins horaires : {len(stats)}")
print(stats[["hour_bin", "mean", "std", "se", "n"]].head(10).to_string(index=False))