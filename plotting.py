from datetime import timedelta
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.cm as cm

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.geodesic as cgeo
import pandas as pd
from gridding import GridAccumulator
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

def plot_cloud_classification(d: dict, t_min: int, t_max: int) -> None:
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


def plot_temperature(d: dict, t_min: int, t_max: int) -> None:
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


def plot_temperature_and_classification(d: dict, t_min: int, t_max: int) -> None:
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
                            vmin=None, vmax=None, extra_twin=None) -> None:
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


def plot_ice_water_content(d: dict, t_min: int, t_max: int) -> None:
    _plot_water_content_2d(d, "iwc_plot", "Ice water content", "IWC /($g/m³$)",
                            t_min, t_max, vmin=0.0, vmax=0.20)


def plot_liquid_water_content(d: dict, t_min: int, t_max: int) -> None:
    _plot_water_content_2d(d, "lwc_plot", "Liquid water content", "LWC / ($g/m³$)",
                            t_min, t_max)


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


def plot_water_paths(d: dict, t_min: int, t_max: int) -> None:
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
        if day is not None:
            counts = grid.count(day)
        elif d1 is not None and d2 is not None:
            counts = grid.count_range(d1, d2)
        else:
            counts = grid.count_range(grid.dates[0], grid.dates[-1])

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
        if day is not None:
            counts = grid.count(day)
        elif d1 is not None and d2 is not None:
            counts = grid.count_range(d1, d2)
        else:
            counts = grid.count_range(grid.dates[0], grid.dates[-1])

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


