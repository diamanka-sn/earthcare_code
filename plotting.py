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
import xarray as xr
from gridding import GridAccumulator
from config import (TITLE_COLOR, TITLE_WEIGHT,PARTICLE_COLORS, PARTICLE_LABEL,
    CIRCLES_CONFIG,LON_REF, LAT_REF,ORBIT_FRAME, DATE_END, DATE_START)


CMAP_PART  = mcolors.ListedColormap([PARTICLE_COLORS[i] for i in range(14)])
BOUNDS_PART = np.arange(-0.5, 14.5, 1)
NORM_PART  = mcolors.BoundaryNorm(BOUNDS_PART, CMAP_PART.N)

CMAP_TEMP  = plt.cm.rainbow
NORM_TEMP  = mcolors.Normalize(vmin=-48, vmax=-12)


def _add_time_labels(ax, t, t0_utc, local_times, t_min, t_max,
                     y_utc=-0.18, y_local=-0.25):
    idx_start = np.argmin(np.abs(t - t_min))
    idx_end = np.argmin(np.abs(t - t_max))
    dt_start = t0_utc + timedelta(seconds=t_min)
    dt_end = t0_utc + timedelta(seconds=t_max)
    lst_start = local_times[idx_start]
    lst_end = local_times[idx_end]

    kw = dict(transform=ax.transAxes, fontsize=10, color=TITLE_COLOR, fontweight=TITLE_WEIGHT)

    for y, left_dt, right_dt, center_label in [
        (y_utc, dt_start, dt_end,"Time / UTC"),
        (y_local, lst_start, lst_end, "Time / Local")]:
        ax.text(0.0, y, left_dt.strftime("%H:%M:%S"),  ha="left",**kw)
        ax.text(1.0, y, right_dt.strftime("%H:%M:%S"), ha="right", **kw)
        ax.text(0.5, y, center_label, ha="center", **kw)


def _add_particle_legend(ax_leg, present_types):
    patches = [
        mpatches.Patch(
            facecolor=PARTICLE_COLORS[i], edgecolor="grey",
            linewidth=0.5, label=PARTICLE_LABEL[i])
        for i in present_types ]
    ax_leg.axis("off")
    ax_leg.legend(handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.85),
        ncol=min(5, len(present_types)),
        fontsize=10,
        frameon=True,
        edgecolor="grey",
        title="cloud particle type",
        title_fontsize=9,
        borderaxespad=4)


def _make_polar_map(header_text):
   
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    fig.patch.set_facecolor("white")
    fig.text(0.02, 0.98, header_text, fontsize=7, va="top")

    ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)

    theta = np.linspace(0, 2 * np.pi, 100)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * 0.5 + [0.5, 0.5])
    ax.set_boundary(circle, transform=ax.transAxes)

    return fig, ax


def _add_distance_circles(ax):
    gd = cgeo.Geodesic()
    for cfg in CIRCLES_CONFIG:
        cp = gd.circle(
            lon=LON_REF, lat=LAT_REF,
            radius=cfg["radius"] * 1000,
            n_samples=100, endpoint=True )
        ax.plot(
            cp[:, 0], cp[:, 1],
            color=cfg["color"], linestyle=cfg["linestyle"],
            linewidth=2, transform=ccrs.PlateCarree(),
            zorder=10, label=f"Radius: {cfg['label']}")


def _fig_header(t_utc_start, t_utc_end, orbit_id=None):
    orbit_id = orbit_id or ORBIT_FRAME
    return (
        f"ECA_JXBA_ACM_CLP_2B_20251230T215005Z_20251230T234304Z_{orbit_id}.h5\n"
        f"From : {t_utc_start} to {t_utc_end}\norbit: {orbit_id}")

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

    c = ax.pcolormesh(d["T2D"], d["HGT"], np.ma.masked_invalid(d["temp_c"]),
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
    ax.contour(
            T2D, HGT, cloud_mask,
            levels=[0.5],
            colors="black",
            linewidths=linewidth,
            alpha=alpha)

def _add_cloud_contours(ax, T2D, HGT, particle_type_raw, present_types,
                        linewidth=0.8, alpha=0.9):
    
    for ptype in present_types:
        binary = np.where(particle_type_raw == ptype, 1.0, 0.0)
        ax.contour(
            T2D, HGT, binary,
            levels=[0.5],
            colors=[PARTICLE_COLORS[ptype]],
            linewidths=linewidth,
            alpha=alpha)
 
def plot_temperature_and_classification(d: dict, t_min: int, t_max: int) -> None:
    fig, (ax, ax_leg) = plt.subplots(
        2, 1, figsize=(16, 7),
        gridspec_kw={"height_ratios": [5, 1], "hspace": 0.154})
    fig.patch.set_facecolor("white")
 
    c = ax.pcolormesh(d["T2D"], d["HGT"], np.ma.masked_invalid(d["temp_c"]),
                      cmap=CMAP_TEMP, norm=NORM_TEMP, shading="auto")
    cb = plt.colorbar(c, ax=ax, pad=0.01, aspect=25, shrink=0.95,
                      ticks=np.arange(-48, -11, 4))
    cb.set_label("Temperature °C", fontsize=8)
 
    _add_cloud_contours_global(ax, d["T2D"], d["HGT"],
                        d["particle_type"])
 
    ax.set_ylabel("Altitude / m", fontsize=10)
    ax.set_xlabel("Time / s", fontsize=9)
    ax.set_title("Temperature and cloud classification",
                 fontsize=12, color=TITLE_COLOR, fontweight=TITLE_WEIGHT)
    ax.set_ylim(3000, 6000)
    ax.set_xlim(t_min, t_max)
 
    _add_time_labels(ax, d["t"], d["t0_utc"], d["local_times"],
                     t_min, t_max, y_utc=-0.20, y_local=-0.25)
   # _add_particle_legend(ax_leg, d["present_type"])
    plt.show()


def _plot_water_content_2d(d: dict, data_key: str, title: str, cbar_label: str,
                            t_min: int, t_max: int,
                            vmin=None, vmax=None, extra_twin=None) -> None:
    fig, (ax1, ax_leg) = plt.subplots(
        2, 1, figsize=(16, 7),
        gridspec_kw={"height_ratios": [5, 1], "hspace": 0.15},
    )
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
    contour = ax1.contour(
            d["T2D"], d["HGT"], cloud_mask,
            levels=[0.5],
            colors="black",
            linewidths=0.8,
            alpha=0.9)
    ax1.set_ylabel("Altitude/m", fontsize=10)
    ax1.set_xlabel("time / s", fontsize=9)
    ax1.set_title(title, fontsize=13, color=TITLE_COLOR, fontweight=TITLE_WEIGHT)
    ax1.set_ylim(3000, 6000)
    ax1.set_xlim(t_min, t_max)
    

    if extra_twin:
        extra_twin(ax1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, l= contour.legend_elements()

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

    ax.plot(d["t"], d["lon"], color=TITLE_COLOR, linewidth=1.5, label="Longitude")
    ax.axhline(y=LON_REF, color=TITLE_COLOR, linestyle="--", linewidth=1, alpha=0.6)
    ax_r.plot(d["t"], d["lat"], color="#FFA500", linewidth=1.5, label="Latitude")
    ax_r.axhline(y=LAT_REF, color="#FFA500", linestyle="--", linewidth=1, alpha=0.6)

    ax.set_ylabel("Longitude / deg", fontsize=9, color=TITLE_COLOR)
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
    ax.set_xlabel("Time / s", fontsize=10)
    ax.set_title("Distance to Concordia", fontsize=13,color=TITLE_COLOR, fontweight=TITLE_WEIGHT)
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


def plot_multi_orbit_lwp(orbites: list[dict], n_orbites_label: str) -> None:
    fig, ax = _make_polar_map(
        f"ACM_CLP_2B\nOrbits: {n_orbites_label}\nStart date: {DATE_START} \nEnd Date: {DATE_END}")

    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False,ylocs=np.arange(-90, -55, 10))
    gl.ylabel_style = {"size": 10}
    gl.xlabel_style = {"size": 10}
    gl.top_labels   = False
    gl.right_label  = False

    ax.plot(LON_REF, LAT_REF, color="#FFA500", marker=".", linestyle="none",
            markersize=5, label="Dome C", transform=ccrs.PlateCarree())

    sc_lwp = None
    for orb in orbites:
        sc_lwp = ax.scatter(orb["lon"], orb["lat"], c=orb["lwp"],cmap="rainbow", s=5,
                             transform=ccrs.PlateCarree(), vmin=0, vmax=40)

    _add_distance_circles(ax)

    if sc_lwp is not None:
        cbar = plt.colorbar(sc_lwp, ax=ax, orientation="vertical", shrink=0.7, pad=0.05)
        cbar.set_label("LWP ($g/m²$)", fontsize=10)

    ax.set_title("Liquid Water Path", fontsize=10,
                 fontweight=TITLE_WEIGHT, color=TITLE_COLOR)
    ax.legend(loc="upper right")
    plt.show()


def plot_multi_orbit_on_relief(orbites: list[dict], param: str = "lwp",
                               vmin: float = 0, vmax: float = 40,
                               relief_alpha: float = 0.6,
                               cache_dir: str = "./data/cache") -> None:
    """Scatter LWP ou IWP superposé sur le relief ETOPO 2022."""
    _LABELS = {
        "lwp": ("Liquid Water Path", "LWP ($g/m²$)"),
        "iwp": ("Ice Water Path",    "IWP ($g/m²$)"),
    }
    if param not in _LABELS:
        raise ValueError(f"param doit être 'lwp' ou 'iwp', pas '{param}'")
    title, cbar_label = _LABELS[param]

    lons, lats, elev = _load_etopo_antarctica(cache_dir=cache_dir)
    LON2D, LAT2D = np.meshgrid(lons, lats)

    n_orb = len(orbites)
    fig, ax = _make_polar_map(
        f"ACM_CLP_2B — {title} sur relief ETOPO 2022\n"
        f"{n_orb} orbites  |  {DATE_START} → {DATE_END}")

    ax.gridlines(draw_labels=True, dms=False, x_inline=False,
                 y_inline=False, alpha=0.3,
                 ylocs=np.arange(-90, -55, 10))

    # fond relief
    norm_relief = mcolors.TwoSlopeNorm(vmin=-3000, vcenter=0, vmax=4500)
    pc = ax.pcolormesh(LON2D, LAT2D, elev,
                       cmap="terrain", norm=norm_relief, alpha=relief_alpha,
                       transform=ccrs.PlateCarree(), shading="auto", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="black", zorder=2)

    # scatter LWP / IWP
    sc = None
    for orb in orbites:
        values = orb.get(param)
        if values is None:
            continue
        v = np.asarray(values, dtype=float)
        mask = ~np.isnan(v) & (v > 0)
        if not mask.any():
            continue
        sc = ax.scatter(orb["lon"][mask], orb["lat"][mask],
                        c=v[mask], cmap="hot_r", s=4,
                        vmin=vmin, vmax=vmax,
                        transform=ccrs.PlateCarree(), zorder=5)

    _add_distance_circles(ax)
    ax.plot(LON_REF, LAT_REF, color="#FFA500", marker="*", markersize=8,
            linestyle="none", label="Dome C", transform=ccrs.PlateCarree(), zorder=10)

    # colorbar relief (gauche)
    cbar_r = fig.colorbar(pc, ax=ax, orientation="vertical",
                          shrink=0.45, pad=0.05, location="left")
    cbar_r.set_label("Élévation (m)", fontsize=9)
    cbar_r.ax.axhline(y=0, color="navy", linewidth=1, linestyle="--")

    # colorbar param (droite)
    if sc is not None:
        cbar_p = fig.colorbar(sc, ax=ax, orientation="vertical",
                              shrink=0.45, pad=0.02)
        cbar_p.set_label(cbar_label, fontsize=9)

    ax.set_title(f"{title} — ETOPO 2022", fontsize=11,
                 fontweight=TITLE_WEIGHT, color=TITLE_COLOR)
    ax.legend(loc="upper right")
    plt.show()


def plot_multi_orbit_elevation(orbites: list[dict], n_orbites_label: str,
                               vmin: float = -500, vmax: float = 4000) -> None:
    fig, ax = _make_polar_map(
        f"ACM_CLP_2B\nOrbits: {n_orbites_label}\nStart date: {DATE_START} \nEnd Date: {DATE_END}")

    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False,
                      y_inline=False, alpha=0.3,
                      ylocs=np.arange(-90, -55, 10))
    gl.ylabel_style = {"size": 10}
    gl.xlabel_style = {"size": 10}
    gl.top_labels   = False
    gl.right_label  = False

    ax.plot(LON_REF, LAT_REF, color="#FFA500", marker=".", linestyle="none",
            markersize=5, label="Dome C", transform=ccrs.PlateCarree())

    sc = None
    for orb in orbites:
        elev = orb.get("surface_elevation")
        if elev is None:
            continue
        sc = ax.scatter(orb["lon"], orb["lat"], c=elev,
                        cmap="terrain", s=5,
                        transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)

    _add_distance_circles(ax)

    if sc is not None:
        cbar = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.7, pad=0.05)
        cbar.set_label("Surface elevation (m)", fontsize=10)

    ax.set_title("Surface elevation", fontsize=10,
                 fontweight=TITLE_WEIGHT, color=TITLE_COLOR)
    ax.legend(loc="upper right")
    plt.show()


def plot_multi_orbit_iwp(orbites: list[dict], n_orbites_label: str) -> None:
    fig, ax = _make_polar_map(
        f"ACM_CLP_2B\nOrbits: {n_orbites_label}\nStart date: {DATE_START} \nEnd Date: {DATE_END}")

    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False,
                      y_inline=False, alpha=0.3,
                      ylocs=np.arange(-90, -55, 10))
    gl.ylabel_style = {"size": 10}
    gl.xlabel_style = {"size": 10}
    gl.top_labels   = False
    gl.right_label  = False

    ax.plot(LON_REF, LAT_REF, color="#FFA500", marker=".", linestyle="none",
            markersize=5, label="Dome C", transform=ccrs.PlateCarree())

    sc_lwp = None
    for orb in orbites:
        sc_lwp = ax.scatter(orb["lon"], orb["lat"], c=orb["iwp"],
                             cmap="rainbow", s=5,
                             transform=ccrs.PlateCarree(), vmin=0, vmax=40)

    _add_distance_circles(ax)

    if sc_lwp is not None:
        cbar = plt.colorbar(sc_lwp, ax=ax, orientation="vertical", shrink=0.7, pad=0.05)
        cbar.set_label("IWP ($g/m²$)", fontsize=10)

    ax.set_title("Ice Water Path", fontsize=10,
                 fontweight=TITLE_WEIGHT, color=TITLE_COLOR)
    ax.legend(loc="upper right")
    plt.show()


 
def _polar_grid_map(lon_bins, lat_bins, data_2d, title, cbar_label,
                    vmin=None, vmax=None, cmap="rainbow",
                    n_orbits=None, label=None):
   
    LON2D, LAT2D = np.meshgrid(lon_bins, lat_bins)
 
    subtitle = f"{len(lat_bins)}×{len(lon_bins)} cells"
    if n_orbits is not None:
        subtitle += f"  | {n_orbits} orbits \n{label}"
 
    fig, ax = _make_polar_map(subtitle)
    ax.gridlines(draw_labels=True, dms=False, x_inline=False,
                 y_inline=False, alpha=0.3,
                 ylocs=np.arange(-90, -55, 10))
 
    _add_distance_circles(ax)
    ax.plot(LON_REF, LAT_REF, color="#FFA500", marker=".", markersize=6,
            linestyle="none", label="Dome C", transform=ccrs.PlateCarree())
 
    pc = ax.pcolormesh(LON2D, LAT2D, data_2d, cmap=cmap, vmin=vmin, vmax=vmax,
                       transform=ccrs.PlateCarree(), shading="auto")
 
    cbar = plt.colorbar(pc, ax=ax, orientation="vertical", shrink=0.7, pad=0.05)
    cbar.set_label(cbar_label, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)
    ax.legend(loc="upper right")
    plt.show()
 
 
def plot_grid_lwp(grid, vmax=None) -> None:
    
    means = grid.mean()
    _polar_grid_map(
        grid.lon_bins, grid.lat_bins,
        means["lwp"],
        title=f"Liquid Water Path - Mean ({grid.label})",
        cbar_label="LWP ($g/m²$)",
        vmin=0, vmax=vmax,
        label=grid.label,
        n_orbits=grid.n_orbits)
 
 
def plot_grid_mean(grid, param: str = "lwp",
                   cbar_label: str | None = None,
                   vmin=None, vmax=None) -> None:
    means = grid.mean()
    if param not in means:
        raise KeyError(f"Paramtre '{param}' absent de la grille. "
                       f"Disponibles : {list(means)}")
    _polar_grid_map(grid.lon_bins, grid.lat_bins,
        means[param],
        title=f"{param.upper()} - Mean ({grid.label})",
        cbar_label=cbar_label or f"{param.upper()} ($g/m²$)",
        vmin=vmin, vmax=vmax, label=grid.label,
        n_orbits=grid.n_orbits)
 
 
def plot_grid_std(grid, param: str = "lwp",
                  cbar_label: str | None = None,
                  vmin=0, vmax=50) -> None:
    
    stds = grid.std()
    if param not in stds:
        raise KeyError(f"Parametre '{param}' absent de la grille."
                       f"Disponibles : {list(stds)}")
    _polar_grid_map(
        grid.lon_bins, grid.lat_bins,
        stds[param],
        title=f"{param.upper()} - std dev ({grid.label})",
        cbar_label=cbar_label or f"std {param}",
        cmap="rainbow",
        vmin=vmin, vmax=vmax,label=grid.label,
        n_orbits=grid.n_orbits)
 
 
def plot_grid_lwp_iwp(grid) -> None:
    
    means = grid.mean()
    stds  = grid.std()
    LON2D, LAT2D = np.meshgrid(grid.lon_bins, grid.lat_bins)
 
    proj = ccrs.SouthPolarStereo()
    fig  = plt.figure(figsize=(20, 8))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"LWP and IWP  |  {grid.n_orbits} orbits  | "f"grid {grid.dlat}°×{grid.dlon}°\n{grid.label}",
        fontsize=12, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)
 
    specs = [("lwp", "LWP mean ($g/m²$)",0, 20,  "rainbow"),
        ("iwp", "IWP mean ($g/m²$)",0, 100,  "rainbow"),
        ("lwp", "LWP std ($g/m²$)",0, 20,  "rainbow"),
        ("iwp", "IWP std ($g/m²$)", 0, 50,  "rainbow")]
    sources = [means["lwp"], means["iwp"], stds["lwp"], stds["iwp"]]
    labels  = ["LWP mean", "IWP mean", "LWP std", "IWP std"]
 
    for col, (data, (param, cbar_label, vmin, vmax, cmap), label) in enumerate(
        zip(sources, specs, labels)):
        ax = fig.add_subplot(1, 4, col + 1, projection=proj)
        ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
 
        theta= np.linspace(0, 2 * np.pi, 100)
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * 0.5 + [0.5, 0.5])
        ax.set_boundary(circle, transform=ax.transAxes)
        ax.gridlines(alpha=0.3)
 
        pc = ax.pcolormesh(LON2D, LAT2D, data,cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(), shading="auto")
        ax.plot(LON_REF, LAT_REF, color="#FFA500", marker=".", markersize=5,
                linestyle="none", transform=ccrs.PlateCarree())
        _add_distance_circles(ax)
 
        plt.colorbar(pc, ax=ax, orientation="horizontal",
                     shrink=0.8, pad=0.04, label=cbar_label)
        ax.set_title(label, fontsize=10, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)
 
    plt.tight_layout()
    plt.show()




_ETOPO_URL = (
    "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/60s/"
    "60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc"
)


def _load_etopo_antarctica(cache_dir: str = "./data/cache",
                            lat_min: float = -90, lat_max: float = -55):
    """Charge ETOPO 2022 pour la région antarctique, avec cache local."""
    cache_path = Path(cache_dir) / "etopo_antarctica.nc"

    if cache_path.exists():
        ds = xr.open_dataset(cache_path)
    else:
        print("Téléchargement ETOPO 2022 (subset Antarctique) via OPeNDAP...")
        remote = xr.open_dataset(_ETOPO_URL)
        subset = remote.sel(lat=slice(lat_min, lat_max))
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        subset.to_netcdf(str(cache_path))
        remote.close()
        ds = xr.open_dataset(cache_path)
        print(f"  Sauvegardé → {cache_path}")

    # nom de variable selon la version du fichier
    var = next((v for v in ("z", "elevation", "Band1") if v in ds), None)
    if var is None:
        raise KeyError(f"Variable d'élévation introuvable. Variables : {list(ds)}")

    lons = ds["lon"].values
    lats = ds["lat"].values
    elev = ds[var].values
    ds.close()
    return lons, lats, elev


def plot_antarctica_relief(vmin: float = -3000, vmax: float = 4500,
                           cache_dir: str = "./data/cache") -> None:
    """Relief de l'Antarctique depuis ETOPO 2022 (DEM externe, résolution 1')."""
    lons, lats, elev = _load_etopo_antarctica(cache_dir=cache_dir)

    LON2D, LAT2D = np.meshgrid(lons, lats)
    cmap = plt.cm.terrain
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    fig, ax = _make_polar_map("ETOPO 2022 — Relief de l'Antarctique (résolution 1')")

    ax.gridlines(draw_labels=True, dms=False, x_inline=False,
                 y_inline=False, alpha=0.4,
                 ylocs=np.arange(-90, -55, 10))

    _add_distance_circles(ax)
    ax.plot(LON_REF, LAT_REF, color="#FFA500", marker="*", markersize=8,
            linestyle="none", label="Dome C", transform=ccrs.PlateCarree(), zorder=10)

    pc = ax.pcolormesh(LON2D, LAT2D, elev,
                       cmap=cmap, norm=norm,
                       transform=ccrs.PlateCarree(), shading="auto")

    cbar = plt.colorbar(pc, ax=ax, orientation="vertical",
                        shrink=0.7, pad=0.05, extend="both")
    cbar.set_label("Élévation (m)", fontsize=10)
    cbar.ax.axhline(y=0, color="navy", linewidth=1.2, linestyle="--")

    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="black", zorder=5)

    ax.set_title("Relief de l'Antarctique — ETOPO 2022",
                 fontsize=12, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)
    ax.legend(loc="upper right")
    plt.show()


def orbites_above_threshold(orbites: list[dict],param: str = "lwp",
                             threshold: float = 100.0, plot: bool = True) -> list[dict]:

    series = {}
    for orb in orbites:
        values = orb.get(param)
        if values is None:
            continue
        col_id = orb.get("orbit_id") or orb.get("nom_orbite", "unknown")
        arr = np.asarray(values, dtype=float)
        series[col_id] = pd.Series(arr)

    if not series:
        print("Aucune donnee pour " + param)
        return []

    df = pd.DataFrame(series)

    cols_above = df.columns[(df > threshold).any()].tolist()

    print(param.upper() + " > " + str(threshold) + " g/m2")
    print("  " + str(len(cols_above)) + "/" + str(len(orbites)) + " orbite concernee :")
    for col in cols_above:
        n_above  = int((df[col] > threshold).sum())
        max_val  = float(df[col].max())
        orb_info = next((o for o in orbites
                         if (o.get("orbit_id") or o.get("nom_orbite")) == col), {})
        date_str = orb_info.get("start_day", "")
        print(col+ " date: " + date_str+ " max=" + f"{max_val:.1f}" + " n>" + str(threshold) + ": " + str(n_above))

    if not cols_above:
        print("Aucune orbite ne depasse ce seuil.")
        return []

    filtered = [
        orb for orb in orbites
        if (orb.get("orbit_id") or orb.get("nom_orbite")) in cols_above
    ]

    if plot:
        LABELS = {
            "lwp": ("Liquid Water Path", "LWP ($g/m^2$)"),
            "iwp": ("Ice Water Path",    "IWP ($g/m^2$)"),
        }
        long_name, cbar_label = LABELS.get(param, (param.upper(), param))
        vmax_plot = float(df[cols_above].max().max())

        header = ("ACM_CLP\n"+ param.upper() + " > " + str(threshold) + " g/m2  "
                  + "(" + str(len(filtered)) + "/" + str(len(orbites)) + " orbites)")

        fig, ax = _make_polar_map(header)
        ax.gridlines(draw_labels=True, dms=False,
                     x_inline=False, y_inline=False,
                     alpha=0.3, ylocs=np.arange(-90, -55, 10))
        ax.plot(LON_REF, LAT_REF, color="#FFA500", marker=".",
                markersize=5, linestyle="none",
                label="Dome C", transform=ccrs.PlateCarree())

        for orb in orbites:
            if orb.get("param") is None:
                pass
            ax.scatter(orb["lon"], orb["lat"],
                       color="lightgrey", s=2, alpha=0.4,
                       transform=ccrs.PlateCarree(), zorder=1)

        sc = None
        for orb in filtered:
            values = np.asarray(orb.get(param, []), dtype=float)
            values = np.where(values < 0, np.nan, values)
            values = np.where(values<=threshold, np.nan, values)
            mask = ~np.isnan(values)
            sc = ax.scatter(orb["lon"][mask], orb["lat"][mask],
                            c=values[mask], cmap="rainbow", s=6,vmin=threshold, vmax=vmax_plot,
                            transform=ccrs.PlateCarree(), zorder=5)

        if sc is not None:
            cbar = plt.colorbar(sc, ax=ax, orientation="vertical",shrink=0.7, pad=0.05)
            cbar.set_label(cbar_label, fontsize=10)
            cbar.ax.axhline(y=threshold, color="red",
                            linewidth=1.5, linestyle="--")

        _add_distance_circles(ax)
        ax.set_title(
            long_name + "orbites avec "
            + param.upper() + " > " + str(threshold) + " g/m2",
            fontsize=10, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)
        ax.legend(loc="upper right")
        plt.show()

    return filtered


def plot_grid_results(grid: GridAccumulator, day: str | None = None,
                      d1: str | None = None,d2: str | None = None) -> None:
    
    if day is not None:
        means = grid.mean(day)
        stds  = grid.std(day)
        orbits= grid._days[day]["_n_orbits"]
        label = day
    elif d1 is not None and d2 is not None:
        means = grid.mean_range(d1, d2)
        stds  = grid.std_range(d1, d2)
        orbits=grid.n_orbits_range(d1, d2)
        label = f"{d1} to {d2}"
    else:
        d1 = grid.dates[0]
        d2 = grid.dates[-1]
        means = grid.mean_range(d1, d2)
        stds= grid.std_range(d1, d2)
        orbits=grid.n_orbits
        label = f"{d1} to {d2}"


    class _FakeGrid:
        pass
    fg = _FakeGrid()
    fg.lon_bins = grid.lon_bins
    fg.lat_bins = grid.lat_bins
    
    fg.dlat = grid.dlat
    fg.dlon = grid.dlon
    fg.n_orbits = orbits
    fg.label = label
    fg._means = means
    fg._stds= stds
    fg.mean= lambda: means
    fg.std= lambda: stds
    return fg


def plot_grid_count(grid_or_dict,
                    param: str = "lwp",day: str | None = None,
                    d1: str | None = None,
                    d2: str | None = None,
                    log_scale: bool = False) -> None:
  
    if isinstance(grid_or_dict, dict):
        counts = grid_or_dict
        lon_bins  = None
        lat_bins  = None
        n_orbits  = None
        label     = param
    else:
        grid = grid_or_dict
        lon_bins = grid.lon_bins
        lat_bins = grid.lat_bins
        if day is not None:
            counts  = grid.count(day)
            label   = day
            n_orbits = grid._days[day]["_n_orbits"]
        elif d1 is not None and d2 is not None:
            counts  = grid.count_range(d1, d2)
            label   = d1 + " to " + d2
            n_orbits = grid.n_orbits_range(d1, d2)
        else:
            d1 = grid.dates[0]
            d2 = grid.dates[-1]
            counts  = grid.count_range(d1, d2)
            label   = d1 + " to " + d2
            n_orbits = grid.n_orbits

    if param not in counts:
        raise KeyError("Parametre '" + param + "' absent des counts.")

    data= counts[param].astype(float)
    data= np.where(data == 0, np.nan, data)   

    if log_scale and np.nanmax(data) > 0:
        vmin_log = max(1, np.nanmin(data[data > 0]))
        vmax_log = np.nanmax(data)
        norm  = mcolors.LogNorm(vmin=vmin_log, vmax=vmax_log)
        cmap  = "rainbow"
    else:
        
        norm  = None
        cmap  = "rainbow"

    title = param.upper() + " - Number of observations per bin  (" + label + ")"
    header = "ACM_CLP_2B - " + label
    if n_orbits is not None:
        header += "  (" + str(n_orbits) + " orbits)"

    LON2D, LAT2D = np.meshgrid(lon_bins, lat_bins)

    fig, ax = _make_polar_map(header)
    ax.gridlines(draw_labels=True, dms=False, x_inline=False,
                 y_inline=False, alpha=0.3,
                 ylocs=np.arange(-90, -55, 10))

    _add_distance_circles(ax)
    ax.plot(LON_REF, LAT_REF, color="#FFA500", marker=".", markersize=6,
            linestyle="none", label="Dome C", transform=ccrs.PlateCarree())

    pc = ax.pcolormesh(LON2D, LAT2D, data,
                       cmap=cmap, norm=norm, 
                       transform=ccrs.PlateCarree(), shading="auto")

    cbar = plt.colorbar(pc, ax=ax, orientation="vertical", shrink=0.7, pad=0.05)
    cbar.set_label("Number of observations" + (" (echelle log)" if log_scale else ""),
                   fontsize=10)

    ax.set_title(title, fontsize=10, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)
    ax.legend(loc="upper right")
    plt.show()


def plot_grid_count_histogram(grid_or_dict,param: str = "lwp",
                               day: str | None = None,
                               d1: str | None = None,d2: str | None = None) -> None:
  
    if isinstance(grid_or_dict, dict):
        counts = grid_or_dict
        label  = param
    else:
        grid = grid_or_dict
        if day is not None:
            counts = grid.count(day);       label = day
        elif d1 is not None and d2 is not None:
            counts = grid.count_range(d1, d2); label = d1 + " -> " + d2
        else:
            counts = grid.count_range(grid.dates[0], grid.dates[-1])
            label  = grid.dates[0] + " -> " + grid.dates[-1]

    if param not in counts:
        raise KeyError("Parametre '" + param + "' absent des counts.")

    data = counts[param].ravel()
    data = data[data > 0] 

    if len(data) == 0:
        print("[plot] Aucune cellule avec des donnees.")
        return

    fig, ax1 = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        param.upper() + "- number of observations per bins  ("
        + label + ")",
        fontsize=12, fontweight=TITLE_WEIGHT, color=TITLE_COLOR,
    )

    ax1.hist(data, bins=40, color=TITLE_COLOR, edgecolor="white",
             linewidth=0.3, alpha=0.85)
    ax1.set_xlabel("Number of observations", fontsize=10)
    ax1.set_ylabel("number of cells", fontsize=10)
    ax1.set_title("Echelle", fontsize=10)

    #Statistiques
    stats = pd.Series(data)
    txt = ("n cells = " + str(len(data)) + "\n"
           + "min  = " + str(int(stats.min())) + "\n"
           + "max  = " + str(int(stats.max())) + "\n"
           + "mean = " + f"{stats.mean():.1f}" + "\n"
           + "med  = " + f"{stats.median():.1f}")
    ax1.text(0.97, 0.97, txt, transform=ax1.transAxes,
             va="top", ha="right", fontsize=8,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    # Histogramme log
    #ax2.hist(data, bins=40, color=TITLE_COLOR, edgecolor="white", linewidth=0.3, alpha=0.85, log=True)
   # ax2.set_xlabel("Number of observations", fontsize=10)
   # ax2.set_ylabel("number of cells (log)", fontsize=10)
   # ax2.set_title("Echelle log", fontsize=10)

    plt.tight_layout()
    plt.show()