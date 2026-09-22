"""
Code for storing utility functions for data pre-processing and plotting for analysis
------------------------------------------------------------------------------------
Author: Sreevathsa G. (sg13n23@soton.ac.uk; ORCID ID: 0000-0003-4084-9677)
Last updated: 21 September 2026
"""

import numpy as np
import xarray as xr
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Dask cluster
import dask
from dask.distributed import Client, LocalCluster

# Plotting and mapping
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from cdo import Cdo
cdo = Cdo(tempdir='/dssgfs01/scratch/sg13n23/ATTRIBUTION_EXPS/NPD_Wind_Exp_Diagnostics/tmp/')

# Cartopy land/coastline/topography features, at the resolutions used by the map panels
land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='grey')
land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='face', facecolor='grey')
coast_50m = cfeature.NaturalEarthFeature('physical', 'coastline', '50m', edgecolor='black', facecolor='none')
coast_10m = cfeature.NaturalEarthFeature('physical', 'coastline', '10m', edgecolor='black', facecolor='none')
topog = cfeature.NaturalEarthFeature(category='raster', name='natural_earth_hypsometric', scale='50m', facecolor='none')

# =============================================================================
# GENERIC HELPER FUNCTIONS THAT ARE BROADLY USED ACROSS THE REPOSITORY
# =============================================================================

def setup_dask_cluster(n_workers=6, threads_per_worker=3, memory_limit="48GB"):
    """
    Dask cluster setup (on ANEMONE HPC @ NOCS, UK):
    ----------------------------------------------
    Initialize and return a Dask cluster and client.
    """
    # Keeping Dask's spill/scratch files on the scratch filesystem rather than in $TMPDIR
    dask.config.set({"temporary_directory": "/dssgfs01/scratch/sg13n23/temp/",
                     "local_directory": "/dssgfs01/scratch/sg13n23/temp/"})
    cluster = LocalCluster(n_workers=n_workers,
                           threads_per_worker=threads_per_worker,
                           memory_limit=memory_limit,
                           dashboard_address=":8787",)
    client = Client(cluster)
    return cluster, client

def renamer(ds_f):
    """
    Standardise coordinate, dimension and variable names across datasets coming from
    different sources (the NEMO eORCA025 experiments, the HG3 coupled model and the
    climatological baselines). Conservative temperature is written as 'thetao_con' by some
    of the NEMO file-defs and as 'toce_con' by others (minor discrepancy when the model was 
    setup for experimentation), and the depth/time/index names vary likewise; renaming them 
    up front avoids conflicts when two such datasets are differenced or merged later on.

    Parameters
    ----------
    ds_f (xr.Dataset) : Dataset whose names are to be standardised.

    Returns
    -------
    xr.Dataset : The same dataset, with every recognised name renamed.
    """
    # Coordinate names:
    coord_map = {
        'deptht': 'depth',
        'lev': 'depth',
        'longitude': 'nav_lon',
        'latitude': 'nav_lat',
        'lon':'nav_lon',
        'lat':'nav_lat',
        'i': 'x',
        'j': 'y',
        'time_counter': 'time'
    }
    # Dimension names
    dim_map = {'x_2': 'x', 'y_2': 'y'}

    # Variable names
    var_map = {'thetao_con':'toce_con'}

    # Only the names actually present are renamed, so the same call works on any of the files
    for old, new in coord_map.items():
        if old in ds_f.coords:
            ds_f = ds_f.rename({old: new})

    for old, new in dim_map.items():
        if old in ds_f.dims:
            ds_f = ds_f.rename({old: new})

    for old, new in var_map.items():
        if old in list(ds_f.variables):
            ds_f = ds_f.rename({old: new})

    return ds_f

def z20_calculator(ds):
    """
    Calculate the depth of the 20degC isotherm (Z20) by identifying, for each
    (time, lat, lon), the depth level whose conservative temperature is closest
    to 20degC.

    Parameters
    ----------
    ds (xr.Dataset) : Dataset containing 'toce_con' (conservative temperature)
                      and a 'depth' coordinate.

    Returns
    -------
    xr.DataArray : DataArray named 'z20' giving the depth of the 20degC isotherm
                   for each (time, lat, lon) where valid data exist.
    """
    # Absolute difference |T - 20degC| at all depths
    abs_diff = abs(ds['toce_con'] - 20)
    # Identifying grid points that have at least one valid depth value
    valid_mask = abs_diff.notnull().any(dim='depth')
    # Masking out locations where all depths are NaN
    abs_diff = abs_diff.where(valid_mask)
    # Finding the depth index where |T - 20| is smallest (closest to 20degC)
    min_index = abs_diff.fillna(np.inf).argmin(dim='depth')
    # Extracting the actual depth values at those indices
    z20 = ds['depth'].isel(depth=min_index).where(valid_mask).rename('z20')
    return z20

def ond_average(exp='BASELINE', freq='1m'):
    """
    October-November-December (OND) mean of conservative temperature

    Parameters
    ----------
    exp (str)  : Experiment/internal ID as it appears in the file names (e.g. 'CTRL_2023',
                 'ANWSUP', 'BASELINE_NPD', 'HG3-1995').
    freq (str) : Output frequency of the files to read ('1m' monthly-mean or '1d' daily-mean).

    Returns
    -------
    xr.Dataset : The three monthly files merged and averaged down to a single OND mean.
    """
    # Merging the October (M10), November (M11) and December (M12) files, then time-averaging
    return renamer(xr.merge([xr.open_dataset('./data/TOCE_CON/TOCE_CON_{exp}_{freq}_M10.nc'.format(exp=exp, freq=freq)),
                             xr.open_dataset('./data/TOCE_CON/TOCE_CON_{exp}_{freq}_M11.nc'.format(exp=exp, freq=freq)),
                             xr.open_dataset('./data/TOCE_CON/TOCE_CON_{exp}_{freq}_M12.nc'.format(exp=exp, freq=freq))], compat='no_conflicts')).mean(dim='time')

def nino_ta_calc(nb='N34', exp=None, freq='1d'):
    """
    Niño-box conservative temperature anomaly (experiment minus its climatological baseline) over a full year

    Parameters
    ----------
    nb (str)   : Niño box to read ('N12', 'N3', 'N34' or 'N4').
    exp (str)  : Experiment/internal ID as it appears in the file names.
    freq (str) : Output frequency of the files to read ('1m' or '1d').

    Returns
    -------
    xr.Dataset : The twelve monthly anomalies merged into one year-long dataset.
    """
    nb_ta = []
    if 'HG3-' not in exp:
        # JRA55-do forced runs are differenced against the NPD baseline (1996-2023)
        for m in tqdm(np.arange(1,13)):
            exp_ds = xr.open_dataset('./data/NINO_BOXES/{nb}_TOCE_CON_{exp}_{freq}_M{m:02d}.nc'.format(nb=nb, exp=exp, m=m, freq=freq))
            bs_ds = xr.open_dataset('./data/NINO_BOXES/{nb}_TOCE_CON_BASELINE_NPD_{freq}_M{m:02d}.nc'.format(nb=nb, m=m, freq=freq))
            # Differencing against .values, since the two files carry different time labels
            nb_ta += [renamer(exp_ds)['toce_con'] - renamer(bs_ds)['toce_con'].values]
    else:
        # HG3 has no daily (1d) output, so HG3-1995 is read at 1m and differenced against the monthly HG3 baseline (1976-2005) instead
        for m in tqdm(np.arange(1,13)):
            exp_ds = xr.open_dataset('./data/NINO_BOXES/{nb}_TOCE_CON_{exp}_1m_M{m:02d}.nc'.format(nb=nb, exp=exp, m=m))
            bs_ds = xr.open_dataset('./data/NINO_BOXES/{nb}_TOCE_CON_BASELINE_HG3_1m_M{m:02d}.nc'.format(nb=nb, m=m))
            nb_ta += [renamer(exp_ds)['toce_con'] - renamer(bs_ds)['toce_con'].values]

    nb_ta = renamer(xr.merge(nb_ta, compat='no_conflicts'))
    return nb_ta

def ohc_t300m_calc(m=1, freq='1m', exp=None):
    """
    Upper-300m ocean heat content for one calendar month of one run, computed as
    rho*cp*sum(e3t*T) over the depth levels down to 300 m.

    Parameters
    ----------
    m (int)    : Calendar month (1-12) to read.
    freq (str) : Output frequency of the files to read ('1m' or '1d').
    exp (str)  : Experiment/internal ID as it appears in the file names.

    Returns
    -------
    xr.DataArray : DataArray named 'ohc_t300m', in J.m^(-2), on the grid of the input file.
    """
    cp = 3991.86795711963  # specific heat capacity (J.kg⁻¹.K⁻¹)
    rho = 1026.0           # reference density (kg.m⁻³)

    toce_con_ds = renamer(xr.open_dataset('./data/TOCE_CON/TOCE_CON_{exp}_{freq}_M{m:02d}.nc'.format(exp=exp, m=m, freq=freq)))
    # Layer thicknesses (e3t) were only cropped alongside temperature for the single-year NEMO experiments, 
    # so the baselines and the HG3 runs have to source e3t from elsewhere:
    if exp == 'BASELINE_NPD':
        # Borrowing e3t from one of the experiments or single-year control-runs will work here.
        e3t = renamer(xr.open_dataset('./data/TOCE_CON/TOCE_CON_CTRL_2023_{freq}_M{m:02d}.nc'.format(m=m, freq=freq)))['e3t']
    elif (exp == 'HG3-1995') | (exp == 'BASELINE_HG3'):
        # HG3 is on the eORCA12 grid with a depth-only (1D) vertical coordinate, so e3t_1d is
        # read from its domain file and broadcast across (time, y, x) to match the T field
        e3t_1d_values = xr.open_dataset('./data/domain_cfg_HG3_eORCA12.nc')['e3t_1d'].values

        e3t = e3t_1d_values[:, :, None, None]
        e3t = np.broadcast_to(e3t, toce_con_ds['toce_con'].shape)
        e3t_da = toce_con_ds['toce_con'].copy(deep=True)
        e3t_da = e3t_da.rename('e3t')
        e3t_da.values = e3t
        e3t = e3t_da

    else:
        e3t = toce_con_ds['e3t']

    # Surface layer collapsed to a 1/NaN mask, so land points stay NaN through the depth sum
    mask = e3t[:,0]
    mask = xr.where(np.isnan(mask), mask, 1)

    # Depth-integrating the layer heat content over the top 300 m of the water column
    ohc_t300m = (e3t.values*(toce_con_ds['toce_con'])).sel(depth=slice(0,301)).sum(dim='depth')*cp*rho*mask.values
    return ohc_t300m.rename('ohc_t300m')

def net_wnd_speed_tavg(ds_dict, ax, levs=np.arange(0,8.1,0.25), cbar=False):
    """
    Plot the time-averaged net wind speed as filled contours, with the mean wind vectors
    overlaid as quiver arrows (used for the Figure 2 panels a and b).

    Parameters
    ----------
    ds_dict (dict)  : Dictionary holding the 'uas' and 'vas' DataArrays of one wind product.
    ax (GeoAxes)    : Cartopy axes to draw on.
    levs (sequence) : Contour levels for the wind-speed shading.
    cbar (bool)     : Whether xarray attaches its own colorbar; False when a shared colorbar
                      is added later with add_colorbar.

    Returns
    -------
    c (QuadContourSet) : The filled-contour set, to be reused as a colorbar mappable.
    """
    # Time-mean zonal and meridional components, combined into a net wind speed
    uas_avg = ds_dict['uas'].mean(dim=['time'])
    vas_avg = ds_dict['vas'].mean(dim=['time'])
    net_wnd_speed = np.sqrt(uas_avg**2 + vas_avg**2)
    c = net_wnd_speed.plot.contourf(x='lon', y='lat', cmap='rainbow',
                               levels=levs, transform=ccrs.PlateCarree(), ax=ax, extend='max', add_colorbar=cbar)

    # Wind direction shown as arrows, thinned onto a coarser 30x30 grid for legibility
    ax.quiver(net_wnd_speed.lon, net_wnd_speed.lat, uas_avg, vas_avg, transform=ccrs.PlateCarree(),
              scale=400, regrid_shape=30, width=0.002, color='black')

    return c

def n4_box_positive_anom_mean(uas_1sd, lat_slice=slice(-10, 10), lon_slice=slice(120, 140)):
    """
    Area-weighted, mean-centred, positive-only box average of the >= 1 std. dev. (SD) zonal wind anomalies
    (Figure 1 panel k). The default box (10S-10N, 120E-140E) is the western Pacific region
    in which much of the anomalous westerlies are generated (i.e., WWBs).

    Parameters
    ----------
    uas_1sd (xr.DataArray) : Zonal wind anomalies already masked to >= mean + 1SD.
    lat_slice (slice)      : Latitude band to average over.
    lon_slice (slice)      : Longitude band to average over.

    Returns
    -------
    xr.DataArray : Box-mean time series with the negative (easterly) side masked out.
    """
    box = uas_1sd.sel(lat=lat_slice, lon=lon_slice)
    # Relabelling the time axis onto a common 3-hourly 2023 calendar lets the anomalies of the
    # different wind products be overlaid on one axis; only Jan-Jun is kept, which is when the
    # anomalous westerlies of interest occur
    box['time'] = pd.date_range(start='2023-01-01', periods=box.time.size, freq='3h')
    box = box.sel(time=slice('2023-01-01', '2023-06-30'))
    # cos(lat) weighting, so the poleward grid cells are not over-counted in the box average
    lat_weights = np.cos(np.deg2rad(box['lat']))
    box_mean = box.weighted(lat_weights).mean(dim=['lat', 'lon'])
    # Re-centring on the box mean, then keeping only the positive (westerly) anomalies
    box_mean = box_mean - box_mean.mean()
    return box_mean.where(box_mean >= 0)

def add_colorbar(fig, axes, mappable, label, ticks, side="right",
                 width=0.010, height=0.018, pad=0.007, ticksize=10,
                 labelsize=12, height_scale=0.90,):
    """
    Add a colorbar aligned with a block of axes.

    Parameters
    ----------
    fig (matplotlib.figure.Figure) : Figure object.
    axes (sequence of Axes)        : Axes that define the block to align the colorbar with.
    mappable (ScalarMappable)      : Object returned by contourf/pcolormesh/etc.
    label (str)                    : Colorbar label.
    ticks (sequence)               : Tick locations for the colorbar.
    side ({"right", "bottom"})     : Where to place the colorbar relative to the axes block.
                                     - "right": vertical bar to the right (uses 'width', 'height_scale').
                                     - "bottom": horizontal bar below (uses 'height').
    width (float)                  : Thickness of the bar when side="right" (in figure fraction).
    height (float)                 : Thickness of the bar when side="bottom" (in figure fraction).
    pad (float)                    : Padding between the axes block and the colorbar (figure fraction)
    ticksize (int)                 : Font size for ticks.
    labelsize (int)                : Font size for label.
    height_scale (float)           : Only used for side="right": fraction of axes block height to use.

    Returns
    -------
    cb (matplotlib.colorbar.Colorbar) : Colorbar object
    """
    # Compute bounding box enclosing all axes
    row_box = Bbox.union([ax.get_position(fig).frozen() for ax in axes])

    # Decide placement and create colorbar axis
    if side == "right":
        # Vertical bar placed to the right of the axes block
        cax = fig.add_axes([row_box.x1 + pad, row_box.y0, width, row_box.height * height_scale,])
        orientation = "vertical"
    elif side == "bottom":
        # Horizontal bar placed below the axes block
        cax = fig.add_axes([row_box.x0, row_box.y0 - pad, row_box.width, height,])
        orientation = "horizontal"
    else:
        raise ValueError(f"Unsupported side: {side!r}")
    # Create and format the colorbar
    cb = fig.colorbar(mappable, cax=cax, orientation=orientation, ticks=ticks)
    cb.set_label(label, fontsize=labelsize)
    cb.ax.tick_params(labelsize=ticksize)

    if side == "right":
        # Ensure ticks appear on the right for vertical bars
        cax.yaxis.set_ticks_position("right")
        cax.xaxis.set_visible(False)

    return cb

def merge_pngs_horizontal(paths, out_path):
    """
    Merge multiple PNG files horizontally into one PNG.

    Parameters
    ----------
    paths (list)     : List of figure paths to merge.
    out_path (str)   : Output path for the merged figure.
    """
    # Loading all images
    imgs = [Image.open(p) for p in paths]
    # Determining max height to align all images
    max_h = max(im.height for im in imgs)

    # Resizing images to have a uniform height
    imgs_resized = []
    for im in imgs:
        if im.height != max_h:
            new_w = int(im.width * max_h / im.height)
            imgs_resized.append(im.resize((new_w, max_h), Image.LANCZOS))
        else:
            imgs_resized.append(im)

    # Total width for final merged canvas
    total_w = sum(im.width for im in imgs_resized)
    combined = Image.new("RGB", (total_w, max_h), color="white")

    # Pasting images side by side
    x = 0
    for im in imgs_resized:
        combined.paste(im, (x, 0))
        x += im.width

    combined.save(out_path)

def merge_pngs_vertical(paths, out_path):
    """
    Merge multiple PNG files vertically into one PNG. 

    Parameters
    ----------
    paths (list)     : List of figure paths to merge.
    out_path (str)   : Output path for the merged figure.
    """
    # Loading all images
    imgs = [Image.open(p) for p in paths]
    # Determining max width to align all images
    max_w = max(im.width for im in imgs)

    # Resizing images to have a uniform width
    imgs_resized = []
    for im in imgs:
        if im.width != max_w:
            new_h = int(im.height * max_w / im.width)
            imgs_resized.append(im.resize((max_w, new_h), Image.LANCZOS))
        else:
            imgs_resized.append(im)

    # Total height for final merged canvas
    total_h = sum(im.height for im in imgs_resized)
    combined = Image.new("RGB", (max_w, total_h), color="white")

    # Pasting images one below another
    y = 0
    for im in imgs_resized:
        combined.paste(im, (0, y))
        y += im.height

    combined.save(out_path)

def format_axis(ax, kind, row, n_rows, xlabel=None, ylabel=None, extent=None, time=None,
                ylim=None, yticks=None, x_tick_intervals=None, y_tick_intervals=None):
    """
    Shared axis formatting for every panel type used across the repository: extents, limits,
    ticks, locators, gridlines, tick-label sizes and axis labels. Panel titles, legends,
    reference lines and the plotted data itself stay with the calling plot function.

    Parameters
    ----------
    ax (Axes)                : Axes (cartopy GeoAxes for the map kinds) to be formatted.
    kind (str)               : Panel type being formatted:
                               - 'map'        : SST/OHC anomaly maps of figure_3/5/7 panels A and B.
                               - 'zonal'      : equatorial zonal cross-sections of figure_3/5/7 panel C.
                               - 'z20'        : Z20-vs-longitude lines of figure_4/6/8 panel C.
                               - 'wind_time'  : Hovmoller-style time-longitude wind panels of
                                                figure_1 (Section 2.2).
                               - 'wind_map'   : wind-comparison maps of figure_2 (Section 2.2).
                               - 'cumulative' : cumulative heat attribution time series of
                                                figure_4/6/8 panel A.
                               - 'trajectory' : zonal-vs-vertical advection trajectories of
                                                figure_4/6/8 panel B.
    row (int)                : Row index of this panel within its stacked column of panels.
    n_rows (int)             : Total number of rows, so only the bottom row is given an x-label.
    xlabel (str)             : X-axis label; each kind falls back to its own default.
    ylabel (str)             : Y-axis label; each kind falls back to its own default.
    extent (sequence)        : [lon0, lon1, lat0, lat1] for kind='map' (default: tropical Pacific).
    time (sequence)          : Time coordinate values, required for kind='wind_time' to place
                               the month ticks.
    ylim (tuple)             : Y-axis limits (bottom, top), required for kind='z20'.
    yticks (sequence)        : Y-axis tick locations, required for kind='z20'.
    x_tick_intervals (float) : Major x-tick spacing, required for kind='trajectory'.
    y_tick_intervals (float) : Major y-tick spacing, required for kinds 'cumulative'/'trajectory'.
    """
    # only the bottom row of a stacked column of panels gets an x-axis label
    is_last_row = (row == n_rows - 1)

    # 0-360 longitudes relabelled into the E/W convention; shared by 'zonal' and 'z20'
    def format_longitude(lon):
        if lon > 180:
            return f'{360 - lon:.0f}°W'
        return f'{lon:.0f}°E'

    if kind == 'map':
        # crop to the tropical Pacific and add coastlines for a cartopy map panel
        ax.set_extent(extent or [140, 290, -31, 31], crs=ccrs.PlateCarree())
        ax.add_feature(land_50m)
        ax.add_feature(coast_50m)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                           linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = plt.FixedLocator(range(-180, 181, 20))
        gl.ylocator = plt.FixedLocator(range(-90, 91, 10))
        gl.xlabel_style = {'size': 16, 'color': 'black'}
        gl.ylabel_style = {'size': 16, 'color': 'black'}
        ax.set_ylabel(ylabel or 'Latitude', fontsize=18)
        ax.set_xlabel((xlabel or 'Longitude') if is_last_row else '', fontsize=18)
    elif kind == 'zonal':
        # fixed depth ticks for an equatorial cross-section panel
        ax.set_yticks(ticks=np.arange(0, 300.1, 30))
        ax.grid(True, alpha=0.5, ls='--', color='k')
        ax.tick_params(axis='both', labelsize=16)
        ax.set_ylabel(ylabel or 'Depth (m)', fontsize=18)
        ax.set_xlabel((xlabel or 'Longitude') if is_last_row else '', fontsize=18)

        # Longitude range and ticks
        major_ticks = np.arange(140, 281, 20)
        minor_ticks = np.arange(140, 281, 10)

        ax.set_xlim(140, 280)
        ax.set_xticks(major_ticks)
        ax.set_xticklabels([format_longitude(lon) for lon in major_ticks])
        ax.set_xticks(minor_ticks, minor=True)

        # Lighter grid lines for the 10 degree minor longitude ticks.
        ax.grid(which='minor', axis='x', color='gray', alpha=0.65, linestyle='--', linewidth=0.8)
        ax.tick_params(axis='x', which='minor', length=4)
    elif kind == 'z20':
        # longitude ticks relabelled from 0-360 to -180..180 for a Z20 line panel
        ax.grid(True, alpha=0.5, color='gray')
        ax.set_xlabel(xlabel or 'Longitude', fontsize=16, labelpad=10)

        # Longitude range and ticks
        major_ticks = np.arange(130, 271, 20)
        minor_ticks = np.arange(130, 271, 10)

        ax.set_xlim(130, 270)
        ax.set_xticks(major_ticks)
        ax.set_xticklabels([format_longitude(lon) for lon in major_ticks])
        ax.set_xticks(minor_ticks, minor=True)

        # Depth (left panel) or anomaly (right panel) range, passed in by the caller. Depth
        # increases downward, so its limits arrive the other way round, i.e. high to low
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.set_ylabel(ylabel or '', fontsize=16)

        ax.tick_params(axis='both', labelsize=14)
    elif kind == 'wind_time':
        # month-labelled time y-axis vs longitude x-axis, used for the Figure 1
        # zonal-wind Hovmoller-style panels (Section 2.2)
        month_starts = [t for t in time if pd.Timestamp(t).day == 1 and pd.Timestamp(t).hour == 0]
        ax.set_yticks(ticks=month_starts,
                       labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.set_xticks(ticks=np.arange(120, -75 + 360 + 0.1, 20))
        ax.grid(True, alpha=0.3, ls='--', color='k')
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlabel(xlabel or '', fontsize=16)
        ax.set_ylabel(ylabel or '', fontsize=16)
    elif kind == 'wind_map':
        # coarser gridlines, no extent/land layer, used for the Figure 2
        # wind-comparison maps (Section 2.2)
        ax.add_feature(coast_50m)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                           linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = plt.FixedLocator(range(-180, 181, 40))
        gl.ylocator = plt.FixedLocator(range(-90, 91, 30))
        gl.xlabel_style = {'size': 16, 'color': 'black'}
        gl.ylabel_style = {'size': 16, 'color': 'black'}
    elif kind == 'cumulative':
        # month-labelled time x-axis for the cumulative heat-attribution panels; every panel
        # keeps its own tick labels despite the shared axes, so each can be read on its own
        ax.xaxis.set_major_locator(mdates.MonthLocator(range(1, 13)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.grid(True, ls='--', alpha=0.5, color='gray')
        ax.tick_params(axis='y', labelsize=15)
        ax.tick_params(axis='x', labelsize=14, rotation=0)
        ax.yaxis.set_major_locator(MultipleLocator(y_tick_intervals))
        # only the panels the caller labelled, i.e. the leftmost column, get a y-axis label
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=16)
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.yaxis.set_tick_params(labelleft=True)
    elif kind == 'trajectory':
        # tick labels kept on every panel as above; the margins leave room for the
        # end-of-trajectory markers to sit clear of the axes spines
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.yaxis.set_tick_params(labelleft=True)
        ax.xaxis.set_major_locator(MultipleLocator(x_tick_intervals))
        ax.yaxis.set_major_locator(MultipleLocator(y_tick_intervals))
        ax.margins(0.13)
        ax.grid(True, ls='--', alpha=0.4, color='gray')
        ax.set_xlabel(xlabel or '', fontsize=16)
        ax.set_ylabel(ylabel or '', fontsize=16)
        ax.tick_params(axis='both', labelsize=15)
    else:
        raise ValueError(f"Unsupported kind: {kind!r}")

def compute_nino_ssta_change(exps, period, nb_regions=('N3', 'N34', 'N4'), exp_overrides=None):
    """
    Niño-box surface temperature anomaly (exp - baseline) averaged over 'period', one row per
    (Niño box, experiment) pair. Used as the text annotation inside the figure_3/5/7 SSTA maps.

    Parameters
    ----------
    exps (sequence)       : Experiment/internal IDs to compute the anomaly for.
    period (tuple)        : (start, end) date strings to average the anomaly over (e.g. OND).
    nb_regions (sequence) : Niño boxes to loop over, as used in the file names.
    exp_overrides (dict)  : Per-experiment overrides of {'baseline', 'freq', 'reindex_year'},
                            needed by HG3-1995 (monthly HG3 baseline, relabelled time axis).

    Returns
    -------
    pd.DataFrame : Columns ['EXP', 'ANOM_AVG'], indexed by Niño box ('NB_REGION').
    """
    exp_overrides = exp_overrides or {}
    # default file suffix and baseline for a "normal" (non-HG3) experiment
    default_cfg = {'baseline': 'BASELINE_NPD', 'freq': '1d'}
    # accumulator: one row per (Nino box, experiment) combination
    change = pd.DataFrame(columns=['NB_REGION', 'EXP', 'ANOM_AVG'])
    for nb in nb_regions:
        for exp in exps:
            # per-experiment overrides (used by the HG3-1995 case) win over the defaults
            cfg = {**default_cfg, **exp_overrides.get(exp, {})}
            nb_exp = renamer(xr.open_dataset(
                './data/NINO_BOXES/{nb}_TOCE_CON_{exp}_{freq}.nc'.format(nb=nb, exp=exp, freq=cfg['freq'])
            )).sel(depth=0, method='nearest')
            if 'reindex_year' in cfg:
                # relabel the monthly HG3 climatology onto the comparison year so its
                # time coordinate lines up with the experiment being compared against
                nb_exp['time'] = pd.to_datetime([
                    '{}-{:02d}-01'.format(cfg['reindex_year'], pd.to_datetime(t).month) for t in nb_exp['time'].values
                ])
            nb_bs = renamer(xr.open_dataset(
                './data/NINO_BOXES/{nb}_TOCE_CON_{baseline}_{freq}.nc'.format(nb=nb, baseline=cfg['baseline'], freq=cfg['freq'])
            )).sel(depth=0, method='nearest')
            # align the baseline's time labels to the experiment's before differencing
            nb_bs['time'] = nb_exp['time']
            # box-mean anomaly (experiment minus baseline), averaged over the given months
            mean_anom = (nb_exp - nb_bs)['toce_con'].sel(time=slice(*period)).mean().item()
            change.loc[len(change)] = [nb, exp, mean_anom]
    change.set_index('NB_REGION', inplace=True)
    return change

def compute_nino_ohca_change(exps, months=(10, 11, 12), nb_regions=('N3', 'N34', 'N4'), exp_overrides=None):
    """
    Niño-box upper-300m ocean heat content anomaly (exp - baseline), in GJ.m^(-2), one row per
    (Niño box, experiment) pair. Used as the text annotation inside the figure_3/5/7 OHCA maps.

    Parameters
    ----------
    exps (sequence)       : Experiment/internal IDs to compute the anomaly for.
    months (sequence)     : Calendar months to average over (October-November-December).
    nb_regions (sequence) : Unused here, the boxes being cut by cdo below instead; kept so the
                            signature matches compute_nino_ssta_change.
    exp_overrides (dict)  : Per-experiment override of {'baseline'}, needed by HG3-1995.

    Returns
    -------
    pd.DataFrame : Columns ['EXP', 'ANOM_AVG'], indexed by Niño box ('NB_REGION').
    """
    exp_overrides = exp_overrides or {}
    # Nino 3/3.4/4 box extents, as "lon_min,lon_max,lat_min,lat_max" for cdo sellonlatbox
    boxes = {'N3': '-150,-90,-5,5', 'N34': '-170,-120,-5,5', 'N4': '160,210,-5,5'}
    change = pd.DataFrame(columns=['NB_REGION', 'EXP', 'ANOM_AVG'])
    for exp in exps:
        # HG3-1995 compares against the HG3 baseline, everything else against NPD
        baseline = exp_overrides.get(exp, {}).get('baseline', 'BASELINE_NPD')
        # OND-mean upper-300m ocean heat content for the experiment and its baseline
        mean_ohc_exp = xr.merge([ohc_t300m_calc(m=m, freq='1m', exp=exp) for m in months])
        mean_ohc_bs = xr.merge([ohc_t300m_calc(m=m, freq='1m', exp=baseline) for m in months])
        for nb, box in boxes.items():
            # spatial box average, then difference exp minus baseline
            nb_ohc_exp = cdo.sellonlatbox(box, input=mean_ohc_exp, returnXDataset=True).mean()
            nb_ohc_bs = cdo.sellonlatbox(box, input=mean_ohc_bs, returnXDataset=True).mean()
            anom = (nb_ohc_exp['ohc_t300m'] - nb_ohc_bs['ohc_t300m'].values).item()
            change.loc[len(change)] = [nb, exp, anom / 1e9]  # joules to gigajoules
    change.set_index('NB_REGION', inplace=True)
    return change

def add_nino_box_outlines(ax, linewidth=1.75, alpha=0.75, label_fontsize=12):
    """
    Overlay Niño 3/3.4/4 region outlines (unfilled, black) on a cartopy GeoAxes.

    Parameters
    ----------
    ax (GeoAxes)           : Cartopy axes to draw the outlines on.
    linewidth (float)      : Line width of the box edges.
    alpha (float)          : Opacity shared by the box edges and their labels.
    label_fontsize (int)   : Font size of the box labels.
    """
    # name: (lon0, lon1, lat0, lat1, linestyle, label_lon)
    boxes = {
        'Niño 3':   (210, 270, -5, 5, '-',  253),   # 150W-90W, 5S-5N
        'Niño 3.4': (190, 240, -5, 5, '--', 215),   # 170W-120W, 5S-5N
        'Niño 4':   (160, 210, -5, 5, '-',  177),   # 160E-150W, 5S-5N
    }
    for name, (lon0, lon1, lat0, lat1, ls, label_lon) in boxes.items():
        ax.add_patch(mpatches.Rectangle(
            (lon0, lat0), lon1 - lon0, lat1 - lat0,
            fill=False, edgecolor='black', linestyle=ls, linewidth=linewidth,
            alpha=alpha, transform=ccrs.PlateCarree(), zorder=10))
        ax.text(label_lon, lat0 - 0.8, name, transform=ccrs.PlateCarree(),
                ha='center', va='top', fontsize=label_fontsize, fontweight='bold',
                color='black', alpha=alpha, zorder=10)

def truncate_cmap(name, clip=(0, 1), n=256):
    """
    Return a copy of a named colormap with its lightest 'clip' fraction dropped,
    used so the palest end of a sequential colormap never reads as near-white.

    Parameters
    ----------
    name (str)    : Name of the matplotlib colormap to truncate.
    clip (tuple)  : (lo, hi) fractions of the colormap to keep.
    n (int)       : Number of colours to sample the truncated colormap at.

    Returns
    -------
    mcolors.LinearSegmentedColormap : The truncated colormap.
    """
    lo, hi = clip
    base = plt.get_cmap(name)
    return mcolors.LinearSegmentedColormap.from_list(f'trunc_{name}', base(np.linspace(lo, hi, n)))

# =============================================================================
# FIGURE-PANEL PLOTTING FUNCTIONS (FIGURES 3-8 OF THE MANUSCRIPT)
# =============================================================================
# Each of Figures 3-8 is split into separate A/B/C panels, one per quantity or
# diagnostic being plotted, saved as its own PNG and merged afterwards with
# merge_pngs_horizontal/merge_pngs_vertical. Reasoning for this is to control the
# orientation and layout of each individual panel, so that they can be adapted
# (e.g. for poster presentations) without reworking the whole figure.
# =============================================================================

def plot_anomaly_map_panel(kind, exps, fig_titles, change_df, ctrl_exp, panel_index, n_rows,
                           levels, cbar_kwargs, savepath, suptitle, suptitle_y=None,
                           exp_overrides=None,
                           show_nino_boxes=False, nino_boxes_n_axes=1,
                           nino_box_kwargs=None):
    """
    SST ('ssta') or OHC ('ohc') anomaly map panel, stacked n_rows x 1 by experiment
    (used for figure_3/5/7 panels A and B).

    Parameters
    ----------
    kind ({'ssta', 'ohc'})    : Quantity to map, OND-mean surface temperature anomaly or
                                OND-mean upper-300m ocean heat content anomaly.
    exps (sequence)           : Experiment/internal IDs, one map row each, control first.
    fig_titles (dict)         : Per-experiment subplot titles, keyed by experiment ID.
    change_df (pd.DataFrame)  : Niño-box anomaly table from compute_nino_ssta_change or
                                compute_nino_ohca_change, annotated onto each map.
    ctrl_exp (str)            : Experiment ID treated as the control, i.e. the row against
                                which the other rows' Niño-box changes (∆) are quoted.
    panel_index (int)         : Index of this panel within its figure, offsetting the panel
                                lettering so panel B carries on where panel A stopped.
    n_rows (int)              : Number of stacked map rows (i.e. of experiments).
    levels (sequence)         : Contour levels for the anomaly shading.
    cbar_kwargs (dict)        : Keyword arguments forwarded to add_colorbar.
    savepath (str)            : Output path for the panel PNG.
    suptitle (str)            : Figure-level title.
    suptitle_y (float)        : Optional y position for the suptitle, to be nudged as the
                                number of stacked rows changes.
    exp_overrides (dict)      : Per-experiment override of {'baseline'}, needed by HG3-1995.
    show_nino_boxes (bool)    : If True, overlay the Niño 3/3.4/4 region outlines on the maps.
    nino_boxes_n_axes (int)   : Number of axes (counted from the top) on which to draw the
                                outlines. 1 = first (top) axis only, 2 = first two, etc.
    nino_box_kwargs (dict)    : Extra keyword arguments passed through to add_nino_box_outlines
                                (e.g. dict(label_fontsize=14)).
    """
    exp_overrides = exp_overrides or {}
    nino_box_kwargs = nino_box_kwargs or {}
    unit_suffix = '˚C' if kind == 'ssta' else ' GJ.m⁻²'  # shown after each Nino-box number in the text annotation

    # Pacific-centred map projection, one row of map axes per experiment (sharing the
    # same extent and colour scale so the panels are directly comparable)
    projection = ccrs.PlateCarree(central_longitude=180)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 5 * n_rows),
                             subplot_kw={'projection': projection}, sharex=True, sharey=True)
    axes = axes.flatten()
    # the suptitle sits just above the top panel by default; suptitle_y lets the caller nudge
    # it, which is needed as the number of stacked rows (and so the figure height) changes
    suptitle_kwargs = dict(fontsize=20, x=0.515, y=1.00)
    if suptitle_y is not None:
        suptitle_kwargs['y'] = suptitle_y
    fig.suptitle(suptitle, **suptitle_kwargs)

    for row, (exp, ax) in enumerate(zip(exps, axes)):
        baseline = exp_overrides.get(exp, {}).get('baseline', 'BASELINE_NPD')

        if kind == 'ssta':
            # OND-mean SST anomaly at the surface, experiment minus baseline
            field = (ond_average(exp, freq='1m')['toce_con']
                     - ond_average(baseline, freq='1m')['toce_con'].values).sel(depth=0, method='nearest')
        elif kind == 'ohc':
            # OND-mean upper-300m ocean heat content anomaly, converted to GJ.m^(-2), experiment minus baseline
            mean_exp = xr.merge([ohc_t300m_calc(m=m, freq='1m', exp=exp) for m in (10, 11, 12)])
            mean_bs = xr.merge([ohc_t300m_calc(m=m, freq='1m', exp=baseline) for m in (10, 11, 12)])
            field = (mean_exp['ohc_t300m'] - mean_bs['ohc_t300m'].values).mean(dim='time') / 1e9
        else:
            raise ValueError(f"Unsupported kind: {kind!r}")

        # shift longitude from -180..180 to 0..360 so the Pacific is not split across the edge
        field['nav_lon'] = (field['nav_lon'] + 360) % 360
        c = field.plot.contourf(x='nav_lon', y='nav_lat', add_colorbar=False, cmap='RdBu_r',
                                levels=levels, transform=ccrs.PlateCarree(), ax=ax, extend='both')

        # Niño region outlines, drawn on the first 'nino_boxes_n_axes' axes only
        if show_nino_boxes and row < nino_boxes_n_axes:
            add_nino_box_outlines(ax, **nino_box_kwargs)

        # panel letters run a, b, c... row by row; panel_index shifts the starting
        # letter so panel B picks up numbering right where panel A left off
        panel_letter = chr(ord('a') + panel_index * n_rows + row)
        ax.set_title(f'{panel_letter}) ' + fig_titles[exp], fontsize=20, pad=12)

        # build the Nino 3/3.4/4 text annotation, adding the change relative to
        # the control experiment on every row except the control's own row
        nb_labels = {'N3': 'Niño 3   : ', 'N34': 'Niño 3.4: ', 'N4': 'Niño 4   : '}
        exp_rows = change_df[change_df['EXP'] == exp][['ANOM_AVG']]
        text_lines = []
        for nb in ('N3', 'N34', 'N4'):
            val = exp_rows.at[nb, 'ANOM_AVG']
            line = f"{nb_labels[nb]}{val:.2f}{unit_suffix}"
            if exp != ctrl_exp:
                ctrl_rows = change_df[change_df['EXP'] == ctrl_exp][['ANOM_AVG']]
                delta = val - ctrl_rows.at[nb, 'ANOM_AVG']
                line += f' (∆ = {delta:+.2f})'
            text_lines.append(line)
        text = '\n'.join(text_lines)
        ax.text(0.01, 0.975, text, transform=ax.transAxes, ha='left', va='top', fontsize=20, color='black',
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.85, boxstyle='round,pad=0.25'))

        format_axis(ax, 'map', row, n_rows)

    add_colorbar(fig, axes=list(axes), mappable=c, **cbar_kwargs)
    fig.tight_layout()
    fig.savefig(savepath, bbox_inches='tight', dpi=300)

def plot_zonal_panel(exps, fig_titles, panel_index, n_rows, levels, cbar_kwargs, savepath, suptitle,
                      suptitle_y=None, tight_layout_pad=1.85, default_baseline='BASELINE_NPD',
                      default_y_index=125, exp_overrides=None):
    """
    Equatorial zonal cross-section panel of the OND-mean ocean temperature anomaly, stacked
    n_rows x 1 by experiment (used for figure_3/5/7 panel C).

    Parameters
    ----------
    exps (sequence)          : Experiment/internal IDs, one cross-section row each, control first.
    fig_titles (dict)        : Per-experiment subplot titles, keyed by experiment ID.
    panel_index (int)        : Index of this panel within its figure, offsetting the panel
                               lettering so it carries on where the previous panel stopped.
    n_rows (int)             : Number of stacked rows (i.e. of experiments).
    levels (sequence)        : Contour levels for the anomaly shading.
    cbar_kwargs (dict)       : Keyword arguments forwarded to add_colorbar.
    savepath (str)           : Output path for the panel PNG.
    suptitle (str)           : Figure-level title.
    suptitle_y (float)       : Optional y position for the suptitle, to be nudged as the
                               number of stacked rows changes.
    tight_layout_pad (float) : Padding passed to fig.tight_layout, likewise row-count dependent.
    default_baseline (str)   : Baseline run every experiment is differenced against by default.
    default_y_index (int)    : Grid row index of the equator on the eORCA025 grid.
    exp_overrides (dict)     : Per-experiment override of {'baseline', 'y_index'}, needed by
                               HG3-1995, which sits on the eORCA12 grid.
    """
    exp_overrides = exp_overrides or {}

    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 5 * n_rows))
    axes = axes.flatten()

    # as in plot_anomaly_map_panel, the suptitle is placed just above the top panel unless the
    # caller nudges it with suptitle_y
    suptitle_kwargs = dict(fontsize=20, x=0.515, y=1.00)
    if suptitle_y is not None:
        suptitle_kwargs['y'] = suptitle_y
    fig.suptitle(suptitle, **suptitle_kwargs)

    for row, (exp, ax) in enumerate(zip(exps, axes)):
        override = exp_overrides.get(exp, {})
        # HG3-1995 lives on a different ocean grid, so it needs its own baseline and
        # equatorial row index (y_index) rather than the eORCA025 defaults
        baseline = override.get('baseline', default_baseline)
        y_index = override.get('y_index', default_y_index)

        # depth vs longitude slice along the equator (fixed y_index), experiment minus baseline
        zonal_cs = (ond_average(exp, freq='1m')['toce_con']
                    - ond_average(baseline, freq='1m')['toce_con'].values).sel(depth=slice(0, 310))[:, y_index, :]
        zonal_cs['nav_lon'] = (zonal_cs['nav_lon'] + 360) % 360
        c = zonal_cs.plot.contourf(x='nav_lon', y='depth', ax=ax, yincrease=False, levels=levels,
                                    cmap='RdBu_r', add_colorbar=False, extend='both')

        # same row-major lettering scheme as plot_anomaly_map_panel above
        panel_letter = chr(ord('a') + panel_index * n_rows + row)
        ax.set_title(f'{panel_letter}) ' + fig_titles[exp], fontsize=20, pad=12)

        format_axis(ax, 'zonal', row, n_rows)

    add_colorbar(fig, axes=list(axes), mappable=c, **cbar_kwargs)
    fig.tight_layout(pad=tight_layout_pad)
    fig.savefig(savepath, bbox_inches='tight', dpi=300)

def plot_cumulative_attribution_panel(exp_ids, ctrl, ctrl_label, savepath, y_tick_intervals=1, exp_ds_dict=None,
                                      depth_ranges=['0-60', '60-120', '120-180']):
    """
    Cumulative (experiment minus control) contribution to box heat content, one
    column per depth bracket and one row per experiment (used for figure_4/6/8 panel A).

    Parameters
    ----------
    exp_ids (sequence)        : Experiment/internal IDs, one row of panels each.
    ctrl (xr.Dataset)         : Budget dataset of the control equivalent to difference against.
    ctrl_label (str)          : Display name of that control, used in the panel titles.
    savepath (str)            : Output path for the panel PNG.
    y_tick_intervals (float)  : Major y-tick spacing, in ZJ.
    exp_ds_dict (dict)        : Budget datasets keyed by experiment ID (the notebook's EXPS).
    depth_ranges (list)       : Depth brackets to plot as the three columns, as labelled on the
                                'depth_range' coordinate of the budget files.
    """
    time = ctrl.time_counter.values
    # seconds between successive daily time steps, needed to turn a power (W) into
    # an energy (J) when cumulatively summing below
    dt = np.diff(time).astype('timedelta64[s]').astype(float)
    dt = np.append(dt, dt[-1])

    # drop day 1 (dhdt is NaN there by construction), once, and apply it everywhere
    sl = slice(1, None)
    time_c = time[sl]
    cum = lambda da: np.cumsum(np.asarray(da)[sl] * dt[sl]) / 1e21  # watts to cumulative zettajoules

    n_rows = len(exp_ids)
    # sec 3.1 only ever has one experiment row, everything else has two
    fig, axes = plt.subplots(n_rows, 3, figsize=(20, 4 if n_rows == 1 else 6), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)  # keep 2D indexing valid even when n_rows == 1

    # ri walks down the experiment rows, di walks across the three depth-bracket columns
    for ri, exp_id in enumerate(exp_ids):
        exp_ds = exp_ds_dict[exp_id]
        for di, dr in enumerate(depth_ranges):
            e, c = exp_ds.sel(depth_range=dr), ctrl.sel(depth_range=dr)

            # reference-independent gradient advection, split by direction
            dvert = cum(e.heat_grad_vert) - cum(c.heat_grad_vert)
            dzon = cum(e.heat_grad_zonal) - cum(c.heat_grad_zonal)
            dmer = cum(e.heat_grad_merid) - cum(c.heat_grad_merid)

            # online non-advective bundle: mixing plus convection plus surface forcing
            # (this carries the surface flux the offline budget is blind to at 0-60 m)
            dnadv_res = cum(e.online_non_adv_sum) - cum(c.online_non_adv_sum)

            # daily-mean advective sampling error: online total advection minus the
            # offline gradient advection (the part u.gradT from daily means cannot see)
            drect = (cum(e.online_advection) - cum(c.online_advection)) - (dvert + dzon + dmer)

            # net cumulative heat change; the lines above are a decomposition of this total
            dnet = cum(e.dhdt) - cum(c.dhdt)

            ax = axes[ri, di]
            ax.axhline(0, color='0.6', lw=0.7)
            ax.plot(time_c, dvert, color='#c0392b', lw=1.8, label='Vertical advection')
            ax.plot(time_c, dzon, color='#2471a3', lw=1.8, label='Zonal advection')
            ax.plot(time_c, dmer, color='#27ae60', lw=1.2, label='Meridional advection')
            ax.plot(time_c, dnadv_res, color='#505050', lw=1.2, ls='--',
                    label='Non-advective (mixing + convection + surf. flux)')
            ax.plot(time_c, drect, color='#7f8c8d', lw=1.2, ls=':', label='Advection residual (offline vs online)')
            ax.plot(time_c, dnet, color='k', lw=2.2, label='Net (= $\\Delta H_{EXP-CTRL}$)')

            panel_letter = chr(ord('a') + ri * 3 + di)
            ax.set_title(f'{panel_letter}) {exp_id} - {ctrl_label} | {dr} m', fontsize=18, pad=12)

            # only the leftmost column of the row is given a y-axis label
            format_axis(ax, 'cumulative', ri, n_rows, ylabel='EXP - CTRL (ZJ)' if di == 0 else '',
                        y_tick_intervals=y_tick_intervals)

    fig.suptitle('Cumulative advective and non-advective contribution to box-integrated Nino 3.4 heat anomaly: EXP - CTRL',
                 fontsize=20, y=1)

    # collect one legend entry per unique label across every subplot, so the six
    # line styles above are only listed once in the shared legend at the bottom
    handles, labels = [], []
    for a in axes.flat:
        for handle, label in zip(*a.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.115), frameon=True, fontsize=13)
    fig.tight_layout(h_pad=2, w_pad=1)
    fig.savefig(savepath, bbox_inches='tight', dpi=300)

def plot_trajectory_panel(runs, n_exp_rows, group_title, savepath, x_tick_intervals=1, y_tick_intervals=0.5, clim=None, depth_ranges=['0-60', '60-120', '120-180']):
    """
    Zonal vs vertical cumulative advective heat anomaly trajectory relative to
    climatology, one subplot per depth bracket (used for figure_4/6/8 panel B).

    Parameters
    ----------
    runs (list)               : (label, dataset, marker, cmap_name) tuples, control first; every
                                run is overlaid on each subplot as its own colour-graded trajectory.
    n_exp_rows (int)          : Number of experiment rows that panel A used, so that the panel
                                lettering here carries on from it.
    group_title (str)         : Figure-level title naming the experiment group.
    savepath (str)            : Output path for the panel PNG.
    x_tick_intervals (float)  : Major x-tick spacing, in degC.
    y_tick_intervals (float)  : Major y-tick spacing, in degC.
    clim (xr.Dataset)         : 365-day day-of-year budget climatology of BASELINE_NPD, prepared
                                in the notebook's reader cell, used as the trajectory reference.
    depth_ranges (list)       : Depth brackets to plot as the three subplots, as labelled on the
                                'depth_range' coordinate of the budget files.
    """
    rho_cp = 4.0956565e6  # rho times c_p, in J per m3 per K
    clip = (0.25, 0.85)  # drop the lightest 25% of each colormap
    norm = Normalize(1, 12)  # month 1 to month 12, for the colour gradient along each trajectory

    # clim must already be the 365-day day-of-year climatology, day-aligned with the
    # runs (built in the reader cell). Re-running the MM-DD groupby here would not be
    # idempotent and would corrupt clim on a second execution, so only assert it here.
    assert clim.sizes['time_counter'] == 365, 'clim is not the 365-day cycle; run the climatology-prep cell first.'

    # panel A used one letter per (experiment row, depth bracket) pair, so panel B
    # picks up numbering right after that
    letter_offset = n_exp_rows * 3

    # one subplot per depth bracket; every entry in 'runs' is overlaid on each
    # subplot as its own colour-graded trajectory through the year
    fig, axes = plt.subplots(1, 3, figsize=(20, 4), sharex=True, sharey=True)
    for di, dr in enumerate(depth_ranges):
        ax = axes[di]
        for lab, run, mk, cmap_name in runs:
            cmap = truncate_cmap(cmap_name, clip=clip)
            sm = ScalarMappable(norm=norm, cmap=cmap)
            tt = run.time_counter.values
            # seconds between successive time steps and the calendar month of each,
            # used to weight and colour the trajectory segments below
            dtt = np.diff(tt).astype('timedelta64[s]').astype(float)
            dtt = np.append(dtt, dtt[-1])
            mon = run.time_counter.dt.month.values
            rcv = rho_cp * run.sel(depth_range=dr).vol_box.values  # exact, time-varying box volume times rho*cp

            # gradient terms relative to climatology, in the same units as the run
            zon = run.sel(depth_range=dr).heat_grad_zonal.values - clim.sel(depth_range=dr).heat_grad_zonal.values
            ver = run.sel(depth_range=dr).heat_grad_vert.values - clim.sel(depth_range=dr).heat_grad_vert.values
            assert np.isfinite(zon).all() and np.isfinite(ver).all(), f'{lab} {dr}: NaN in gradient terms'

            # cumulative degC contribution from each direction, division done inside the sum
            x = np.cumsum(zon * dtt / rcv)
            y = np.cumsum(ver * dtt / rcv)
            # build one line segment per time step so each can be coloured by its month
            pts = np.array([x, y]).T.reshape(-1, 1, 2)
            seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(seg, cmap=cmap, norm=norm, linewidths=2.0, zorder=3)
            lc.set_array(mon[:-1])
            ax.add_collection(lc)
            # open marker at the trajectory's final point
            ax.scatter(x[-1], y[-1], marker=mk, s=90, facecolor='white',
                       edgecolor=sm.to_rgba(12), lw=2.2, zorder=4)

        # zero lines and a star marking the common starting point of every trajectory
        ax.axhline(0, color='k', lw=0.6)
        ax.axvline(0, color='k', lw=0.6)
        ax.scatter(0, 0, marker='*', s=180, facecolor='white', edgecolor='k', lw=1.2, zorder=6)

        # only the leftmost subplot is given a y-axis label
        format_axis(ax, 'trajectory', 0, 1,
                    xlabel='Cumulative zonal adv. warming anom. ($-u\\,\\partial_x T$) [°C]',
                    ylabel='Cumulative vertical adv. \nwarming anom. ($-w\\,\\partial_z T$) [°C]' if di == 0 else '',
                    x_tick_intervals=x_tick_intervals, y_tick_intervals=y_tick_intervals)
        panel_letter = chr(ord('a') + letter_offset + di)
        ax.set_title(f'{panel_letter}) {dr} m', fontsize=18, pad=12)

    # month colorbar, aligned to the rightmost panel
    pos = axes[2].get_position()
    cax = fig.add_axes([pos.x1 + 0.1, pos.y0, 0.012, pos.height])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=truncate_cmap('Greys', clip=clip)), cax=cax, ticks=range(1, 13))
    cb.ax.set_yticklabels(list('JFMAMJJASOND'))
    cb.set_label('Month (light → dark = Jan → Dec)', fontsize=16, labelpad=10)
    cb.ax.tick_params(labelsize=14)

    # shared legend of the run marker/colour styling, built by hand since each trajectory is a
    # colour-graded LineCollection rather than a single labelled line
    h1 = [Line2D([0], [0], color=plt.get_cmap(cm)(0.85), marker=m, ls='-', ms=7,
                 markerfacecolor='white', markeredgecolor=plt.get_cmap(cm)(1.0), markeredgewidth=2, label=l)
          for l, _, m, cm in runs]
    h1 += [Line2D([0], [0], marker='*', mfc='white', mec='k', mew=1.2, ls='none', ms=12, label='Year start (Jan 1)')]
    fig.legend(handles=h1, loc='upper center', bbox_to_anchor=(0.5125, axes[1].get_position().y0 - 0.095),
               ncol=4, frameon=True, fontsize=16)

    fig.suptitle(group_title, fontsize=20)
    fig.tight_layout(h_pad=2, w_pad=1)
    fig.savefig(savepath, bbox_inches='tight', dpi=300)

def plot_z20_panel(exps, labels, period, letter_offset, savepath, figsize=(20, 4.5),
                    default_baseline='BASELINE_NPD', default_freq='1d',
                    baseline_styles=None, exp_overrides=None, extra_left_lines=None,
                    ylim_right=(30, -20), yticks_right=None,
                    legend_kwargs_left=None, legend_kwargs_right=None):
    """
    Depth of the 20C isotherm (left axis) and its anomaly against a climatology
    baseline (right axis), both averaged over 'period', one line per experiment
    (used for figure_4/6/8 panel C).

    Parameters
    ----------
    exps (sequence)            : Experiment/internal IDs to draw, control first.
    labels (sequence)          : Legend label per experiment, in the same order as 'exps'.
    period (tuple)             : (start, end) date strings to average over (e.g. OND).
    letter_offset (int)        : Panel-letter offset, so the two axes here carry on the
                                 lettering from the preceding panels of the figure.
    savepath (str)             : Output path for the panel PNG.
    figsize (tuple)            : Figure size, in inches.
    default_baseline (str)     : Baseline run every experiment is differenced against by default.
    default_freq (str)         : Output frequency of the Z20 files to read by default.
    baseline_styles (dict)     : {'label', 'color', 'ls'} per baseline, for its reference line.
    exp_overrides (dict)       : Per-experiment override of {'baseline', 'freq', 'reindex_year'},
                                 needed by HG3-1995 (monthly HG3 baseline, relabelled time axis).
    extra_left_lines (list)    : Optional (data, label, colour, linestyle) reference curves for
                                 the left axis only. Used to show the Z20 of the 'donor' years
                                 (CTRL_2023, HG3-1995) whose anomalous westerlies were
                                 transplanted onto JRA55-2013 to build the ANWTRN experiments.
    ylim_right (tuple)         : Anomaly-axis limits, given (positive, negative).
    yticks_right (sequence)    : Anomaly-axis tick locations; derived from ylim_right if omitted.
    legend_kwargs_left (dict)  : Extra keyword arguments for the left axis' legend.
    legend_kwargs_right (dict) : Extra keyword arguments for the right axis' legend.
    """
    exp_overrides = exp_overrides or {}
    baseline_styles = baseline_styles or {}
    extra_left_lines = extra_left_lines or []
    legend_kwargs_left = legend_kwargs_left or {}
    legend_kwargs_right = legend_kwargs_right or {}

    if yticks_right is None:
        yticks_right = np.arange(ylim_right[0], ylim_right[1] - 1, -10)

    def read_z20(exp, freq):
        # equatorial Pacific crop of one run's year-long Z20 file, with the x dimension
        # relabelled to longitudes (taken along the middle row of the crop) in 0-360
        ds = cdo.sellonlatbox(
            '130,270,-5,5',
            input=f'./data/Z20/Z20_{exp}_{freq}_M01-12.nc',
            returnXDataset=True
        )
        ds['x'] = ds['nav_lon'][ds['nav_lon'].shape[0] // 2, :].values
        ds['x'] = (ds['x'] + 360) % 360
        return ds

    # left axis: absolute Z20 depth for every experiment plus the climatology
    # reference line(s); right axis: each experiment's Z20 anomaly vs its baseline
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes = axes.flatten()

    baseline_cache = {}
    drawn_baselines = set()
    for exp, label in zip(exps, labels):
        # per-experiment overrides (used by HG3-1995) win over the section-wide defaults
        cfg = {'baseline': default_baseline, 'freq': default_freq, **exp_overrides.get(exp, {})}
        exp_z20 = read_z20(exp, cfg['freq'])
        if 'reindex_year' in cfg:
            # relabel a monthly climatology onto the comparison year for time alignment
            exp_z20['time'] = pd.to_datetime([
                '{}-{:02d}-01'.format(cfg['reindex_year'], pd.to_datetime(t).month) for t in exp_z20['time'].values
            ])

        # read each distinct baseline file only once and cache it across experiments
        cache_key = (cfg['baseline'], cfg['freq'])
        if cache_key not in baseline_cache:
            baseline_cache[cache_key] = read_z20(cfg['baseline'], cfg['freq'])
        bs_z20 = baseline_cache[cache_key]
        bs_z20['time'] = exp_z20['time']

        if cfg['baseline'] not in drawn_baselines:
            # draw each distinct climatology reference line only the first time it is used
            style = baseline_styles.get(cfg['baseline'], {'label': 'Clim. mean', 'color': 'k', 'ls': '-'})
            bs_z20['Z20'].sel(time=slice(*period)).mean(dim=['y', 'time']).plot(
                yincrease=False, label=style['label'], ax=axes[0], color=style['color'], ls=style['ls'], lw=1.25)
            drawn_baselines.add(cfg['baseline'])

        # this experiment's own Z20 depth line, and its anomaly against the baseline above
        exp_z20_anoms = exp_z20['Z20'] - bs_z20['Z20'].values
        exp_z20['Z20'].sel(time=slice(*period)).mean(dim=['y', 'time']).plot(
            yincrease=False, label=label, ax=axes[0], lw=1.25)
        exp_z20_anoms.sel(time=slice(*period)).mean(dim=['y', 'time']).plot(
            yincrease=True, label=label, ax=axes[1], lw=1.75)

    # any extra pre-computed reference curves (e.g. donor-wind Z20), left axis only
    for data, label, color, ls in extra_left_lines:
        data.sel(time=slice(*period)).mean(dim=['y', 'time']).plot(
            yincrease=False, label=label, ax=axes[0], color=color, ls=ls, lw=1.25)

    # depth increases downward on the left axis, so the limits are given high to low.
    # ylim_right is given as (positive, negative), e.g. (30, -20): passing the
    # positive bound as the axis bottom and the negative bound as the axis top
    # makes the anomaly axis read negative at the top, positive at the bottom
    format_axis(axes[0], 'z20', 0, 1, ylabel='Z20 (m)', ylim=(220, 0), yticks=np.arange(0, 221, 30))
    format_axis(axes[1], 'z20', 0, 1, ylabel='Z20 Anomaly (m)', ylim=ylim_right, yticks=yticks_right)

    axes[0].legend(fontsize=14, **legend_kwargs_left)
    axes[1].legend(fontsize=14, **legend_kwargs_right)
    letter_a, letter_b = chr(ord('a') + letter_offset), chr(ord('a') + letter_offset + 1)
    axes[0].set_title(f'{letter_a}) Depth of 20˚C isotherm (Z20)', fontsize=18, pad=12)
    axes[1].set_title(f'{letter_b}) Z20 Anomaly w.r.t. climatological mean', fontsize=18, pad=12)
    axes[1].axhline(0, color='k', linewidth=2, linestyle='--')

    fig.tight_layout()
    fig.savefig(savepath, bbox_inches='tight', dpi=300)
