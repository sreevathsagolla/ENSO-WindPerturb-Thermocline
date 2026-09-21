"""
Offline volume and heat budgets for a 3D box in the NEMO eORCA025 model output, computed by the
face-transport method.

Processing in this script includes:
-------------------------------------
1. Locating the box in grid-index space from the requested lat/lon/depth bounds (find_box_indices).

2. Volume budget from the face transports, checked against d(Volume)/dt from the time-varying e3t.

3. Heat budget in FLUX form, rho*cp*sum(V*T_face) through the six box faces (top, bottom, east, 
west, north, south), heat content of the box + its dH/dt tendency.

4. Heat budget in GRADIENT form, -rho*cp*integral(u.gradT)dV, split into its zonal (u dT/dx),
meridional (v dT/dy) and vertical (w dT/dz) parts.

5. Volume-integrating whichever of NEMO's own online heat trends are present in the T-grid output.

Inputs:
  gridT (dst) : toce_con, e3t; optionally the online trends (ocontempadvect, ocontempdiff,
                ocontemppmdiff, ocontemptend, ttrd_evd_li, ttrd_bbl_li, ttrd_atf_li, ttrd_qns_li,
                rsdoabsorb) and the surface fluxes (hfds/qt_oce, sowaflup/empmr)
  gridU/V/W   : uo and e3u / vo and e3v / wo
  domain      : e1t, e2t, e2u, e1v, from mesh_mask.nc or domain_cfg.nc

NEMO C-grid: U(i,j) sits on the eastern face of T-cell (i,j) and V(i,j) on its northern face, W(k)
at the cell top/bottom with k increasing downward; uo, vo and wo are positive eastward, northward and
upward. The surface fluxes are not written into the 1d output of these experiments, so for a
surface box vol_top and heat_top come out as NaN and the surface forcing enters the budget only
through the online trends.

-------------------------------------------------------------------------------------------
Author: Sreevathsa G. (sg13n23@soton.ac.uk; ORCID ID: 0000-0003-4084-9677)
Last updated: 21 September 2026
"""

# Import statements
import numpy as np
import xarray as xr

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

RHO_0  = 1026.0            # reference seawater density (kg.m^(-3))
CP     = 3991.86795711963  # specific heat capacity of seawater (J.kg^(-1).K^(-1))
RHO_CP = RHO_0 * CP        # volumetric heat capacity (J.m^(-3).K^(-1))

# =============================================================================
# GENERIC HELPER FUNCTIONS USED BY THE BUDGET CALCULATION
# =============================================================================

def nearest_index(arr, value):
    # Index of the element of 'arr' lying closest to 'value'.
    return int(np.argmin(np.abs(np.asarray(arr) - value)))

def clean_fill_values(da):
    # Zero out the NEMO fill values (|val| > 1e10)
    return da.where(np.abs(da) < 1e10, 0.0)

# =============================================================================
# LOCATING THE BUDGET BOX IN GRID-INDEX SPACE
# =============================================================================

def find_box_indices(dst, lat_bounds, lon_bounds, depth_bounds):
    """
    Convert the geographic bounds of the budget box into (y, x, deptht) grid indices, and report the 
    box those indices describe. The lookup uses median lat/lon profiles along each axis,
    which is accurate on the ORCA tripolar grid in the tropics and cheaper than searching the full
    2D nav_lat/nav_lon fields.

    Parameters
    ----------
    dst (xr.Dataset)     : T-grid output, read for its nav_lat, nav_lon and deptht coordinates.
    lat_bounds (tuple)   : (south, north) bounds of the box, in degrees north.
    lon_bounds (tuple)   : (west, east) bounds of the box, in degrees east.
    depth_bounds (tuple) : (top, bottom) bounds of the box, in metres.

    Returns
    -------
    dict : Inclusive box indices {'j_s', 'j_n', 'i_w', 'i_e', 'k_top', 'k_bot'}, plus 'surface_box'
           (bool), True when the box reaches the surface and so is closed at the top by a flux.
    """
    nav_lat = dst['nav_lat'].values
    nav_lon = dst['nav_lon'].values
    deptht  = dst['deptht'].values

    # Median latitude profile along y and median longitude profile along x
    lat_1d = np.nanmedian(nav_lat, axis=1)
    lon_1d = np.nanmedian(nav_lon, axis=0)

    # Nearest grid index to each of the six requested bounds
    j_s   = nearest_index(lat_1d, lat_bounds[0])
    j_n   = nearest_index(lat_1d, lat_bounds[1])
    i_w   = nearest_index(lon_1d, lon_bounds[0])
    i_e   = nearest_index(lon_1d, lon_bounds[1])
    k_top = nearest_index(deptht, depth_bounds[0])
    k_bot = nearest_index(deptht, depth_bounds[1])

    # Swapping the pairs where needed, so the bounds can be given in either order
    if j_s > j_n:     
        j_s, j_n = j_n, j_s
    if i_w > i_e:     
        i_w, i_e = i_e, i_w
    if k_top > k_bot: 
        k_top, k_bot = k_bot, k_top

    surface_box = (k_top == 0)

    # Reporting the box in index and geographic space
    print("=" * 60)
    print("BOX DEFINITION")
    print("=" * 60)
    print(f"  y : {j_s} to {j_n}  "
          f"({lat_1d[j_s]:.2f}N to {lat_1d[j_n]:.2f}N)")
    print(f"  x : {i_w} to {i_e}  "
          f"({lon_1d[i_w]:.2f}E to {lon_1d[i_e]:.2f}E)")
    print(f"  z : {k_top} to {k_bot}  "
          f"({deptht[k_top]:.1f} m to {deptht[k_bot]:.1f} m)")
    print(f"  Surface box: {surface_box}")

    # Warnings: the W/S faces are read one cell outside the box and the centred gradients below
    # need a one-cell halo, neither of which exists at the edge of the supplied data
    ny, nx = nav_lat.shape
    if i_w < 1 or j_s < 1:
        print("  WARNING: box abuts western/southern edge of domain.")
    if i_e >= nx - 2 or j_n >= ny - 2:
        print("  WARNING: box abuts eastern/northern edge of domain.")
    if k_bot + 1 >= len(deptht):
        print("  WARNING: box reaches deepest model level.")
    print("=" * 60)

    return dict(j_s=j_s, j_n=j_n, i_w=i_w, i_e=i_e, k_top=k_top, k_bot=k_bot, surface_box=surface_box)

# =============================================================================
# VOLUME AND HEAT BUDGET COMPUTATION FOR ONE BOX
# =============================================================================

def compute_box_budgets(dst, dsu, dsv, dsw, domain,
                        lat_bounds, lon_bounds, depth_bounds,
                        rho_cp=RHO_CP, interp_T=True):
    """
    Volume and heat budgets for one 3D box, as a set of 1D time series on the time axis of the
    input files (daily, for every use in this repository).

    Parameters
    ----------
    dst (xr.Dataset)     : T-grid output (temperature, e3t, and any online heat trends).
    dsu (xr.Dataset)     : U-grid output (zonal velocity and e3u).
    dsv (xr.Dataset)     : V-grid output (meridional velocity and e3v).
    dsw (xr.Dataset)     : W-grid output (vertical velocity).
    domain (xr.Dataset)  : Static mesh_mask/domain_cfg scale factors, on the same (y, x) region.
    lat_bounds (tuple)   : (south, north) bounds of the box, in degrees north.
    lon_bounds (tuple)   : (west, east) bounds of the box, in degrees east.
    depth_bounds (tuple) : (top, bottom) bounds of the box, in metres.
    rho_cp (float)       : Volumetric heat capacity (J.m^(-3).K^(-1)).
    interp_T (bool)      : If True, temperature at a velocity point is the average of the two
                           T-cells either side of it; if False, the co-located T-cell value.

    Returns
    -------
    xr.Dataset : All budget terms as 1D time series: the volume transports and storage (vol_*), the
                 flux-form heat transports and content (heat_*), the gradient-form directional
                 advection (heat_grad_*, reference-independent and the ones to use for attribution)
                 and whichever online tendencies were found (online_*). The box indices, bounds,
                 rho_cp, interp_T and surface-flux availability go into the attributes.
    """
    # Box indices
    idx = find_box_indices(dst, lat_bounds, lon_bounds, depth_bounds)
    j_s, j_n = idx['j_s'], idx['j_n']
    i_w, i_e = idx['i_w'], idx['i_e']
    k_top, k_bot = idx['k_top'], idx['k_bot']
    surface_box = idx['surface_box']

    # Slices covering the interior of the box, i.e. the T-cells the budget is computed for
    ys = slice(j_s, j_n + 1)
    xs = slice(i_w, i_e + 1)
    ks = slice(k_top, k_bot + 1)

    # Static scale factors from the domain file
    e1t = domain['e1t'].squeeze()   # zonal width at T-points (m)
    e2t = domain['e2t'].squeeze()   # meridional width at T-points (m)
    e2u = domain['e2u'].squeeze()   # meridional width at U-points (m), i.e. the E/W face widths
    e1v = domain['e1v'].squeeze()   # zonal width at V-points (m), i.e. the N/S face widths

    # The box indices are shared, so a mismatch would silently compute the budget for the wrong box
    assert e1t.shape[-2:] == dst['toce_con'].shape[-2:], \
        "Domain (y,x) shape does not match output. Subset to the same region."

    # Time-varying fields, with the fill values zeroed out
    T   = clean_fill_values(dst['toce_con'])  # conservative temperature (degC)
    e3t = clean_fill_values(dst['e3t'])       # T-cell thickness (m), time-varying under QCO/VVL
    uo  = clean_fill_values(dsu['uo'])        # zonal velocity, thickness-weighted daily mean
    e3u = clean_fill_values(dsu['e3u'])       # U-cell thickness (m)
    vo  = clean_fill_values(dsv['vo'])        # meridional velocity, thickness-weighted daily mean
    e3v = clean_fill_values(dsv['e3v'])       # V-cell thickness (m)
    wo  = clean_fill_values(dsw['wo'])        # vertical velocity, simple daily mean

    time = dst['time_counter']
    nz   = dst.sizes['deptht']

    # Horizontal area of every T-cell in the box, used to integrate the online trends
    area = e1t.isel(y=ys, x=xs) * e2t.isel(y=ys, x=xs)

    # =========================================================================
    # VOLUME BUDGET
    # =========================================================================
    '''
    Sign convention, carried through to the heat budget: POSITIVE means INTO the box, so the
    eastern, northern and top faces are negated and the western, southern and bottom ones are not.
        East (x = i_e)     : +uo eastward = out    North (y = j_n)     : +vo northward = out
        West (x = i_w - 1) : +uo eastward = in     South (y = j_s - 1) : +vo northward = in
        Top  (k = k_top)   : +wo upward   = out    Bot   (k = k_bot+1) : +wo upward    = in
    '''
    print("Computing volume transports ...")

    # Lateral faces: velocity x cell thickness x face width. Each uvol_*/vvol_* 
    # is kept as its own variable because the heat budget reuses it further below.
    # -------------------------------------------------------------------------
    # Eastern face, at the U-points east of the last column of T-cells in the box
    uvol_e = (uo.isel(x=i_e, deptht=ks, y=ys) * e3u.isel(x=i_e, deptht=ks, y=ys) * e2u.isel(x=i_e, y=ys))
    vol_east = -(uvol_e.sum(dim=['deptht', 'y']))

    # Western face, one cell outside the box (U(i_w - 1) is the west side of T-cell i_w)
    uvol_w = (uo.isel(x=i_w-1, deptht=ks, y=ys) * e3u.isel(x=i_w-1, deptht=ks, y=ys) * e2u.isel(x=i_w-1, y=ys))
    vol_west = uvol_w.sum(dim=['deptht', 'y'])

    # Northern face, at the V-points north of the last row of T-cells in the box
    vvol_n = (vo.isel(y=j_n, deptht=ks, x=xs) * e3v.isel(y=j_n, deptht=ks, x=xs) * e1v.isel(y=j_n, x=xs))
    vol_north = -(vvol_n.sum(dim=['deptht', 'x']))

    # Southern face, one cell outside the box (V(j_s - 1) is the south side of T-cell j_s)
    vvol_s = (vo.isel(y=j_s-1, deptht=ks, x=xs) * e3v.isel(y=j_s-1, deptht=ks, x=xs) * e1v.isel(y=j_s-1, x=xs))
    vol_south = vvol_s.sum(dim=['deptht', 'x'])

    # Vertical faces; the flags record whether a top face had to be left as NaN
    # -------------------------------------------------------------------------
    vol_top_missing  = False
    heat_top_missing = False

    # Top face: a surface box is closed by the net freshwater flux, there being no W-point above
    # the first model level ('sowaflup' in some NEMO file-defs, 'empmr' in others)
    if surface_box and ('sowaflup' in dst or 'empmr' in dst):
        empmr = clean_fill_values(dst['sowaflup' if 'sowaflup' in dst else 'empmr']).isel(y=ys, x=xs)
        # Negated as the flux is positive out of the ocean, over RHO_0 to make it a volume flux
        vol_top = -(empmr / RHO_0 * e1t.isel(y=ys, x=xs) * e2t.isel(y=ys, x=xs)).sum(dim=['y', 'x'])
    elif surface_box:
        print("  Note: sowaflup/empmr not found; vol_top set to NaN.")
        vol_top = xr.full_like(vol_east, np.nan)
        vol_top_missing = True
    else:
        # Interior box: W(k_top) sits on the top face, +ve upward hence negated
        vol_top = -(wo.isel(deptht=k_top, y=ys, x=xs) * e1t.isel(y=ys, x=xs) * e2t.isel(y=ys, x=xs)).sum(dim=['y', 'x'])

    # Bottom face, at W(k_bot + 1); +ve upward is into the box here, so no negation
    if k_bot + 1 < nz:
        vol_bot = (wo.isel(deptht=k_bot+1, y=ys, x=xs) * e1t.isel(y=ys, x=xs) * e2t.isel(y=ys, x=xs)).sum(dim=['y', 'x'])
    else:
        # No W-level below the box, so there is nothing to exchange with
        vol_bot = xr.zeros_like(vol_east)
        print("  Note: box reaches deepest level; bottom transport = 0.")

    vol_total = vol_east + vol_west + vol_north + vol_south + vol_top + vol_bot

    # Volume storage: e3t is time-varying, so the box volume breathes with the sea surface height, 
    # and d(vol_box)/dt is what the net transport must match
    vol_box = (e3t.isel(deptht=ks, y=ys, x=xs) * e1t.isel(y=ys, x=xs) * e2t.isel(y=ys, x=xs)).sum(dim=['deptht', 'y', 'x'])

    # Time step in seconds, labelled on the second and later steps so that it divides the diffs
    tsec = time.values.astype('datetime64[s]').astype('float64')
    dt = np.diff(tsec)
    dt_da = xr.DataArray(dt, dims='time_counter', coords={'time_counter': vol_box['time_counter'][1:]})

    dvol_dt = vol_box.diff(dim='time_counter') / dt_da

    # Closure: transport minus storage change, ~0, the check on the face reconstruction and signs
    vol_residual = vol_total.sel(time_counter=dvol_dt.time_counter) - dvol_dt

    # =========================================================================
    # HEAT BUDGET, FLUX FORM (OFFLINE FACE RECONSTRUCTION)
    # =========================================================================
    '''
    Heat transport through a face is its volume transport weighted by the temperature of the water
    crossing it, rho_cp * sum(V * T_face). Temperature is carried at the T-points, so interp_T=True
    averages the two T-cells either side of the face (the default here) and interp_T=False takes
    the co-located T-cell value.
    '''
    print("Computing heat transports ...")

    # Temperature interpolated onto the velocity points
    def T_at_U(i):
        # Temperature at the U-face at x-index i, i.e. between T-cells i and i+1.
        if interp_T and 0 < i < dst.sizes['x'] - 1:
            return 0.5 * (T.isel(x=i) + T.isel(x=i+1))
        return T.isel(x=i)

    def T_at_V(j):
        # Temperature at the V-face at y-index j, i.e. between T-cells j and j+1.
        if interp_T and 0 < j < dst.sizes['y'] - 1:
            return 0.5 * (T.isel(y=j) + T.isel(y=j+1))
        return T.isel(y=j)

    def T_at_W(k):
        # Temperature at the W-face at deptht-index k, i.e. between T-cells k-1 and k.
        if interp_T and k > 0:
            return 0.5 * (T.isel(deptht=k-1) + T.isel(deptht=k))
        return T.isel(deptht=k)

    # Lateral heat transports, reusing the per-cell volume transports from above
    heat_east  = -(uvol_e * T_at_U(i_e).isel(deptht=ks, y=ys) * rho_cp).sum(dim=['deptht', 'y'])
    heat_west  = (uvol_w * T_at_U(i_w-1).isel(deptht=ks, y=ys) * rho_cp).sum(dim=['deptht', 'y'])
    heat_north = -(vvol_n * T_at_V(j_n).isel(deptht=ks, x=xs) * rho_cp).sum(dim=['deptht', 'x'])
    heat_south = (vvol_s * T_at_V(j_s-1).isel(deptht=ks, x=xs) * rho_cp).sum(dim=['deptht', 'x'])

    # Vertical heat transports; top face mirroring the volume budget: the surface heat flux for a surface box 
    # (positive into the ocean, hence no negation), a transport through W(k_top) otherwise
    if surface_box and ('hfds' in dst or 'qt_oce' in dst):
        hflx = clean_fill_values(dst['hfds' if 'hfds' in dst else 'qt_oce']).isel(y=ys, x=xs)
        heat_top = (hflx * e1t.isel(y=ys, x=xs) * e2t.isel(y=ys, x=xs)).sum(dim=['y', 'x'])
    elif surface_box:
        print("  Note: hfds/qt_oce not found; heat_top set to NaN.")
        heat_top = xr.full_like(heat_east, np.nan)
        heat_top_missing = True
    else:
        wvol_top = (wo.isel(deptht=k_top, y=ys, x=xs) * e1t.isel(y=ys, x=xs) * e2t.isel(y=ys, x=xs))
        heat_top = -(wvol_top * T_at_W(k_top).isel(y=ys, x=xs) * rho_cp).sum(dim=['y', 'x'])

    # Bottom face, at W(k_bot + 1)
    if k_bot + 1 < nz:
        wvol_bot = (wo.isel(deptht=k_bot+1, y=ys, x=xs) * e1t.isel(y=ys, x=xs) * e2t.isel(y=ys, x=xs))
        heat_bot = (wvol_bot * T_at_W(k_bot+1).isel(y=ys, x=xs) * rho_cp).sum(dim=['y', 'x'])
    else:
        heat_bot = xr.zeros_like(heat_east)

    heat_adv = heat_east + heat_west + heat_north + heat_south + heat_top + heat_bot

    # Heat content of the box and its tendency dH/dt
    heat_content = (T.isel(deptht=ks, y=ys, x=xs) * e3t.isel(deptht=ks, y=ys, x=xs) * e1t.isel(y=ys, x=xs)
                    * e2t.isel(y=ys, x=xs) * rho_cp).sum(dim=['deptht', 'y', 'x'])

    dhdt = heat_content.diff(dim='time_counter') / dt_da

    # Unlike the volume residual this does not vanish: it holds the diffusion, convection and
    # surface forcing the offline advection is blind to, plus the daily-mean sampling error.
    heat_residual = dhdt - heat_adv.sel(time_counter=dhdt.time_counter)

    # =========================================================================
    # HEAT BUDGET, GRADIENT FORM (REFERENCE-INDEPENDENT DIRECTIONAL ADVECTION)
    # =========================================================================
    '''
    The flux-form face pairs above split the net advection using face temperatures, so each
    directional term depends on the arbitrary zero of the temperature scale and is not
    interpretable on its own. The gradient form, -rho_cp * integral( u.grad T ) dV, splits the SAME
    net advection using temperature GRADIENTS, which makes each term invariant to adding a constant
    to T and maps it onto the ENSO feedback framework (zonal-advective ~ u dT/dx, thermocline
    ~ w dT/dz). Velocities are averaged onto the T-points and the gradients are centred differences 
    on the e1t/e2t/e3t spacing, z positive upward; cells abutting the edge of the supplied data fall 
    back on a one-sided difference.
    '''
    print("Computing gradient-form directional advection ...")

    # Face velocities averaged onto the T-points (+x eastward, +y northward, +z upward)
    u_T = 0.5 * (uo + uo.shift(x=1))             # 0.5*(uo[i-1] + uo[i])
    v_T = 0.5 * (vo + vo.shift(y=1))             # 0.5*(vo[j-1] + vo[j])
    w_T = 0.5 * (wo + wo.shift(deptht=-1))       # 0.5*(wo[k]   + wo[k+1])

    # Centred spacings between neighbouring T-points: half of each neighbour plus all of this cell
    dx_c = 0.5 * e1t.shift(x=1) + e1t + 0.5 * e1t.shift(x=-1)
    dy_c = 0.5 * e2t.shift(y=1) + e2t + 0.5 * e2t.shift(y=-1)
    dz_c = 0.5 * e3t.shift(deptht=1) + e3t + 0.5 * e3t.shift(deptht=-1)

    # Centred temperature gradients at the T-points
    dTdx = (T.shift(x=-1) - T.shift(x=1)) / dx_c                  # (T[i+1]-T[i-1])/dx
    dTdy = (T.shift(y=-1) - T.shift(y=1)) / dy_c                  # (T[j+1]-T[j-1])/dy
    dTdz = (T.shift(deptht=1) - T.shift(deptht=-1)) / dz_c        # (T[k-1]-T[k+1])/dz, z up

    # One-sided fallbacks; each fillna only touches the edge the line above it left as NaN
    dTdx = dTdx.fillna((T.shift(x=-1) - T) / (0.5 * e1t + 0.5 * e1t.shift(x=-1)))   # west edge
    dTdx = dTdx.fillna((T - T.shift(x=1)) / (0.5 * e1t.shift(x=1) + 0.5 * e1t))     # east edge
    dTdy = dTdy.fillna((T.shift(y=-1) - T) / (0.5 * e2t + 0.5 * e2t.shift(y=-1)))   # south edge
    dTdy = dTdy.fillna((T - T.shift(y=1)) / (0.5 * e2t.shift(y=1) + 0.5 * e2t))     # north edge
    dTdz = dTdz.fillna((T - T.shift(deptht=-1)) / (0.5 * e3t + 0.5 * e3t.shift(deptht=-1)))  # top (k=0)
    dTdz = dTdz.fillna((T.shift(deptht=1) - T) / (0.5 * e3t.shift(deptht=1) + 0.5 * e3t))    # bottom

    # Volume of every T-cell in the box, time-varying through e3t as in vol_box above
    cell_vol = (e3t.isel(deptht=ks, y=ys, x=xs) * e1t.isel(y=ys, x=xs) * e2t.isel(y=ys, x=xs))

    # Advective-tendency integrals over the box; positive means warming the box
    heat_grad_zonal = -(rho_cp * u_T.isel(deptht=ks, y=ys, x=xs)
                        * dTdx.isel(deptht=ks, y=ys, x=xs) * cell_vol).sum(dim=['deptht', 'y', 'x'])
    heat_grad_merid = -(rho_cp * v_T.isel(deptht=ks, y=ys, x=xs)
                        * dTdy.isel(deptht=ks, y=ys, x=xs) * cell_vol).sum(dim=['deptht', 'y', 'x'])
    heat_grad_vert  = -(rho_cp * w_T.isel(deptht=ks, y=ys, x=xs)
                        * dTdz.isel(deptht=ks, y=ys, x=xs) * cell_vol).sum(dim=['deptht', 'y', 'x'])
    heat_grad_adv = heat_grad_zonal + heat_grad_merid + heat_grad_vert

    # =========================================================================
    # ONLINE TENDENCY DIAGNOSTICS
    # =========================================================================
    '''
    NEMO's own heat trends, which unlike everything above carry no offline reconstruction error and
    are the only route to the diffusion, convection and surface-forcing terms. Which of them were
    written depends on the file-defs of the run, so whichever are present in dst are used and the
    rest skipped. They are already layer-integrated (W.m⁻²), hence integrated over 'area' alone.
    '''
    print("Checking online tendency diagnostics ...")

    # Output variable names of the 3D online trends, mapped onto the labels used in the output
    online_3d_vars = {
        'ocontempadvect': 'advection',
        'ocontemppmdiff': 'iso_diffusion',
        'ocontempdiff':   'vert_diffusion',
        'ttrd_evd_li':    'evd_convection',
        'ttrd_bbl_li':    'bbl',
        'ttrd_atf_li':    'asselin_filter',
        'ocontemptend':   'total_tendency',
    }

    online = {}
    for varname, label in online_3d_vars.items():
        if varname in dst:
            online[label] = (clean_fill_values(dst[varname]).isel(deptht=ks, y=ys, x=xs)
                             * area).sum(dim=['deptht', 'y', 'x'])

    # Surface non-solar flux and runoff, a 2D field and so only meaningful for a surface box
    if surface_box and 'ttrd_qns_li' in dst:
        online['sfc_nonsolar'] = (clean_fill_values(dst['ttrd_qns_li']).isel(y=ys, x=xs)
                                  * area).sum(dim=['y', 'x'])

    # Penetrative solar heating, which reaches below the surface level and so applies to every box
    if 'rsdoabsorb' in dst:
        online['sfc_solar'] = (clean_fill_values(dst['rsdoabsorb']).isel(deptht=ks, y=ys, x=xs)
                               * area).sum(dim=['deptht', 'y', 'x'])

    # The non-advective terms bundled into one series, which utils.py plots as its 'mixing +
    # convection + surface flux' line; the surface forcing joins only for a surface box.
    non_adv_keys = ['iso_diffusion', 'vert_diffusion', 'evd_convection',
                    'bbl', 'asselin_filter']
    if surface_box:
        non_adv_keys += ['sfc_nonsolar', 'sfc_solar']

    online_non_adv = None
    for key in non_adv_keys:
        if key in online:
            online_non_adv = online[key].copy() if online_non_adv is None else online_non_adv + online[key]

    n_online = len(online)
    if n_online > 0:
        print(f"  Found {n_online} online tendency fields.")
    else:
        print("  No online tendency fields found; skipping online diagnostics.")

    # =========================================================================
    # ASSEMBLING THE OUTPUT DATASET
    # =========================================================================
    # Everything above is still lazily loaded, and is evaluated in the single .compute() 
    # at the end of this section, so that Dask gets one graph per call of this function
    out = xr.Dataset()

    # Volume budget
    out['vol_east']     = vol_east
    out['vol_west']     = vol_west
    out['vol_north']    = vol_north
    out['vol_south']    = vol_south
    out['vol_top']      = vol_top
    out['vol_bot']      = vol_bot
    out['vol_total']    = vol_total
    out['dvol_dt']      = dvol_dt
    out['vol_residual'] = vol_residual
    out['vol_box']      = vol_box
    out['vol_box'].attrs.update(units='m3', long_name='e3t-weighted box volume (time-varying, VVL)')

    # Heat budget, flux form
    out['heat_east']     = heat_east
    out['heat_west']     = heat_west
    out['heat_north']    = heat_north
    out['heat_south']    = heat_south
    out['heat_top']      = heat_top
    out['heat_bot']      = heat_bot
    out['heat_adv']      = heat_adv
    out['dhdt']          = dhdt
    out['heat_residual'] = heat_residual
    out['heat_content']  = heat_content

    # Heat budget, gradient form; reference-independent
    out['heat_grad_zonal'] = heat_grad_zonal
    out['heat_grad_merid'] = heat_grad_merid
    out['heat_grad_vert']  = heat_grad_vert
    out['heat_grad_adv']   = heat_grad_adv

    # Online tendencies, only those that were found in dst
    for label, da in online.items():
        out[f'online_{label}'] = da
    if online_non_adv is not None:
        out['online_non_adv_sum'] = online_non_adv

    # Evaluating the whole graph in one pass
    out = out.compute()

    # Metadata: the box used, the settings, and whether a surface flux was available for the top
    out.attrs.update(dict(
        lat_bounds             = lat_bounds,
        lon_bounds             = lon_bounds,
        depth_bounds           = depth_bounds,
        j_s=j_s, j_n=j_n, i_w=i_w, i_e=i_e,
        k_top=k_top, k_bot=k_bot,
        surface_box            = int(surface_box),
        surface_flux_available = int(not (vol_top_missing or heat_top_missing)),
        rho_cp                 = rho_cp,
        interp_T               = int(interp_T),
    ))
    if vol_top_missing or heat_top_missing:
        nan_vars = ', '.join(
            ([' vol_top'] if vol_top_missing  else []) +
            (['heat_top'] if heat_top_missing else [])
        )
        out.attrs['surface_flux_note'] = (
            f"{nan_vars} are NaN: surface flux variables "
            "(sowaflup/empmr for freshwater, hfds/qt_oce for heat) "
            "were not found in dst. Derived net flux totals "
            "(vol_total, heat_adv) do not include the surface contribution."
        )

    print("Done. :)")
    return out