import numpy

from landlab.core.utils import as_id_array

from cyth_scripts import _accumulate_bw,_accumulate_flux, _accumulate_flux_basins, _accumulate_flux_basins_pll, _accumulate_active_flux_basins_pll, _accumulate_active_flux_basins

def get_active_flux_basins(
    s, r, active_starts,active_ends, node_cell_area=1.0, tracer_rate =1.0, boundary_nodes=None
):

    """Calculate the total flux of some tracer (e.g. water discharge from runoff) at each node by
    looping through individual drainage basins, splitting the work in parallel across the cores of the machine.

    Parameters
    ----------
    s : ndarray of int
        Ordered (downstream to upstream) array of node IDs
    r : ndarray of int
        Receiver IDs for each node
    starts : ndarray of int 
        location within 's' of sink nodes (i.e. outlets for separate basins)
    node_cell_area : float or ndarray
        Cell surface areas for each node. If it's an array, must have same
        length as s (that is, the number of nodes). 
    tracer_rate : float or ndarray
        Variable tracer flux at each cell. in units M per T (for mass, and chemical fluxes) 
        or L per T (for volumetric fluxes e.g water discharge). If set as 1.0, flux is equal 
        to upstream area.
    boundary_nodes: list, optional
        Array of boundary nodes to have discharge and drainage area set to zero.
        Default value is None.
    Returns
    -------
    ndarray flux """
    # Number of active basins
    nb = len(active_starts)
    # number of nodes
    ns = len(s)
    # Initialize the flux array. Flux starts out as the cell's local 
    # tracer rate times the cell's surface area then (unless the cell has no
    # donors) grows from there.
    flux = numpy.zeros(ns, dtype=int) + node_cell_area * tracer_rate

    # Optionally zero out drainage area and flux at boundary nodes
    if boundary_nodes is not None:
        drainage_area[boundary_nodes] = 0
        flux[boundary_nodes] = 0
        
    # Call the cfunc to work accumulate from upstream to downstream
    _accumulate_active_flux_basins(nb, s, r, active_starts, active_ends, flux)
    return flux

def get_active_flux_basins_pll(
    s, r, active_starts,active_ends, node_cell_area=1.0, tracer_rate =1.0, boundary_nodes=None
):

    """Calculate the total flux of some tracer (e.g. water discharge from runoff) at each node by
    looping through individual drainage basins, splitting the work in parallel across the cores of the machine.

    Parameters
    ----------
    s : ndarray of int
        Ordered (downstream to upstream) array of node IDs
    r : ndarray of int
        Receiver IDs for each node
    starts : ndarray of int 
        location within 's' of sink nodes (i.e. outlets for separate basins)
    node_cell_area : float or ndarray
        Cell surface areas for each node. If it's an array, must have same
        length as s (that is, the number of nodes). 
    tracer_rate : float or ndarray
        Variable tracer flux at each cell. in units M per T (for mass, and chemical fluxes) 
        or L per T (for volumetric fluxes e.g water discharge). If set as 1.0, flux is equal 
        to upstream area.
    boundary_nodes: list, optional
        Array of boundary nodes to have discharge and drainage area set to zero.
        Default value is None.
    Returns
    -------
    ndarray flux """
    # Number of active basins
    nb = len(active_starts)
    # number of nodes
    ns = len(s)
    # Initialize the flux array. Flux starts out as the cell's local 
    # tracer rate times the cell's surface area then (unless the cell has no
    # donors) grows from there.
    flux = numpy.zeros(ns, dtype=int) + node_cell_area * tracer_rate

    # Optionally zero out drainage area and flux at boundary nodes
    if boundary_nodes is not None:
        drainage_area[boundary_nodes] = 0
        flux[boundary_nodes] = 0
        
    # Call the cfunc to work accumulate from upstream to downstream
    _accumulate_active_flux_basins_pll(nb, s, r, active_starts, active_ends, flux)
    return flux


def get_flux_basins_pll(
    s, r, starts, node_cell_area=1.0, tracer_rate =1.0, boundary_nodes=None
):

    """Calculate the total flux of some tracer (e.g. water discharge from runoff) at each node by
    looping through individual drainage basins, splitting the work in parallel across the cores of the machine.

    Parameters
    ----------
    s : ndarray of int
        Ordered (downstream to upstream) array of node IDs
    r : ndarray of int
        Receiver IDs for each node
    starts : ndarray of int 
        location within 's' of sink nodes (i.e. outlets for separate basins)
    node_cell_area : float or ndarray
        Cell surface areas for each node. If it's an array, must have same
        length as s (that is, the number of nodes). 
    tracer_rate : float or ndarray
        Variable tracer flux at each cell. in units M per T (for mass, and chemical fluxes) 
        or L per T (for volumetric fluxes e.g water discharge). If set as 1.0, flux is equal 
        to upstream area.
    boundary_nodes: list, optional
        Array of boundary nodes to have discharge and drainage area set to zero.
        Default value is None.
    Returns
    -------
    ndarray flux """
    # Number of basins
    nb = len(starts)
    # Number of nodes
    ns = len(s)
    # Initialize the flux array. Flux starts out as the cell's local 
    # tracer rate times the cell's surface area then (unless the cell has no
    # donors) grows from there.
    flux = numpy.zeros(ns, dtype=int) + node_cell_area * tracer_rate

    # Optionally zero out drainage area and flux at boundary nodes
    if boundary_nodes is not None:
        drainage_area[boundary_nodes] = 0
        flux[boundary_nodes] = 0
        
    # Call the cfunc to work accumulate from upstream to downstream
    _accumulate_flux_basins_pll(nb, ns, s, r, starts, flux)
    return flux



def get_flux_basins(
    s, r, starts, node_cell_area=1.0, tracer_rate =1.0, boundary_nodes=None
):

    """Calculate the total flux of some tracer (e.g. water discharge from runoff) at each node by
    looping through individual drainage basins.

    Parameters
    ----------
    s : ndarray of int
        Ordered (downstream to upstream) array of node IDs
    r : ndarray of int
        Receiver IDs for each node
    starts : ndarray of int 
        location within 's' of sink nodes (i.e. outlets for separate basins)
    node_cell_area : float or ndarray
        Cell surface areas for each node. If it's an array, must have same
        length as s (that is, the number of nodes). 
    tracer_rate : float or ndarray
        Variable tracer flux at each cell. in units M per T (for mass, and chemical fluxes) 
        or L per T (for volumetric fluxes e.g water discharge). If set as 1.0, flux is equal 
        to upstream area.
    boundary_nodes: list, optional
        Array of boundary nodes to have discharge and drainage area set to zero.
        Default value is None.
    Returns
    -------
    ndarray flux """
    # Number of basins
    nb = len(starts)
    # Number of nodes
    ns = len(s)
    # Initialize the flux array. Flux starts out as the cell's local 
    # tracer rate times the cell's surface area then (unless the cell has no
    # donors) grows from there.
    flux = numpy.zeros(ns, dtype=int) + node_cell_area * tracer_rate

    # Optionally zero out drainage area and flux at boundary nodes
    if boundary_nodes is not None:
        drainage_area[boundary_nodes] = 0
        flux[boundary_nodes] = 0
        
    # Call the cfunc to work accumulate from upstream to downstream
    _accumulate_flux_basins(nb, ns, s, r, starts, flux)
    return flux

def get_flux(
    s, r, node_cell_area=1.0, tracer_rate =1.0, boundary_nodes=None
):

    """Calculate the total flux of some tracer (e.g. water discharge from runoff) at each node.

    Parameters
    ----------
    s : ndarray of int
        Ordered (downstream to upstream) array of node IDs
    r : ndarray of int
        Receiver IDs for each node
    node_cell_area : float or ndarray
        Cell surface areas for each node. If it's an array, must have same
        length as s (that is, the number of nodes). 
    tracer_rate : float or ndarray
        Variable tracer flux at each cell. in units M per T (for mass, and chemical fluxes) 
        or L per T (for volumetric fluxes e.g water discharge). If set as 1.0, flux is equal 
        to upstream area.
    boundary_nodes: list, optional
        Array of boundary nodes to have discharge and drainage area set to zero.
        Default value is None.
    Returns
    -------
    ndarray flux """
    # Number of points
    np = len(s)

    # Initialize the flux array. Flux starts out as the cell's local 
    # tracer rate times the cell's surface area then (unless the cell has no
    # donors) grows from there.
    flux = numpy.zeros(np, dtype=int) + node_cell_area * tracer_rate

    # Optionally zero out drainage area and flux at boundary nodes
    if boundary_nodes is not None:
        drainage_area[boundary_nodes] = 0
        flux[boundary_nodes] = 0

    # Call the cfunc to work accumulate from upstream to downstream
    _accumulate_flux(np, s, r, flux)

    return flux


def get_drainage_area_and_discharge(
    s, r, node_cell_area=1.0, runoff=1.0, boundary_nodes=None
):

    """Calculate the drainage area and water discharge at each node.

    Parameters
    ----------
    s : ndarray of int
        Ordered (downstream to upstream) array of node IDs
    r : ndarray of int
        Receiver IDs for each node
    node_cell_area : float or ndarray
        Cell surface areas for each node. If it's an array, must have same
        length as s (that is, the number of nodes).
    runoff : float or ndarray
        Local runoff rate at each cell (in water depth per time). If it's an
        array, must have same length as s (that is, the number of nodes).
    boundary_nodes: list, optional
        Array of boundary nodes to have discharge and drainage area set to zero.
        Default value is None.
    Returns
    -------
    tuple of ndarray
        drainage area and discharge

    Notes
    -----
    -  If node_cell_area not given, the output drainage area is equivalent
       to the number of nodes/cells draining through each point, including
       the local node itself.
    -  Give node_cell_area as a scalar when using a regular raster grid.
    -  If runoff is not given, the discharge returned will be the same as
       drainage area (i.e., drainage area times unit runoff rate).
    -  If using an unstructured Landlab grid, make sure that the input
       argument for node_cell_area is the cell area at each NODE rather than
       just at each CELL. This means you need to include entries for the
       perimeter nodes too. They can be zeros.

    Examples
    --------
    >>> import numpy as np
    >>> from landlab.components.flow_accum import (
    ...     find_drainage_area_and_discharge)
    >>> r = np.array([2, 5, 2, 7, 5, 5, 6, 5, 7, 8])-1
    >>> s = np.array([4, 1, 0, 2, 5, 6, 3, 8, 7, 9])
    >>> a, q = find_drainage_area_and_discharge(s, r)
    >>> a
    array([  1.,   3.,   1.,   1.,  10.,   4.,   3.,   2.,   1.,   1.])
    >>> q
    array([  1.,   3.,   1.,   1.,  10.,   4.,   3.,   2.,   1.,   1.])
    """
    # Number of points
    np = len(s)

    # Initialize the drainage_area and discharge arrays. Drainage area starts
    # out as the area of the cell in question, then (unless the cell has no
    # donors) grows from there. Discharge starts out as the cell's local runoff
    # rate times the cell's surface area.
    drainage_area = numpy.zeros(np, dtype=int) + node_cell_area
    discharge = numpy.zeros(np, dtype=int) + node_cell_area * runoff

    # Optionally zero out drainage area and discharge at boundary nodes
    if boundary_nodes is not None:
        drainage_area[boundary_nodes] = 0
        discharge[boundary_nodes] = 0

    # Call the cfunc to work accumulate from upstream to downstream, permitting
    # transmission losses
    _accumulate_bw(np, s, r, drainage_area, discharge)
    # nodes at channel heads can still be negative with this method, so...
    discharge = discharge.clip(0.0)

    return drainage_area, discharge
