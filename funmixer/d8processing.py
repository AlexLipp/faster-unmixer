# Preprocessing functions for data input
# 1. Snap samples to drainage
#    - Snap to nearest drainage line greater than a certain threshold
#    - Optionally take a dictionary of "nudges" to move points to the nearest drainage line
#    - Plot up before and after with labelled points + arrows from original to snapped
#    - Include a plot of the sub-catchments + the connected network.

import numpy as np
from osgeo import gdal
from typing import Tuple, List

import funmixer.flow_acc_cfuncs as cf


def read_geo_file(filename: str) -> Tuple[np.ndarray, gdal.Dataset]:
    """Reads a geospatial file"""
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    return arr, ds


def write_geotiff(filename: str, arr: np.ndarray, ds: gdal.Dataset) -> None:
    """Writes a numpy array to a geotiff"""
    if arr.dtype == np.float32:
        arr_type = gdal.GDT_Float32
    else:
        arr_type = gdal.GDT_Int32
    print("Type of array is", arr.dtype)
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(filename, arr.shape[1], arr.shape[0], 1, arr_type)
    out_ds.SetProjection(ds.GetProjection())
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    band = out_ds.GetRasterBand(1)
    band.WriteArray(arr)
    band.FlushCache()
    band.ComputeStatistics(False)


def check_D8(
    flowdirs_filename: str,
) -> bool:
    arr, _ = read_geo_file(flowdirs_filename)
    # Cast to int
    arr = arr.astype(int)
    # Check that the only values present are 0, 1, 2, 4, 8, 16, 32, 64, 128 using sets
    unique_values = set(np.unique(arr))
    values_are_valid = unique_values == {0, 1, 2, 4, 8, 16, 32, 64, 128}
    if not values_are_valid:
        print(
            f"VALUE CHECK RESULT: Fail. Invalid values present in D8 flow direction grid: {unique_values}. \n Expected values are { {0, 1, 2, 4, 8, 16, 32, 64, 128} }"
        )
    else:
        print("VALUE CHECK RESULT: Pass.")
    # Check that the boundaries are all 0
    boundaries_are_zero = (
        np.all(arr[0, :] == 0)
        and np.all(arr[-1, :] == 0)
        and np.all(arr[:, 0] == 0)
        and np.all(arr[:, -1] == 0)
    )
    if not boundaries_are_zero:
        print("BOUNDARY CHECK RESULT: Fail. Boundaries of D8 flow direction grid are not all 0.")
    else:
        print("BOUNDARY CHECK RESULT: Pass.")
    # Return True if both boundaries_are_zero and values_are_valid
    result = boundaries_are_zero and values_are_valid
    print("-" * 50)
    if result:
        print("D8 flow direction grid is valid.")
    else:
        print("D8 flow direction grid is INVALID.")
    return result


def set_d8_boundaries_to_zero(flowdirs_filename: str) -> None:
    """
    Sets the boundaries of a D8 flow direction grid to zero. Writes the result to a new geotiff with a prefix "zero_bounds_"

    Args:
        flowdirs_filename (str): The filename of the D8 flow direction grid
    """
    arr, ds = read_geo_file(flowdirs_filename)
    arr[0, :] = 0
    arr[-1, :] = 0
    arr[:, 0] = 0
    arr[:, -1] = 0

    # Save the new file with prefix fixed_bounds_[original_filename].tif, overwriting the original file extension
    name, _ = flowdirs_filename.split(".")
    new_filename = f"zero_bounds_{name}.tif"
    print("Writing new file with zeroed boundaries to", new_filename)
    write_geotiff(new_filename, arr, ds)


def elevation_to_d8(elevation_filename: str) -> None:
    """
    Converts a geospatial raster of elevations to a D8 flow grid and writes it to a file called [filename]_d8.tif.
    Follows convention of ESRI D8 flow directions: right = 1, lower right = 2, bottom = 4, etc.
    All boundary nodes are set to be sinks.

    Args:
        elevation_filename (str): The filename of the elevation raster
    """
    elev, ds = read_geo_file(elevation_filename)
    d8 = cf.topo_to_d8(elev.astype(float))
    # Set boundaries to zero
    d8[0, :] = 0
    d8[-1, :] = 0
    d8[:, 0] = 0
    d8[:, -1] = 0

    name, _ = elevation_filename.split(".")
    new_filename = f"d8_{name}.tif"
    print("Writing valid D8 to", new_filename)
    write_geotiff(new_filename, d8, ds)


class D8Accumulator:
    """Class to accumulate flow on a D8 flow grid. This class can be used to calculate drainage area and discharge,
    and to accumulate any other tracer across a drainage network. The class assumes that all boundary
    nodes are sinks (i.e., no flow leaves the grid). This class can be used with any geospatial file that GDAL can read.

    Parameters
    ----------
    filename : str
        Path to the D8 flow grid geospatial file (e.g., .tif, .asc, etc.)
        This can be to any file that GDAL can read. Expects a single band raster (ignores other bands).
        This raster should be a 2D array of D8 flow directions according to ESRI convention:

            Sink [no flow]= 0
            Right = 1
            Lower right = 2
            Bottom = 4
            Lower left = 8
            Left = 16
            Upper left = 32
            Top = 64
            Upper right = 128

    Attributes
    ----------
    receivers : np.ndarray
        Array of receiver nodes (i.e., the ID of the node that receives the flow from the i'th node)
    order : np.ndarray
        Array of nodes in order of upstream to downstream (breadth-first)
    baselevel_nodes : np.ndarray
        Array of baselevel nodes (i.e., nodes that do not donate flow to any other nodes)
    arr : np.ndarray
        Array of D8 flow directions
    ds : gdal.Dataset
        GDAL Dataset object of the D8 flow grid. If the array is manually set, this will be None
    extent : List[float]
        Extent of the array in the accumulator as [xmin, xmax, ymin, ymax]. Can be used for plotting.

    Methods
    -------
    accumulate(weights : np.ndarray = None)
        Accumulate flow on the grid using the D8 flow directions
    """

    def __init__(self, filename: str):
        """
        Parameters
        ----------
        filename : str
            Path to the D8 flow grid
        """
        # Check that filename is a string
        if not isinstance(filename, str):
            raise TypeError("Filename must be a string")
        self._arr, self._ds = read_geo_file(filename)
        self._arr = self._arr.astype(int)
        self._receivers = cf.d8_to_receivers(self.arr)
        self._baselevel_nodes = np.where(self.receivers == np.arange(len(self.receivers)))[0]
        self._order = cf.build_ordered_list_iterative(self.receivers, self.baselevel_nodes)

    def accumulate(self, weights: np.ndarray = None) -> np.ndarray:
        """Accumulate flow on the grid using the D8 flow directions

        Parameters
        ----------
        weights : np.ndarray [ndim = 2], optional
            Array of weights for each node, defaults to giving each node a weight of 1, resulting in a map of the number of upstream nodes.
            If the area of each node is known, this can be used to calculate drainage area. If run-off at each node is known,
            this can be used to calculate discharge.

        Returns
        -------
        np.ndarray [ndim = 2]
            Array of accumulated weights (or number of upstream nodes if no weights are passed)
        """
        if weights is None:
            # If no weights are passed, assume all nodes have equal weight of 1.
            # Output is array of # upstream nodes
            weights = np.ones(len(self.receivers))
        else:
            if weights.shape != self.arr.shape:
                raise ValueError("Weights must be have same shape as D8 array")
            weights = weights.flatten()

        return cf.accumulate_flow(self.receivers, self.order, weights=weights).reshape(
            self._arr.shape
        )

    @property
    def receivers(self) -> np.ndarray:
        """Array of receiver nodes (i.e., the ID of the node that receives the flow from the i'th node)"""
        return np.asarray(self._receivers)

    @property
    def baselevel_nodes(self) -> np.ndarray:
        """Array of baselevel nodes (i.e., nodes that do not donate flow to any other nodes)"""
        return self._baselevel_nodes

    @property
    def order(self) -> np.ndarray:
        """Array of nodes in order of upstream to downstream"""
        return np.asarray(self._order)

    @property
    def arr(self):
        """Array of D8 flow directions"""
        return self._arr

    @property
    def ds(self):
        """GDAL Dataset object of the D8 flow grid"""
        return self._ds

    @property
    def extent(self) -> List[float]:
        """
        Get the extent of the array in the accumulator. Can be used for plotting.
        """
        trsfm = self.ds.GetGeoTransform()
        minx = trsfm[0]
        maxy = trsfm[3]
        maxx = minx + trsfm[1] * self.arr.shape[1]
        miny = maxy + trsfm[5] * self.arr.shape[0]
        return [minx, maxx, miny, maxy]
