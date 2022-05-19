import numpy as np
cimport numpy as np
cimport cython

from cython.parallel cimport prange
from cython import boundscheck, wraparound

cimport numpy as np
cimport cython



DTYPE_INT = np.int
ctypedef np.int_t DTYPE_INT_t

DTYPE_FLOAT = np.double
ctypedef np.double_t DTYPE_FLOAT_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _accumulate_bw(DTYPE_INT_t np,
                     np.ndarray[DTYPE_INT_t, ndim=1] s,
                     np.ndarray[DTYPE_INT_t, ndim=1] r,
                     np.ndarray[DTYPE_FLOAT_t, ndim=1] drainage_area,
                     np.ndarray[DTYPE_FLOAT_t, ndim=1] discharge):
    """
    Accumulates drainage area and discharge, permitting transmission losses.
    """
    cdef int donor, recvr, i
    cdef float accum

    # Iterate backward through the list, which means we work from upstream to
    # downstream.
    for i in range(np-1, -1, -1):
        donor = s[i]
        recvr = r[donor]
        if donor != recvr:
            drainage_area[recvr] += drainage_area[donor]
            accum = discharge[recvr] + discharge[donor]
            if accum < 0.:
                accum = 0.
            discharge[recvr] = accum



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _accumulate_flux(DTYPE_INT_t np,
                     np.ndarray[DTYPE_INT_t, ndim=1] s,
                     np.ndarray[DTYPE_INT_t, ndim=1] r,
                     np.ndarray[DTYPE_FLOAT_t, ndim=1] flux):
    """
    Accumulates ONLY flux of some tracer in same way as defined above.
    """
    cdef int donor, recvr, i

    # Iterate backward through the list, which means we work from upstream to
    # downstream.
    for i in range(np-1, -1, -1):
        donor = s[i]
        recvr = r[donor]
        if donor != recvr:
            flux[recvr] += flux[donor]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _accumulate_flux_basins(DTYPE_INT_t nb,
                     DTYPE_INT_t ns,
                     np.ndarray[DTYPE_INT_t, ndim=1] s,
                     np.ndarray[DTYPE_INT_t, ndim=1] r,
                     np.ndarray[DTYPE_INT_t, ndim=1] starts,
                     np.ndarray[DTYPE_FLOAT_t, ndim=1] flux):
    """
    Accumulates flux by accumulating individual basins one after the other explicitly.
    """
    cdef int donor, recvr, i, j

    for i in range(nb-1):
        for j in range(starts[i+1]-1,starts[i]-1,-1):
            donor = s[j]
            recvr = r[donor]
            if donor != recvr:
                flux[recvr] += flux[donor]

    for j in range(starts[nb-1],ns-1,-1):
        donor = s[j]
        recvr = r[donor]
        if donor != recvr:
            flux[recvr] += flux[donor]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _accumulate_flux_basins_pll(DTYPE_INT_t nb,
                     DTYPE_INT_t ns,
                     np.ndarray[DTYPE_INT_t, ndim=1] s,
                     np.ndarray[DTYPE_INT_t, ndim=1] r,
                     np.ndarray[DTYPE_INT_t, ndim=1] starts,
                     np.ndarray[DTYPE_FLOAT_t, ndim=1] flux):
    """
    Accumulates flux by accumulating individual basins explicitly, but in parallel.
    """
    cdef int donor, recvr, i, j

    with nogil:
        for i in prange(nb-1,schedule='static'):
            for j in range(starts[i+1]-1,starts[i]-1,-1):
                donor = s[j]
                recvr = r[donor]
                if donor != recvr:
                    flux[recvr] += flux[donor]

    for j in range(starts[nb-1],ns-1,-1):
        donor = s[j]
        recvr = r[donor]
        if donor != recvr:
            flux[recvr] += flux[donor]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _accumulate_active_flux_basins_pll(DTYPE_INT_t nb,
                     np.ndarray[DTYPE_INT_t, ndim=1] s,
                     np.ndarray[DTYPE_INT_t, ndim=1] r,
                     np.ndarray[DTYPE_INT_t, ndim=1] actv_starts,
                     np.ndarray[DTYPE_INT_t, ndim=1] actv_ends,
                     np.ndarray[DTYPE_FLOAT_t, ndim=1] flux):
    """
    Accumulates flux by accumulating only active individual basins explicitly in parallel.
    """
    cdef int donor, recvr, i, j
    with nogil:
        for i in prange(nb,schedule='static'):
            for j in range(actv_ends[i]-1,actv_starts[i]-1,-1):
                donor = s[j]
                recvr = r[donor]
                if donor != recvr:
                    flux[recvr] += flux[donor]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _accumulate_active_flux_basins(DTYPE_INT_t nb,
                     np.ndarray[DTYPE_INT_t, ndim=1] s,
                     np.ndarray[DTYPE_INT_t, ndim=1] r,
                     np.ndarray[DTYPE_INT_t, ndim=1] actv_starts,
                     np.ndarray[DTYPE_INT_t, ndim=1] actv_ends,
                     np.ndarray[DTYPE_FLOAT_t, ndim=1] flux):
    """
    Accumulates flux by accumulating only active individual basins explicitly in serial. 
    """
    cdef int donor, recvr, i, j

    for i in range(nb):
        for j in range(actv_ends[i]-1,actv_starts[i]-1,-1):
            donor = s[j]
            recvr = r[donor]
            if donor != recvr:
                flux[recvr] += flux[donor]
