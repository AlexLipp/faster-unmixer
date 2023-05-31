# Unmixing nested observed concentrations in river networks for source regions

This repository implements an efficient solution to the unmixing of nested concentrations in a (river) network using convex optimisation. The method is described in our [EGU abstract](https://meetingorganizer.copernicus.org/EGU23/EGU23-5368.html) 

## Compiling

Check out submodules:
```
git submodule update --init --recursive
```

Install prereqs:
```
sudo apt install pybind11-dev cmake
```

Compile the C++ code:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_GDAL=ON ..
make
cd ..
```

Run the example optimisation problem:
```
python3 unmix_mwe.py
```

If this returns: `ModuleNotFoundError: No module named 'pyfastunmix'`, you may need to add the `build/` directory to your path using: 
```
export PYTHONPATH=$PYTHONPATH:build/
```

## Cite 

If you use this please cite: 

Lipp, A. and Barnes, R.: Identifying tracer and pollutant sources in drainage networks from point observations using an efficient convex unmixing scheme, EGU General Assembly 2023, Vienna, Austria, 24–28 Apr 2023, EGU23-5368, [DOI: 10.5194/egusphere-egu23-5368](https://doi.org/10.5194/egusphere-egu23-5368), 2023.