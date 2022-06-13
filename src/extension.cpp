#include "faster-unmixer.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(pyfastunmix, m) {
  m.doc() = "fastunmix module"; // optional module docstring

  py::class_<SampleData>(m, "SampleData", py::dynamic_attr())
      .def_readonly("x", &SampleData::x)
      .def_readonly("y", &SampleData::y)
      .def_readonly("name", &SampleData::name);

  py::class_<SampleNode>(m, "SampleNode", py::dynamic_attr())
      .def_readonly("downstream_node", &SampleNode::downstream_node)
      .def_readonly("children", &SampleNode::children)
      .def_readonly("area", &SampleNode::area)
      .def_readwrite("total_area", &SampleNode::total_area)
      .def_readonly("data", &SampleNode::data);

  m.def("fastunmix", &faster_unmixer, "Get a graph for fast unmixing", py::arg("data_dir"));
}