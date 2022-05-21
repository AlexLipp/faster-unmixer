cmake_minimum_required (VERSION 3.9)

project(faster-unmixer
  VERSION 0.0.1
  DESCRIPTION "High-performance unmixing"
  LANGUAGES CXX
)

find_package(pybind11 REQUIRED)

add_subdirectory(submodules/richdem EXCLUDE_FROM_ALL)

add_library(faster_unmixer
  src/faster-unmixer.cpp
)
target_include_directories(faster_unmixer PRIVATE src/)
target_link_libraries(faster_unmixer PRIVATE richdem)
target_compile_options(faster_unmixer PRIVATE -Wall -Wextra -pedantic -Werror)
target_compile_features(faster_unmixer PRIVATE cxx_std_20)
set_property(TARGET faster_unmixer PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(faster-unmixer.x
  src/main.cpp
)
target_link_libraries(faster-unmixer.x PRIVATE faster_unmixer)
target_compile_options(faster-unmixer.x PRIVATE -Wall -Wextra -pedantic -Werror)

pybind11_add_module(pyfastunmix
  src/extension.cpp
)
target_link_libraries(pyfastunmix PRIVATE faster_unmixer)
target_compile_options(pyfastunmix PRIVATE -Wall -Wextra -pedantic -Werror)
