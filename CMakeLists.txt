cmake_minimum_required (VERSION 3.9)

project(faster-unmixer
  VERSION 0.0.1
  DESCRIPTION "High-performance unmixing"
  LANGUAGES CXX
)

add_subdirectory(submodules/richdem EXCLUDE_FROM_ALL)

add_executable(faster-unmixer.x
  faster-unmixer.cpp
)
target_link_libraries(faster-unmixer.x PRIVATE richdem)
target_compile_options(faster-unmixer.x PRIVATE -Wall -Wextra -pedantic -Werror)