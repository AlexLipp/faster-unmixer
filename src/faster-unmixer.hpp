#pragma once

#include <string>
#include <vector>

struct SampleData {
  int x;
  int y;
  std::string name;
};

struct SampleNode {
  const size_t parent;
  std::vector<size_t> children;
  size_t area = 0;
  SampleData data;
  SampleNode(size_t parent, const SampleData& data) : parent(parent), data(data) {}
};

std::vector<SampleNode> faster_unmixer(const std::string& data_dir);