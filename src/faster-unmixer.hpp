#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct SampleData {
  int x;
  int y;
  std::string name;
};

struct SampleNode {
  const size_t downstream_node;
  std::vector<size_t> children;
  size_t area = 0;
  size_t total_area = 0;
  SampleData data;
  SampleNode(size_t downstream_node, const SampleData& data) : downstream_node(downstream_node), data(data) {}
};

// Used to store graph edges
struct PairHash {
  std::size_t operator()(const std::pair<uint32_t, uint32_t>& sp) const {
    // Boost hash combine function: https://stackoverflow.com/a/27952689/752843
    return sp.first ^ (sp.second + 0x9e3779b9 + (sp.first << 6) + (sp.first >> 2));
  }
};

using adjacency_graph_t = std::unordered_map<std::pair<uint32_t, uint32_t>, int32_t, PairHash>;

std::pair<std::vector<SampleNode>, adjacency_graph_t>  faster_unmixer(const std::string& data_dir);
