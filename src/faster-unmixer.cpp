#include "faster-unmixer.hpp"

#include <richdem/common/Array2D.hpp>
#include <richdem/common/grid_cell.hpp>
#include <richdem/common/iterators.hpp>
#include <richdem/depressions/Barnes2014.hpp>
#include <richdem/flowmet/d8_flowdirs.hpp>
#include <richdem/methods/flow_accumulation.hpp>
#include <richdem/misc/conversion.hpp>

#include <fstream>
#include <queue>
#include <string>
#include <unordered_map>

using namespace richdem;

std::vector<SampleData> get_sample_data(const std::string &data_dir){
  std::vector<SampleData> sample_data;

  {
    std::ifstream fin(data_dir + "/fitted_samp_locs.dat");
    double sx;
    double sy;
    while(fin>>sx>>sy){
      sample_data.emplace_back();
      sample_data.back().x = sx;
      sample_data.back().y = sy;
    }
  }

  {
    std::ifstream fin(data_dir + "/samples.dat");
    std::string name;
    int x1;
    double x2;
    double x3;
    auto current_sample = sample_data.begin();
    while(fin>>name>>x1>>x2>>x3){
      current_sample->name = name;
      current_sample++;
    }
  }

  return sample_data;
}

std::pair<std::vector<SampleNode>, adjacency_graph_t> faster_unmixer(const std::string& data_dir){
  // Load data
  Array2D<d8_flowdir_t> arc_flowdirs(data_dir + "/d8.asc");

  // Convert raw flowdirs to RichDEM flowdirs
  auto flowdirs = Array2D<d8_flowdir_t>::make_from_template(arc_flowdirs);
  convert_arc_flowdirs_to_richdem_d8(arc_flowdirs, flowdirs);

  flowdirs.saveGDAL("/z/rd_flowdirs.tif");

  // Get sample locations and put them in a set using flat-indexing for fast
  // look-up
  std::unordered_map<uint32_t, SampleData> sample_locs;
  for(const auto &sample: get_sample_data(data_dir)){
    sample_locs[flowdirs.xyToI(sample.x, sample.y)] = sample;
  }

  // Graph of how the samples are connected together.
  std::vector<SampleNode> sample_parent_graph;

  // Identify cells which do not flow to anywhere. These are the start of our
  // region-identifying procedure
  std::queue<GridCell> q;
  iterate_2d(flowdirs, [&](const auto x, const auto y){
    if(flowdirs(x,y)==NO_FLOW){
      q.emplace(x,y);
    }
  });

  // Indicates that the cell doesn't correspond to any label
  constexpr auto NO_LABEL = 0;
  auto sample_label = Array2D<uint32_t>::make_from_template(flowdirs, NO_LABEL);
  sample_label.setNoData(NO_LABEL);
  sample_parent_graph.emplace_back(0, SampleData{});

  // Iterate in a wave from all the flow endpoints to the headwaters, labeling
  // cells as we go.
  while(!q.empty()){
    const auto c = q.front();
    q.pop();

    // We have identified a new station!
    if(sample_locs.count(flowdirs.xyToI(c.x,c.y))!=0){
      const auto &data = sample_locs.at(flowdirs.xyToI(c.x,c.y));
      // Generate a new label for the sample
      const auto my_new_label = sample_parent_graph.size();
      // The current label will become the parent label
      const auto my_current_label = sample_label(c.x,c.y);
      auto& parent = sample_parent_graph.at(my_current_label);
      parent.children.push_back(my_new_label);
      sample_parent_graph.emplace_back(my_current_label, data);
      // Update the sample's label
      sample_label(c.x,c.y) = my_new_label;
    }

    const auto my_label = sample_label(c.x,c.y);
    sample_parent_graph.at(my_label).area++;

    // Loop over all my neighbours
    for(int n=1;n<=8;n++){
      const int nx = c.x + d8x[n];
      const int ny = c.y + d8y[n];
      // If cell flows into me, it gets my label. If the cell happens to have a
      // sample, then this label will be overwritten when the cell is popped
      // from the queue. D8 means that upstream cells are only added to the
      // queue once.
      if(flowdirs.inGrid(nx,ny) && flowdirs(nx,ny)==d8_inverse[n]){
        sample_label(nx,ny) = sample_label(c.x,c.y);
        q.emplace(nx,ny);
      }
    }
  }

  // Sanity check
  {
    size_t total_area=0;
    for(const auto &x: sample_parent_graph){
      total_area+=x.area;
    }
    if(total_area!=flowdirs.size()){
      throw std::runtime_error("Total area in graph " + std::to_string(total_area) + " does not equal total area in DEM " + std::to_string(flowdirs.size()) + "!");
    }
  }

  adjacency_graph_t adjacency_graph;

  iterate_2d(sample_label, [&](const auto x, const auto y){
    const auto my_label = sample_label(x, y);
    if(my_label == sample_label.noData()){
      return;
    }
    for(int n=1;n<=8;n++){
      const int nx = x + d8x[n];
      const int ny = y + d8y[n];
      if(sample_label.inGrid(nx,ny)){
        const auto n_label = sample_label(nx, ny);
        // Less-than comparison creates a preferred ordering to prevent
        // double-counting and also prevents self-loops
        if(n_label != sample_label.noData() && my_label < n_label){
          adjacency_graph[{my_label,n_label}]++;
        }
      }
    }
  });

  // Save regions output
  sample_label.saveGDAL("/z/labels.tif");

  std::ofstream fout_sg("/z/sample_graph.dot");
  fout_sg<<"# dot -Tpng sample_graph.dot -o sample_graph.png"<<std::endl;
  fout_sg<<"digraph sample_graph {"<<std::endl;
  for(size_t i=0;i<sample_parent_graph.size();i++){
    const auto& self = sample_parent_graph.at(i);
    fout_sg<<"    "<<i<<" [label=\""<<self.data.name<<"\\n"<<self.area<<"\"];"<<std::endl;
    fout_sg<<"    "<<i<<" -> "<<self.downstream_node<<";"<<std::endl;
  }
  fout_sg<<"}"<<std::endl;

  return {sample_parent_graph, adjacency_graph};
}