#include "faster-unmixer.hpp"

#include <richdem/common/Array2D.hpp>
#include <richdem/common/grid_cell.hpp>
#include <richdem/common/iterators.hpp>
#include <richdem/flowmet/d8_flowdirs.hpp>
#include <richdem/methods/flow_accumulation.hpp>
#include <richdem/depressions/Barnes2014.hpp>

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

std::vector<SampleNode> faster_unmixer(const std::string& data_dir){
  // Load data
  Array2D<float> dem(data_dir + "/topo.dat");

  // Determine the location of streams via area thresholding
  // auto accum = Array2D<double>::make_from_template(dem);
  // FA_D8(dem, accum);
  // constexpr auto area_threshold = 25*1000000.0;
  // for(auto i=accum.i0();i<accum.size();i++){
  //   if(accum(i) < area_threshold){
  //     accum(i) = 0;
  //   }
  // }

  // Get sample locations and put them in a set using flat-indexing for fast
  // look-up
  std::unordered_map<uint32_t, SampleData> sample_locs;
  for(const auto &x: get_sample_data(data_dir)){
    sample_locs[dem.xyToI(x.x, x.y)] = x;
  }


  // Get D8 flow directions while ignoring depressions
  auto flowdirs = Array2D<d8_flowdir_t>::make_from_template(dem);
  PriorityFloodFlowdirs_Barnes2014(dem, flowdirs);

  // Graph of how the samples are connected together.
  std::vector<SampleNode> sample_parent_graph;

  // Identify cells which do not flow to anywhere. These are the start of our
  // region-identifying procedure
  std::queue<GridCell> q;
  iterate_2d(flowdirs, [&](const auto x, const auto y){
    if(flowdirs.isEdgeCell(x,y)){
      q.emplace(x,y);
    } else if(flowdirs(x,y)==NO_FLOW){
      q.emplace(x,y);
    }
  });

  // Indicates that the cell doesn't correspond to any label
  constexpr auto NO_LABEL = 0;
  auto sample_label = Array2D<uint32_t>::make_from_template(dem, NO_LABEL);
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
    if(total_area!=dem.size()){
      throw std::runtime_error("Total area in graph does not equal total area in DEM!");
    }
  }

  // Save regions output
  sample_label.setNoData(0);
  sample_label.saveGDAL("/z/out.tif");

  std::ofstream fout_sg("/z/sample_graph.dot");
  fout_sg<<"# dot -Tpng sample_graph.dot -o sample_graph.png"<<std::endl;
  fout_sg<<"digraph sample_graph {"<<std::endl;
  for(size_t i=0;i<sample_parent_graph.size();i++){
    const auto& self = sample_parent_graph.at(i);
    fout_sg<<"    "<<i<<" [label=\""<<self.data.name<<"\\n"<<self.area<<"\"];"<<std::endl;
    fout_sg<<"    "<<i<<" -> "<<self.parent<<";"<<std::endl;
  }
  fout_sg<<"}"<<std::endl;

  return sample_parent_graph;
}