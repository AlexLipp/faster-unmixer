#include <richdem/common/Array2D.hpp>
#include <richdem/common/grid_cell.hpp>
#include <richdem/common/iterators.hpp>
#include <richdem/flowmet/d8_flowdirs.hpp>
#include <richdem/methods/flow_accumulation.hpp>
#include <richdem/depressions/Barnes2014.hpp>

#include <fstream>
#include <iostream>
#include <queue>
#include <string>

using namespace richdem;

int main(int argc, char **argv){
  if(argc != 3){
    std::cerr<<"Syntax: "<<argv[0]<<" <Input file> <Sample Locs>"<<std::endl;
    return -1;
  }

  const std::string in_name = argv[1];
  const std::string sample_locs_name = argv[2];

  // Load data
  Array2D<float> dem(in_name);

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
  std::unordered_set<uint32_t> sample_locs;
  {
    std::ifstream fin(sample_locs_name);
    double sx;
    double sy;
    while(fin>>sx>>sy){
      sample_locs.insert(flowdirs.xyToI(sx,sy));
    }
  }

  // Get D8 flow directions while ignoring depressions
  auto flowdirs = Array2D<d8_flowdir_t>::make_from_template(dem);
  PriorityFloodFlowdirs_Barnes2014(dem, flowdirs);

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

  // Label to be applied to the next sensor we encounter
  auto current_label = 1;

  // Iterate in a wave from all the flow endpoints to the headwaters, labeling
  // cells as we go.
  while(!q.empty()){
    const auto c = q.front();
    q.pop();

    // We have identified a new station!
    if(sample_locs.count(flowdirs.xyToI(c.x,c.y))!=0){
      sample_label(c.x,c.y) = current_label++;
    }

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

  // Save regions output
  sample_label.saveGDAL("/z/out.tif");

  return 0;
}