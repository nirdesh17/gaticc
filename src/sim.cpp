#include "pch.h"
// #include <algorithm>
// #include <cmath>
// #include <cstdint>
// #include <functional>
// #include <iostream>
// #include <iterator>
// #include <numeric>
// #include <utility>

// #include <boost/graph/adjacency_list.hpp>
// #include <boost/graph/graph_traits.hpp>

#include "sim.h"
#include "utils.h"

/* auxillary functions */
Point get_cartesian_cord(int index, int r, int c) {
  int row_n = std::floor((float)index / r);
  int col_n = index % c;
  return Point(row_n, col_n);
}

std::vector<float> compute_output_scale(const std::vector<float>& x_scale,
    const std::vector<float>& w_scale, const std::vector<float>& y_scale) {
  auto new_x_scale = broadcast_vec(x_scale, w_scale.size());
  auto new_y_scale = broadcast_vec(y_scale, w_scale.size());
  std::vector<float> ret(w_scale.size());
  for (int i = 0; i < w_scale.size(); ++i) {
    ret[i] = new_y_scale[i] / (new_x_scale[i] * w_scale[i] );
  }
  return ret;
}
