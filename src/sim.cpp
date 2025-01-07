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

std::vector<float> compute_output_scale(const std::vector<float> &x_scale,
                                        const std::vector<float> &w_scale,
                                        const std::vector<float> &y_scale) {
  auto new_x_scale = broadcast_vec(x_scale, w_scale.size());
  auto new_y_scale = broadcast_vec(y_scale, w_scale.size());
  std::vector<float> ret(w_scale.size());
  for (int i = 0; i < w_scale.size(); ++i) {
    ret[i] = new_y_scale[i] / (new_x_scale[i] * w_scale[i]);
  }
  return ret;
}

std::vector<int> permute(const std::vector<int> &v, std::vector<int> perm) {
  std::for_each(perm.begin(), perm.end(),
                [&v](int i) { assert((i < v.size()) ? true : false); });
  std::vector<int> ret(v.size());
  for (int i = 0; i < v.size(); ++i) {
    ret.at(i) = v.at(perm.at(i));
  }
  return ret;
}

/* n-dimensional-index incrementer
 * ii is a n-dimensional index
 * limit_shape is the shape of the tensor that ii is indexing
 *
 * for example,
 * consider, limit_shape = [3,4,2]
 * first,
 *  ii = [0,0,0]
 * calling increment_shape on it makes it,
 *  ii = [0,0,1]
 * then,
 *  ii = [0,1,0]
 *  ii = [0,1,1]
 *  ii = [0,2,0]
 *  ii = [0,2,1]
 *  ii = [0,3,0]
 *  ii = [0,3,1]
 *  ii = [1,0,0]
 * and so on till
 *  ii = [2,3,1]
 */
void increment_shape(std::vector<int> &ii,
                     const std::vector<int> &limit_shape) {
  assert(ii.size() == limit_shape.size());
  int current_index = ii.size() - 1;
  while (current_index >= 0) {
    ii.at(current_index)++;
    if (ii.at(current_index) >= limit_shape.at(current_index)) {
      ii.at(current_index) = 0;
      current_index--;
    } else {
      break;
    }
  }
  if (ii.at(0) >= limit_shape.at(0)) {
    log_fatal("Cannot increment past limit_shape\n");
  }
}

/* Deduces and removes -1/0 from old_shape to return
 * a correct new_shape.
 * See https://onnx.ai/onnx/operators/onnx__Reshape.html#reshape
 *
 * TODO: handle 0s in shape (does not do it presently)
 */
std::vector<int64_t> deduce_new_shape(std::vector<int64_t> old_shape,
                                      int input_total_size) {
  auto itr = std::find(old_shape.begin(), old_shape.end(), -1);
  if (itr != old_shape.end()) {
    int remaining_size = std::abs(prod(old_shape.begin(), old_shape.end(), 1));
    assert(input_total_size % remaining_size == 0 &&
           "unable to deduce new shape");
    int remaining_dim = input_total_size / remaining_size;
    *itr = remaining_dim;
  }
  return old_shape;
}

int calc_shift_val(float inverted) {
  int shift_val = 16;
  double min_diff = std::numeric_limits<double>::max();
  for (int shift = 1; shift <= 24; shift++) {
    int int_scale = floor(inverted * (1 << shift));
    double check = ((double)(int_scale) / (double)(1 << shift));
    double ok = (check / inverted);
    double diff = (1 - ok);
    if (diff < min_diff && diff >= 0) {
      min_diff = diff;
      shift_val = shift;
    }
  }
  return shift_val;
}