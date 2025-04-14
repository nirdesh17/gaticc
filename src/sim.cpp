#include "sim.h"
#include "utils.h"

std::vector<int> permute(const std::vector<int> &v, std::vector<int> perm) {
  std::for_each(perm.begin(), perm.end(), [&v](int i) {
    ignore_unused(i);
    assert((i < v.size()) ? true : false);
  });
  std::vector<int> ret(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
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

#if 0
int calc_shift_val(float inverted) {
  int shift_val = 16;
  std::vector<int> values = {
      512,  2147483647, 1023, 178956970, 256, 987654321, 768, 134217728,
      900,  50331648,   100,  268435456, 750, 100663296, 50,  67108864,
      1022, 33554432,   200,  16777216,  600, 8388608,   300, 4194304,
      400,  2097152,    850,  1048576,   923, 524288,    650, 262144,
      450,  131072,     700,  65536,     123, 32768,     333, 16384,
      432,  8192,       876,  4096,      64,  2048,      987, 1024,
      999,  1};
  double min_diff = std::numeric_limits<double>::max();
  for (uint32_t shift = 1; shift <= 100; shift++) {
    // std::cout<<"shift: "<<shift<<std::endl;
    uint32_t shifted_val = (1 << shift);
    uint32_t shifted_m1 = (1 << (shift - 1));

    double mean_diff1 = 0;
    for (int i = 0; i < 50; i++) {
      uint32_t int_scale = floor(inverted * shifted_val);
      double check =
          (double)(values[i] * int_scale + shifted_m1) / (double)shifted_val;
      double ori = values[i] * inverted;
      // std::cout<<(int)check<<" "<<(int)ori<<std::endl;
      mean_diff1 += abs(ori - check);
    }

    double mean_diff = mean_diff1 / 50;
    if (mean_diff < min_diff) {
      min_diff = mean_diff;
      shift_val = shift;
    }
  }
  return 10;
}
#endif

int calc_shift_val(float inverted) {
  int shift_val = 16;
  int calib_scale = inverted * (1<<shift_val);
  while (calib_scale > (1<<15)) {
    shift_val--;
    calib_scale = inverted * (1<<shift_val);
  }
  return shift_val;
}

