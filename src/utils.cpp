#include "tensor.h"
#include "utils.h"
#include <cstdint>
#include <typeinfo>

/* Used by run_* functions in executor to free under-lying Tensor
 * pointers. This could very well be templated by that requires the
 * caller to know the type of the under-lying data that is being
 * abstracted by std::any. This is not true for us, thus the if-else
 * ladder.
 */
void TensorPool::free(int index) {
  std::any v = pool.at(index);
  if (v.type() == typeid(Tensor<int8_t> *)) {
    Tensor<int8_t> *dd = std::any_cast<Tensor<int8_t> *>(v);
    /* TODO: temporary hack, find a cleaner workaround */
    if (dd->freeable()) {
      delete dd;
    }
  } else if (v.type() == typeid(Tensor<int16_t> *)) {
    Tensor<int16_t> *dd = std::any_cast<Tensor<int16_t> *>(v);
    if (dd->freeable()) {
      delete dd;
    }
  } else if (v.type() == typeid(Tensor<int> *)) {
    Tensor<int> *dd = std::any_cast<Tensor<int> *>(v);
    if (dd->freeable()) {
      delete dd;
    }
  } else if (v.type() == typeid(Tensor<int64_t> *)) {
    Tensor<int64_t> *dd = std::any_cast<Tensor<int64_t> *>(v);
    if (dd->freeable()) {
      delete dd;
    }
  } else if (v.type() == typeid(Tensor<int32_t> *)) {
    Tensor<int32_t> *dd = std::any_cast<Tensor<int32_t> *>(v);
    if (dd->freeable()) {
      delete dd;
    }
  } else if (v.type() == typeid(Tensor<float> *)) {
    Tensor<float> *dd = std::any_cast<Tensor<float> *>(v);
    if (dd->freeable()) {
      delete dd;
    }
  } else if (v.type() == typeid(Tensor<double> *)) {
    Tensor<double> *dd = std::any_cast<Tensor<double> *>(v);
    if (dd->freeable()) {
      delete dd;
    }
  } else {
    log_fatal("Unknown type: %s, cannot free. Support has to be added",
              v.type().name());
  }
  pool.at(index).reset();
}

void TensorPool::free() {
  for (int i = 0; i < pool.size(); ++i) {
    pool.at(i).reset();
  }
}


bool TensorPool::has_value(int index) { return pool.at(index).has_value(); }

void TensorPool::resize(int size) {
  pool.resize(size);
}

Point::Point(int a, int b): first {a}, second {b} {}

std::ostream& operator<<(std::ostream &os, const Point& p) {
  os << p.first << ',' << p.second;
  return os;
}

/* path: such as "/usr/bin/file.txt"
 * returns: "file.txt"
 */
std::filesystem::path extract_basename(const std::string &path) {
  std::filesystem::path fs_path(path);
  return fs_path.filename();
}

/* path: such as "/usr/bin/file.txt"
 * returns: "/usr/bin"
 */
std::filesystem::path extract_dirname(const std::string &path) {
  std::filesystem::path fs_path(path);
  return fs_path.remove_filename();
}

/* true if two shapes are broadcastable 
 * see https://numpy.org/doc/stable/user/basics.broadcasting.html
 */
bool is_broadcastable(const std::vector<int> &shape1, const std::vector<int>& shape2) {
  if (shape1.size() == 1 || shape2.size() == 1) {
    return true;
  }

  if (shape1.size() == shape2.size()) {
    /* iterate from rhs */
    for (int i = shape1.size()-1; i > 0; --i) {
      if (shape1[i] != shape2[i]) {
        return false;
      }
    }
    return true;
  }

  return false;
}

std::vector<int> get_dims_after_pad(std::vector<int> current_dims, const std::vector<int>& pad) {
  auto last_dim = current_dims.rbegin();
  auto second_last_dim = current_dims.rbegin() + 1;
  
  *second_last_dim = *second_last_dim + pad[I_UP] + pad[I_DOWN];
  *last_dim = *last_dim + pad[I_LEFT] + pad[I_RIGHT];
  return current_dims;
}

bool islying(int i, int j, int rows, int cols, const std::vector<int> &pad) {
  if (((j >= 0 && j < pad[0]) || (j >= (cols + pad[0]) && j < (cols + pad[0] + pad[2]))) ||
      ((i >= 0 && i < pad[1]) || (i >= (rows + pad[1]) && i < (rows + pad[1] + pad[3])))) {
    return true;
  } else {
    return false;
  }
}
