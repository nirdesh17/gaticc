#pragma once
#include "onnx.pb.h"
#include "onnx_parser.h"
#include "utils.h"
#include <iostream>
#include <thread>
#include <vector>

/* TODO: iterator mechanism for tensors
 * TODO: destructors
 */

/* A general purpose interface to an n-dimensional tensor
 *
 * Implementation Details:
 * 
 * The Tensor Base Class is abstract and defines a 
 * blueprint for underlying implementations. The
 * implementations inherit and override neccessarily
 * the pure functions and optionally the regular virtual
 * functions. All 'read' type of functions i.e. functions
 * that do not mutate the underlying tensor are pure and
 * need to be defined by every derived class. Regular
 * virtual functions are 'read+write' mutating functions, they 
 * should only be implemented if the derived class wishes
 * to.
 *
 * Derived classes implement (or wrap around) different
 * types of concrete data structures to create a common
 * interface that of the Tensor base class. See, currently
 * implemented derived classes TensorExtant and TensorCreate
 * below.
 */
template <typename T> class Tensor {
public:
  /* Read functions */

  virtual T at(std::vector<int> &at) const = 0;
  virtual T at(std::vector<int> &&at) const = 0;
  virtual T at(int index) const = 0;
  virtual int dims_size() const = 0;
  virtual int dims_at(int index) const = 0;
  virtual std::vector<int> get_dims() const = 0;
  virtual int dims_iterator(int index) const = 0;
  virtual int size() const = 0;
  virtual std::vector<T> get() const = 0;
  virtual void print() const = 0;
  /* Derived classes implement this and return whether delete can be called
   * on the underlying tensor. For derived types (such as TensorExtant and
   * TensorSlice) that wrap around some other type and do not fully own their
   * tensors, freeable() returns false. For TensorCreate(), true is returned
   * as it fully own the underlying Tensor
   * */
  virtual bool freeable() const = 0;
  virtual ~Tensor() = 0;

  /* Write functions */

  /* insert one element at a time */
  virtual void insert(std::vector<int> &at, T data);
  virtual void push_back(T data);
  virtual void push_back(const std::vector<T>& data);
  virtual void set_dims(std::vector<int> const &temp_dims);
  virtual void clear();
  virtual void shrink_to_fit();
  virtual void set(int index, T val);
  virtual Tensor<T>& operator=(const Tensor<T>& rhs);
  virtual typename std::vector<T>::iterator begin();
  virtual typename std::vector<T>::iterator end();
};

template <typename T>
void Tensor<T>::insert(std::vector<int> &at, T data) {
  log_fatal("Un-implemented function");
}
template <typename T>
void Tensor<T>::push_back(T data) {
  log_fatal("Un-implemented function");
}
template <typename T>
void Tensor<T>::push_back(const std::vector<T>& data) {
  log_fatal("Un-implemented function");
}
template <typename T>
void Tensor<T>::set_dims(std::vector<int> const &temp_dims) {
  log_fatal("Un-implemented function");
}
template <typename T>
void Tensor<T>::clear() {
  log_fatal("Un-implemented function");
}

template <typename T>
void Tensor<T>::shrink_to_fit() {
  log_fatal("Un-implemented function");
}
template <typename T>
void Tensor<T>::set(int index, T val) {
  log_fatal("Un-implemented function");
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& rhs) {
  log_fatal("Un-implemented function");
}

template <typename T>
typename std::vector<T>::iterator Tensor<T>::begin() {
  log_fatal("Un-implemented function");
}

template <typename T>
typename std::vector<T>::iterator Tensor<T>::end() {
  log_fatal("Un-implemented function");
}

template <typename T>
Tensor<T>::~Tensor() {
}

/* TensorExtant - Wrapper around onnx::TensorProto
 *
 * TensorExtant deduces where actual data is stored
 * in a onnx::TensorProto object (where weights and biases
 * of a NN are stored) and keeps a pointer
 * to it. It is read-only, does not allow mutating
 * weights
 */
template <typename T> class TensorExtant : public Tensor<T> {
private:
  std::vector<int> dims;
  const onnx::TensorProto *ptr;
  /* Where the actual data resides in memory */
  const T *data;
  /* Initialize `dims` and `ptr`, `data` is initialized
   * by template specialized constructors
   */
  void init_dims(const onnx::TensorProto *ptr);
public:
  /* There are no generic constructors for TensorExtant,
   * all are specialized. See tensor.cpp.
   */
  TensorExtant(const onnx::TensorProto *ptr);
  T at(std::vector<int> &at) const override;
  T at(std::vector<int> &&at) const override;
  T at(int index) const override;
  int dims_size() const override;
  int dims_at(int index) const override;
  std::vector<int> get_dims() const override;
  int dims_iterator(int index) const override;
  int size() const override;
  bool freeable() const override;
  /* Expensive function, creates a copy of the
   * underlying data
   */
  std::vector<T> get() const override;
  void print() const override;
  ~TensorExtant();
};


template <typename T>
void TensorExtant<T>::init_dims(const onnx::TensorProto *ptr) {
  dims.resize(ptr->dims_size());
  std::copy(ptr->dims().begin(), ptr->dims().end(), dims.begin());
  this->ptr = ptr;
}

template <typename T> T TensorExtant<T>::at(int index) const {
  assert(index < this->dims_iterator(-1));
  return data[index];
}

template <typename T> T TensorExtant<T>::at(std::vector<int> &index) const {
  assert(index.size() == dims.size());
  int sum = 0;
  for (int i = 0; i < index.size(); i++) {
    sum = sum + index[i] * dims_iterator(i);
  }
  return at(sum);
}

template <typename T> T TensorExtant<T>::at(std::vector<int> &&index) const {
  return at(index);
}

template <typename T>
void TensorExtant<T>::print() const {
  for (int i = 0; i < dims_iterator(-1); ++i) {
    if (i % 9 == 0) {
      std::cout << '\n';
    }
    std::cout << data[i] << '\t';
  }
}

template <typename T>
std::vector<int> TensorExtant<T>::get_dims() const {
  return dims;
}

template <typename T>
int TensorExtant<T>::dims_size() const { return dims.size(); }

template <typename T> int TensorExtant<T>::dims_at(int index) const {
  assert(index < dims.size());
  return dims[index];
}

template <typename T> int TensorExtant<T>::dims_iterator(int index) const {
  int a = 1;
  for (int i = 1; i < dims.size() - index; i++) {
    a *= dims[index + i];
  }
  return a;
}

template <typename T> int TensorExtant<T>::size() const {
  return dims_iterator(-1);
}

template <typename T> std::vector<T> TensorExtant<T>::get() const {
  std::vector<T> ret (dims_iterator(-1));
  for (int i = 0; i < this->size(); ++i) {
    ret[i] = data[i];
  }
  return ret;
}

template <typename T>
bool TensorExtant<T>::freeable() const {
  return false;
}

template <typename T>
TensorExtant<T>::~TensorExtant() {
  // frees nothing as it owns nothing
}

template <typename T> class TensorCreate : public Tensor<T> {
  std::vector<int> dims;
  std::vector<int> stride;
  std::vector<T> vec;
public:
  TensorCreate() = delete;

  TensorCreate(std::vector<T> const &v, std::vector<int> const &dim) {
    dims = dim;
    vec = v;
    stride = get_stride_from_shape(dim);
  }

  TensorCreate(std::vector<int> const &dim) {
    dims = dim;
    vec.resize(dims_iterator(-1), 0);
    stride = get_stride_from_shape(dim);
  }

  T at(std::vector<int> &at) const override {
    assert(at.size() == dims.size());
    int sum = 0;
    for (int i = 0; i < at.size(); i++) {
      assert(at[i] <= dims[i]);
      sum += at[i] * stride[i];
    }
    return vec.at(sum);
  }

  T at(std::vector<int> &&at) const override { 
    return this->at(at);
  }

  T at(int index) const override { return vec.at(index); }

  int dims_size() const override { return dims.size(); }

  int dims_at(int index) const override { return dims.at(index); }
  void push_back(T data) override { vec.push_back(data); }

  void push_back(const std::vector<T>& data) { 
    for (const T& i : data) {
      this->push_back(i);
    }
  }

  void insert(std::vector<int> &at, T data) override {
    assert(at.size() <= dims.size());
    int sum = 0;
    for (int i = 0; i < at.size(); i++) {
      assert(at[i] <= dims[i]);
      sum += at[i] * stride[i];
    }
    vec[sum] = data;
  }

  void set_dims(std::vector<int> const &temp_dims) override {
    dims = temp_dims;
    return;
  }
  std::vector<int> get_dims() const override {
    return dims;
  }
  int dims_iterator(int index) const override {
    int a = 1;
    for (int i = 1; i < dims.size() - index; i++) {
      a *= dims[index + i];
    }
    return a;
  }

  void clear() override { vec.clear(); }

  void shrink_to_fit() override { vec.shrink_to_fit(); }

  int size() const override { return vec.size(); }

  std::vector<T> get() const override { return vec; }

  void set(int index, T val) override { vec.at(index) = val; }

  virtual Tensor<T>& operator=(const Tensor<T>& rhs) {
    this->dims = rhs.get_dims();
    this->vec = rhs.get();
    return *this;
  }

  void print() const override { 
    print_vec("tensor", vec);
  }

  typename std::vector<T>::iterator begin() override {
    return vec.begin();
  }
  typename std::vector<T>::iterator end() override {
    return vec.end();
  }

  bool freeable() const override {
    return true;
  }

  ~TensorCreate();
};

template <typename T>
TensorCreate<T>::~TensorCreate() {
}


template <typename T> class TensorSlice : public Tensor<T> {
  Tensor<T> *src;
  std::vector<int> slice;
  /* Linear offset wrt the original linear representation
   * of src tensor
   */
  int offset;
  /* Linear size upper bound of this slice */
  int slice_size;

  std::vector<int> dims;

public:
  TensorSlice(Tensor<T> *src, std::vector<int> slice);
  T at(std::vector<int> &index) const override;
  T at(std::vector<int> &&index) const override;
  T at(int index) const override;
  int dims_size() const override;
  int dims_at(int index) const override;
  std::vector<int> get_dims() const override;
  int dims_iterator(int index) const override;
  int size() const override;
  std::vector<T> get() const override;
  void print() const override;
  bool freeable() const override;

  ~TensorSlice();

  /* Write functions */

  /* insert one element at a time */
#if 0
  void insert(std::vector<int> &at, T data);
  void push_back(T data);
  void push_back(const std::vector<T>& data);
  void set_dims(std::vector<int> const &temp_dims);
  void clear();
  void shrink_to_fit();
  void set(int index, T val);
  Tensor<T>& operator=(Tensor<T>& rhs);
  typename std::vector<T>::iterator begin();
  typename std::vector<T>::iterator end();
#endif
};

template <typename T>
TensorSlice<T>::TensorSlice(Tensor<T> *src, std::vector<int> slice) {
  assert(slice.size() <= src->dims_size());

  this->slice = slice;
  this->src = src;
  this->offset = 0;
  std::vector<int> strides = get_stride_from_shape(src->get_dims());
  for (int i = 0; i < slice.size(); ++i) {
    this->offset += (strides[i] * slice[i]);
  }
  for (int i = slice.size(); i < src->dims_size(); ++i) {
    this->dims.push_back(src->dims_at(i));
  }
  this->slice_size = prod(dims.begin(), dims.end(), 1);
}

template <typename T>
T TensorSlice<T>::at(std::vector<int> &index) const {
  std::vector<int> new_index = concat(slice, index);
  return src->at(new_index);
}

template <typename T>
T TensorSlice<T>::at(std::vector<int> &&index) const {
  return at(index);
}

template <typename T> T TensorSlice<T>::at(int index) const {
  assert(index >= 0);
  assert(index < slice_size);
  return src->at(offset + index);
}

template <typename T>
int TensorSlice<T>::dims_size() const {
  return dims.size();
}
template <typename T>
int TensorSlice<T>::dims_at(int index) const {
  return dims.at(index);
}
template <typename T>
std::vector<int> TensorSlice<T>::get_dims() const {
  return dims;
}
template <typename T>
int TensorSlice<T>::dims_iterator(int index) const {
  int a = 1;
  for (int i = 1; i < dims.size() - index; i++) {
    a *= dims[index + i];
  }
  return a;
}
template <typename T>
int TensorSlice<T>::size() const {
  return slice_size;
}


template <typename T>
std::vector<T> TensorSlice<T>::get() const {
  /* TODO: expensive function, remove get completely from tensor's 
   * interface
   */
  std::vector<T> ret(slice_size);
  for (int i = 0; i < slice_size; ++i) {
    ret[i] = at(i);
  }
  return ret;
}

template <typename T>
void TensorSlice<T>::print() const {
  for (int i = 0; i < slice_size; ++i) {
    std::cout << at(i) << ' ';
  }
  std::cout << '\n';
  std::cout << "slice print " << slice_size << '\n';
}

template <typename T>
bool TensorSlice<T>::freeable() const {
  return false;
}

template <typename T>
TensorSlice<T>::~TensorSlice() {
  // frees nothing as it owns nothing
}

template <typename T>
Tensor<T>* tensor_sub_zp(const Tensor<T> *input, const std::vector<int>& zp) {
  assert(input->dims_size() == 4 && "tensor_pad assumes 4d inputs");
  std::vector<int> new_dims = input->get_dims();
  Tensor<T> *output = new TensorCreate<T>(new_dims);
  for (int i = 0; i < new_dims[0]; ++i) {
    for (int j = 0; j < new_dims[1]; ++j) {
      for (int k = 0; k < new_dims[2]; ++k) {
        for (int l = 0; l < new_dims[3]; ++l) {
          std::vector<int> out_index {i, j, k, l};
          T v = input->at(out_index) - zp[j];
          output->insert(out_index, v);
        }
      }
    }
  }
  return output;
}


template <typename T>
Tensor<T>* tensor_pad(const Tensor<T> *input, const std::vector<int>& pads) {
  assert(input->dims_size() == 4 && "tensor_pad assumes 4d inputs");
  std::vector<int> new_dims = get_dims_after_pad(input->get_dims(), pads);
  Tensor<T> *output = new TensorCreate<T>(new_dims);
  for (int i = 0; i < new_dims[0]; ++i) {
    for (int j = 0; j < new_dims[1]; ++j) {
      for (int k = 0; k < new_dims[2]; ++k) {
        for (int l = 0; l < new_dims[3]; ++l) {
          std::vector<int> out_index {i, j, k, l};
          if (islying(k, l, input->dims_at(2), input->dims_at(3), pads)) {
            output->insert(out_index, 0);
          } else {
            std::vector<int> in_index {i, j, k-pads[1], l-pads[0]};
            output->insert(out_index, input->at(in_index));
          }
        }
      }
    }
  }
  return output;
}
