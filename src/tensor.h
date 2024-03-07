#pragma once
#include "onnx.pb.h"
#include "onnx_parser.h"
#include "utils.h"
#include <iostream>
#include <thread>
#include <vector>

/* TODO: refactor this, make it cleaner 
 * 1. Tensor as pure abstract class?
 * 2. remove redundant functions
 * 3. document
 * */

template <typename T> class Tensor {
public:
  Tensor() {} 
  virtual T at(std::vector<int> &at) { return -1; }
  virtual T at(std::vector<int> &&at) { return -1; }
  virtual T at(int index) { return -1; }
  virtual int dims_size() { return 0; }
  virtual int dims_at(int index) { return index; }
  virtual void push_back(T data) { return; }
  /* insert one element at a time */
  virtual void insert(std::vector<int> &at, T data) { return; }
  virtual void set_dims(std::vector<int> const &temp_dims) { return; }
  virtual std::vector<int> get_dims() {
    return std::vector<int>();
  }
  virtual int dims_iterator(int index) { return index; }
  virtual void clear() { return; }
  virtual void shrink_to_fit() { return; }
  virtual int size() { return 0; }
  virtual std::vector<T> get() { return std::vector<T>(); }
  virtual Mat<T> get_mat(int index) { return 0; }
  virtual void set(int index, T val) { return; }
  virtual Tensor<T>& operator=(Tensor<T>& rhs) {
  }
  virtual void print() { return; }
};

template <typename T> class TensorExtant : public Tensor<T> {
private:
  std::vector<int> dims;
  const onnx::TensorProto *ptr;

public:
  TensorExtant(const onnx::TensorProto *ptr) {
    dims = std::vector<int>(ptr->dims_size());
    for (int i = 0; i < dims.size(); i++) {
      dims[i] = ptr->dims(i);
    }
    this->ptr = ptr;
  }

  T at(std::vector<int> &at) override {
    assert(at.size() == dims.size());

    int sum = 0;
    for (int i = 0; i < at.size(); i++) {
      sum = sum + at[i] * dims_iterator(i);
    }
    if (typeid(T) == typeid(float))
      return (ptr->float_data(sum));
    else if (typeid(T) == typeid(int32_t))
      return ((int8_t)ptr->raw_data().at(sum));
    // return (ptr->int32_data(sum));
    else if (typeid(T) == typeid(int64_t))
      return (ptr->int64_data(sum));
    else if (typeid(T) == typeid(int8_t))
      return ((int8_t)ptr->raw_data().at(sum));
    else
      return -1;
  }
  int dims_size() override { return dims.size(); }

  int dims_at(int index) override {
    assert(index < dims.size());
    return dims[index];
  }

  int dims_iterator(int index) override {
    int a = 1;
    for (int i = 1; i < dims.size() - index; i++) {
      a *= dims[index + i];
    }
    return a;
  }

  void print() override {
    /* TODO: needs refactoring */
    for (int i = 0; i < dims_iterator(-1); ++i) {
      if (typeid(T) == typeid(float)) {
        std::cout << ptr->float_data(i) << '\n';
      } else if (typeid(T) == typeid(int8_t)) {
        std::cout << (int8_t)ptr->raw_data().at(i) << '\n';
      } else {
        std::cout << -1 << '\n';
      }
    }
  }
};

template <typename T> class TensorCreate : public Tensor<T> {
  std::vector<int> dims;
  std::vector<T> vec;

public:
  TensorCreate() = delete;

  TensorCreate(std::vector<T> const &v, std::vector<int> const &dim) {
    dims = dim;
    vec = v;
  }

  TensorCreate(std::vector<int> const &dim) {
    dims = dim;
    vec.resize(dims_iterator(-1), 0);
  }

  T at(std::vector<int> &at) override {
    assert(at.size() == dims.size());

    int sum = 0;
    for (int i = 0; i < at.size(); i++) {
      assert(at[i] <= dims[i]);
      sum = sum + at[i] * dims_iterator(i);
    }
    return vec.at(sum);
  }

  T at(std::vector<int> &&at) override { 
    return this->at(at);
  }

  T at(int index) { return vec.at(index); }

  int dims_size() override { return dims.size(); }

  int dims_at(int index) override { return dims.at(index); }
  void push_back(T data) override { vec.push_back(data); }

  void insert(std::vector<int> &at, T data) override {
    assert(at.size() <= dims.size());
    int sum = 0;
    for (int i = 0; i < at.size(); i++) {
      assert(at[i] <= dims[i]);
      sum = sum + at[i] * dims_iterator(i);
    }
    vec[sum] = data;
  }

  void set_dims(std::vector<int> const &temp_dims) override {
    dims = temp_dims;
    return;
  }
  std::vector<int> get_dims() override {
    return dims;
  }
  int dims_iterator(int index) override {
    int a = 1;
    for (int i = 1; i < dims.size() - index; i++) {
      a *= dims[index + i];
    }
    return a;
  }

  void clear() override { vec.clear(); }

  void shrink_to_fit() override { vec.shrink_to_fit(); }

  int size() override { return vec.size(); }

  std::vector<T> get() override { return vec; }

  void set(int index, T val) override { vec.at(index) = val; }

  /* TODO: re-work */
  Mat<T> get_mat(int index) override { 
    assert(this->dims_size() == 3 && "not a 3d tensor, cant get mat");
    std::vector<int> itr {0, 0, 0};
    Mat<T> ret(this->dims_at(1), std::vector<T>(this->dims_at(2)));
    for (int i = 0; i < this->dims_at(1); ++i) {
      for (int j = 0; j < this->dims_at(2); ++j) {
        ret.at(i, j) = this->at(std::vector({index, i, j}));
      }
    }
    return ret;
  }

  virtual Tensor<T>& operator=(Tensor<T>& rhs) {
    this->dims = rhs.get_dims();
    this->vec = rhs.get();
    return *this;
  }
};
