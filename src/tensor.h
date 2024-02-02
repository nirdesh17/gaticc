#pragma once
#include "onnx.pb.h"
#include "onnx_parser.h"
#include <iostream>
#include <thread>
#include <vector>

template <typename T1> class Tensor {
public:
  Tensor(){};

  virtual T1 at(std::vector<int> &at) { return -1; }

  virtual int dims_size() { return 0; }

  virtual int dims_at(int index) { return index; }

  virtual void push_back(T1 data) { return; }

  virtual void insert(std::vector<int> &at, T1 data) { return; }

  virtual void set_dims(std::vector<int> temp_dims, int start_index) { return; }

  virtual int dims_iterator(int index) { return index; }

  virtual void clear() { return; }

  virtual void shrink_to_fit() { return; }

  virtual int size() { return 0; }

  virtual std::vector<T1> get() { 
    std::vector<T1> temp_vec;
    return temp_vec;}
  
  virtual T1 get(int index) { return 0 ;}

};

template <typename T1> class TensorExtant : public Tensor<T1> {
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

  T1 at(std::vector<int> &at) override {
    assert(at.size() == dims.size());

    int sum = 0;
    for (int i = 0; i < at.size(); i++) {
      sum = sum + at[i] * dims_iterator(i);
    }
    if (typeid(T1) == typeid(float))
      return (ptr->float_data(sum));
    else if (typeid(T1) == typeid(int32_t))
      return (ptr->int32_data(sum));
    else if (typeid(T1) == typeid(int64_t))
      return (ptr->int64_data(sum));
    else if (typeid(T1) == typeid(int8_t))
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
};

template <typename T1> class TensorCreate : public Tensor<T1> {
private:
  std::vector<int> dims;
  std::vector<T1> vec;

public:
  TensorCreate(){};
  TensorCreate(std::vector<int> &dim) {
    dims = dim;
    vec = std::vector<T1>(dims_iterator(-1));
  }

  T1 at(std::vector<int> &at) override {
    assert(at.size() == dims.size());

    int sum = 0;
    for (int i = 0; i < at.size(); i++) {
      assert(at[i] <= dims[i]);
      sum = sum + at[i] * dims_iterator(i);
    }
    return vec.at(sum);
  }

  T1 at(int index) { return (T1)vec.at(index); }

  int dims_size() override { return dims.size(); }

  int dims_at(int index) override {
    assert(index < dims.size());
    return dims[index];
  }
  void push_back(T1 data) override {
    vec.push_back(data);
  }

  void insert(std::vector<int> &at, T1 data) override {
    assert(at.size() <= dims.size());
    int sum = 0;
    for (int i = 0; i < at.size(); i++) {
      assert(at[i] <= dims[i]);
      sum = sum + at[i] * dims_iterator(i);
    }
    vec[sum] = data;
  }

  void set_dims(std::vector<int> temp_dims, int start_index) override {
    if(dims.size()==temp_dims.size()){
      return;
    }
    dims.push_back(temp_dims.at(start_index));
    start_index++;
    set_dims(temp_dims,start_index);
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

  std::vector<T1> get() override { return vec;}
  
  T1 get(int index) override { return vec.at(index);}
};