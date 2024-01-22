#pragma once
#include "onnx.pb.h"
#include "onnx_parser.h"
#include <iostream>
#include <thread>
#include <vector>
class Tensor {
public:
  Tensor(){};

  virtual int at(std::vector<int> &at) { return -1;}

  virtual int dims_size() {return 0;}
  
  virtual int dims_at(int index) {return index;}

  virtual void push_back(int data) {return;}

  virtual void insert(std::vector<int> &at, int data) {return;}

  virtual int dims_iterator(int index) {return index;}

  virtual void clear(){return;}
};

class TensorExtant : public Tensor{
private : 
  std::vector<int> dims;
  const onnx::TensorProto * ptr;
public:
  TensorExtant(const onnx::TensorProto * ptr) {
    dims = std::vector<int>(ptr->dims_size());
    for (int i = 0; i < dims.size(); i++) {
      dims[i] = ptr->dims(i);
    }
    this->ptr = ptr; 
  } 

  int at(std::vector<int> &at) override {
    assert(at.size() == dims.size());

    int sum = 0;
    for (int i = 0; i < at.size(); i++) {
      sum = sum + at[i] * dims_iterator(dims.size() - 1 - i);
    }
    if (typeid(int) == typeid(float))
      return (ptr->float_data(sum));
    else if (typeid(int) == typeid(int32_t))
      return (ptr->int32_data(sum));
    else if (typeid(int) == typeid(int64_t))
      return (ptr->int64_data(sum));
		else
		return -1;

    // have functionality for raw data too
  }

  int dims_iterator(int index) override {
    // std::cout << " value of index " << index << dims.size()<<std::endl;
    // assert(index < dims.size());
    int a = 1;
    for (int i = 1; i < dims.size() - index; i++) {
      a *= dims[index+i];
    }
    return a;
  }

  // this should belong to parent only?
};

class TensorCreate : public Tensor { 
  private:
    std::vector<int> dims;
    std::vector<int> vec;
  public:
    TensorCreate(std::vector<int> &dim) {
      dims = dim;
      vec = std::vector<int>(dims_iterator(-1));
    }

    int at(std::vector<int> &at) override {
      assert(at.size() == dims.size());

      int sum = 0;
      for (int i = 0; i < at.size(); i++) {
      assert(at[i] <= dims[i]);
      sum = sum + at[i] * dims_iterator(i);
      // printf(" \nat i %d,%d,%d\n", i,at[i],sum);
      }
      return vec[sum];
    }

    int dims_size() override { 
      return dims.size();
    }

    int dims_at(int index) override {
      assert(index < dims.size());
      return dims[index];
    }

    void push_back(int data) override {
      static int i = 0;
      if (i == 0) {
      vec.clear();
      // vec.shrink_to_fit();
      i++;
      }
      printf(" vec size %d and dims_iterator %d \n",vec.size(),dims_iterator(-1));
      assert(vec.size() <= dims_iterator(-1));
      vec.push_back(data);
    // printf(" vec value %d\n", vec[index]);
    // std::cout<< "vec value "<< vec[index]<<std::cout;
    }

    void insert(std::vector<int> &at, int data) override {
    assert(at.size() <= dims.size());
    int sum = 0;
    for (int i = 0; i < at.size(); i++) {
      assert(at[i] <= dims[i]);
      sum = sum + at[i] * dims_iterator(i);
    }
    vec[sum] = data;
  }

   int dims_iterator(int index) override {
    // std::cout << " value of index " << index << dims.size()<<std::endl;
    // assert(index < dims.size());
    int a = 1;
    for (int i = 1; i < dims.size() - index; i++) {
      a *= dims[index+i];
    }
    return a;
  }

  void clear() override {
    vec.clear();
  }
};