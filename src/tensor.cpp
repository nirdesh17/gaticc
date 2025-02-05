#include "pch.h"
#include "tensor.h"
#include "onnx_parser.h"


template <>
TensorExtant<float>::TensorExtant(const onnx::TensorProto *ptr) {
  init_dims(ptr);
  if (ptr->float_data_size() != 0) {
    data = (const float *) (ptr->float_data().data());
  } else {
    if (Op::dtype_eq(ptr->data_type(), onnx::TensorProto_DataType_FLOAT)) {
      data = (const float *) (ptr->raw_data().c_str());
    } else {
      log_fatal("Unable to deduce type for tensor or un-implemented: {}\n", ptr->name());
    }
  }
}

template <>
TensorExtant<int32_t>::TensorExtant(const onnx::TensorProto *ptr) {
  init_dims(ptr);
  if (ptr->int32_data_size() != 0) {
    data = (const int32_t *) (ptr->int32_data().data());
  } else {
    if (Op::dtype_eq(ptr->data_type(), onnx::TensorProto_DataType_INT32)) {
      data = (const int32_t *) (ptr->raw_data().c_str());
    } else {
      log_fatal("Unable to deduce type for tensor or un-implemented: {}\n", ptr->name());
    }
  }
}


template <>
TensorExtant<int64_t>::TensorExtant(const onnx::TensorProto *ptr) {
  init_dims(ptr);
  if (ptr->int64_data_size() != 0) {
    data = (const int64_t *) (ptr->int64_data().data());
  } else {
    if (Op::dtype_eq(ptr->data_type(), onnx::TensorProto_DataType_INT64)) {
      data = (const int64_t *) (ptr->raw_data().c_str());
    } else {
      log_fatal("Unable to deduce type for tensor or un-implemented: {}\n", ptr->name());
    }
  }
}

template <>
TensorExtant<int8_t>::TensorExtant(const onnx::TensorProto *ptr) {
  init_dims(ptr);
  if (Op::dtype_eq(ptr->data_type(), onnx::TensorProto_DataType_INT8)) {
    data = (const int8_t *) (ptr->raw_data().c_str());
  } else {
    log_fatal("Unable to deduce type for tensor or un-implemented: {}\n", ptr->name());
  }
}

template <>
TensorExtant<uint8_t>::TensorExtant(const onnx::TensorProto *ptr) {
  init_dims(ptr);
  if (Op::dtype_eq(ptr->data_type(), onnx::TensorProto_DataType_UINT8)) {
    data = (const uint8_t *) (ptr->raw_data().c_str());
  } else {
    log_fatal("Unable to deduce type for tensor or un-implemented: {}\n", ptr->name());
  }
}

template <>
TensorExtant<double>::TensorExtant(const onnx::TensorProto *ptr) {
  init_dims(ptr);
  if (Op::dtype_eq(ptr->data_type(), onnx::TensorProto_DataType_DOUBLE)) {
    data = (const double *) (ptr->raw_data().c_str());
  } else {
    log_fatal("Unable to deduce type for tensor or un-implemented: {}\n", ptr->name());
  }
}
