#pragma once

#define NO_IMPORT_ARRAY
#include "numpy_init.h"

/* From libpython */
#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#endif

#include "tensor.h"
#include "utils.h"
#include <filesystem>
#include <string>
#include <typeinfo>
#include <vector>

#define py_fatal_err_check(var, func_name)                                     \
  do {                                                                         \
    if (var == NULL) {                                                         \
      PyErr_Print();                                                           \
      fprintf(stderr, "%s: %d: %s\n", __FILE__, __LINE__, func_name);          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/* TODO: refactor into above function, probably not needed */
/* return value check */
#define py_fatal_rv_check(var, comment)                                        \
  do {                                                                         \
    if (!var) {                                                                \
      PyErr_Print();                                                           \
      fprintf(stderr, "%s: %d: %s", __FILE__, __LINE__, comment);              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

template <typename T> int deduce_npy_typenum() {
  if (typeid(T) == typeid(float)) {
    return NPY_FLOAT32;
  } else if (typeid(T) == typeid(int)) {
    return NPY_INT32;
  } else if (typeid(T) == typeid(int8_t)) {
    return NPY_INT8;
  } else if (typeid(T) == typeid(uint8_t)) {
    return NPY_UINT8;
  } else {
    log_fatal("Cannot deduce typenum or unimplemented\n");
  }
}

/* Use PyEngine objects to load python files and call
 * functions inside it. Basically a wrapper around the
 * `import <module>` statement.
 */
class PyEngine {
private:
  PyObject *mod;
  PyObject *dict;

public:
  /**
   * mod_name: name of the module
   * mod_dir: absolute path to the dir where mod_name resides
   */
  PyEngine(std::string const &mod_name, std::filesystem::path &mod_dir);
  ~PyEngine();
  PyObject *call_func(std::string const &func_name, PyObject *args) const;
  void print_object(PyObject *obj);
};

/* List (in python) to std::vector */
template <typename T> std::vector<T> il2iv(PyObject *list) {
  if (!PyList_Check(list)) {
    log_fatal("Input not a list\n");
  }
  std::vector<T> vec;

  for (Py_ssize_t i = 0; i < PyList_Size(list); ++i) {
    PyObject *item = PyList_GetItem(list, i);
    py_fatal_err_check(item, "PyList_GetItem");

    T value;
    if (is_int_like<T>(value)) {
      if (!PyLong_Check(item)) {
        log_fatal("heterogenous types\n");
      }
      value = PyLong_AsLong(item);
      if (value == -1 && PyErr_Occurred()) {
        log_fatal("Unable to extract long from obj\n");
      }
    } else if (is_float_like<T>(value)) {
      if (!PyFloat_Check(item)) {
        log_fatal("heterogenous types\n");
        continue;
      }
      value = static_cast<T>(PyFloat_AsDouble(item));
      if (value == -1.0 && PyErr_Occurred()) {
        log_fatal("Unable to extract float from obj\n");
      }
    } else {
      log_fatal("Unsupported type: {}\n", typeid(T).name());
    }
    vec.push_back(value);
  }
  return vec;
}

template <typename T> PyObject *iv2il(std::vector<T> const &v) {
  PyObject *l = PyList_New(v.size());
  py_fatal_rv_check(l, "PyList_New");
  for (int i = 0; i < v.size(); ++i) {
    PyObject *value = PyLong_FromLong(v.at(i));
    if (PyList_SetItem(l, i, value) == -1) {
      log_fatal("PyList_SetItem: out of bounds\n");
    }
  }
  return l;
}

template <typename T>
PyObject *iv2np(std::vector<T> const &v, std::vector<int> const &dims) {
  /* Create a flattened array then call reshape on it */
  Py_intptr_t retdims[1]{prod(dims.begin(), dims.end(), 1)};
  int typenum = deduce_npy_typenum<T>();
  PyObject *nparr = PyArray_SimpleNew(1, retdims, typenum);
  for (int i = 0; i < PyArray_Size(nparr); ++i) {
    T *ptr = (T *)PyArray_GETPTR1((PyArrayObject *)nparr, i);
    *ptr = v[i];
  }
  PyObject *shape = PyTuple_New(dims.size());
  for (int i = 0; i < dims.size(); ++i) {
    PyTuple_SET_ITEM(shape, i, PyLong_FromLong(dims[i]));
  }
  PyObject *ret = PyArray_Reshape((PyArrayObject *)nparr, shape);
  Py_XDECREF(nparr);
  Py_XDECREF(shape);
  return ret;
}

template <typename T>
std::vector<T> np2iv(PyObject *nparr, std::vector<int> &dims) {
  Py_intptr_t *shape = PyArray_SHAPE((PyArrayObject *)nparr);
  int total_dims = PyArray_NDIM((PyArrayObject *)nparr);
  assert(dims.size() == 0 && "shape info is filled by this function via "
                             "push_back, expect 'dims' to be empty");
  for (int i = 0; i < total_dims; ++i) {
    dims.push_back(shape[i]);
  }

  int nparrsz = PyArray_SIZE((PyArrayObject *)nparr);
  std::vector<T> ret(nparrsz);
  PyObject *flattened = PyArray_Flatten((PyArrayObject *)nparr, NPY_CORDER);
  for (int i = 0; i < nparrsz; ++i) {
    ret[i] = *((T *)PyArray_GETPTR1((PyArrayObject *)flattened, i));
  }
}

template <typename T> Tensor<T> *np2t(PyObject *nparr) {
  Py_intptr_t *shape = PyArray_SHAPE((PyArrayObject *)nparr);
  int total_dims = PyArray_NDIM((PyArrayObject *)nparr);
  std::vector<int> dims;
  for (int i = 0; i < total_dims; ++i) {
    dims.push_back(shape[i]);
  }

  int nparrsz = PyArray_Size(nparr);
  Tensor<T> *ret = new TensorCreate<T>(dims);
  PyObject *flattened = PyArray_Flatten((PyArrayObject *)nparr, NPY_CORDER);
  if (!flattened) {
    log_fatal("Could not flatten array\n");
  }
  for (int i = 0; i < nparrsz; ++i) {
    ret->set(i, *((T *)PyArray_GETPTR1((PyArrayObject *)flattened, i)));
  }
  Py_XDECREF(flattened);
  return ret;
}

template <typename T> PyObject *t2np(const Tensor<T> *t) {
  /* Create a flattened array then call reshape on it */
  std::vector<int> dims = t->get_dims();
  Py_intptr_t retdims[1]{prod(dims.begin(), dims.end(), 1)};
  int typenum = deduce_npy_typenum<T>();
  PyObject *nparr = PyArray_SimpleNew(1, retdims, typenum);
  for (int i = 0; i < PyArray_Size(nparr); ++i) {
    T *ptr = (T *)PyArray_GETPTR1((PyArrayObject *)nparr, i);
    *ptr = t->at(i);
  }
  PyObject *shape = PyTuple_New(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    PyTuple_SET_ITEM(shape, i, PyLong_FromLong(dims[i]));
  }
  PyObject *ret = PyArray_Reshape((PyArrayObject *)nparr, shape);
  Py_XDECREF(nparr);
  Py_XDECREF(shape);
  return ret;
}

template <typename T>
void pickle_tensor(const Tensor<T> *t, std::string filename) {
  PyObject *t_obj = t2np(t);
  std::filesystem::path mod_path = "src";
  PyEngine engine("ml_inference", mod_path);
  PyObject *args = Py_BuildValue("(sO)", filename.c_str(), t_obj);
  engine.call_func("save_tensor", args);
  Py_XDECREF(t_obj);
}
