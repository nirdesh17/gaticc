#pragma once

#include "Python.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ndarrayobject.h"
#include "utils.h"
#include <string>
#include <vector>
#include <filesystem>

#define log_err(err) fprintf(stderr, "%s: %d: %s\n", __FILE__, __LINE__, err);

#define py_fatal_err_check(var, func_name)                                     \
  do {                                                                         \
    if (var == NULL) {                                                         \
      PyErr_Print();                                                           \
      fprintf(stderr, "%s: %d: %s\n", __FILE__, __LINE__, func_name);          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/* return value check */
#define py_fatal_rv_check(var, comment)                                        \
  do {                                                                         \
    if (!var) {                                                                \
      PyErr_Print();                                                           \
      fprintf(stderr, "%s: %d: %s", __FILE__, __LINE__, comment);              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/* Use PyEngine objects to load python files and call
 * functions inside it. Basically a wrapper around the
 * `import <module>` statement.
 */
class PyEngine {
private:
  PyObject *mod;

public:
  /**
   * mod_name: name of the module
   * mod_dir: absolute path to the dir where mod_name resides
   */
  PyEngine(std::string const &mod_name, std::filesystem::path &mod_dir);
  ~PyEngine();
  PyObject *call_func(std::string const &func_name, PyObject *args);
  /* Integer List (in python) to Integer Vector (in cpp) */

  template <typename T> 
  std::vector<T> il2iv(PyObject *list) {
    std::vector<T> vec;
    for (Py_ssize_t i = 0; i < PyList_Size(list); ++i) {
      PyObject *item = PyList_GetItem(list, i);
      py_fatal_err_check(item, "PyList_GetItem");
      if (!PyLong_Check(item)) {
        continue;
      }
      long value = PyLong_AsLong(item);
      if (value == -1 && PyErr_Occurred()) {
        log_err("PyLong_AsLong");
      }
      vec.push_back((T)value);
    }
    return vec;
  }

  template <typename T>
  PyObject *iv2il(std::vector<T> const& v) {
    PyObject *l = PyList_New(v.size());
    py_fatal_rv_check(l, "PyList_New");
    for (int i = 0; i < v.size(); ++i) {
      PyObject *value = PyLong_FromLong(v.at(i));
      if (PyList_SetItem(l, i, value) == -1) {
        log_fatal("PyList_SetItem: out of bounds");
      }
    }
    return l;
  }

  template <typename T>
  std::vector<T> np2iv(PyArrayObject *np) {
    std::vector<T> v;
    PyObject *flattend = PyArray_Flatten(np, NPY_ANYORDER);
    for (int i = 0; i < PyArray_Size(flattend); ++i) {
      T* value = (T*) PyArray_GETPTR1(flattend, i);
      v.push_back(*value);
    }
    Py_XDECREF(flattend);
    return v;
  }

};

std::vector<int> py_read_img(PyEngine &engine, std::string const &filepath);
std::vector<int> py_fetch_kernel(PyEngine &engine, int layer, int n, int c);

PyObject *py_preprocess(PyEngine &engine, std::string const &image_path);
PyObject *py_infer_layer_torch(PyEngine &engine, std::string const &model, PyObject *ifm, int layer);
