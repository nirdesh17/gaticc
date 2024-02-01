#pragma once

#include "Python.h"
#include <string>
#include <vector>
#include <filesystem>

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
      //py_fatal_err_check(item, "PyList_GetItem");
      //if (!PyLong_Check(item)) {
      //  continue;
      //}
      long value = PyLong_AsLong(item);
      //if (value == -1 && PyErr_Occurred()) {
      //  log_err("PyLong_AsLong");
      //}
      vec.push_back((T)value);
    }
    return vec;
  }
};

std::vector<int> py_read_img(PyEngine &engine, std::string const &filepath);
std::vector<int> py_fetch_kernel(PyEngine &engine, int layer, int n, int c);

PyObject *py_preprocess(PyEngine &engine, std::string const &image_path);
PyObject *py_infer_layer_torch(PyEngine &engine, std::string const &model, PyObject *ifm, int layer);

PyObject *py_np2l(PyEngine &engine, PyObject *nparr);
