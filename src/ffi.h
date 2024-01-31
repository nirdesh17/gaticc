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
  std::vector<int> il2iv(PyObject *list);
};

std::vector<int> py_read_img(PyEngine &engine, std::string const &filepath);
std::vector<int> py_fetch_kernel(PyEngine &engine, int layer, int n, int c);

PyObject *py_preprocess(PyEngine &engine, std::string const &image_path);
PyObject *py_infer_layer_torch(PyEngine &engine, std::string const &model, PyObject *ifm, int layer);
