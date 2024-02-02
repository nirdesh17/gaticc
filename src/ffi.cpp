#include "Python.h"
#include "ffi.h"
#include <cstdlib>
#include <iostream>
#include <vector>

/*
 * On reference counting,
 * idea is to use refs to make sure that an element is not de-allocated “Owning
 * a reference” means being responsible for calling Py_DECREF on it when the
 * reference is no longer needed.
 * return values of user-defined functions are considered to pass ownership.
 * the reference returned in the custody of the caller and not the callee.
 */

/* list: borrowed reference */
int py_list_sum(PyObject *list) {
  long sum = 0;
  for (Py_ssize_t i = 0; i < PyList_Size(list); ++i) {
    PyObject *item = PyList_GetItem(list, i);
    if (!PyLong_Check(item)) {
      continue;
    }
    long value = PyLong_AsLong(item);
    if (value == -1 && PyErr_Occurred()) {
      return -1;
    }
    sum += value;
  }
  return sum;
}

/* list: borrowed reference */
void py_list_print(PyObject *list) {
  for (Py_ssize_t i = 0; i < PyList_Size(list); ++i) {
    PyObject *item = PyList_GetItem(list, i);
    if (!PyLong_Check(item)) {
      continue;
    }
    long value = PyLong_AsLong(item);
    if (value == -1 && PyErr_Occurred()) {
      return;
    }
    printf("%ld ", value);
  }
  printf("\n");
}

PyEngine::PyEngine(std::string const &mod_name, std::filesystem::path &mod_dir) {
  /* No-op when called for the second-time */
  Py_Initialize();
  PyObject *sys = PyImport_ImportModule("sys");
  PyObject *path = PyObject_GetAttrString(sys, "path");

  if (PyList_Append(path, PyUnicode_FromString(mod_dir.c_str())) == -1) {
    log_err("PyList_Append");
    exit(EXIT_FAILURE);
  }
  this->mod = PyImport_Import(PyUnicode_FromString(mod_name.c_str()));
  py_fatal_err_check(mod, "PyImport_Import");
  Py_XDECREF(sys);
  Py_XDECREF(path);
}

PyEngine::~PyEngine() {
  Py_XDECREF(mod);
  /* Let the OS handle freeing the interpreter */
  //Py_FinalizeEx();
}

PyObject *PyEngine::call_func(std::string const &func_name, PyObject *args) {
  PyObject *dict = PyModule_GetDict(mod);
  py_fatal_err_check(dict, "PyModule_GetDict");
  PyObject *sum_fn =
      PyDict_GetItem(dict, PyUnicode_FromString(func_name.c_str()));
  py_fatal_err_check(dict, "PyDict_GetItem");
  PyObject *ret = PyObject_CallObject(sum_fn, args);
  py_fatal_err_check(ret, "PyObject_CallObject");

  Py_XDECREF(dict);
  Py_XDECREF(sum_fn);
  return ret;
}

std::vector<int> py_read_img(PyEngine &engine, std::string const &filepath) {
  PyObject *args = Py_BuildValue("(s)", filepath.c_str());
  py_fatal_err_check(args, "Py_BuildValue");
  PyObject *img_array = engine.call_func("read_img", args);
  py_fatal_rv_check(PyList_Check(img_array),
                    "read_img return value not a list");
  std::vector<int> img_vec = engine.il2iv<int>(img_array);
  Py_XDECREF(args);
  Py_XDECREF(img_array);
  return img_vec;
}

std::vector<int> py_fetch_kernel(PyEngine &engine, int layer, int n, int c) {
  PyObject *args = Py_BuildValue("(iii)", layer, n, c);
  py_fatal_err_check(args, "Py_BuildValue");
  PyObject *kernel_array = engine.call_func("fetch_kernel", args);
  py_fatal_rv_check(PyList_Check(kernel_array),
                    "fetch_kernel return value not a list");
  std::vector<int> kernel_vec = engine.il2iv<int>(kernel_array);
  Py_XDECREF(args);
  Py_XDECREF(kernel_array);
  return kernel_vec;
}

PyObject *py_preprocess(PyEngine &engine, std::string const &image_path) {
  PyObject *args = Py_BuildValue("(s)", image_path.c_str());
  py_fatal_err_check(args, "Py_BuildValue");
  PyObject *img_array = engine.call_func("preprocess", args);
  Py_XDECREF(args);
  return img_array;
}

PyObject *py_infer_layer_torch(PyEngine &engine, std::string const &model, PyObject *ifm, int layer) {
  PyObject *args = Py_BuildValue("(sOi)", model.c_str(), ifm, layer);
  py_fatal_err_check(args, "Py_BuildValue");
  PyObject *result = engine.call_func("infer_layer_torch", args);
  py_fatal_rv_check(PyList_Check(result),
                    "infer_layer_torch return value not a list");
  /* Py_BuildValue increments ref for ifm so decrease here 
   * should be DECREF by the caller too
   * */
  Py_XDECREF(ifm);
  Py_XDECREF(args);
  return result;
}

void PyEngine::print_object(PyObject *obj) {
  PyObject_Print(obj, stdout, 0);
  std::cout << '\n';
}
