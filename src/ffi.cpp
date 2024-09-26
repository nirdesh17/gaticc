#include "pch.h"
#define NO_IMPORT_ARRAY
#include "numpy_init.h"

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif
// #include "Python.h"
#include "ffi.h"
// #include <cstdlib>
// #include <iostream>
// #include <vector>


/*
 * On reference counting,
 * idea is to use refs to make sure that an element is not de-allocated “Owning
 * a reference” means being responsible for calling Py_DECREF on it when the
 * reference is no longer needed.
 * return values of user-defined functions are considered to pass ownership.
 * the reference returned in the custody of the caller and not the callee.
 */

PyEngine::PyEngine(std::string const &mod_name, std::filesystem::path &mod_dir) {
  PyObject *sys = PyImport_ImportModule("sys");
  PyObject *path = PyObject_GetAttrString(sys, "path");

  if (PyList_Append(path, PyUnicode_FromString(mod_dir.c_str())) == -1) {
    log_fatal("PyList_Append");
  }

  if (gbl_args.has_option("venv-path")) {
    std::string venv_path = gbl_args["venv-path"].as<std::string>();
    if (PyList_Insert(path, 0, PyUnicode_FromString(venv_path.c_str())) == -1) {
      log_fatal("PyList_Append");
    }
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
  py_fatal_err_check(sum_fn, "PyDict_GetItem");
  PyObject *ret = PyObject_CallObject(sum_fn, args);
  py_fatal_err_check(ret, "PyObject_CallObject");
  return ret;
}

void PyEngine::print_object(PyObject *obj) {
  PyObject_Print(obj, stdout, 0);
  std::cout << '\n';
}
