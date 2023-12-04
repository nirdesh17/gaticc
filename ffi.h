#pragma once

#include "Python.h"
#include <string>
#include <vector>

class PyEngine {
    private:
    PyObject *mod;
    public:
        PyEngine(std::string const &mod_name);
        ~PyEngine();
        PyObject *call_func(std::string const &func_name, PyObject *args);
        std::vector<int> il2iv(PyObject *list);
};

std::vector<int> py_read_img(PyEngine &engine, std::string const &filepath);
std::vector<int> py_fetch_kernel(PyEngine &engine, int layer, int n, int c);
