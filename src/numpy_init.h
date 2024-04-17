#pragma once

/*
 * include numpy libraries by:
 *  #define NO_IMPORT_ARRAY
 *  #include "numpy_init.h" 
 * in all *.cpp,h files except main.cpp where defining NO_IMPORT_ARRAY
 * can be omitted
 *
 *  See https://pythonextensionpatterns.readthedocs.io/en/latest/cpp_and_numpy.html#c-and-the-numpy-c-api
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL sysim_ARRAY_API

#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
