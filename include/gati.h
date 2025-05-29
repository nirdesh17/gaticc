#pragma once

#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using std::string;
using vss = std::vector<std::pair<string, string>>;
namespace py = pybind11;

void init();
void compile(const string& onnx_path, const string &gml_path, const vss& );
void summary(const string& onnx_path, const vss& rest);
void sim(const string& onnx_path, const string& loadpy, const string& preprocfn, const string& postprocfn, const vss& rest);
void run(const string& onnx_path, const string& gml_path, const string& loadpy, const string& preprocfn, const string& postprocfn, const vss& rest);
__attribute__((visibility("default"))) py::array sim2(const std::string& onnx_path, py::array arr, const vss& rest);
