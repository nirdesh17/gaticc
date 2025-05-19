#pragma once

#include <string>
#include <vector>

using std::string;
using vss = std::vector<std::pair<string, string>>;

void init();
void compile(const string& onnx_path, const string &gml_path, const vss& );
void summary(const string& onnx_path, const vss& rest);
void sim(const string& onnx_path, const string& loadpy, const string& preprocfn, const string& postprocfn, const vss& rest);
void run(const string& onnx_path, const string& gml_path, const string& loadpy, const string& preprocfn, const string& postprocfn, const vss& rest);
