#pragma once

#include <cstddef>
#include <string>
#include "onnx_parser.h"

class Fstream {
  char *m_buf;
  size_t m_size;
public:
  Fstream(std::string& filename);
  ~Fstream();
};

class Runner {
  public:
    Runner();
    void run(Op::Parser &parser, const std::string& gml_file);
};
