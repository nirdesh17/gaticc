#pragma once

#include "onnx_parser.h"

void split_large_kernel(Op::Graph &g);
void dump_graph(const Op::Graph &g, const std::string &tag);