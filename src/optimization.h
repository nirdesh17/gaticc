#pragma once

#include "onnx_parser.h"

void split_large_kernel(Op::Graph &g);
void update_channel_offsets(Op::Graph g);

