#pragma once

#include "onnx_parser.h"
#include <bitset>
#include <queue>

bool is_miniblock_opener(Op::LayerBase *l);

bool is_miniblock(const Op::LayerBase *l);



class InstGen {
  InstBlob instructions;
  void set_device(const std::vector<Op::LayerBase *> &order);
public:
  InstGen(const std::vector<Op::LayerBase *> &order);
  InstBlob get_blob();
};
