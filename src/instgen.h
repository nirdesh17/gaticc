#pragma once

#include "onnx_parser.h"
#include <bitset>
#include <queue>

#define OP_CONV 0
#define Opcode_LOW 0
#define Opcode_HIGH 3
#define Opcode_COUNT 4
#define IW_LOW 4
#define IW_HIGH 13
#define IW_COUNT 10
#define IH_LOW 14
#define IH_HIGH 23
#define IH_COUNT 10
#define OW_LOW 24
#define OW_HIGH 33
#define OW_COUNT 10
#define OH_LOW 34
#define OH_HIGH 43
#define OH_COUNT 10
#define IC_LOW 44
#define IC_HIGH 53
#define IC_COUNT 10
#define KN_LOW 54
#define KN_HIGH 63
#define KN_COUNT 10
#define KW_LOW 64
#define KW_HIGH 67
#define KW_COUNT 4
#define KH_LOW 68
#define KH_HIGH 71
#define KH_COUNT 4
#define Stride_LOW 72
#define Stride_HIGH 75
#define Stride_COUNT 4
#define Pad_LOW 76
#define Pad_HIGH 78
#define Pad_COUNT 3
#define ChannelItr_LOW 79
#define ChannelItr_HIGH 90
#define ChannelItr_COUNT 12
#define KernelItr_LOW 91
#define KernelItr_HIGH 102
#define KernelItr_COUNT 12
#define ImageStartAddress_LOW 103
#define ImageStartAddress_HIGH 134
#define ImageStartAddress_COUNT 32
#define ImageEndAddress_LOW 135
#define ImageEndAddress_HIGH 166
#define ImageEndAddress_COUNT 32
#define WeightStartAddress_LOW 167
#define WeightStartAddress_HIGH 198
#define WeightStartAddress_COUNT 32
#define WeightEndAddress_LOW 199
#define WeightEndAddress_HIGH 230
#define WeightEndAddress_COUNT 32

bool is_miniblock_opener(Op::LayerBase *l);

bool is_miniblock(const Op::LayerBase *l);

class InstGen {
  InstBlob instructions;
  void set_device(const std::vector<Op::LayerBase *> &order);
public:
  InstGen(const std::vector<Op::LayerBase *> &order);
  InstBlob get_blob();
};
