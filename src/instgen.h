#pragma once

#include "onnx.pb.h"
#include "onnx_parser.h"
#include <bitset>

/* Automatically generated code follows from
 * https://github.com/vicharak-in/Gati/blob/main/docs/source/instructions/inst.rst
 * DO NOT EDIT MANUALLY
 */

/* ============ INST BEGIN ======================*/

#define OP_CONV 0x00
#define CONV_Opcode_LOW 0
#define CONV_Opcode_HIGH 3
#define CONV_Opcode_COUNT 4
#define CONV_IW_LOW 4
#define CONV_IW_HIGH 13
#define CONV_IW_COUNT 10
#define CONV_IH_LOW 14
#define CONV_IH_HIGH 23
#define CONV_IH_COUNT 10
#define CONV_OW_LOW 24
#define CONV_OW_HIGH 33
#define CONV_OW_COUNT 10
#define CONV_OH_LOW 34
#define CONV_OH_HIGH 43
#define CONV_OH_COUNT 10
#define CONV_IC_LOW 44
#define CONV_IC_HIGH 53
#define CONV_IC_COUNT 10
#define CONV_KN_LOW 54
#define CONV_KN_HIGH 63
#define CONV_KN_COUNT 10
#define CONV_KW_LOW 64
#define CONV_KW_HIGH 67
#define CONV_KW_COUNT 4
#define CONV_KH_LOW 68
#define CONV_KH_HIGH 71
#define CONV_KH_COUNT 4
#define CONV_Stride_LOW 72
#define CONV_Stride_HIGH 75
#define CONV_Stride_COUNT 4
#define CONV_Pad_LOW 76
#define CONV_Pad_HIGH 78
#define CONV_Pad_COUNT 3
#define CONV_ImageStartAddress_LOW 79
#define CONV_ImageStartAddress_HIGH 110
#define CONV_ImageStartAddress_COUNT 32
#define CONV_ImageEndAddress_LOW 111
#define CONV_ImageEndAddress_HIGH 142
#define CONV_ImageEndAddress_COUNT 32
#define CONV_WeightStartAddress_LOW 143
#define CONV_WeightStartAddress_HIGH 174
#define CONV_WeightStartAddress_COUNT 32
#define CONV_WeightEndAddress_LOW 175
#define CONV_WeightEndAddress_HIGH 206
#define CONV_WeightEndAddress_COUNT 32

#define OP_FC 0x03
#define FC_Opcode_LOW 0
#define FC_Opcode_HIGH 3
#define FC_Opcode_COUNT 4
#define FC_WeightRows_LOW 4
#define FC_WeightRows_HIGH 19
#define FC_WeightRows_COUNT 16
#define FC_WeightCols_LOW 20
#define FC_WeightCols_HIGH 35
#define FC_WeightCols_COUNT 16
#define FC_InputRows_LOW 36
#define FC_InputRows_HIGH 51
#define FC_InputRows_COUNT 16
#define FC_DropoutConstant_LOW 52
#define FC_DropoutConstant_HIGH 59
#define FC_DropoutConstant_COUNT 8
#define FC_Flatten_LOW 60
#define FC_Flatten_HIGH 60
#define FC_Flatten_COUNT 1
#define FC_ImageDim_LOW 61
#define FC_ImageDim_HIGH 80
#define FC_ImageDim_COUNT 20
#define FC_ImageStartAddress_LOW 81
#define FC_ImageStartAddress_HIGH 112
#define FC_ImageStartAddress_COUNT 32
#define FC_ImageEndAddr_LOW 113
#define FC_ImageEndAddr_HIGH 144
#define FC_ImageEndAddr_COUNT 32
#define FC_WeightStartAddress_LOW 145
#define FC_WeightStartAddress_HIGH 176
#define FC_WeightStartAddress_COUNT 32
#define FC_WeightEndAddress_LOW 177
#define FC_WeightEndAddress_HIGH 208
#define FC_WeightEndAddress_COUNT 32
#define FC_Vec2MatCols_LOW 209
#define FC_Vec2MatCols_HIGH 224
#define FC_Vec2MatCols_COUNT 16

#define OP_OutputBlock 0x02
#define OutputBlock_Opcode_LOW 0
#define OutputBlock_Opcode_HIGH 3
#define OutputBlock_Opcode_COUNT 4
#define OutputBlock_AccumulantAddr_LOW 4
#define OutputBlock_AccumulantAddr_HIGH 35
#define OutputBlock_AccumulantAddr_COUNT 32
#define OutputBlock_OutputAddr_LOW 36
#define OutputBlock_OutputAddr_HIGH 67
#define OutputBlock_OutputAddr_COUNT 32
#define OutputBlock_ChannelItr_LOW 68
#define OutputBlock_ChannelItr_HIGH 79
#define OutputBlock_ChannelItr_COUNT 12
#define OutputBlock_KernelItr_LOW 80
#define OutputBlock_KernelItr_HIGH 91
#define OutputBlock_KernelItr_COUNT 12
#define OutputBlock_ImageDimOutput_LOW 92
#define OutputBlock_ImageDimOutput_HIGH 107
#define OutputBlock_ImageDimOutput_COUNT 16
#define OutputBlock_ImageDimAcc_LOW 108
#define OutputBlock_ImageDimAcc_HIGH 123
#define OutputBlock_ImageDimAcc_COUNT 16
#define OutputBlock_AccEn_LOW 124
#define OutputBlock_AccEn_HIGH 124
#define OutputBlock_AccEn_COUNT 1

#define OP_START 0xf
#define START_Opcode_LOW 0
#define START_Opcode_HIGH 3
#define START_Opcode_COUNT 4
#define START_LayerNumber_LOW 4
#define START_LayerNumber_HIGH 15
#define START_LayerNumber_COUNT 12
#define START_TotalLayers_LOW 16
#define START_TotalLayers_HIGH 27
#define START_TotalLayers_COUNT 12

#define OP_TailBlock 0x01
#define TailBlock_Opcode_LOW 0
#define TailBlock_Opcode_HIGH 3
#define TailBlock_Opcode_COUNT 4
#define TailBlock_BNEn_LOW 4
#define TailBlock_BNEn_HIGH 4
#define TailBlock_BNEn_COUNT 1
#define TailBlock_BNChannels_LOW 5
#define TailBlock_BNChannels_HIGH 14
#define TailBlock_BNChannels_COUNT 10
#define TailBlock_BNStartAddress_LOW 15
#define TailBlock_BNStartAddress_HIGH 46
#define TailBlock_BNStartAddress_COUNT 32
#define TailBlock_BNEndAddress_LOW 47
#define TailBlock_BNEndAddress_HIGH 78
#define TailBlock_BNEndAddress_COUNT 32
#define TailBlock_ActEn_LOW 79
#define TailBlock_ActEn_HIGH 79
#define TailBlock_ActEn_COUNT 1
#define TailBlock_ActType_LOW 80
#define TailBlock_ActType_HIGH 83
#define TailBlock_ActType_COUNT 4
#define TailBlock_ActParam_LOW 84
#define TailBlock_ActParam_HIGH 91
#define TailBlock_ActParam_COUNT 8
#define TailBlock_QuantEn_LOW 92
#define TailBlock_QuantEn_HIGH 92
#define TailBlock_QuantEn_COUNT 1
#define TailBlock_QuantScale_LOW 93
#define TailBlock_QuantScale_HIGH 108
#define TailBlock_QuantScale_COUNT 16
#define TailBlock_QuantShift_LOW 109
#define TailBlock_QuantShift_HIGH 113
#define TailBlock_QuantShift_COUNT 5
#define TailBlock_PoolEn_LOW 114
#define TailBlock_PoolEn_HIGH 114
#define TailBlock_PoolEn_COUNT 1
#define TailBlock_PoolType_LOW 115
#define TailBlock_PoolType_HIGH 117
#define TailBlock_PoolType_COUNT 3
#define TailBlock_PoolWidth_LOW 118
#define TailBlock_PoolWidth_HIGH 121
#define TailBlock_PoolWidth_COUNT 4
#define TailBlock_PoolHeight_LOW 122
#define TailBlock_PoolHeight_HIGH 125
#define TailBlock_PoolHeight_COUNT 4
#define TailBlock_PoolStride_LOW 126
#define TailBlock_PoolStride_HIGH 129
#define TailBlock_PoolStride_COUNT 4
#define TailBlock_PoolPadding_LOW 130
#define TailBlock_PoolPadding_HIGH 133
#define TailBlock_PoolPadding_COUNT 4
#define TailBlock_BiasEn_LOW 134
#define TailBlock_BiasEn_HIGH 134
#define TailBlock_BiasEn_COUNT 1
#define TailBlock_FCBiasEn_LOW 135
#define TailBlock_FCBiasEn_HIGH 135
#define TailBlock_FCBiasEn_COUNT 1
#define TailBlock_BiasStartAddress_LOW 136
#define TailBlock_BiasStartAddress_HIGH 167
#define TailBlock_BiasStartAddress_COUNT 32
#define TailBlock_BiasEndAddress_LOW 168
#define TailBlock_BiasEndAddress_HIGH 199
#define TailBlock_BiasEndAddress_COUNT 32

#define ACT_RELU 0x00
#define POOL_MAX 0x00
#define POOL_AVERAGE 0x01
#define POOL_GLOBAL_AVG 0x02

inline std::map<std::string, int>
get_conv_table(const std::bitset<INST_SIZE_BITS> &inst) {
  std::map<std::string, int> tbl;
  tbl.insert({"Opcode", bitset_range_get<CONV_Opcode_COUNT, INST_SIZE_BITS>(
                            inst, CONV_Opcode_LOW, CONV_Opcode_HIGH)});
  tbl.insert({"IW", bitset_range_get<CONV_IW_COUNT, INST_SIZE_BITS>(
                        inst, CONV_IW_LOW, CONV_IW_HIGH)});
  tbl.insert({"IH", bitset_range_get<CONV_IH_COUNT, INST_SIZE_BITS>(
                        inst, CONV_IH_LOW, CONV_IH_HIGH)});
  tbl.insert({"OW", bitset_range_get<CONV_OW_COUNT, INST_SIZE_BITS>(
                        inst, CONV_OW_LOW, CONV_OW_HIGH)});
  tbl.insert({"OH", bitset_range_get<CONV_OH_COUNT, INST_SIZE_BITS>(
                        inst, CONV_OH_LOW, CONV_OH_HIGH)});
  tbl.insert({"IC", bitset_range_get<CONV_IC_COUNT, INST_SIZE_BITS>(
                        inst, CONV_IC_LOW, CONV_IC_HIGH)});
  tbl.insert({"KN", bitset_range_get<CONV_KN_COUNT, INST_SIZE_BITS>(
                        inst, CONV_KN_LOW, CONV_KN_HIGH)});
  tbl.insert({"KW", bitset_range_get<CONV_KW_COUNT, INST_SIZE_BITS>(
                        inst, CONV_KW_LOW, CONV_KW_HIGH)});
  tbl.insert({"KH", bitset_range_get<CONV_KH_COUNT, INST_SIZE_BITS>(
                        inst, CONV_KH_LOW, CONV_KH_HIGH)});
  tbl.insert({"Stride", bitset_range_get<CONV_Stride_COUNT, INST_SIZE_BITS>(
                            inst, CONV_Stride_LOW, CONV_Stride_HIGH)});
  tbl.insert({"Pad", bitset_range_get<CONV_Pad_COUNT, INST_SIZE_BITS>(
                         inst, CONV_Pad_LOW, CONV_Pad_HIGH)});
  tbl.insert(
      {"ImageStartAddress",
       bitset_range_get<CONV_ImageStartAddress_COUNT, INST_SIZE_BITS>(
           inst, CONV_ImageStartAddress_LOW, CONV_ImageStartAddress_HIGH)});
  tbl.insert({"ImageEndAddress",
              bitset_range_get<CONV_ImageEndAddress_COUNT, INST_SIZE_BITS>(
                  inst, CONV_ImageEndAddress_LOW, CONV_ImageEndAddress_HIGH)});
  tbl.insert(
      {"WeightStartAddress",
       bitset_range_get<CONV_WeightStartAddress_COUNT, INST_SIZE_BITS>(
           inst, CONV_WeightStartAddress_LOW, CONV_WeightStartAddress_HIGH)});
  tbl.insert(
      {"WeightEndAddress",
       bitset_range_get<CONV_WeightEndAddress_COUNT, INST_SIZE_BITS>(
           inst, CONV_WeightEndAddress_LOW, CONV_WeightEndAddress_HIGH)});
  return tbl;
}
inline void pretty_print_conv(const std::bitset<INST_SIZE_BITS> &inst) {
  auto tbl = get_conv_table(inst);
  print_table(tbl);
}
inline std::map<std::string, int>
get_fc_table(const std::bitset<INST_SIZE_BITS> &inst) {
  std::map<std::string, int> tbl;
  tbl.insert({"Opcode", bitset_range_get<FC_Opcode_COUNT, INST_SIZE_BITS>(
                            inst, FC_Opcode_LOW, FC_Opcode_HIGH)});
  tbl.insert(
      {"WeightRows", bitset_range_get<FC_WeightRows_COUNT, INST_SIZE_BITS>(
                         inst, FC_WeightRows_LOW, FC_WeightRows_HIGH)});
  tbl.insert(
      {"WeightCols", bitset_range_get<FC_WeightCols_COUNT, INST_SIZE_BITS>(
                         inst, FC_WeightCols_LOW, FC_WeightCols_HIGH)});
  tbl.insert({"InputRows", bitset_range_get<FC_InputRows_COUNT, INST_SIZE_BITS>(
                               inst, FC_InputRows_LOW, FC_InputRows_HIGH)});
  tbl.insert({"DropoutConstant",
              bitset_range_get<FC_DropoutConstant_COUNT, INST_SIZE_BITS>(
                  inst, FC_DropoutConstant_LOW, FC_DropoutConstant_HIGH)});
  tbl.insert({"Flatten", bitset_range_get<FC_Flatten_COUNT, INST_SIZE_BITS>(
                             inst, FC_Flatten_LOW, FC_Flatten_HIGH)});
  tbl.insert({"ImageDim", bitset_range_get<FC_ImageDim_COUNT, INST_SIZE_BITS>(
                              inst, FC_ImageDim_LOW, FC_ImageDim_HIGH)});
  tbl.insert({"ImageStartAddress",
              bitset_range_get<FC_ImageStartAddress_COUNT, INST_SIZE_BITS>(
                  inst, FC_ImageStartAddress_LOW, FC_ImageStartAddress_HIGH)});
  tbl.insert(
      {"ImageEndAddr", bitset_range_get<FC_ImageEndAddr_COUNT, INST_SIZE_BITS>(
                           inst, FC_ImageEndAddr_LOW, FC_ImageEndAddr_HIGH)});
  tbl.insert(
      {"WeightStartAddress",
       bitset_range_get<FC_WeightStartAddress_COUNT, INST_SIZE_BITS>(
           inst, FC_WeightStartAddress_LOW, FC_WeightStartAddress_HIGH)});
  tbl.insert({"WeightEndAddress",
              bitset_range_get<FC_WeightEndAddress_COUNT, INST_SIZE_BITS>(
                  inst, FC_WeightEndAddress_LOW, FC_WeightEndAddress_HIGH)});
  tbl.insert(
      {"Vec2MatCols", bitset_range_get<FC_Vec2MatCols_COUNT, INST_SIZE_BITS>(
                          inst, FC_Vec2MatCols_LOW, FC_Vec2MatCols_HIGH)});
  return tbl;
}
inline void pretty_print_fc(const std::bitset<INST_SIZE_BITS> &inst) {
  auto tbl = get_fc_table(inst);
  print_table(tbl);
}
inline std::map<std::string, int>
get_outputblock_table(const std::bitset<INST_SIZE_BITS> &inst) {
  std::map<std::string, int> tbl;
  tbl.insert(
      {"Opcode", bitset_range_get<OutputBlock_Opcode_COUNT, INST_SIZE_BITS>(
                     inst, OutputBlock_Opcode_LOW, OutputBlock_Opcode_HIGH)});
  tbl.insert(
      {"AccumulantAddr",
       bitset_range_get<OutputBlock_AccumulantAddr_COUNT, INST_SIZE_BITS>(
           inst, OutputBlock_AccumulantAddr_LOW,
           OutputBlock_AccumulantAddr_HIGH)});
  tbl.insert(
      {"OutputAddr",
       bitset_range_get<OutputBlock_OutputAddr_COUNT, INST_SIZE_BITS>(
           inst, OutputBlock_OutputAddr_LOW, OutputBlock_OutputAddr_HIGH)});
  tbl.insert(
      {"ChannelItr",
       bitset_range_get<OutputBlock_ChannelItr_COUNT, INST_SIZE_BITS>(
           inst, OutputBlock_ChannelItr_LOW, OutputBlock_ChannelItr_HIGH)});
  tbl.insert(
      {"KernelItr",
       bitset_range_get<OutputBlock_KernelItr_COUNT, INST_SIZE_BITS>(
           inst, OutputBlock_KernelItr_LOW, OutputBlock_KernelItr_HIGH)});
  tbl.insert(
      {"ImageDimOutput",
       bitset_range_get<OutputBlock_ImageDimOutput_COUNT, INST_SIZE_BITS>(
           inst, OutputBlock_ImageDimOutput_LOW,
           OutputBlock_ImageDimOutput_HIGH)});
  tbl.insert(
      {"ImageDimAcc",
       bitset_range_get<OutputBlock_ImageDimAcc_COUNT, INST_SIZE_BITS>(
           inst, OutputBlock_ImageDimAcc_LOW, OutputBlock_ImageDimAcc_HIGH)});
  tbl.insert(
      {"AccEn", bitset_range_get<OutputBlock_AccEn_COUNT, INST_SIZE_BITS>(
                    inst, OutputBlock_AccEn_LOW, OutputBlock_AccEn_HIGH)});
  return tbl;
}
inline void pretty_print_outputblock(const std::bitset<INST_SIZE_BITS> &inst) {
  auto tbl = get_outputblock_table(inst);
  print_table(tbl);
}
inline std::map<std::string, int>
get_start_table(const std::bitset<INST_SIZE_BITS> &inst) {
  std::map<std::string, int> tbl;
  tbl.insert({"Opcode", bitset_range_get<START_Opcode_COUNT, INST_SIZE_BITS>(
                            inst, START_Opcode_LOW, START_Opcode_HIGH)});
  tbl.insert({"LayerNumber",
              bitset_range_get<START_LayerNumber_COUNT, INST_SIZE_BITS>(
                  inst, START_LayerNumber_LOW, START_LayerNumber_HIGH)});
  tbl.insert({"TotalLayers",
              bitset_range_get<START_TotalLayers_COUNT, INST_SIZE_BITS>(
                  inst, START_TotalLayers_LOW, START_TotalLayers_HIGH)});
  return tbl;
}
inline void pretty_print_start(const std::bitset<INST_SIZE_BITS> &inst) {
  auto tbl = get_start_table(inst);
  print_table(tbl);
}
inline std::map<std::string, int>
get_tailblock_table(const std::bitset<INST_SIZE_BITS> &inst) {
  std::map<std::string, int> tbl;
  tbl.insert(
      {"Opcode", bitset_range_get<TailBlock_Opcode_COUNT, INST_SIZE_BITS>(
                     inst, TailBlock_Opcode_LOW, TailBlock_Opcode_HIGH)});
  tbl.insert({"BNEn", bitset_range_get<TailBlock_BNEn_COUNT, INST_SIZE_BITS>(
                          inst, TailBlock_BNEn_LOW, TailBlock_BNEn_HIGH)});
  tbl.insert({"BNChannels",
              bitset_range_get<TailBlock_BNChannels_COUNT, INST_SIZE_BITS>(
                  inst, TailBlock_BNChannels_LOW, TailBlock_BNChannels_HIGH)});
  tbl.insert(
      {"BNStartAddress",
       bitset_range_get<TailBlock_BNStartAddress_COUNT, INST_SIZE_BITS>(
           inst, TailBlock_BNStartAddress_LOW, TailBlock_BNStartAddress_HIGH)});
  tbl.insert(
      {"BNEndAddress",
       bitset_range_get<TailBlock_BNEndAddress_COUNT, INST_SIZE_BITS>(
           inst, TailBlock_BNEndAddress_LOW, TailBlock_BNEndAddress_HIGH)});
  tbl.insert({"ActEn", bitset_range_get<TailBlock_ActEn_COUNT, INST_SIZE_BITS>(
                           inst, TailBlock_ActEn_LOW, TailBlock_ActEn_HIGH)});
  tbl.insert(
      {"ActType", bitset_range_get<TailBlock_ActType_COUNT, INST_SIZE_BITS>(
                      inst, TailBlock_ActType_LOW, TailBlock_ActType_HIGH)});
  tbl.insert(
      {"ActParam", bitset_range_get<TailBlock_ActParam_COUNT, INST_SIZE_BITS>(
                       inst, TailBlock_ActParam_LOW, TailBlock_ActParam_HIGH)});
  tbl.insert(
      {"QuantEn", bitset_range_get<TailBlock_QuantEn_COUNT, INST_SIZE_BITS>(
                      inst, TailBlock_QuantEn_LOW, TailBlock_QuantEn_HIGH)});
  tbl.insert({"QuantScale",
              bitset_range_get<TailBlock_QuantScale_COUNT, INST_SIZE_BITS>(
                  inst, TailBlock_QuantScale_LOW, TailBlock_QuantScale_HIGH)});
  tbl.insert({"QuantShift",
              bitset_range_get<TailBlock_QuantShift_COUNT, INST_SIZE_BITS>(
                  inst, TailBlock_QuantShift_LOW, TailBlock_QuantShift_HIGH)});
  tbl.insert(
      {"PoolEn", bitset_range_get<TailBlock_PoolEn_COUNT, INST_SIZE_BITS>(
                     inst, TailBlock_PoolEn_LOW, TailBlock_PoolEn_HIGH)});
  tbl.insert(
      {"PoolType", bitset_range_get<TailBlock_PoolType_COUNT, INST_SIZE_BITS>(
                       inst, TailBlock_PoolType_LOW, TailBlock_PoolType_HIGH)});
  tbl.insert({"PoolWidth",
              bitset_range_get<TailBlock_PoolWidth_COUNT, INST_SIZE_BITS>(
                  inst, TailBlock_PoolWidth_LOW, TailBlock_PoolWidth_HIGH)});
  tbl.insert({"PoolHeight",
              bitset_range_get<TailBlock_PoolHeight_COUNT, INST_SIZE_BITS>(
                  inst, TailBlock_PoolHeight_LOW, TailBlock_PoolHeight_HIGH)});
  tbl.insert({"PoolStride",
              bitset_range_get<TailBlock_PoolStride_COUNT, INST_SIZE_BITS>(
                  inst, TailBlock_PoolStride_LOW, TailBlock_PoolStride_HIGH)});
  tbl.insert(
      {"PoolPadding",
       bitset_range_get<TailBlock_PoolPadding_COUNT, INST_SIZE_BITS>(
           inst, TailBlock_PoolPadding_LOW, TailBlock_PoolPadding_HIGH)});
  tbl.insert(
      {"BiasEn", bitset_range_get<TailBlock_BiasEn_COUNT, INST_SIZE_BITS>(
                     inst, TailBlock_BiasEn_LOW, TailBlock_BiasEn_HIGH)});
  tbl.insert(
      {"FCBiasEn", bitset_range_get<TailBlock_FCBiasEn_COUNT, INST_SIZE_BITS>(
                       inst, TailBlock_FCBiasEn_LOW, TailBlock_FCBiasEn_HIGH)});
  tbl.insert(
      {"BiasStartAddress",
       bitset_range_get<TailBlock_BiasStartAddress_COUNT, INST_SIZE_BITS>(
           inst, TailBlock_BiasStartAddress_LOW,
           TailBlock_BiasStartAddress_HIGH)});
  tbl.insert(
      {"BiasEndAddress",
       bitset_range_get<TailBlock_BiasEndAddress_COUNT, INST_SIZE_BITS>(
           inst, TailBlock_BiasEndAddress_LOW, TailBlock_BiasEndAddress_HIGH)});
  return tbl;
}
inline void pretty_print_tailblock(const std::bitset<INST_SIZE_BITS> &inst) {
  auto tbl = get_tailblock_table(inst);
  print_table(tbl);
}

/* ============ INST END ======================*/

/* Corresponds to AXI_ADDR_WIDTH */
#define WORD_SIZE 32
#define ACC_SIZE 32
/* What address do instructions start from */
#define GATI_INST_ORG 0
#define DWP_HEADER_BYTES 12

/* Megablock and Miniblock
 *
 * All operators, implemented or not, can be divided into two sects: Megablock
 * and Miniblock
 *
 * Megablocks are a set of miniblocks that execute in a pipeline.  Input to a
 * megablock comes from dram and output from a megablock is written back to
 * dram. As miniblocks are arranged in a pipeline, input comes from a previous
 * miniblock. A megablock opener is the first miniblock of a pipeline. Only one
 * megablock can execute at a time. All miniblocks execute at the same time.
 *
 * Currently, (TODO: this should be updated later), there are two megablocks:
 * conv and fc and many miniblocks: relu, maxpool, bias, quantizer,
 * outputpipeline etc. When a convolution is happening, these miniblocks form a
 * megablock: conv, bias, quantizer, relu, maxpool, output When a FC is
 * happening, these miniblocks form a megablock: fc, bias, quantizer, relu,
 * output
 *
 * Some miniblocks can be skipped, for example, maxpool is skipped if a maxpool
 * op does not follow convolution.
 */

bool is_megablock(const Op::LayerBase *l);
bool is_miniblock(const Op::LayerBase *l);

/* InstGen generates according to the ISA
 *
 * It does this in multiple different passes passing over the execution order
 * as returned by parser. Instructions in the isa are compact, for example, the
 * tail instruction has information related to relu, quantization, batchnorm,
 * bias etc. On the other hand, onnx represents these as separate layers or as
 * a part of a layer corresponding to an entirely different instruction (for
 * example, bias info can be found in conv nodes). To deal with this, InstGen
 * generates the final instructions in a emit-merge strategy. Each node in onnx
 * emits all the instructions it is capable of in a InstBlob, later a pass over
 * InstBlob merges like instructions into one by ORing them together. 
 *
 * Example: If an onnx graph contains CONV -> RELU -> MAXPOOL -> FC -> RELU, 
 *
 * In the emit phase, these instructions will be generated (in order):
 *
 *  CONV, OutputBlock (from conv node), Tail (from bias), Tail (from relu), 
 *  Tail (from maxpool) FC, OutputBlock (from fc node), Tail (from fc bias),
 *  Tail (from relu)
 *
 * In the merge phase, like instructions will be combined thusly to result in
 * these instructions:
 *  
 *  CONV, OutputBlock (from conv node), Tail (bias, relu, maxpool), 
 *  FC, OutputBlock (from fc node), Tail (fc bias, relu)
 *
 */

/* TODO: explain DWP */

class InstGen {
  InstBlob ret_inst;
  /* Total bytes to be allocated including instructions, weights, io
   * data, and partial sum data
   */
  int total_model_size;
  int total_dwp_packets;
public:
  InstGen(Op::Parser &parser);
  InstBlob get_blob();
  int model_size();
  int dwp_packets();
};

/*
 * AddressGen generates addresses to be substituted in config instructions.
 * It does this by separating the address space (ideally all of the available
 * ram) in 4 distinct regions as shown below.
 *
 * +----------+---------------------+--------------------+--------------------+
 * |          |                     |                    |                    |
 * | Config   |  Weights & Biases   |    Input/Output    |    Accumulants     |
 * |          |                     |                    |                    |
 * +----------+---------------------+--------------------+--------------------+
 * 0                                                                         MAX
 *
 * Config starts at address 0 and its size is known a priori. Same for weights
 * and biases. Input/Output are final activations of layers i.e. intermidiate
 * values of the model and are stored in I/O region. Accumulants are
 * intermidiate values of a layer (as opposed to a model), they tend to be
 * greater in width than I/O (where I/O would be 8bit, Accumulants would be
 * 32bits), are stored in the final segement. Data in config region is allocated
 * all  at once, it fits all the instructions. Data is w/b region is allocated
 * on a FCFS basis. As a result, weights/biases for first layer to be executed
 * will come first in the ram. Data is I/O is allocated based on VirtualAddress
 * registers assigned to each LayerBase by RegisterAllocator. Data is
 * Accumulants is allocated in the same fashion as I/O but with a fixed offset
 * and data width.
 */

class AddressGen {
  /* pointer to the current address from which ram
   * addresses can be assigned
   */
  uint32_t current_address;
  /* Size (in words) occupied by inst region */
  int inst_region_size;
  int io_region_register_size;
  int weight_region_size;
  int max_io_reg;

  uint32_t ram_size_max;

  void addr_incr(uint32_t size);

  int get_total_instructions(const std::vector<Op::LayerBase *> &order);
  int get_io_region_register_size(const std::vector<Op::LayerBase *> &order);
  int get_weight_size(const std::vector<Op::LayerBase *> &order);
  int get_max_io_reg(const std::vector<Op::LayerBase *> &order);

public:
  AddressGen(const std::vector<Op::LayerBase *> &order);
  /* get a address in weights/bias region */
  uint32_t alloc(uint32_t size);
  /* get a address in io region */
  uint32_t io_addr_from_register(Op::VirtualAddress reg);
  /* get a address in accumulant region */
  uint32_t ps_addr_from_register(Op::VirtualAddress reg);
  int io_reg_size();
  int get_model_size();
};

void pretty_print(const InstBlob &blob);
void pretty_print(const std::bitset<INST_SIZE_BITS>& inst);

template <typename T>
uint32_t aligned_conv_weight(const T &wdims) {
  assert(wdims.size() == 4);
  auto w = wdims;
  auto sa_arch = get_sa_arch();
  w[TENSOR_4D_CHANNELS] = ceil_mod(w[TENSOR_4D_CHANNELS], sa_arch[2]); 
  w[TENSOR_4D_BATCH] = ceil_mod(w[TENSOR_4D_BATCH], sa_arch[1]);
  uint32_t ret = prod(w.begin(), w.end(), 1); 
  return ret;
}

template <typename T>
uint32_t aligned_conv_bias(const T &dims) {
  assert(dims.size() == 1);
  auto sa_arch = get_sa_arch();
  uint32_t ret = ceil_mod(dims[0], sa_arch[2]);
  return ret;
}

template <typename T>
uint32_t aligned_conv_input(const T &dims) {
  assert(dims.size() == 4);
  auto sa_arch = get_sa_arch();
  auto i = dims;
  i[TENSOR_4D_CHANNELS] = ceil_mod(i[TENSOR_4D_CHANNELS], sa_arch[2]); 
  int ret = prod(i.begin(), i.end(), 1); 
  return ret;
}

template <typename T>
uint32_t aligned_conv_output(const T &dims) {
  assert(dims.size() == 4);
  auto sa_arch = get_sa_arch();
  auto i = dims;
  i[TENSOR_4D_CHANNELS] = ceil_mod(i[TENSOR_4D_CHANNELS], sa_arch[1]); 
  int ret = prod(i.begin(), i.end(), 1); 
  return ret;
}

template <typename T>
uint32_t aligned_fc_weight(const T &dims) {
  assert(dims.size() == 2);
  auto va_size = get_va_size();
  auto w = dims;
  w[0] = ceil_mod(w[0], WORD_SIZE);
  w[1] = ceil_mod(w[1], va_size);
  int ret = prod(w.begin(), w.end(), 1); 
  return ret;
}

template <typename T>
uint32_t aligned_fc_bias(const T &dims) {
  assert(dims.size() == 1);
  auto va_size = get_va_size();
  uint32_t ret = ceil_mod(dims[0], va_size);
  return ret;
}

template <typename T>
uint32_t aligned_fc_io(const T &dims) {
  assert(dims.size() == 2);
  assert(dims[0] == 1);
  uint32_t ret = ceil_mod(dims[1], WORD_SIZE);
  return ret;
}

enum ENGINES {
  ENGINE_UNKNOWN,
  ENGINE_SA,
  ENGINE_FC,
  ENGINE_BIAS,
};

struct InitAddrRow {
  uint32_t addr;
  const onnx::TensorProto *data;
  int engine;
};

class InitializerTable {
  std::vector<InitAddrRow> tbl;
  public:
    void push_back(uint32_t addr, const onnx::TensorProto *data, int engine);
};

/* get nth byte (0 being LSB), of a */
template <typename T> inline char get_byte(T a, int n) {
  assert(n < sizeof(T) && n >= 0);
  char c = (a >> (n * 8)) & 0xff;
  return c;
}

class BinBlob {
  char *m_data;
  size_t m_size;
  /* byte wise index into data */
  size_t m_ptr;

  template <typename T> void generic_append(T a) {
    /* reverse iteration for big endian */
    for (int i = sizeof(T) - 1; i >= 0; --i) {
      char c = get_byte(a, i);
      m_data[m_ptr++] = c;
    }
  }

public:
  BinBlob(char *data, size_t size);
  void append(int a);
  void append(uint8_t a);
  void append(int8_t a);

  template <typename T>
  void append(const std::vector<T> &vec) {
    assert(vec.size() > 0);
    std::cout << "vec size " << vec.size() << '\n';
    std::cout << "m_size - ptr " << m_size - m_ptr << '\n';
    assert(vec.size() * sizeof(vec[0]) <= (m_size - m_ptr));
    for (T i : vec) {
      generic_append(i);
    }
  }
  void append(const InstBlob& instblob, int page_num, uint32_t addr);
  //void append(const InitializerTable &tbl);
  /* do not allow type that are not explicityly implemented */
  template <typename T> void append(T i) = delete;
  void print();
};

/* Prepares and optionally serializes gml model into
 * gml files
 */
class GmlGen {
  /* origin address */
  uint32_t m_org;

public:
  GmlGen(uint32_t org);
  BinBlob generate_gml(Op::Parser &parser);
};
