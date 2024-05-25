#pragma once

#include "onnx_parser.h"
#include <bitset>

/* Instructions copied from
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
#define CONV_ChannelItr_LOW 79
#define CONV_ChannelItr_HIGH 90
#define CONV_ChannelItr_COUNT 12
#define CONV_KernelItr_LOW 91
#define CONV_KernelItr_HIGH 102
#define CONV_KernelItr_COUNT 12
#define CONV_ImageStartAddress_LOW 103
#define CONV_ImageStartAddress_HIGH 134
#define CONV_ImageStartAddress_COUNT 32
#define CONV_ImageEndAddress_LOW 135
#define CONV_ImageEndAddress_HIGH 166
#define CONV_ImageEndAddress_COUNT 32
#define CONV_WeightStartAddress_LOW 167
#define CONV_WeightStartAddress_HIGH 198
#define CONV_WeightStartAddress_COUNT 32
#define CONV_WeightEndAddress_LOW 199
#define CONV_WeightEndAddress_HIGH 230
#define CONV_WeightEndAddress_COUNT 32

#define OP_FC 0x04
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
#define FC_KernelIteration_LOW 145
#define FC_KernelIteration_HIGH 160
#define FC_KernelIteration_COUNT 16
#define FC_RWAddressCountFlatten_LOW 161
#define FC_RWAddressCountFlatten_HIGH 176
#define FC_RWAddressCountFlatten_COUNT 16

#define OP_OutputBlock 0x03
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

#define OP_START 0xff
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

/* ============ INST END ======================*/

/* Corresponds to AXI_ADDR_WIDTH */
#define WORD_SIZE 32
#define ACC_SIZE 32

bool is_megablock(const Op::LayerBase *l);
bool is_miniblock(const Op::LayerBase *l);

class InstGen {
  InstBlob instructions;
  void set_device(const std::vector<Op::LayerBase *> &order);
public:
  InstGen(Op::Parser &parser);
  InstBlob get_blob();
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
 * values of the model and are stored in I/O region. Accumulants are intermidiate
 * values of a layer (as opposed to a model), they tend to be greater in width 
 * than I/O (where I/O would be 8bit, Accumulants would be 32bits), are stored
 * in the final segement. Data in config region is allocated all  at once, it
 * fits all the instructions. Data is w/b region is allocated on a FCFS basis.
 * As a result, weights/biases for first layer to be executed will come first
 * in the ram. Data is I/O is allocated based on VirtualAddress registers assigned
 * to each LayerBase by RegisterAllocator. Data is Accumulants is allocated
 * in the same fashion as I/O but with a fixed offset and data width.
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

  int get_total_instructions(const std::vector<Op::LayerBase*> &order);
  int get_io_region_register_size(const std::vector<Op::LayerBase*> &order);
  int get_weight_size(const std::vector<Op::LayerBase*> &order);
  int get_max_io_reg(const std::vector<Op::LayerBase*> &order);
public:
  AddressGen(const std::vector<Op::LayerBase*> &order);
  /* get a address in weights/bias region */
  uint32_t alloc(uint32_t size);
  /* get a address in io region */
  uint32_t io_addr_from_register(Op::VirtualAddress reg);
  /* get a address in accumulant region */
  uint32_t ps_addr_from_register(Op::VirtualAddress reg);
  int io_reg_size();
};
