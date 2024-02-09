#include "../src/sim.h"
#include "../src/transformers.h"
#include "../src/utils.h"
#include "../src/onnx_parser.h"
#include <numeric>
#include <stdio.h>

int main(int argc, char *argv[]) {

  std::vector<int> expected{
      70,   70,   70,   56,   42,   28,   14,   0,    546,  581,  616,
      651,  686,  721,  721,  721,  1036, 1120, 1204, 1288, 1372, 1456,
      1456, 1456, 1526, 1659, 1792, 1925, 2058, 2191, 2191, 2191, 2016,
      2198, 2380, 2562, 2744, 2926, 2926, 2926, 2506, 2737, 2968, 3199,
      3430, 3661, 3661, 3661, 2506, 2737, 2968, 3199, 3430, 3661, 3661,
      3661, 2506, 2737, 2968, 3199, 3430, 3661, 3661, 3661};

  const int SA_rows = 7;
  const int SA_columns = 7;
  const int input_rows = 7;
  const int input_columns = 7;

  std::vector<int> weight(SA_rows * SA_columns);
  std::vector<int> input_matrix(input_rows * input_columns);

  std::iota(weight.begin(), weight.end(), -12);
  std::iota(input_matrix.begin(), input_matrix.end(), -12);

  SA<int, int> SA1(SA_rows, SA_columns);
  SA1.load_weights(weight);

  Chain c1;
  c1.push(new Chainblock());

  GemmTransformer<int, int> GT1(input_rows, input_columns, SA_rows, SA_columns);
  auto out = GT1.transform(input_matrix);

  SA1.propagate(out, c1);
  auto t1 = SA1.get_output();
  auto computed = GT1.untransform(t1);

  Op::MaxpoolParams mp{.imap{input_rows, input_columns},
                   .k{4, 4},
                   .pad{2, 2, 2, 2},
                   .stride{1, 1},
                   .dilation{1, 1}};

  Pooler<int> p1;
  auto temp_mat = v2mat<int>(computed, 7, 7);
  Mat<int> pooler_output = p1.max_pooler(temp_mat, mp);
  std::vector<int> output = mat2v<int>(pooler_output, 7, 7);
  bool status = generate_report<int, int>(argv[0], expected, output);
  return status;
}
