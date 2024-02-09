#include "../src/sim.h"
#include "../src/transformers.h"
#include "../src/utils.h"
#include <numeric>
#include <stdio.h>

Argparse gbl_args;
int main(int argc, char *argv[]) {
  gbl_args.parse(argc, argv);

#if 0
  std::vector<float> expected{-42.4375, -104.125, -142.625, -128.625,
                              147.875,  306.25,   327.25,   257.25,
                              613.375,  1335.25,  1552.25,  1286.25,
                              721.875,  1580.25,  1853.25,  1543.5};
#endif

  std::vector<int> expected{-42, -104, -142, -128, 147, 306,  327,  257,
                              613, 1335, 1552, 1286, 721, 1580, 1853, 1543};

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
                       .stride{2, 2},
                       .dilation{1, 1}};
  Pooler<int> p1;
  auto temp_mat = v2mat<int>(computed, 7, 7);
  auto pooler_output = p1.average_pooler(temp_mat, mp);

  auto output = mat2v<int>(pooler_output, 7, 7);
  bool status = generate_report<int, int>(argv[0], expected, output);

  return status;
}
