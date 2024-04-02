#include "../src/onnx_parser.h"
#include "../src/sim.h"
#include "../src/transformers.h"
#include "../src/utils.h"
#include <numeric>

Argparse gbl_args;
int main(int argc, char *argv[]) {
  gbl_args.parse(argc, argv);
  std::vector<int> expected{
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  2,  2,  2,  0,  0,  0,
      0,  0,  0,  3,  3,  3,  0,  0,  0,  1,  1,  0,  4,  4,  4,  7,  7,  0,
      2,  2,  2,  5,  5,  5,  8,  8,  8,  3,  3,  3,  0,  6,  6,  9,  9,  9,
      4,  4,  4,  7,  7,  0,  10, 10, 10, 5,  5,  5,  8,  8,  8,  11, 11, 11,
      0,  6,  6,  9,  9,  9,  12, 12, 12, 7,  7,  0,  10, 10, 10, 0,  0,  0,
      8,  8,  8,  11, 11, 11, 11, 11, 11, 9,  9,  9,  0,  12, 12, 12, 12, 12,
      10, 10, 10, 13, 13, 0,  0,  0,  0,  11, 11, 11, 14, 14, 14, 14, 14, 14,
      0,  12, 12, 15, 15, 15, 15, 15, 15, 13, 13, 0,  16, 16, 16, 16, 16, 16,
      14, 14, 14, 17, 17, 17, 17, 17, 17, 15, 15, 15, 0,  18, 18, 18, 18, 18,
      16, 16, 16, 19, 19, 0,  0,  0,  0,  17, 17, 17, 20, 20, 20, 20, 20, 20,
      0,  18, 18, 21, 21, 21, 21, 21, 21, 19, 19, 0,  22, 22, 22, 22, 22, 22,
      20, 20, 20, 23, 23, 23, 23, 23, 23, 21, 21, 21, 0,  24, 24, 24, 24, 24,
      22, 22, 22, 25, 25, 0,  0,  0,  0,  23, 23, 23, 26, 26, 26, 26, 26, 26,
      0,  24, 24, 27, 27, 27, 27, 27, 27, 25, 25, 0,  28, 28, 28, 28, 28, 28,
      26, 26, 26, 29, 29, 29, 29, 29, 29, 27, 27, 27, 0,  30, 30, 30, 30, 30,
      28, 28, 28, 31, 31, 0,  0,  0,  0,  29, 29, 29, 32, 32, 32, 32, 32, 32,
      0,  30, 30, 33, 33, 33, 33, 33, 33, 0,  0,  0,  34, 34, 34, 34, 34, 34,
      0,  0,  0,  35, 35, 35, 35, 35, 35, 0,  0,  0,  0,  36, 36, 36, 36, 36,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0};

  int input_row = 6;
  int input_col = 6;
  int ker_row = 3;
  int ker_col = 3;

  int sa_rows = ker_row * ker_col;
  int sa_col = 3;

  Op::ConvParams cp{.imap{input_row, input_col},
                    .k{ker_row, ker_col},
                    .pad{1, 1, 1, 1},
                    .stride{1, 1}};
  SaDims sa_dims{.rows{sa_rows}, .cols{sa_col}};

  ConvTransformer<int, int> t(cp, sa_dims);

  std::vector<int> v(input_row * input_col);
  std::iota(v.begin(), v.end(), 1);

  auto transformed_input = t.transform(v);
  std::vector<int> computed = transformed_input.flatten();
  bool status = generate_report<int, int>(argv[0], expected, computed);
  return status;
}