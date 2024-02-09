#include "../src/sim.h"
#include "../src/transformers.h"
#include "../src/utils.h"
#include <numeric>

Argparse gbl_args;

int main(int argc, char *argv[]) {
  gbl_args.parse(argc, argv);
  std::vector<int> expected{
      1,  0,  0,  0,  0,  0,  0,  0,  0,  2,  2,  0,  0,  0,  0,  0,  0,  0,
      3,  3,  3,  0,  0,  0,  0,  0,  0,  4,  4,  4,  7,  0,  0,  0,  0,  0,
      7,  5,  5,  8,  8,  0,  0,  0,  0,  8,  8,  6,  9,  9,  9,  0,  0,  0,
      9,  9,  9,  10, 10, 10, 13, 0,  0,  10, 10, 10, 13, 11, 11, 14, 14, 0,
      13, 11, 11, 14, 14, 12, 15, 15, 15, 14, 14, 12, 15, 15, 15, 16, 16, 16,
      15, 15, 15, 16, 16, 16, 19, 17, 17, 16, 16, 16, 19, 17, 17, 20, 20, 18,
      19, 17, 17, 20, 20, 18, 21, 21, 21, 20, 20, 18, 21, 21, 21, 22, 22, 22,
      21, 21, 21, 22, 22, 22, 25, 23, 23, 22, 22, 22, 25, 23, 23, 26, 26, 24,
      0,  23, 23, 26, 26, 24, 27, 27, 27, 0,  0,  24, 27, 27, 27, 28, 28, 28,
      0,  0,  0,  28, 28, 28, 31, 29, 29, 0,  0,  0,  0,  29, 29, 32, 32, 30,
      0,  0,  0,  0,  0,  30, 33, 33, 33, 0,  0,  0,  0,  0,  0,  34, 34, 34,
      0,  0,  0,  0,  0,  0,  0,  35, 35, 0,  0,  0,  0,  0,  0,  0,  0,  36};

  int input_rows = 6;
  int input_columns = 6;
  /* 0 for srows/scols works here as srows/scols are only required by
   * the untransform function and its not used here  */
  Op::ConvParams cp{
      .imap{input_rows, input_columns},
      .k{3, 3},
  };

  SaDims sa_dims{.rows{0}, .cols{0}};
  ConvTransformer<int, int> t(cp, sa_dims);
  std::vector<int> v(input_rows * input_columns);
  std::iota(v.begin(), v.end(), 1);
  auto tmp = t.transform(v);
  auto computed = tmp.flatten();
  bool status = generate_report<int, int>(argv[0], expected, computed);
  return status;
}
