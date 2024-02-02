#include "../src/sim.h"
#include "../src/transformers.h"
#include "../src/utils.h"
#include "../src/onnx_parser.h"
#include <numeric>

int main(int argc, char *argv[]) {
  std::vector<int> expected{
      474,  519,  564,  609,  744,  789,  834,  879,  1014, 1059, 1104, 1149,
      1284, 1329, 1374, 1419, 1122, 1248, 1374, 1500, 1878, 2004, 2130, 2256,
      2634, 2760, 2886, 3012, 3390, 3516, 3642, 3768, 1770, 1977, 2184, 2391,
      3012, 3219, 3426, 3633, 4254, 4461, 4668, 4875, 5496, 5703, 5910, 6117};

  int input_rows = 6;
  int input_columns = 6;
  int kernel_rows = 3;
  int kernel_cols = 3;

  int array_rows = 9;
  int array_cols = 3;

  Op::ConvParams cp {
    .imap {input_rows, input_columns},
    .k {kernel_rows, kernel_cols},
    .pad {0,0,0,0},
    .stride {1,1}
  };

  SaDims sa_dims {
    .rows {array_rows},
    .cols {array_cols}
  };

  SA a1(array_rows, array_cols, true);
  ConvTransformer t(cp, sa_dims);

  std::vector<int> v(input_rows * input_columns);
  std::iota(v.begin(), v.end(), 1);

  std::vector<int> w(array_rows * array_cols);
  std::iota(w.begin(), w.end(), 1);

  auto wt = t.transform_weights(w, array_rows, array_cols);

  a1.load_weights(wt);
  auto transformed_input = t.transform(v);

  Chain c1;
  c1.push(new Chainblock());
  a1.propagate(transformed_input, c1);
  auto output = a1.get_output();
  auto computed = t.untransform(output);
  bool status = generate_report<int, int>(argv[0], expected, computed);
  return status;
}
