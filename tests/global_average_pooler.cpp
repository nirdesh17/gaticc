#include "../src/sim.h"
#include "../src/transformers.h"
#include "../src/utils.h"
#include <numeric>
#include <stdio.h>

int main(int argc, char *argv[]) {

  //std::vector<int> expected{487.57025146};
  std::vector<int> expected{487};

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
                   .k{2, 2},
                   .pad{2, 2, 2, 2},
                   .stride{1, 1},
                   .dilation{1, 1}};

  Pooler<int> p1;
  auto temp_mat = v2mat<int>(computed, 7, 7);
  auto pooler_output = p1.global_average_pooler(temp_mat, mp);
  auto output = mat2v<int>(pooler_output, 7, 7);
  bool status = generate_report<int, int>(argv[0], expected, output);
  return status;
}
