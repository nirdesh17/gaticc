#include "../src/sim.h"
#include "../src/transformers.h"
#include "../src/utils.h"
#include <numeric>

int main(int argc, char *argv[]) {
  std::vector<int> expected = {15,  18, 21,  42,  54,  66,  69, 90,
                               111, 96, 126, 156, 123, 162, 201};

  const int array_rows = 3;
  const int array_columns = 3;

  const int input_rows = 5;
  const int input_columns = 3;

  std::vector<int> w(array_rows * array_columns);
  std::vector<int> v(input_rows * input_columns);

  std::iota(w.begin(), w.end(), 0);
  std::iota(v.begin(), v.end(), 0);

  SA<int, int> a1(array_rows, array_columns);
  a1.load_weights(w);

  Chain c1;
  c1.push(new Chainblock());
  GemmTransformer<int, int> t(input_rows, input_columns, array_rows, array_columns);
  auto out = t.transform(v);
  a1.propagate(out, c1);
  auto t1 = a1.get_output();
  auto computed = t.untransform(t1);
  bool status = generate_report<int, int>(argv[0], expected, computed);
  return status;
}
