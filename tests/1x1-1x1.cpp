#include <numeric>
#include "../sim.h"
#include "../utils.h"
#include "../transformers.h"

int main(int argc, char *argv[]) {
    std::vector<int> expected = {0};

    const int array_rows = 1;
    const int array_columns = 1;

    const int input_rows = 1;
    const int input_columns = 1;

    std::vector<int> w(array_rows * array_columns);
    std::vector<int> v(input_rows*input_columns);

    std::iota(w.begin(), w.end(), 0);
    std::iota(v.begin(), v.end(), 0);

    SA a1(array_rows,array_columns);
    a1.load_weights(w);

    Chain c1;
    c1.push(new Chainblock());
    GemmTransformer t(input_rows, input_columns, array_rows, array_columns);
    auto out = t.transform(v);
    a1.propagate(out, c1);
    auto t1 = a1.get_output();
    auto computed = t.untransform(t1);
    bool status = generate_report<int>(argv[0], expected, computed);
    return status;
}
