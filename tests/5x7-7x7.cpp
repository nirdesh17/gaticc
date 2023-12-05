#include <numeric>
#include "../src/sim.h"
#include "../src/utils.h"
#include "../src/transformers.h"

int main(int argc, char *argv[]) {
    std::vector<int> expected = { 637,  658,  679,  700,  721,  742,  763, 1666, 1736, 1806, 1876, 1946, 2016, 2086, 2695, 2814, 2933, 3052, 3171, 3290, 3409, 3724, 3892, 4060, 4228, 4396, 4564, 4732, 4753, 4970, 5187, 5404, 5621, 5838, 6055};

    const int array_rows = 7;
    const int array_columns = 7;

    const int input_rows = 5;
    const int input_columns = 7;

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
