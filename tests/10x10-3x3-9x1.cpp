#include <numeric>
#include "../src/sim.h"
#include "../src/utils.h"
#include "../src/transformers.h"

int main(int argc, char *argv[]) {
    std::vector<int> expected {
       726,  771,  816,  861,  906,  951,  996, 1041, 1176,
       1221, 1266, 1311, 1356, 1401, 1446, 1491, 1626, 1671,
       1716, 1761, 1806, 1851, 1896, 1941, 2076, 2121, 2166,
       2211, 2256, 2301, 2346, 2391, 2526, 2571, 2616, 2661,
       2706, 2751, 2796, 2841, 2976, 3021, 3066, 3111, 3156,
       3201, 3246, 3291, 3426, 3471, 3516, 3561, 3606, 3651,
       3696, 3741, 3876, 3921, 3966, 4011, 4056, 4101, 4146,
       4191
    };

    int input_rows = 10;
    int input_columns = 10;
    int kernel_rows = 3;
    int kernel_cols = 3;

    int array_rows = 9;
    int array_cols = 1;
    
    SA a1(array_rows, array_cols, true);
    ConvTransformer t(input_rows, input_columns, kernel_rows, kernel_cols, array_rows, array_cols);

    std::vector<int> v(input_rows*input_columns);
    std::iota(v.begin(), v.end(), 1);
    
    std::vector<int> w(array_rows*array_cols);
    std::iota(w.begin(), w.end(), 1);

    a1.load_weights(w);
    auto transformed_input = t.transform(v);

    Chain c1;
    c1.push(new Chainblock());

    a1.propagate(transformed_input, c1);
    auto output = a1.get_output();
    auto computed = t.untransform(output);
    bool status = generate_report<int,int>(argv[0], expected, computed);
    return status;
}
