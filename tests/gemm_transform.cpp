#include <numeric>
#include "../src/sim.h"
#include "../src/utils.h"
#include "../src/transformers.h"


int main(int argc, char *argv[]) {
    std::vector<int> expected {1, 0, 0, 0, 0, 0 ,
                                7, 2, 0, 0, 0, 0,
                                13, 8, 3, 0, 0, 0,
                                19, 14, 9, 4, 0, 0,
                                25, 20, 15, 10, 5, 0,
                                31, 26, 21, 16, 11, 6,
                                0, 32, 27, 22, 17, 12,
                                0, 0, 33, 28, 23, 18,
                                0, 0, 0, 34, 29, 24,
                                0, 0, 0, 0, 35, 30,
                                0, 0, 0, 0, 0, 36,
                                0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0
    };


    int input_rows = 6;
    int input_columns = 6;
    GemmTransformer t(input_rows, input_columns, 0, 0);
    std::vector<int> v(input_rows*input_columns);
    std::iota(v.begin(), v.end(), 1);
    auto tmp = t.transform(v);
    auto computed = flatten<int>(tmp);
    bool status = generate_report<int>(argv[0], expected, computed);
    return status;
}
