#include <numeric>
#include "transformers.h"
#include "sim.h"
#include "utils.h"
#include "ffi.h"
#include "ops.h"

int main() {
    PyEngine engine("preprocess");
    Imgdims id {3, 224, 224};
    Kerneldims kd {1, 3, 3};
    int sa_rows = 9; int sa_cols = 1;

    std::vector<int> img = py_read_img(engine, std::string("images/mug.jpg"));
    auto weights = py_fetch_kernel(engine, 0, 0, 1);
    SA a1(sa_rows, sa_cols);
    Conv2d cc(id, kd);
    auto out = cc(img, weights, a1);
    std::cout << "out size: " << out.size() << '\n';
}
