#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <numeric>
#include <tuple>
#include <queue>
#include <cstdlib>

class foo {
    private:
        int *ptr;
    public:
        foo(int sz) {
            ptr = (int *) malloc(sz * sizeof(int));
        }
        ~foo() {
        }
};

int main() {
    std::vector<int> v {1,2,3,4,5};
    v.~vector();
}
