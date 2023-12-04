#include <iostream>
#include <utility>
#include <cassert>
#include <numeric>
#include "ops.h"
#include "transformers.h"

/* extract nth channel from img */
std::vector<int> extract_channel(std::vector<int>& img, Imgdims &dims, int n) {
    int c = dims.c;
    int h = dims.h;
    int w = dims.w;
    assert(img.size() == (c*h*w));
    auto start_itr = img.begin() + (h*w*n);
    auto end_itr = start_itr + (h*w);
    assert(end_itr <= img.end());
    return std::vector<int>(start_itr, end_itr);
}

Conv2d::Conv2d(Imgdims id, Kerneldims kd):
            id{id}, kd{kd}, m_stride{1}, m_padding{0} {
}

std::vector<int> Conv2d::operator()(std::vector<int> &input, std::vector<int> &weights, 
        SA &a1) {
    assert(weights.size() == (a1.get_rows() * a1.get_cols()));
    a1.load_weights(weights);
    ConvTransformer tr(id.w, id.h, kd.w, kd.h, a1.get_rows(), a1.get_cols());

    Chain chain;
    chain.push(new Relu());

    /* TODO: include stride/padding below */
    int hout = (id.w - kd.w) + 1;
    int wout = (id.h - kd.h) + 1;
    std::vector<int> ret;
    for (int i = 0; i < id.c; ++i) {
        auto channel = extract_channel(input, id, i);
        auto transformed = tr.transform(channel);
        a1.propagate(transformed, chain);
        auto output_array = a1.get_output();
        std::vector<int> untr = tr.untransform(output_array);
        ret.insert(ret.end(), untr.begin(), untr.end());
        std::cout << untr.size() << '\n';
    }
    return ret;
}
