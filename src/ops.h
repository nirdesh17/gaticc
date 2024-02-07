#ifndef OPS_H
#define OPS_H

#include <vector>
#include "sim.h"

struct Imgdims {
    int c;
    int h;
    int w;
};
using Imgdims = struct Imgdims;
using Kerneldims = struct Imgdims;

std::vector<int> extract_channel(std::vector<int>& img, Imgdims &dims, int n);

#if 0
class Conv2d {
    Imgdims id;
    Kerneldims kd;
    int m_stride;
    int m_padding;
    public:
        Conv2d(Imgdims id, Kerneldims kd);
        std::vector<int> operator()(std::vector<int> &input, std::vector<int> &weights, 
                SA &a1);
};
#endif

#endif
