#include <vector>
#include <numeric>
#include <utility>
#include <algorithm>

#include "sim.h"
#include "transformers.h"
#include "utils.h"
#include "onnx_parser.h"

#if 0
/* remove n elements from the start of vector and return them */
std::vector<int> GemmTransformer::pick(std::vector<int>& v, int n) {
    assert(n <= v.size());
    assert(v.empty() != true);
    std::vector<int> removed(v.begin(), v.begin()+n);
    v.erase(std::remove_if(v.begin(), v.begin()+n, [](int i) { return true; }), v.begin()+n);
    return removed;
}

/* pad n zeros after v */
void GemmTransformer::zero_pad_after(std::vector<int> &v, int n) {
    for (int i = 0; i < n; ++i) {
        v.push_back(0);
    }
}

/* pad n zeros before v */
void GemmTransformer::zero_pad_before(std::vector<int> &v, int n) {
    for (int i = 0; i < n; ++i) {
        v.insert(v.begin(), 0);
    }
}

/* returns an array of the form 
 *  1 2 3 3 3 2 1
 * this is the order in which the systolic array consumes elements
 * first cycle, 1 element is consumed
 * second cycle, 2 elements are consumed
 * third, fourth, fifth cycles, 3 (== columns) (max) elements are consumed
 * in the latter cycles, the order is reversed, i.e. 2,1.
 */
std::vector<int> GemmTransformer::get_access_frequency(int rows, int cols) {
    std::vector<int> frequency;
    int total = 0, i = 1, i_sum = 0;
    while (i < cols) {
        frequency.push_back(i);
        i_sum += i;
        total += i;
        ++i;
    }
    while (((rows * cols) - total) > i_sum) {
        frequency.push_back(i);
        total += cols;
    }
    for (int j = (cols-1); j > 0; --j) {
        frequency.push_back(j);
    }
    return frequency;
}

/* transpose input matrix into systolic order 
 *  if v = {1,2,3,4,5,6,7,8,9}
 *  which can also be viewed as a 2D array like so
 *      1 2 3
 *      4 5 6
 *      7 8 9
 *  transpose converts v into 1 4 2 7 5 3 8 6 9
 */
std::vector<int> GemmTransformer::to_sys_major(std::vector<int> &v, int rows, int columns) {
    Mat m = v2mat<int,int>(v, rows, columns);
    Tree t = Tree(m, rows, columns);
    std::vector<int> s = t.breadth_first_order();
    return s;
}

/* pad zeros for prologue and epilogue */
void GemmTransformer::zero_pad(Mat& out, std::vector<int>& frequency, int columns) {
    auto itr = out.begin();
    int count = std::count(frequency.begin(), frequency.end(), columns);
    for (int i = 1; i < columns; ++i) {
        zero_pad_after(*itr, columns-i);
        itr++;
    }
    for (int i = 0; i < count; ++i) {
        itr++;
    }
    for (int i = columns-1; i > 0; --i) {
        zero_pad_before(*itr, columns-i);
        itr++;
    }
}
#endif


#if 0
Mat GemmTransformer::transform(std::vector<int> &a) {   // retun val should be changed to float?
    if (arows < acolumns) {
        /* TODO: implement this in to_systolic_order() */
        printf("[ERR]: Input rows less than input columns prohibited. This is an "
                "un-implemented feature\n");
        std::exit(1);
    }
    std::vector<int> sys_major = to_sys_major(a, arows, acolumns);
    std::vector<int> frequency = get_access_frequency(arows, acolumns);
    int frequency_sum = std::accumulate(frequency.begin(), frequency.end(), 0);
    assert(frequency_sum == (arows * acolumns));

    Mat out;
    std::for_each(frequency.begin(), frequency.end(), 
            [&out, &sys_major, this](int n) { out.push_back(pick(sys_major, n)); }
            );

    zero_pad(out, frequency, acolumns);
    /* add zero vectors */
    for (int i = 0; i < (acolumns-2); ++i) {
        std::vector<int> tmp(acolumns);
        out.push_back(tmp);
    }
    return out;
}

GemmTransformer::GemmTransformer(int arows, int acolumns, int brows, int bcolumns): 
    arows{arows}, acolumns{acolumns}, brows{brows}, bcolumns{bcolumns} {
}

/* shift nth column by stride times in upward direction */
void shift_columns_up(Mat& m, int n, int stride) {
    for (int i = 0; i < m.size()-stride; ++i) {
        m[i][n] = m[i+stride][n];
    }
}

std::vector<int> GemmTransformer::untransform(Mat &a) {
    std::vector<int> out;
    int out_rows = bcolumns;
    int out_cols = arows;
    for (int i = 0; i < out_cols; i++) {
        for (int j = 0; j < out_rows; ++j) {
            out.push_back(a.at(j).at(i));
        }
    }
    return out;
}


/* For multi-channel kernels , in order to load them in a 2-D SA
* we need to pack the channels into one single vector. Concatenate channels
* in a single vector and call 'transform_weights(vec v,SA_row,SA_column)' on it.
* It reorders them in a specific order so that when the 'load_weights()' is called.
* it will load the 1st kernel in the 1st column of the SA and the 2nd kernel in the 2nd column of the SA.
*/

/* 
* transform_weights() : takes elements one by one from the input vector in increasing order
* and places them in the return vector in order i+ SA_row, so when load_weights is called on return vector
* USED WHEN MULTI_CHANNEL KERNEL AND 2D SA is used.
*
*    kernel_1 2x2:             1   2
*                              3   4
*
*
*    kernel_2 2x2:             5   6
*                              7   8      
*
*
*
*    input_vector: 1,2,3,4,5,6,7,8  (concatenate kernel_1 & kernel_2)
*
*    transform weights(4,2) return vector : 1,5,2,6,3,7,4,8
*
*    load_weigths(return vector, 4,2) :      1   5
*                                            2   6
*                                            3   7
*                                            4   8   (4x2)
*/
std::vector<int> ConvTransformer::transform_weights(std::vector<int>& w, int out_row, int out_col) {
    assert(w.size() == out_row*out_col);
    std::vector<int> out(out_row*out_col, 0);
    for (int i = 0; i < out_col; ++i) {
        for (int j = 0; j < out_row; ++j) {
            out.at(i+j*out_col) = w.at(i*out_row+j);
        }
    }
    return out;
}

std::vector<int> ConvTransformer::untransform(Mat &a) {
    std::vector<int> out;
    int hout = sa_odims_row(m_cp);
    int wout = sa_odims_cols(m_cp);
    for (int i = 0; i < sa_dims.cols; i++) {
        for (int j = 0; j < hout*wout; ++j) {
            out.push_back(Mat_at(a, i, j));
        }
    }
    return out;
}

TransformerType GemmTransformer::get_type() {
    return GEMM_TF;
}
#endif

#if 0

TransformerType ConvTransformer::get_type() {
    return CONV_TF;
}

/* true if index is last slide first element */
bool ConvTransformer::is_lsfe(Point const &index) {
    int y = index.second; 
    return (y == (m_cp.imap[0] - m_cp.k[0])) ? true : false;
}
/* true if index is last slide middle element */
bool ConvTransformer::is_lsme(Point const &index) {
    int y = index.second; 
    return ((y > (m_cp.imap[1] - m_cp.k[1])) && (y < m_cp.imap[0] - 1)) ? true : false;
}
/* true if index is last slide last element */
bool ConvTransformer::is_lsle(Point const &index) {
    int y = index.second; 
    return (y == m_cp.imap[0]-1);
}

/* true if index is at the last position in the current slide */
bool ConvTransformer::is_kern_edge(Point const &index) {
    int y = index.second;
    return (y == m_cp.k[0]-1);
}

/* x = x' ; y = y' */
void ConvTransformer::xxyy(Point &current, Point const &left) {
    current.first = left.first;
    current.second = left.second;
}

/* x = x' + 1 ; y = 0 */
void ConvTransformer::xp1y0(Point &current, Point const &above) {
    current.first = above.first + 1;
    current.second = 0;
}

/* x = x' ; y = y' + 1 */
void ConvTransformer::xyp1(Point &current, Point const &above) {
    current.first = above.first;
    current.second = above.second + 1;
}

/* x = x' + 1 ; y = y' - 1 */
void ConvTransformer::xp1ym1(Point &current, Point const &above) {
    current.first = above.first + 1;
    current.second = -1;
}

bool ConvTransformer::has_occured(Point const &p, std::vector<bool> const &occurence) {
    int y = p.second;
    int lsfe = (m_cp.imap[0] - m_cp.k[0]);
    return occurence.at(y % lsfe);
}

bool ConvTransformer::is_zero(Point const &p) {
    return (p.first == 0 && p.second == 0) ? true : false;
}

/* true if p is the first element of kernel at last sliding position */
bool ConvTransformer::is_last_kernel(Point const &p) {
    return ((p.first == (m_cp.imap[0] - m_cp.k[0])) && (p.second == (m_cp.imap[1] - m_cp.k[1])));
}

void ConvTransformer::mark_occured(Point const &p, std::vector<bool> &occurence) {
    int y = p.second;
    int lsfe = (m_cp.imap[0]-m_cp.k[0]);
    occurence.at(y % lsfe) = 1;
}

/* fill 'out' matrix with values from 'input' at co-ordinates present in ibuf starting
 * from offset till n
 */
void ConvTransformer::fill_index(Mat &out, Mat const &input, std::vector<Point> const &ibuf, int n, int offset) {
    assert(ibuf.size() == (m_cp.k[0]*m_cp.k[1]));
    assert(n <= (m_cp.k[0]*m_cp.k[1]));
    std::vector<int> buf(m_cp.k[0]*m_cp.k[1], 0);
    for (int i = offset; i < n; ++i) {
        auto p = ibuf.at(i);
        buf.at(i) = Mat_at(input, p.first, p.second);
    }
    assert(buf.size() <= (m_cp.k[0]*m_cp.k[1]));
    out.push_back(buf);
}

/* generate n indices based on previous indices (stored in ibuf2) and store them in ibuf */
void ConvTransformer::generate_index(std::vector<Point> const &ibuf2, std::vector<Point> &ibuf, int n) {
    assert(ibuf.size() == (m_cp.k[0]*m_cp.k[1]));
    assert(ibuf2.size() == (m_cp.k[0]*m_cp.k[1]));
    assert(n <= (m_cp.k[0]*m_cp.k[1]));
    std::vector<bool> occurence(m_cp.k[0], 0);
    for (int i = 0; i < n; ++i) {
        if (is_lsfe(ibuf2.at(i)) && !has_occured(ibuf2.at(i), occurence)) {
            xp1y0(ibuf.at(i), ibuf2.at(i));
            mark_occured(ibuf2.at(i), occurence);
        }
        else if (is_lsme(ibuf2.at(i)) && !has_occured(ibuf2.at(i), occurence)) {
            (i != 0) 
                ? xxyy(ibuf.at(i), ibuf.at(i-1)) 
                : xxyy(ibuf.at(i), ibuf2.at(i));
            mark_occured(ibuf2.at(i), occurence);
        }
        else if (is_lsle(ibuf2.at(i))) {
            (i != 0) 
                ? xxyy(ibuf.at(i), ibuf.at(i-1)) 
                : xxyy(ibuf.at(i), ibuf2.at(i));
        }
        else {
            xyp1(ibuf.at(i), ibuf2.at(i));
        }
    }
}

#endif


#if 0
/* This transform required to carry out convolution on the systolic array
 * is called 'im2col' [1] in the literature. The output of this function is
 * equal to that of ordinary im2col, but the method of achieving is completely
 * different. 
 *
 * In a real heterogeneous environment, there are two ways to doing im2col:
 * static and dynamic. static would transform input into an expanded matrix and 
 * store it somewhere and later feed it to the systolic array. dynamic involves
 * playing the algorithm in real time so inputs are expanded as they arrive and
 * fed directly to the SA. This reduces the latency and size expansion incurred by
 * static SA.
 *
 * This algorithm is the proof-of-concept implementation of the aforementioned
 * dynamic im2col. 
 *
 * For a convolution involving KWxKH kernel, we require a SA of atleast KWxKH rows.
 * The algorithm generates KWxKH elements at a time and feeds it to the SA. 
 *
 * Consider a convolution of 4x4 input with a 2x2 kernel. We require 4 inputs to be
 * generated at a timestep. For the first timestep, the inputs required are values 
 * at at co-ordinates
 *      (0,0)       0       0       0
 * the zeros are padded as the SA only consumes 1 element in the fist timestep. This
 * is followed by the arrays made of:
 *      (0,1)     (0,1)     0       0
 *      (0,2)     (0,2)   (1,0)     0
 * and so on. The numbers inside the brackets are co-ordinates indexing a matrix and are
 * replaced by their values. 
 *
 * Definitions:
 *  lsfe: last slide first element. 
 *        the first element of the last sliding position of a kernel.
 *        for a 2x2 kernel on 4x4 input, all the co-ordinates with co-ordinates of the
 *        second last column are lsfe.
 *  lsme: last slide middle element
 *        all the elements b/w first element and last of the last sliding position of a
 *        kernel 
 *        for 4x4 kernel on 6x6 input, co-ordinates with y values = 4,5
 *  lsle: last slide last element
 *        all elements of the last column
 *
 * The Algorithm:
 *  int previous[4];
 *  int current[4];
 *  while (1) {
 *      for (i = 0 to 4) {
 *          if (is_lsfe(previous[i]) && first_lsfe)
 *              current[i] = (previous[i].x + 1, 1) 
 *          else if (is_lsme(previous[i]) && first_lsme)
 *              current[i] = previous[i]
 *          else if (is_lsle(previous[i]))
 *              current[i] = previous[i]
 *          else
 *              current[i] = (previous[i].x, previous[i].y + 1)
 *      }
 *  }
 *
 * Explanation:
 *  1. Start with two buffers 'previous' and 'current' of co-ordinates (x,y)
 *  2. iterate over current buffer.
 *  3. during each iteration, compare current buffer's co-ordinates to previous buffer's  
 *  at the same index
 *  4. if its lsfe, increment the x value of previous buffer and set y to 1 and 
 *  only do this once for a buffer.
 *  5. if its lsme, copy the value to the left of the current buffer and only do 
 *  this once for a buffer.
 *  6. if its lsle, copy the value to the left of the current buffer
 *  7. after iteration, replace co-ordinates in current buf to their corresponding values
 *  8. copy current buf's contents of previous buf.
 *
 * Here's a complete set of vectors as generated by this algorithm for 2x2 kernel on a 4x4
 * input:
 *       0,0 0,0 0,0 0,0
 *       0,1 0,1 0,0 0,0
 *       0,2 0,2 1,0 0,0
 *       0,3 0,3 1,1 1,1
 *       0,4 0,4 1,2 1,2
 *       1,0 0,5 1,3 1,3
 *       1,1 1,1 1,4 1,4
 *       1,2 1,2 2,0 1,5
 *       1,3 1,3 2,1 2,1
 *       1,4 1,4 2,2 2,2
 *       2,0 1,5 2,3 2,3
 *       2,1 2,1 2,4 2,4
 *       2,2 2,2 3,0 2,5
 *       0,0 2,3 3,1 3,1
 *       0,0 0,0 3,2 3,2
 *       0,0 0,0 0,0 3,3
 */

Mat ConvTransformer::transform(std::vector<int> &a) {
    Mat input = v2mat<int,int>(a, m_cp.imap[0], m_cp.imap[1]);
    Mat out;
    std::vector<Point> ibuf(m_cp.k[0]*m_cp.k[1], std::make_pair(0,0));
    std::vector<Point> ibuf2(m_cp.k[0]*m_cp.k[1], std::make_pair(0,0));
    ibuf2.at(0).second = -1;
    /* prologue */
    for (int i = 1; i < (m_cp.k[0]*m_cp.k[1]); ++i) {
        generate_index(ibuf2, ibuf, i);
        //print_vec_point("vec", ibuf);
        std::copy(ibuf.begin(), ibuf.end(), ibuf2.begin());
        (is_kern_edge(ibuf.at(i-1)))
            ? xp1ym1(ibuf2.at(i), ibuf.at(i-1))
            : xxyy(ibuf2.at(i), ibuf.at(i-1));
        fill_index(out, input, ibuf, i, 0);
    }
    /* core */
    while (!is_last_kernel(ibuf.at(0))) {
        generate_index(ibuf2, ibuf, m_cp.k[0]*m_cp.k[1]);
        //print_vec_point("vec", ibuf);
        std::copy(ibuf.begin(), ibuf.end(), ibuf2.begin());
        fill_index(out, input, ibuf, m_cp.k[0]*m_cp.k[1], 0);
    }
    /* epilogue */
    for (int i = 1; i < (m_cp.k[0]*m_cp.k[1]); ++i) {
        generate_index(ibuf2, ibuf, m_cp.k[0]*m_cp.k[1]);
        //print_vec_point("vec", ibuf);
        std::copy(ibuf.begin(), ibuf.end(), ibuf2.begin());
        fill_index(out, input, ibuf, m_cp.k[0]*m_cp.k[1], i);
    }
    for (int i = 0; i < (sa_dims.cols-2); ++i) {
        std::vector<int> tmp(m_cp.k[0]*m_cp.k[1], 0);
        out.push_back(tmp);
    }
    return out;
}

ConvTransformer::ConvTransformer(Op::ConvParams const &cp, SaDims const &sa_dims) {
  std::memcpy(&m_cp, &cp, sizeof(Op::ConvParams));
  std::memcpy(&(this->sa_dims), &sa_dims, sizeof(SaDims));
}

#endif

void print_vec_point(const char *s, std::vector<Point>const &v) {
    printf("%s: ", s);
    for (auto &p: v) {
        std::cout << p.first << ',' << p.second << ' ';
    }
    std::cout << '\n';
}
