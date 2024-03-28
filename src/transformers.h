#pragma once

#include "onnx_parser.h"
#include "utils.h"
#include <algorithm>
#include <vector>

/* The transformer interface allows new transformations to be added in a
 * convenient manner. The interface is thus:
 *  The 'Transformer' base class should be inherited by any valid transformer.
 *  It defines virtual methods that should be implemented by the inheriting
 *  class.
 * See for instance, the GemmTransformer class which enables general matrix
 * multiplication on the systolic array.
 *
 * Transformers can be viewed as enablers of computations. The systolic array
 * can be considered to be un-intelligent dataflow devices that simply pass
 * data b/w PE in a regular manner. In such a system, Transformers are the
 * device that do the intelligent work of re-shaping data so that feeding it to
 * the systolic array results in an algorithm's execution.
 *
 */

using TransformerType = enum TransformerType {
  GEMM_TF,
  CONV_TF,
};

/* Abstract base class for all transformers */
template <typename inputT, typename outputT> class Transformer {
public:
  virtual TransformerType get_type() = 0;
  virtual Mat<inputT> transform(std::vector<inputT> &a) = 0;
  virtual std::vector<outputT> untransform(Mat<outputT> &a) = 0;
};

template <typename inputT, typename outputT> class GemmTransformer : public Transformer<inputT, outputT> {
private:
  int arows;
  int acolumns;
  int brows;
  int bcolumns;
  std::vector<inputT> to_sys_major(std::vector<inputT> &v, int rows, int columns);
  std::vector<int> get_access_frequency(int rows, int cols);
  void zero_pad_before(std::vector<inputT> &v, int n);
  void zero_pad_after(std::vector<inputT> &v, int n);
  void zero_pad(Mat<inputT> &out, std::vector<int> &frequency, int columns);
  std::vector<inputT> pick(std::vector<inputT> &v, int n);

public:
  GemmTransformer() = delete;
  GemmTransformer(int r, int c, int ir, int ic);
  TransformerType get_type() override;
  Mat<inputT> transform(std::vector<inputT> &a) override;
  std::vector<outputT> untransform(Mat<outputT> &a) override;
};

template <typename inputT, typename outputT> class ConvTransformer : public Transformer<inputT, outputT> {
private:
  Op::ConvParams m_cp;
  SaDims sa_dims;
  void fill_index(Mat<inputT> &out, Mat<inputT> const &input,
                  std::vector<Point> const &ibuf, int n, int offset);
  void generate_index(std::vector<Point> const &ibuf2, std::vector<Point> &ibuf,
                      int n);
  bool is_lsle(Point const &index);
  bool is_lsme(Point const &index);
  bool is_lsfe(Point const &index);
  bool is_zero(Point const &p);
  void xp1y0(Point &current, Point const &above);
  void xyp1(Point &current, Point const &above);
  void xp1ym1(Point &current, Point const &above);
  void xxyy(Point &current, Point const &left);
  bool is_last_kernel(Point const &p);
  void mark_occured(Point const &p, std::vector<bool> &occurence);
  bool has_occured(Point const &p, std::vector<bool> const &occurence);
  bool is_kern_edge(Point const &i);

public:
  ConvTransformer() = delete;
  ConvTransformer(Op::ConvParams const &cp, SaDims const &sa_dims);
  TransformerType get_type() override;
  Mat<inputT> transform(std::vector<inputT> &a) override;
  std::vector<inputT> transform_weights(std::vector<inputT> &w, int out_rows,
                                   int out_cols);
  std::vector<outputT> untransform(Mat<outputT> &a) override;
};

/* remove n elements from the start of vector and return them */
template <typename inputT, typename outputT>
std::vector<inputT> GemmTransformer<inputT, outputT>::pick(std::vector<inputT> &v, int n) {
  assert(n <= v.size());
  assert(v.empty() != true);
  std::vector<inputT> removed(v.begin(), v.begin() + n);
  v.erase(std::remove_if(v.begin(), v.begin() + n, [](inputT i) { return true; }),
          v.begin() + n);
  return removed;
}

/* pad n zeros after v */
template <typename inputT, typename outputT>
void GemmTransformer<inputT, outputT>::zero_pad_after(std::vector<inputT> &v, int n) {
  for (int i = 0; i < n; ++i) {
    v.push_back(0);
  }
}

/* pad n zeros before v */
template <typename inputT, typename outputT>
void GemmTransformer<inputT, outputT>::zero_pad_before(std::vector<inputT> &v, int n) {
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
template <typename inputT, typename outputT>
std::vector<int> GemmTransformer<inputT, outputT>::get_access_frequency(int rows, int cols) {
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
  for (int j = (cols - 1); j > 0; --j) {
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
template <typename inputT, typename outputT>
std::vector<inputT> GemmTransformer<inputT, outputT>::to_sys_major(std::vector<inputT> &v, int rows,
                                                int columns) {
  Mat<inputT> m = v2mat<inputT>(v, rows, columns);
  // TODO: templated tree
  Tree t = Tree(m, rows, columns);
  std::vector<inputT> s = t.breadth_first_order();
  return s;
}

/* pad zeros for prologue and epilogue */
template <typename inputT, typename outputT>
void GemmTransformer<inputT, outputT>::zero_pad(Mat<inputT> &out, std::vector<int> &frequency,
                                  int columns) {
  auto itr = out.begin();
  int count = std::count(frequency.begin(), frequency.end(), columns);
  for (int i = 1; i < columns; ++i) {
    zero_pad_after(*itr, columns - i);
    itr++;
  }
  for (int i = 0; i < count; ++i) {
    itr++;
  }
  for (int i = columns - 1; i > 0; --i) {
    zero_pad_before(*itr, columns - i);
    itr++;
  }
}

template <typename inputT, typename outputT>
Mat<inputT> GemmTransformer<inputT, outputT>::transform(
    std::vector<inputT> &a) { // retun val should be changed to float?
  if (arows < acolumns) {
    /* TODO: implement this in to_systolic_order() */
    printf("[ERR]: Input rows less than input columns prohibited. This is an "
           "un-implemented feature\n");
    std::exit(1);
  }
  std::vector<inputT> sys_major = to_sys_major(a, arows, acolumns);
  std::vector<int> frequency = get_access_frequency(arows, acolumns);
  int frequency_sum = std::accumulate(frequency.begin(), frequency.end(), 0);
  assert(frequency_sum == (arows * acolumns));

  Mat<inputT> out;
  std::for_each(
      frequency.begin(), frequency.end(),
      [&out, &sys_major, this](int n) { out.push_back(pick(sys_major, n)); });

  zero_pad(out, frequency, acolumns);
  /* add zero vectors */
  for (int i = 0; i < (acolumns - 2); ++i) {
    std::vector<inputT> tmp(acolumns);
    out.push_back(tmp);
  }
  return out;
}

template <typename inputT, typename outputT>
GemmTransformer<inputT, outputT>::GemmTransformer(int arows, int acolumns, int brows,
                                    int bcolumns)
    : arows{arows}, acolumns{acolumns}, brows{brows}, bcolumns{bcolumns} {}

#if 0
/* shift nth column by stride times in upward direction */
void shift_columns_up(Mat& m, int n, int stride) {
    for (int i = 0; i < m.size()-stride; ++i) {
        m[i][n] = m[i+stride][n];
    }
}
#endif

template <typename inputT, typename outputT>
std::vector<outputT> GemmTransformer<inputT, outputT>::untransform(Mat<outputT> &a) {
  std::vector<outputT> out;
  int out_rows = bcolumns;
  int out_cols = arows;
  for (int i = 0; i < out_cols; i++) {
    for (int j = 0; j < out_rows; ++j) {
      out.push_back(a.at(j, i));
    }
  }
  return out;
}

template <typename inputT, typename outputT> TransformerType GemmTransformer<inputT, outputT>::get_type() {
  return GEMM_TF;
}

template <typename inputT, typename outputT> TransformerType ConvTransformer<inputT, outputT>::get_type() {
  return CONV_TF;
}

/* true if index is last slide first element */
template <typename inputT, typename outputT> bool ConvTransformer<inputT, outputT>::is_lsfe(Point const &index) {
  int y = index.second;
  return (y == (m_cp.imap[WIDTH]+m_cp.pad[0]+m_cp.pad[2] - m_cp.k[WIDTH])) ? true : false;
}
/* true if index is last slide middle element */
template <typename inputT, typename outputT> bool ConvTransformer<inputT, outputT>::is_lsme(Point const &index) {
  int y = index.second;
  return ((y > (m_cp.imap[WIDTH]+m_cp.pad[0]+m_cp.pad[2] - m_cp.k[WIDTH])) && (y < m_cp.imap[WIDTH]+m_cp.pad[0]+m_cp.pad[2] - 1)) ? true
                                                                      : false;
}
/* true if index is last slide last element */
template <typename inputT, typename outputT> bool ConvTransformer<inputT, outputT>::is_lsle(Point const &index) {
  int y = index.second;
  return (y == m_cp.imap[WIDTH]+m_cp.pad[0]+m_cp.pad[2] - 1);
}

/* true if index is at the last position in the current slide */
template <typename inputT, typename outputT>
bool ConvTransformer<inputT, outputT>::is_kern_edge(Point const &index) {
  int y = index.second;
  return (y == m_cp.k[WIDTH] - 1);
}

/* x = x' ; y = y' */
template <typename inputT, typename outputT>
void ConvTransformer<inputT, outputT>::xxyy(Point &current, Point const &left) {
  current.first = left.first;
  current.second = left.second;
}

/* x = x' + 1 ; y = 0 */
template <typename inputT, typename outputT>
void ConvTransformer<inputT, outputT>::xp1y0(Point &current, Point const &above) {
  current.first = above.first + 1;
  current.second = 0;
}

/* x = x' ; y = y' + 1 */
template <typename inputT, typename outputT>
void ConvTransformer<inputT, outputT>::xyp1(Point &current, Point const &above) {
  current.first = above.first;
  current.second = above.second + 1;
}

/* x = x' + 1 ; y = y' - 1 */
template <typename inputT, typename outputT>
void ConvTransformer<inputT, outputT>::xp1ym1(Point &current, Point const &above) {
  current.first = above.first + 1;
  current.second = -1;
}

template <typename inputT, typename outputT>
bool ConvTransformer<inputT, outputT>::has_occured(Point const &p,
                                     std::vector<bool> const &occurence) {
  int y = p.second;
  int lsfe = (m_cp.imap[WIDTH]+m_cp.pad[0]+m_cp.pad[2] - m_cp.k[WIDTH]);
  return occurence.at(y % lsfe);
}

template <typename inputT, typename outputT> bool ConvTransformer<inputT, outputT>::is_zero(Point const &p) {
  return (p.first == 0 && p.second == 0) ? true : false;
}

/* true if p is the first element of kernel at last sliding position */
template <typename inputT, typename outputT> bool ConvTransformer<inputT, outputT>::is_last_kernel(Point const &p) {
  return ((p.first == (m_cp.imap[HEIGHT]+m_cp.pad[1]+m_cp.pad[3] - m_cp.k[HEIGHT])) &&
          (p.second == (m_cp.imap[WIDTH]+m_cp.pad[0]+m_cp.pad[2] - m_cp.k[WIDTH])));
}

template <typename inputT, typename outputT>
void ConvTransformer<inputT, outputT>::mark_occured(Point const &p,
                                      std::vector<bool> &occurence) {
  int y = p.second;
  int lsfe = (m_cp.imap[WIDTH]+m_cp.pad[0]+m_cp.pad[2] - m_cp.k[WIDTH]);
  occurence.at(y % lsfe) = 1;
}

/* fill 'out' matrix with values from 'input' at co-ordinates present in ibuf
 * starting from offset till n
 */
template <typename inputT, typename outputT>
void ConvTransformer<inputT, outputT>::fill_index(Mat<inputT> &out, Mat<inputT> const &input,
                                    std::vector<Point> const &ibuf, int n,
                                    int offset) {
  assert(ibuf.size() == (m_cp.k[HEIGHT] * m_cp.k[WIDTH]));
  assert(n <= (m_cp.k[WIDTH] * m_cp.k[HEIGHT]));
  std::vector<inputT> buf(m_cp.k[HEIGHT] * m_cp.k[WIDTH], 0);
  for (int i = offset; i < n; ++i) {
    auto p = ibuf.at(i);
    buf.at(i) = input.at(p.first, p.second);
  }
  assert(buf.size() <= (m_cp.k[HEIGHT] * m_cp.k[WIDTH]));
  out.push_back(buf);
}

/* generate n indices based on previous indices (stored in ibuf2) and store them
 * in ibuf */
template <typename inputT, typename outputT>
void ConvTransformer<inputT, outputT>::generate_index(std::vector<Point> const &ibuf2,
                                        std::vector<Point> &ibuf, int n) {
  assert(ibuf.size() == (m_cp.k[HEIGHT] * m_cp.k[WIDTH]));
  assert(ibuf2.size() == (m_cp.k[HEIGHT] * m_cp.k[WIDTH]));
  assert(n <= (m_cp.k[HEIGHT] * m_cp.k[WIDTH]));
  std::vector<bool> occurence(m_cp.k[WIDTH], 0);
  for (int i = 0; i < n; ++i) {
    if (is_lsfe(ibuf2.at(i)) && !has_occured(ibuf2.at(i), occurence)) {
      xp1y0(ibuf.at(i), ibuf2.at(i));
      mark_occured(ibuf2.at(i), occurence);
    } else if (is_lsme(ibuf2.at(i)) && !has_occured(ibuf2.at(i), occurence)) {
      (i != 0) ? xxyy(ibuf.at(i), ibuf.at(i - 1))
               : xxyy(ibuf.at(i), ibuf2.at(i));
      mark_occured(ibuf2.at(i), occurence);
    } else if (is_lsle(ibuf2.at(i))) {
      (i != 0) ? xxyy(ibuf.at(i), ibuf.at(i - 1))
               : xxyy(ibuf.at(i), ibuf2.at(i));
    } else {
      xyp1(ibuf.at(i), ibuf2.at(i));
    }
  }
}

/* This transform required to carry out convolution on the systolic array
 * is called 'im2col' [1] in the literature. The output of this function is
 * equal to that of ordinary im2col, but the method of achieving is completely
 * different.
 *
 * In a real heterogeneous environment, there are two ways to doing im2col:
 * static and dynamic. static would transform input into an expanded matrix and
 * store it somewhere and later feed it to the systolic array. dynamic involves
 * playing the algorithm in real time so inputs are expanded as they arrive and
 * fed directly to the SA. This reduces the latency and size expansion incurred
 * by static SA.
 *
 * This algorithm is the proof-of-concept implementation of the aforementioned
 * dynamic im2col.
 *
 * For a convolution involving KWxKH kernel, we require a SA of atleast KWxKH
 * rows. The algorithm generates KWxKH elements at a time and feeds it to the
 * SA.
 *
 * Consider a convolution of 4x4 input with a 2x2 kernel. We require 4 inputs to
 * be generated at a timestep. For the first timestep, the inputs required are
 * values at at co-ordinates (0,0)       0       0       0 the zeros are padded
 * as the SA only consumes 1 element in the fist timestep. This is followed by
 * the arrays made of: (0,1)     (0,1)     0       0 (0,2)     (0,2)   (1,0) 0
 * and so on. The numbers inside the brackets are co-ordinates indexing a matrix
 * and are replaced by their values.
 *
 * Definitions:
 *  lsfe: last slide first element.
 *        the first element of the last sliding position of a kernel.
 *        for a 2x2 kernel on 4x4 input, all the co-ordinates with co-ordinates
 * of the second last column are lsfe. lsme: last slide middle element all the
 * elements b/w first element and last of the last sliding position of a kernel
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
 *  3. during each iteration, compare current buffer's co-ordinates to previous
 * buffer's at the same index
 *  4. if its lsfe, increment the x value of previous buffer and set y to 1 and
 *  only do this once for a buffer.
 *  5. if its lsme, copy the value to the left of the current buffer and only do
 *  this once for a buffer.
 *  6. if its lsle, copy the value to the left of the current buffer
 *  7. after iteration, replace co-ordinates in current buf to their
 * corresponding values
 *  8. copy current buf's contents of previous buf.
 *
 * Here's a complete set of vectors as generated by this algorithm for 2x2
 * kernel on a 6x6 input: 0,0 0,0 0,0 0,0 0,1 0,1 0,0 0,0 0,2 0,2 1,0 0,0 0,3
 * 0,3 1,1 1,1 0,4 0,4 1,2 1,2 1,0 0,5 1,3 1,3 1,1 1,1 1,4 1,4 1,2 1,2 2,0 1,5
 *       1,3 1,3 2,1 2,1
 *       1,4 1,4 2,2 2,2
 *       2,0 1,5 2,3 2,3
 *       2,1 2,1 2,4 2,4
 *       2,2 2,2 3,0 2,5
 *       0,0 2,3 3,1 3,1
 *       0,0 0,0 3,2 3,2
 *       0,0 0,0 0,0 3,3
 */

template <typename inputT, typename outputT> Mat<inputT> ConvTransformer<inputT, outputT>::transform(std::vector<inputT> &a) {
  Mat<inputT> input = v2mat<inputT>(a, m_cp.imap[0], m_cp.imap[1]);
  Mat<inputT> out;
  std::vector<Point> ibuf(m_cp.k[HEIGHT] * m_cp.k[WIDTH], std::make_pair(0, 0));
  std::vector<Point> ibuf2(m_cp.k[HEIGHT] * m_cp.k[WIDTH], std::make_pair(0, 0));
  ibuf2.at(0).second = -1;
  /* prologue */
  for (int i = 1; i < (m_cp.k[HEIGHT] * m_cp.k[WIDTH]); ++i) {
    generate_index(ibuf2, ibuf, i);
    // print_vec_point("vec", ibuf);
    std::copy(ibuf.begin(), ibuf.end(), ibuf2.begin());
    (is_kern_edge(ibuf.at(i - 1))) ? xp1ym1(ibuf2.at(i), ibuf.at(i - 1))
                                   : xxyy(ibuf2.at(i), ibuf.at(i - 1));
    fill_index(out, input, ibuf, i, 0);
  }
  /* core */
  while (!is_last_kernel(ibuf.at(0))) {
    generate_index(ibuf2, ibuf, m_cp.k[HEIGHT] * m_cp.k[WIDTH]);
    // print_vec_point("vec", ibuf);
    std::copy(ibuf.begin(), ibuf.end(), ibuf2.begin());
    fill_index(out, input, ibuf, m_cp.k[HEIGHT] * m_cp.k[WIDTH], 0);
  }
  /* epilogue */
  for (int i = 1; i < (m_cp.k[0] * m_cp.k[1]); ++i) {
    generate_index(ibuf2, ibuf, m_cp.k[HEIGHT] * m_cp.k[WIDTH]);
    // print_vec_point("vec", ibuf);
    std::copy(ibuf.begin(), ibuf.end(), ibuf2.begin());
    fill_index(out, input, ibuf, m_cp.k[HEIGHT] * m_cp.k[WIDTH], i);
  }
  for (int i = 0; i < (sa_dims.cols - 2); ++i) {
    std::vector<inputT> tmp(m_cp.k[HEIGHT] * m_cp.k[WIDTH], 0);
    out.push_back(tmp);
  }
  return out;
}

/* For multi-channel kernels , in order to load them in a 2-D SA
 * we need to pack the channels into one single vector. Concatenate channels
 * in a single vector and call 'transform_weights(vec v,SA_row,SA_column)' on
 * it. It reorders them in a specific order so that when the 'load_weights()' is
 * called. it will load the 1st kernel in the 1st column of the SA and the 2nd
 * kernel in the 2nd column of the SA.
 *
 * transform_weights() : takes elements one by one from the input vector in
 * increasing order and places them in the return vector in order i+ SA_row, so
 * when load_weights is called on return vector USED WHEN MULTI_CHANNEL KERNEL
 * AND 2D SA is used.
 *
 *    kernel_1 2x2:             1   2
 *                              3   4
 *
 *    kernel_2 2x2:             5   6
 *                              7   8
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
template <typename inputT, typename outputT>
std::vector<inputT> ConvTransformer<inputT, outputT>::transform_weights(std::vector<inputT> &w,
                                                     int out_row, int out_col) {
  assert(w.size() == out_row * out_col);
  std::vector<inputT> out(out_row * out_col, 0);
  for (int i = 0; i < out_col; ++i) {
    for (int j = 0; j < out_row; ++j) {
      out.at(i + j * out_col) = w.at(i * out_row + j);
    }
  }
  return out;
}

template <typename inputT, typename outputT>
std::vector<outputT> ConvTransformer<inputT, outputT>::untransform(Mat<outputT> &a) {
  std::vector<outputT> out;
  int hout = sa_odims_row(m_cp);
  int wout = sa_odims_cols(m_cp);
  for (int i = 0; i < sa_dims.cols; i++) {
    for (int j = 0; j < hout * wout; ++j) {
      out.push_back(a.at(i, j));
    }
  }
  return out;
}

template <typename inputT, typename outputT>
ConvTransformer<inputT, outputT>::ConvTransformer(Op::ConvParams const &cp,
                                    SaDims const &sa_dims) {
  std::memcpy(&m_cp, &cp, sizeof(Op::ConvParams));
  std::memcpy(&(this->sa_dims), &sa_dims, sizeof(SaDims));
}
