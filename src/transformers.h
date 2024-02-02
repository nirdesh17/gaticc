#pragma once

#include "onnx_parser.h"
#include "sim.h"
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
 */

using TransformerType = enum TransformerType {
  GEMM_TF,
  CONV_TF,
};

/* Abstract base class for all transformers */
class Transformer {
public:
  virtual TransformerType get_type() = 0;
  virtual Mat transform(std::vector<int> &a) = 0;
  virtual std::vector<int> untransform(Mat &a) = 0;
};

class GemmTransformer : public Transformer {
private:
  int arows;
  int acolumns;
  ;
  int brows;
  int bcolumns;
  std::vector<int> to_sys_major(std::vector<int> &v, int rows, int columns);
  std::vector<int> get_access_frequency(int rows, int cols);
  void zero_pad_before(std::vector<int> &v, int n);
  void zero_pad_after(std::vector<int> &v, int n);
  void zero_pad(Mat &out, std::vector<int> &frequency, int columns);
  std::vector<int> pick(std::vector<int> &v, int n);

public:
  GemmTransformer() = delete;
  GemmTransformer(int r, int c, int ir, int ic);
  TransformerType get_type() override;
  Mat transform(std::vector<int> &a) override;
  std::vector<int> untransform(Mat &a) override;
};

class ConvTransformer: public Transformer {
    private: 
        Op::ConvParams m_cp;
        SaDims sa_dims;
        void fill_index(Mat &out, Mat const &input, std::vector<Point> const &ibuf, int n, int offset);
        void generate_index(std::vector<Point> const &ibuf2, std::vector<Point> &ibuf, int n);
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
        Mat transform(std::vector<int> &a) override;
        std::vector<int> transform_weights(std::vector<int>& w, int out_rows, int out_cols);
        std::vector<int> untransform(Mat &a) override;
};
