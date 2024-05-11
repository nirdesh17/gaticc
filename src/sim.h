#pragma once

#include "ffi.h"
#include "onnx_parser.h"
#include "tensor.h"
#include "utils.h"
#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif
#include "Python.h"
#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>
#include <algorithm>
#include <cmath>
#include <valarray>

#if 0
namespace Int_Graph {
using Graph =
    boost::adjacency_list<boost::vecS, boost::listS, boost::directedS, int>;
using Vertex = boost::graph_traits<Graph>::vertex_descriptor;
using Adjacency_iterator = Graph::adjacency_iterator;
using Vertex_Iterator = Graph::vertex_iterator;
}; // namespace Int_Graph

class Tree {
private:
  Int_Graph::Graph g;
  int rows;
  int columns;
  std::map<std::pair<int, int>, Int_Graph::Vertex> vertex_map;
  /* TODO: turn these into maps
   * since they are used only internally during construction by
   * connect_btree, hide them
   */
  std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> edge_memo;
  std::vector<std::pair<int, int>> child_memo;

  void generate_btree(Mat<int> const &v, std::pair<int, int> xy);
  bool is_node_allowed(std::pair<int, int> node);
  void connect_btree(std::pair<int, int> xy);

public:
  Tree(Mat<int> const &v, int rows, int columns);
  Int_Graph::Vertex get_root();
  int total_vertices();
  int total_edges();
  void print_btree();
  std::vector<int> breadth_first_order();
};

//TODO: remove this
//#define sa_output_dims(ip_rows, padding, dilation, kernel_rows, stride)        \
//  (((ip_rows + 2 * padding - dilation * (kernel_rows - 1) - 1) / stride) + 1)

template <typename inputT, typename outputT> class PE {
private:
  int id;
  inputT weight;
  /* for partial sums */
  outputT reg;
  /* this is written by other PEs */
  outputT input_buffer;

  outputT aux_buffer;
  outputT ps_buffer;

public:
  PE(int id, inputT w, outputT r, outputT input_buffer);
  int get_id();
  inputT get_weight();
  outputT get_reg();
  outputT get_input_buffer();
  outputT get_ps_buffer();
  void set_input_buffer(outputT r);
  void set_weight(inputT w);
  void set_reg(outputT r);
  void set_aux_buffer(outputT r);
  void set_ps_buffer(outputT r);
  outputT get_aux_buffer();
  outputT mac();
  void print_pe();
};

template <typename inputT, typename outputT>
PE<inputT, outputT>::PE(int id, inputT w, outputT r, outputT input_buffer)
    : id{id}, weight{w}, reg{r}, input_buffer{input_buffer}, aux_buffer{0},
      ps_buffer{0} {};

template <typename inputT, typename outputT> int PE<inputT, outputT>::get_id() {
  return this->id;
}

template <typename inputT, typename outputT>
inputT PE<inputT, outputT>::get_weight() {
  return this->weight;
}

template <typename inputT, typename outputT>
outputT PE<inputT, outputT>::get_reg() {
  return this->reg;
}

template <typename inputT, typename outputT>
outputT PE<inputT, outputT>::get_input_buffer() {
  return this->input_buffer;
}

template <typename inputT, typename outputT>
outputT PE<inputT, outputT>::get_aux_buffer() {
  return this->aux_buffer;
}

template <typename inputT, typename outputT>
outputT PE<inputT, outputT>::get_ps_buffer() {
  return this->ps_buffer;
}

template <typename inputT, typename outputT>
void PE<inputT, outputT>::set_input_buffer(outputT r) {
  this->input_buffer = r;
}

template <typename inputT, typename outputT>
void PE<inputT, outputT>::set_weight(inputT w) {
  this->weight = w;
}

template <typename inputT, typename outputT>
void PE<inputT, outputT>::set_reg(outputT r) {
  this->reg = r;
}

template <typename inputT, typename outputT>
void PE<inputT, outputT>::set_aux_buffer(outputT r) {
  this->aux_buffer = r;
}

template <typename inputT, typename outputT>
void PE<inputT, outputT>::set_ps_buffer(outputT r) {
  this->ps_buffer = r;
}

template <typename inputT, typename outputT>
outputT PE<inputT, outputT>::mac() {
  outputT ps = this->get_input_buffer() * this->get_weight();
  return ps;
}

template <typename inputT, typename outputT>
void PE<inputT, outputT>::print_pe() {
  printf("Id: %d\tWeight: %d\tReg: %d\tInput: %d\tAux: %d\tPs: %d\n", id,
         weight, reg, input_buffer, aux_buffer, ps_buffer);
}

class Profiler {
  uint64_t cycles;

public:
  void incr_cycles(int count);
  uint64_t get_cycles();
};

template <typename T> Mat<T> Padder(Mat<T> &input, int padding) {
  Mat<T> new_mat;
  std::vector<T> store;
  for (int i = 0; i < input.size() + 2 * padding; i++) {
    for (int j = 0; j < input.at(0).size() + 2 * padding; j++) {
      if (((i < padding) || (i >= input.size() + padding)) ||
          ((j < padding) || (j >= input.at(0).size() + padding))) {
        store.push_back(0);
      } else {
        store.push_back(input.at(i - padding, j - padding));
      }
    }
  }
  new_mat = v2mat<T>(store, input.size() + 2 * padding,
                     input.at(0).size() + 2 * padding);
  return new_mat;
}

/* All output blocks shall inherit from this class. Derived
 * classes are required to implement the 'exec' function in
 * order to be considered a correct output function.
 * See Relu and Quantize classes below
 */
class Chainblock {
public:
  /* the vanilla exec function is simply the identity
   * function
   */
  virtual int exec(int x);
  virtual ~Chainblock();
};

/* Rapper around Chainblocks* array */
class Chain {
  std::vector<Chainblock *> arr;

public:
  void push(Chainblock *b);
  Chainblock *at(int i);
  void replace(int i, Chainblock *b);
  ~Chain();
  int pass_through(int x);
};
#endif

template <typename T>
class Relu {
  int clip_val;

public:
  Relu(int clip_val);
  Relu();
  void exec(Tensor<T> *input, Tensor<T> *output);
};

template <typename T>
Relu<T>::Relu(int clip_val) : clip_val{clip_val} {
}

template <typename T>
Relu<T>::Relu() : clip_val{INT_MAX} {
}

template <typename T>
void Relu<T>::exec(Tensor<T> *input, Tensor<T> *output) {
  for (int i = 0; i < input->size(); ++i) {
    T x = input->at(i);
    T v = (x < 0) ? 0 : ((x > clip_val) ? clip_val : x);
    output->set(i, v);
  }
}

#if 0

/*Quantizer is used to re-encode information.It is used to reduce the size and
 *bandwidth required by 4 times. In our case we are reducing a 32bit-Int to a
 *8bit-Int. the original value is multiplied to a scale value and then clipped
 *into the range of (-127,127)
 */
class Quantize : public Chainblock {
  int scale;
  int shift;

public:
  Quantize(int scale, int shift);
  int exec(int x) override;
  int clipper(int x);
};

class BatchNorm : public Chainblock {
  int mean;
  int sd;
  int gamma;
  int beta;

public:
  BatchNorm(int mean, int sd, int gamma, int beta);
  int exec(int x) override;
};

template <typename T>
void bias_add(Tensor<T> *arr, Op::Layer::Conv *cc) {
  Tensor<T> *bias_arr = new TensorExtant<T>(cc->bias);
  if (arr->dims_size() != 3) {
    log_fatal("input to bias not 3-dimensional");
  }
  int stride = arr->dims_at(1) * arr->dims_at(2);
  for (int i = 0; i < arr->dims_iterator(-1); ++i) {
    arr->at(i);
  }
#if 0
  for (int i = 0; i < arr->dims_at(0); ++i) {
    for (int j = 0; j < stride; ++j) {
      int index = i * stride + j;
      std::cout << i << ' ' << j << '\n';
      //T val = arr->at(index) + bias_arr->at(i);
      //arr->set(index, 3.33);
    }
  }
#endif
  delete bias_arr;
}
#endif

#if 0
/*
* this max pooler pools out the max val when a kernel window is slid over the
* output matrix after convolution
* this max pooler has a kernel window and a stride whos value can be set
* manually, and this process overall reduces the
* size of the input image.

* IF dimension of the input matrix is odd ,
* in valid padding we crop out the last column
* in same padding we add a column of zeroes

* feeding the same new_ mat to the transformer again and repeat the process

* the process is divided into two parts: 1. Movement   2. Action

* Movement : moves the kernel window over the input matrix and call a
* generic action on it which will be decided by the caller.

* Action : performs the told mathematical operation ( max, average etc.)
* on the selected elements of the kernel window
*/

template <typename T> using Action = std::function<T(std::vector<T> &)>;

template <typename T> class Pooler {
  static T max_pooler_action(std::vector<T> &input) {
    auto max = std::max_element(input.begin(), input.end());
    return *max;
  }
  static T average_pooler_action(std::vector<T> &input) {
    T avg = (T)(std::accumulate(input.begin(), input.end(), 0) /
                (std::distance(input.begin(), input.end())));
    return avg;
  }

  Mat<T> movement(Mat<T> &input, Op::MaxpoolParams const &mp, Action<T> action);

public:
  Mat<T> max_pooler(Mat<T> &input, Op::MaxpoolParams const &mp);
  Mat<T> average_pooler(Mat<T> &input, Op::MaxpoolParams const &mp);
  Mat<T> global_average_pooler(Mat<T> &input, Op::MaxpoolParams const &mp);
};

template <typename T>
Mat<T> Pooler<T>::movement(Mat<T> &input, Op::MaxpoolParams const &mp,
                           Action<T> action) {
  std::vector<T> ret;
  std::vector<T> temp_matrix;
  // relation bw stride and dilation: stride + dilation < columns
  for (int i = 0;
       i + (mp.dilation[0] * (mp.k[0] - 1)) < mp.imap[0] + 2 * mp.pad[0];
       i += mp.stride[0]) {
    for (int j = 0;
         j + (mp.dilation[0] * (mp.k[1] - 1)) < mp.imap[1] + 2 * mp.pad[0];
         j += mp.stride[0]) {
      ret.clear();
      for (int k = 0; k < mp.k[0]; k++) {
        for (int l = 0; l < mp.k[1]; l++) {
          ret.push_back(
              input.at(i + (k * mp.dilation[0]), j + (l * mp.dilation[0])));
        }
      }
      temp_matrix.push_back(action(ret));
    }
  }
  int hout = sa_output_dims(mp.imap[0], mp.pad[0], mp.dilation[0], mp.k[0],
                            mp.stride[0]);
  return v2mat<T>(temp_matrix, hout, hout);
}

template <typename T>
Mat<T> Pooler<T>::max_pooler(Mat<T> &input, Op::MaxpoolParams const &mp) {
  Mat<T> padded_in;
  /* TODO: asymmetric padding? */
  if (mp.pad[0] != 0) {
    padded_in = Padder(input, mp.pad[0]);
  }
  return movement(padded_in, mp, max_pooler_action);
}

template <typename T>
Mat<T> Pooler<T>::average_pooler(Mat<T> &input, Op::MaxpoolParams const &mp) {
  Mat<T> padded_in;
  if (mp.pad[0] != 0) {
    padded_in = Padder<T>(input, mp.pad[0]);
  }
  return movement(padded_in, mp, average_pooler_action);
}

template <typename T>
Mat<T> Pooler<T>::global_average_pooler(Mat<T> &input,
                                        Op::MaxpoolParams const &mp) {
  Mat<T> padded_in;
  if (mp.pad[0] != 0) {
    padded_in = Padder<T>(input, mp.pad[0]);
  }
  Op::MaxpoolParams mp_copy = mp;
  mp_copy.stride[0] = mp.stride[0] + mp.imap[0] + 2 * mp.pad[0];
  mp_copy.stride[1] = mp.stride[1] + mp.imap[1] + 2 * mp.pad[3];
  mp_copy.k[0] = mp.imap[0] + 2 * mp.pad[0];
  mp_copy.k[1] = mp.imap[1] + 2 * mp.pad[1];
  return movement(padded_in, mp_copy, average_pooler_action);
}
#endif

#if 0
namespace PE_Graph {

template <typename inputT, typename outputT>
using Graph = boost::adjacency_list<boost::vecS, boost::listS, boost::directedS,
                                    PE<inputT, outputT>>;
template <typename inputT, typename outputT>
using Vertex =
    typename boost::graph_traits<Graph<inputT, outputT>>::vertex_descriptor;
template <typename inputT, typename outputT>
using Adjacency_iterator = typename Graph<inputT, outputT>::adjacency_iterator;
template <typename inputT, typename outputT>
using Neighbours =
    std::pair<typename PE_Graph::Adjacency_iterator<inputT, outputT>,
              typename PE_Graph::Adjacency_iterator<inputT, outputT>>;
} // namespace PE_Graph


template <typename inputT, typename outputT> class SA {
private:
  PE_Graph::Graph<inputT, outputT> g;
  int rows;
  int columns;
  std::vector<PE_Graph::Vertex<inputT, outputT>> vertarray;
  Mat<outputT> output_array;
  std::vector<int> output_array_counts;
  Profiler profiler;
  bool profile_enabled;

  int get_index_from_vertex(PE_Graph::Vertex<inputT, outputT> &v);
  PE<inputT, outputT> &get_pe_from_vertex(PE_Graph::Vertex<inputT, outputT> &v);
  PE<inputT, outputT> &get_pe_from_adjacency_iterator(
      PE_Graph::Adjacency_iterator<inputT, outputT> &itr);
  PE_Graph::Neighbours<inputT, outputT>
  get_neighbouring_vertices(PE_Graph::Vertex<inputT, outputT> &v);
  PE_Graph::Vertex<inputT, outputT> &get_vertex_from_adjacency_iterator(
      PE_Graph::Adjacency_iterator<inputT, outputT> &itr);
  void _propagate(PE_Graph::Vertex<inputT, outputT> &v);
  void load_inputs(std::vector<inputT> const &inputs);
  bool is_output_vertex(PE_Graph::Vertex<inputT, outputT> &v);
  int output_array_hash(int n);
  void push_to_output_array(int h, outputT t1);
  void prepare_queue(std::queue<PE_Graph::Vertex<inputT, outputT>> &exec_queue);
  bool should_pass_partial_sum(PE_Graph::Vertex<inputT, outputT> &vi);
  bool should_pass_input(PE_Graph::Vertex<inputT, outputT> &vi);
  PE_Graph::Neighbours<inputT, outputT>
  assign_vertices(const PE_Graph::Neighbours<inputT, outputT> &neighbours,
                  PE_Graph::Vertex<inputT, outputT> &v);

public:
  SA(int r, int c);
  SA(int r, int c, bool profile_enabled);
  void clear_output();
  void print_array();
  void load_weights(std::vector<inputT> &weights);
  void propagate(Mat<inputT> const &input_mat);
  void generate_profile_report();
  int total_vertices();
  int total_edges();
  int get_rows();
  int get_cols();
  Mat<outputT> const &get_output();
};

template <typename inputT, typename outputT>
void SA<inputT, outputT>::load_weights(std::vector<inputT> &weights) {
  for (int i = 0; i < rows * columns; ++i) {
    get_pe_from_vertex(vertarray[i])
        .set_weight(weights[i]); // temporary cast here
  }
}

template <typename inputT, typename outputT>
Mat<outputT> const &SA<inputT, outputT>::get_output() {
  return output_array;
}

template <typename T>
void exchange_queues(std::queue<T> &dest, std::queue<T> &src) {
  assert(dest.empty() == true);
  std::set<T> s;
  while (!src.empty()) {
    s.insert(src.front());
    src.pop();
  }
  for (auto &i : s) {
    dest.push(i);
  }
}

/* iterate over all PEs in systolic manner and call _propagate on them */
template <typename inputT, typename outputT>
void SA<inputT, outputT>::propagate(Mat<inputT> const &input_mat) {
  /* Special handling for SA of size (m, 1)
   * The general case, following this code can handle this special case
   * but as sasa.h implements its algorithms with (m,1) SAs, specially
   * handling this case trivially, without any queues and bookeeping
   * functions leads to faster code. (atleast 2x in performance)
   * */
  if (columns == 1) {
    for (int i = 0; i != input_mat.size(); ++i) {
      load_inputs(input_mat.at(i));
      for (int j = 0; j < rows; ++j) {
        PE_Graph::Vertex<inputT, outputT> v = vertarray.at(j);
        _propagate(v);
      }
    }
    return;
  }

  /* General case for SA of size (m,n) */
  std::queue<PE_Graph::Vertex<inputT, outputT>> exec_queue;
  std::queue<PE_Graph::Vertex<inputT, outputT>> alt_queue;
  prepare_queue(exec_queue);
  for (int i = 0; i != input_mat.size(); ++i) {
    load_inputs(input_mat.at(i));
    while (!exec_queue.empty()) {
      PE_Graph::Vertex<inputT, outputT> v = exec_queue.front();
      alt_queue.push(v);
      exec_queue.pop();
      _propagate(v);
    }
    // print_array();
    exchange_queues<PE_Graph::Vertex<inputT, outputT>>(exec_queue, alt_queue);
    if (profile_enabled) {
      profiler.incr_cycles(1);
    }
  }
}

template <typename inputT, typename outputT>
void SA<inputT, outputT>::prepare_queue(
    std::queue<PE_Graph::Vertex<inputT, outputT>> &exec_queue) {
  std::vector<int> order(rows * columns);
  std::iota(order.begin(), order.end(), 0);
  Mat<int> order2d = v2mat<int>(order, rows, columns);
  Tree t(order2d, rows, columns);
  std::vector<int> new_order = t.breadth_first_order();
  for (auto i : new_order) {
    exec_queue.push(vertarray[i]);
  }
}

/* load inputs at first n PEs where n = 0 3 6 9 ... for a Mx3 systolic array
 */
template <typename inputT, typename outputT>
void SA<inputT, outputT>::load_inputs(std::vector<inputT> const &inputs) {
  for (int i = 0, vi = 0; i < rows; ++i) {
    get_pe_from_vertex(vertarray[vi]).set_input_buffer(inputs.at(i));
    get_pe_from_vertex(vertarray[vi]).set_aux_buffer(inputs.at(i));
    vi += columns;
  }
}

/* carry out mac on one PE pointed by Vertex v, and pass inputs/partial sums
 * to the neighbouring PEs
 */
template <typename inputT, typename outputT>
void SA<inputT, outputT>::_propagate(PE_Graph::Vertex<inputT, outputT> &v) {
  /* expects ps and input_buffer to be filled */
  outputT res = get_pe_from_vertex(v).mac();

  PE_Graph::Adjacency_iterator<inputT, outputT> right{}, down{};
  auto neighbours = get_neighbouring_vertices(v);
  std::tie(right, down) = assign_vertices(neighbours, v);

  /* vertices to the right are connected before vertices to the
   * south of a PE. following code depends on this fact to check
   * which PE to pass inputs and which to pass partial sums
   *
   * following heavily relies on the fact that the array is 2D with
   * at most 2 connections b/w any PE
   */

  /* pass values */
  if (should_pass_input(v)) {
    // pass inputs
    outputT t1 = get_pe_from_adjacency_iterator(right).get_input_buffer();
    get_pe_from_adjacency_iterator(right).set_aux_buffer(t1);

    outputT t2 = get_pe_from_vertex(v).get_aux_buffer();
    get_pe_from_adjacency_iterator(right).set_input_buffer(t2);
  }

  if (should_pass_partial_sum(v)) {
    // pass partial sum
    outputT t1 = get_pe_from_adjacency_iterator(down).get_ps_buffer();
    get_pe_from_adjacency_iterator(down).set_reg(t1);

    outputT t2 = get_pe_from_vertex(v).get_reg() + res;
    get_pe_from_adjacency_iterator(down).set_ps_buffer(t2);
  }

  if (is_output_vertex(v)) {
    outputT t1 = get_pe_from_vertex(v).get_reg() + res;
    int h = output_array_hash(get_index_from_vertex(v));
    push_to_output_array(h, t1);
  }
}

template <typename inputT, typename outputT>
void SA<inputT, outputT>::push_to_output_array(int h, outputT t1) {
  if (h >= 0 && h <= 1) {
    if (output_array_counts.at(h) >= (rows - 1)
        //&& output_array_counts.at(h) < (rows+rows-1)
    ) {
      output_array.at(h).push_back(t1);
    }
  } else if (output_array_counts.at(h) >= (rows - 2 + h)
             //&& output_array_counts.at(h) < (rows+rows-2+h)
  ) {
    output_array.at(h).push_back(t1);
  }
  output_array_counts.at(h) += 1;
}

Point get_cartesian_cord(int index, int r, int c);

template <typename inputT, typename outputT>
SA<inputT, outputT>::SA(int r, int c) : SA<inputT, outputT>(r, c, false) {}

template <typename inputT, typename outputT>
SA<inputT, outputT>::SA(int r, int c, bool profile_enabled)
    : rows{r}, columns{c}, profile_enabled{profile_enabled},
      output_array{Mat<outputT>(columns)},
      output_array_counts{std::vector<int>(columns)} {
  /* add null vertices to the graph
   * do not connect any vertices with edges
   */
  for (int i = 0; i < r * c; ++i) {
    /* zero initialize weights and partial sums */
    PE_Graph::Vertex<inputT, outputT> v =
        boost::add_vertex(PE<inputT, outputT>(i, 0, 0, 0), g);
    vertarray.push_back(v);
  }
  /* connect vertices through edges in a systolic manner.
   * For example, in a 3x3 array , PE (0,0) will be connected to
   * PE (1,0) and PE (0,1), whereas PE (2,0) will only be connected
   * to PE (2,1) and PE (2,2) will be connected to no one
   */
  for (int i = 0; i < r * c; ++i) {
    auto cord = get_cartesian_cord(i, r, c);
    if (cord.first == (r - 1) && cord.second == (c - 1)) {
      // do nothing
    }
    /* rightwards only */
    else if (cord.first == (r - 1)) {
      boost::add_edge(vertarray[i], vertarray[i + 1], g);
    }
    /* downwards only */
    else if (cord.second == (c - 1)) {
      boost::add_edge(vertarray[i], vertarray[i + c], g);
    }
    /* right and down */
    else {
      /* right is added before down entry */
      boost::add_edge(vertarray[i], vertarray[i + 1], g);
      boost::add_edge(vertarray[i], vertarray[i + c], g);
    }
  }
}

template <typename inputT, typename outputT>
int SA<inputT, outputT>::get_index_from_vertex(
    PE_Graph::Vertex<inputT, outputT> &v) {
  return get_pe_from_vertex(v).get_id();
}
template <typename inputT, typename outputT>
PE<inputT, outputT> &
SA<inputT, outputT>::get_pe_from_vertex(PE_Graph::Vertex<inputT, outputT> &v) {
  return g[v];
}

template <typename inputT, typename outputT>
PE<inputT, outputT> &SA<inputT, outputT>::get_pe_from_adjacency_iterator(
    PE_Graph::Adjacency_iterator<inputT, outputT> &itr) {
  return g[*itr];
}

template <typename inputT, typename outputT>
PE_Graph::Neighbours<inputT, outputT>
SA<inputT, outputT>::get_neighbouring_vertices(
    PE_Graph::Vertex<inputT, outputT> &v) {
  return boost::adjacent_vertices(v, g);
}

template <typename inputT, typename outputT>
PE_Graph::Vertex<inputT, outputT> &
SA<inputT, outputT>::get_vertex_from_adjacency_iterator(
    PE_Graph::Adjacency_iterator<inputT, outputT> &itr) {
  for (PE_Graph::Vertex<inputT, outputT> &v : vertarray) {
    if (v == *itr) {
      return v;
    }
  }
  assert(1 == 0 && "adjacency iterator didn't match up to any vertex");
}

template <typename inputT, typename outputT>
int SA<inputT, outputT>::total_vertices() {
  assert(vertarray.size() == boost::num_vertices(g) &&
         "vertarray and total vertices in 'g' mismatch");
  return boost::num_vertices(g);
}

template <typename inputT, typename outputT>
int SA<inputT, outputT>::total_edges() {
  return boost::num_edges(g);
}

template <typename inputT, typename outputT>
int SA<inputT, outputT>::get_rows() {
  return rows;
}

template <typename inputT, typename outputT>
int SA<inputT, outputT>::get_cols() {
  return columns;
}

template <typename inputT, typename outputT>
bool SA<inputT, outputT>::should_pass_input(
    PE_Graph::Vertex<inputT, outputT> &vi) {
  /* TODO: come up with a general case for this function
   * see linus-torvals linked list good taste argument */
  if (columns <= 1) {
    return false;
  }
  int id = get_pe_from_vertex(vi).get_id();
  if ((id != 0) && (((id + 1) % columns) == 0)) {
    return false;
  }
  return true;
}

template <typename inputT, typename outputT>
bool SA<inputT, outputT>::should_pass_partial_sum(
    PE_Graph::Vertex<inputT, outputT> &vi) {
  /* TODO: come up with a general case for this function
   * see linus-torvals linked list good taste argument */
  if (rows <= 1) {
    return false;
  }
  int id = get_pe_from_vertex(vi).get_id();
  int total_pes = rows * columns;
  if ((id >= (total_pes - columns)) && (id < total_pes)) {
    return false;
  }
  return true;
}

template <typename inputT, typename outputT>
PE_Graph::Neighbours<inputT, outputT> SA<inputT, outputT>::assign_vertices(
    const PE_Graph::Neighbours<inputT, outputT> &neighbours,
    PE_Graph::Vertex<inputT, outputT> &v) {
  PE_Graph::Adjacency_iterator<inputT, outputT> right{}, down{};
  int difference = neighbours.second - neighbours.first;
  if (difference == 2) {
    right = neighbours.first;
    down = neighbours.first + 1;
  } else if (difference == 1) {
    if (should_pass_input(v)) {
      right = neighbours.first;
    } else {
      down = neighbours.first;
    }
  }
  return PE_Graph::Neighbours<inputT, outputT>(right, down);
}

template <typename inputT, typename outputT>
void SA<inputT, outputT>::clear_output() {
  for (int i = 0; i < columns; i++) {
    output_array.at(i).resize(0);
    output_array.at(i).shrink_to_fit();
    output_array_counts.at(i) = 0;
  }
  return;
}

/* true if v is a bottom row PE (PEs responsible for outputs) */
template <typename inputT, typename outputT>
bool SA<inputT, outputT>::is_output_vertex(
    PE_Graph::Vertex<inputT, outputT> &v) {
  int index = get_index_from_vertex(v);
  if (index >= (vertarray.size() - columns) &&
      index <= (vertarray.size() - 1)) {
    return true;
  }
  return false;
}

template <typename inputT, typename outputT>
int SA<inputT, outputT>::output_array_hash(int n) {
  assert(n >= vertarray.size() - columns);
  return n - (vertarray.size() - columns);
}

template <typename inputT, typename outputT>
void SA<inputT, outputT>::generate_profile_report() {
  printf("Total Cycles: %d\n", profiler.get_cycles());
}

template <typename inputT, typename outputT>
void SA<inputT, outputT>::print_array() {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < columns; ++j) {
      std::cout << get_pe_from_vertex(vertarray[i * columns + j]).get_weight()
                << '\t';
#if 0
            printf("%d\t%d\t|\t", get_pe_from_vertex(vertarray[i*columns+j]).get_reg(),
                    get_pe_from_vertex(vertarray[i*columns+j]).get_ps_buffer());
            printf("%d,\t%d,\t%d\t|\t", 
                    get_pe_from_vertex(vertarray[i*columns+j]).get_weight(), 
                    get_pe_from_vertex(vertarray[i*columns+j]).get_input_buffer(),
                    get_pe_from_vertex(vertarray[i*columns+j]).get_reg());
#endif
    }
    std::cout << '\n';
  }
  std::cout << '\n';
}
#endif


template <typename T>
void maxpool(Tensor<T> *input, Tensor<T> *output,
             const Op::MaxpoolParams &mp) {
  int input_batch = input->dims_at(TENSOR_4D_BATCH);
  int input_depth = input->dims_at(TENSOR_4D_CHANNELS);
  int input_height = input->dims_at(TENSOR_4D_HEIGHT);
  int input_width = input->dims_at(TENSOR_4D_WIDTH);
  int output_batch = input_batch;
  int output_depth = input_depth;
  int output_height = mp_odims_row(mp, input->get_dims());
  int output_width = mp_odims_cols(mp, input->get_dims());

  for (int d = 0; d < output_depth; ++d) {
    for (int i = 0; i < output_height; ++i) {
      for (int j = 0; j < output_width; ++j) {
        T max_val = std::numeric_limits<T>::min();
        for (int m = 0; m < mp.k[0]; ++m) {
          for (int n = 0; n < mp.k[1]; ++n) {
            std::vector<int> in_index {input_batch-1, d, i * mp.k[0] + m, j * mp.k[1] + n};
            max_val = std::max(max_val, input->at(in_index));
          }
        }
        std::vector<int> out_index {input_batch-1, d, i, j};
        output->insert(out_index, max_val);
      }
    }
  }
}

template <typename T>
void flatten(Tensor<T> *input, Tensor<T> *output) {
  std::vector<int> new_dims = {1, input->dims_iterator(-1)};
  *output = *input;
  output->set_dims(new_dims);
}

inline std::vector<int> permute(const std::vector<int> &v,
                                  std::vector<int> perm) {
  std::for_each(perm.begin(), perm.end(),
                [&v](int i) { assert((i < v.size()) ? true : false); });
  std::vector<int> ret(v.size());
  for (int i = 0; i < v.size(); ++i) {
    ret.at(i) = v.at(perm.at(i));
  }
  return ret;
}

template <typename T>
std::valarray<T> vec2val(const std::vector<T> &v) {
  std::valarray<T> ret(v.size());
  for (int i = 0; i < v.size(); ++i) {
    ret[i] = v[i];
  }
  return ret;
}

template <typename T>
std::valarray<T> vec2val(std::vector<T> &&v) {
  return vec2val(v);
}

/* n-dimensional-index incrementer
 * ii is a n-dimensional index
 * limit_shape is the shape of the tensor that ii is indexing
 *
 * for example,
 * consider, limit_shape = [3,4,2]
 * first,
 *  ii = [0,0,0]
 * calling increment_shape on it makes it,
 *  ii = [0,0,1]
 * then,
 *  ii = [0,1,0]
 *  ii = [0,1,1]
 *  ii = [0,2,0]
 *  ii = [0,2,1]
 *  ii = [0,3,0]
 *  ii = [0,3,1]
 *  ii = [1,0,0]
 * and so on till
 *  ii = [2,3,1]
 */
inline void increment_shape(std::vector<int> &ii, const std::vector<int> &limit_shape) {
  assert(ii.size() == limit_shape.size());
  int current_index = ii.size() - 1;
  while (current_index >= 0) {
    ii.at(current_index)++;
    if (ii.at(current_index) >= limit_shape.at(current_index)) {
      ii.at(current_index) = 0;
      current_index--;
    } else {
      break;
    }
  }
  if (ii.at(0) >= limit_shape.at(0)) {
    log_fatal("Cannot increment past limit_shape");
  }
}

/* TODO: use valarray where fits */
template <typename T>
void transpose(Tensor<T> *input, Tensor<T> *output, std::vector<int> perm) {
  // FIXME
  // i.e. the first dimension of input/output tensors are currently ignored.
  //if (perm.size() == input->dims_size() + 1) {
  //  assert(perm.at(0) == 0);
  //  /* Hack to turn the perm array to 3d
  //   * because input tensors are 3d not 4d
  //   * 1. remove first element (0th dimension)
  //   * 2. subtract 1 from each dim */
  //  perm.erase(perm.begin(), perm.begin() + 1);
  //  for (int i = 0; i < perm.size(); ++i) {
  //    perm.at(i) -= 1;
  //  }
  //} else {
  //  assert(input->dims_size() == perm.size());
  //}

  output->set_dims(permute(input->get_dims(),  perm));
  std::valarray<int> ishape = vec2val(input->get_dims());

  std::valarray<int> istride = get_stride_from_shape(ishape);
  /* consider making get_stride_from_shape take valarrays */
  std::valarray<int> ostride = get_stride_from_shape(vec2val(output->get_dims()));
  

  /* TODO: use valarray here */
  std::vector<int> ii (input->dims_size(), 0);
  int total_elements = input->dims_iterator(-1);
  for (int i = 0; i < total_elements; ++i) {
    std::valarray<int> t0 = vec2val(ii);
    std::valarray<int> t1 = istride * t0;
    int iindex = std::accumulate(std::begin(t1), std::end(t1), 0);
    std::valarray<int> t2 = vec2val(permute(ii, perm));
    std::valarray<int> t3 = ostride * t2;
    int oindex = std::accumulate(std::begin(t3), std::end(t3), 0);
    output->set(oindex, input->at(iindex));
    increment_shape(ii, input->get_dims());
  }
}

/* Vector Arrays 
 * Used by Gemm/Matmul routines */
template <typename inputT, typename weightT, typename outputT> class VA {
  int wrows;
  int wcols;
  int isize;
  Tensor<weightT> *weights;
  Tensor<weightT> *bias;
  public:
    VA(Op::Layer::Gemm &gp);
    VA(Op::Layer::MatMul &gp);
    VA(Op::Layer::QLinearMatMul &gp);
    void run(Tensor<inputT> *input, Tensor<outputT> *output);
    ~VA() {
      delete weights;
      delete bias;
    }
};


template <typename inputT, typename weightT, typename outputT>
VA<inputT, weightT, outputT>::VA(Op::Layer::Gemm &gp) {
  wrows = gp.m_cp.wr;
  wcols = gp.m_cp.wc;
  isize = gp.input_dims[TENSOR_2D_WIDTH];
  if (gp.m_cp.transB) {
    Tensor<weightT> *tmp = new TensorExtant<inputT>(gp.weights);
    auto dims = tmp->get_dims();
    std::vector<int> new_dims {dims[1], dims[0]};
    weights = new TensorCreate<weightT>(new_dims);
    transpose(tmp, weights, std::vector<int>{1, 0});
    delete tmp;
  } else {
    weights = new TensorExtant<weightT>(gp.weights);
  }
  bias = new TensorExtant<weightT>(gp.bias);
}

template <typename inputT, typename weightT, typename outputT>
VA<inputT, weightT, outputT>::VA(Op::Layer::MatMul &gp) {
  wrows = gp.m_cp.wc;
  wcols = gp.m_cp.wr;
  isize = gp.input_dims[TENSOR_2D_WIDTH];
  weights = new TensorExtant<weightT>(gp.weights);
  bias = nullptr;
}

template <typename inputT, typename weightT, typename outputT>
VA<inputT, weightT, outputT>::VA(Op::Layer::QLinearMatMul &gp) {
  wrows = gp.m_cp.wc;
  wcols = gp.m_cp.wr;
  isize = gp.input_dims[TENSOR_2D_WIDTH];
  weights = new TensorExtant<weightT>(gp.weights);
  bias = nullptr;
}

template <typename inputT, typename weightT, typename outputT>
void VA<inputT, weightT, outputT>::run(Tensor<inputT> *input, Tensor<outputT> *output) {
  assert(input->dims_size() == 2 && weights->dims_size() == 2);
  //assert(input->dims_at(1) == weights->dims_at(0) && "non-matching matrix dimensions");

  int N = input->dims_at(0);
  int M = input->dims_at(1);
  int K = weights->dims_at(1);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < K; ++j) {
      outputT dst = 0;
      for (int k = 0; k < M; ++k) {
        /* TODO: use Tensor->at that returns a reference and += operator
         * part of tensor refactor
         */
        dst += input->at(i * M + k) * weights->at(k * K + j);
      }
      /* For gemm */
      if (bias != nullptr) {
        dst += bias->at(i*K + j);
      }
      output->set(i*K + j, dst);
    }
  }
}

/* Deduces and removes -1/0 from old_shape to return 
 * a correct new_shape.
 * See https://onnx.ai/onnx/operators/onnx__Reshape.html#reshape
 *
 * TODO: handle 0s in shape (does not do it presently)
 */
inline std::vector<int64_t> deduce_new_shape(std::vector<int64_t> old_shape, int input_total_size) {
  auto itr = std::find(old_shape.begin(), old_shape.end(), -1);
  if (itr != old_shape.end()) {
    int remaining_size = std::abs(prod(old_shape.begin(), old_shape.end(), 1));
    assert(input_total_size % remaining_size == 0 && "unable to deduce new shape");
    int remaining_dim = input_total_size / remaining_size;
    *itr = remaining_dim; 
  }
  return old_shape;
}

template <typename T>
void reshape(Tensor<T> *input, Tensor<T> *output, const std::vector<int64_t> &new_shape) {
  /* atmost 1 dimension can be -1 */
  std::vector<int64_t> deduced_shape = deduce_new_shape(new_shape, input->dims_iterator(-1));
  *output = *input;
  std::vector<int> dims (deduced_shape.size());
  std::copy(deduced_shape.begin(), deduced_shape.end(), dims.begin());
  output->set_dims(dims);
}



/* Element wise tensor addition */
template <typename inputT, typename outputT>
void tensor_add(Tensor<outputT> *output, Tensor<inputT> *input1, Tensor<inputT> *input2) {
  assert(input1->dims_iterator(-1) == input2->dims_iterator(-1));
  for (int i = 0; i < input1->dims_iterator(-1); ++i) {
    output->set(i, input1->at(i) + input2->at(i));
  }
}

/* Add a tensor and a vector. Each element of the
 * vector is added to all elements of each channel
 * of the tensor
 *
 *  input_tensor.shape = (X, _, _)
 *  input_vector.shape = (X)
 */
template <typename inputT, typename outputT>
void tensor_vector_add(Tensor<outputT> *output, Tensor<inputT> *input_tensor, Tensor<inputT> *input_vector) {
  assert(input_vector->dims_size() == 1);
  assert(input_vector->dims_at(0) == input_tensor->dims_at(0));
  assert(input_tensor->dims_size() == 3);

  for (int i = 0; i < input_tensor->dims_at(0); ++i) {
    for (int j = 0; j < input_tensor->dims_at(1); ++j) {
      for (int k = 0; k < input_tensor->dims_at(2); ++k) {
        std::vector<int> index {i, j, k};
        outputT t1 = input_tensor->at(index) + input_vector->at(i);
        output->insert(index, t1);
      }
    }
  }
}

template <typename inputT, typename outputT>
inline outputT clip(inputT v, int min_lim, int max_lim) {
  if (v < min_lim) {
    return min_lim;
  } else if (v > max_lim) {
    return max_lim;
  } else {
    return v;
  }
}

template <typename inputT, typename outputT>
inline outputT quantize_fn(inputT v, float scale, int zero_point, int min_lim, int max_lim) {
  return clip<inputT, outputT>((outputT) ((v / scale) + zero_point), min_lim, max_lim);
}

template <typename inputT, typename outputT>
inline outputT dequantize_fn(inputT v, float scale, int zero_point) {
  return ((v * scale) + zero_point);
}

template <typename inputT, typename outputT>
void quantize(Tensor<inputT> *input, Tensor<outputT> *output, std::vector<float> scales, std::vector<int> zero_point) {
  int min_lim = 0;
  int max_lim = 0;
  if (typeid(outputT) == typeid(uint8_t)) {
    min_lim = 0;
    max_lim = 255;
  } else {
    log_fatal("cant find saturation values for quantization (unimplemented)");
  }
  if (input->dims_size() == 4) {
    auto bscales = broadcast_vec(scales, input->dims_at(TENSOR_4D_CHANNELS));
    auto bzero_points = broadcast_vec(zero_point, input->dims_at(TENSOR_4D_CHANNELS));
    for (int i = 0; i < input->dims_at(TENSOR_4D_BATCH); ++i) {
      for (int j = 0; j < input->dims_at(TENSOR_4D_CHANNELS); ++j) {
        for (int k = 0; k < input->dims_at(TENSOR_4D_HEIGHT); ++k) {
          for (int l = 0; l < input->dims_at(TENSOR_4D_WIDTH); ++l) {
            std::vector<int> in_index {i, j, k, l};
            inputT val = input->at(in_index);
            outputT new_val = quantize_fn<inputT, outputT>(val, bscales[j], bzero_points[j], min_lim, max_lim);
            output->insert(in_index, new_val);
          }
        }
      }
    }
  } else if (input->dims_size() == 2) {
    assert(scales.size() == 1);
    assert(zero_point.size() == 1);
    for (int i = 0; i < input->dims_iterator(-1); ++i) {
      inputT val = input->at(i);
      outputT new_val = quantize_fn<inputT, outputT>(val, scales[0], zero_point[0], min_lim, max_lim);
      output->set(i, new_val);
    }
  }
}

template <typename inputT, typename weightT, typename outputT> class ConvEngine {
  //const Op::Layer::Conv *cc;
  Tensor<weightT> *weights;
  Tensor<outputT> *bias;
  int kn;
  int kh;
  int kw;
  std::vector<int> pad_vec;

  void _kernel(int k, const Tensor<inputT> *input, Tensor<outputT> *output);

public:
  ConvEngine(const Op::Layer::Conv *cc);
  ConvEngine(const Op::Layer::QLinearConv *cc);
  ~ConvEngine();
  void run(const Tensor<inputT> *input, Tensor<outputT> *output);
};

template <typename inputT, typename weightT, typename outputT>
ConvEngine<inputT, weightT, outputT>::ConvEngine(const Op::Layer::Conv *cc) {
  weights = new TensorExtant<weightT>(cc->weights);
  bias = new TensorExtant<outputT>(cc->bias);
  kn = cc->m_cp.kn;
  kh = cc->m_cp.k[TENSOR_2D_HEIGHT];
  kw = cc->m_cp.k[TENSOR_2D_WIDTH];
  const int *pad = cc->m_cp.pad;
  pad_vec = std::vector<int>{pad[0], pad[1], pad[2], pad[3]};
}

template <typename inputT, typename weightT, typename outputT>
ConvEngine<inputT, weightT, outputT>::ConvEngine(const Op::Layer::QLinearConv *cc) {
  weights = new TensorExtant<weightT>(cc->weights);
  bias = new TensorExtant<outputT>(cc->bias);
  kn = cc->m_cp.kn;
  kh = cc->m_cp.k[TENSOR_2D_HEIGHT];
  kw = cc->m_cp.k[TENSOR_2D_WIDTH];
  const int *pad = cc->m_cp.pad;
  pad_vec = std::vector<int>{pad[0], pad[1], pad[2], pad[3]};
}


template <typename inputT, typename weightT, typename outputT>
void ConvEngine<inputT, weightT, outputT>::_kernel(int k, const Tensor<inputT> *input,
                                          Tensor<outputT> *output) {

  /* TODO: add bias here */
  int ic = input->dims_at(TENSOR_4D_CHANNELS);
  int oh = output->dims_at(TENSOR_4D_HEIGHT);
  int ow = output->dims_at(TENSOR_4D_WIDTH);

  for (int ici = 0; ici < ic; ++ici) {
    for (int ohi = 0; ohi < oh; ++ohi) {
      for (int owi = 0; owi < ow; ++owi) {
        for (int khi = 0; khi < kh; ++khi) {
          for (int kwi = 0; kwi < kw; ++kwi) {
            // printf("k %d, ici %d, ohi,owi %d,%d, ihi,iwi %d,%d, khi,kwi"
            //        "%d,%d\n",
            //        k, ici, ohi, owi, ohi + khi, owi + kwi, khi, kwi);
            std::vector<int> out_index{0, k, ohi, owi};
            std::vector<int> in_index{0, ici, ohi + khi, owi + kwi};
            std::vector<int> w_index{k, ici, khi, kwi};
            // print_vec("output dims", output->get_dims());
            // print_vec("out_index", out_index);
            outputT val = output->at(out_index);
            // print_vec("input dims", input->get_dims());
            // print_vec("in index", in_index);
            // std::cout << "in : " <<  padded_input->at(in_index) << ' ' <<
            // "weigth " << weights->at(w_index) << '\n';
            outputT val2 = input->at(in_index) * weights->at(w_index);
            // std::cout  << "v1 " << " v2 " << val << ' ' << val2 << '\n';
            output->insert(out_index, val + val2);
          }
        }
      }
    }
  }
}

template <typename inputT, typename weightT, typename outputT>
void ConvEngine<inputT, weightT, outputT>::run(const Tensor<inputT> *input, Tensor<outputT> *output) {
  Tensor<inputT> *padded_input = tensor_pad(input, pad_vec);

  std::vector<std::thread> tc;
  for (int k = 0; k < kn; ++k) {
    tc.push_back(std::thread(&ConvEngine<inputT,weightT,outputT>::_kernel, this, k, padded_input, output));
  }
  for (int k = 0; k < kn; ++k) {
    tc[k].join();
  }
}

template <typename inputT, typename weightT, typename outputT>
ConvEngine<inputT, weightT, outputT>::~ConvEngine() {
  delete weights;
  delete bias;
}


template <typename inputT, typename outputT>
void dequantize(Tensor<inputT> *input, Tensor<outputT> *output, const std::vector<float> &scales, const std::vector<int> &zero_point) {
  /* TODO: refactor this */
  if (input->dims_size() == 4) {
    auto bscales = broadcast_vec(scales, input->dims_at(TENSOR_4D_CHANNELS));
    auto bzero_points = broadcast_vec(zero_point, input->dims_at(TENSOR_4D_CHANNELS));

    for (int i = 0; i < input->dims_at(TENSOR_4D_BATCH); ++i) {
      for (int j = 0; j < input->dims_at(TENSOR_4D_CHANNELS); ++j) {
        for (int k = 0; k < input->dims_at(TENSOR_4D_HEIGHT); ++k) {
          for (int l = 0; l < input->dims_at(TENSOR_4D_WIDTH); ++l) {
            std::vector<int> in_index {i, j, k, l};
            inputT val = input->at(in_index);
            outputT new_val = dequantize_fn<inputT, outputT>(val, bscales[j], bzero_points[j]);
            output->insert(in_index, new_val);
          }
        }
      }
    }
  } else if (input->dims_size() == 2) {
    assert(scales.size() == 1);
    assert(zero_point.size() == 1);
    for (int i = 0; i < input->dims_iterator(-1); ++i) {
      inputT val = input->at(i);
      outputT new_val = dequantize_fn<inputT, outputT>(val, scales[0], zero_point[0]);
      output->set(i, new_val);
    }
  }
}
