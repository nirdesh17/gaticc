#pragma once

#include "onnx_parser.h"
#include "tensor.h"
#include "utils.h"
#include "ffi.h"
#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif
#include "Python.h"
#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

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

#define sa_output_dims(ip_rows, padding, dilation, kernel_rows, stride)        \
  (((ip_rows + 2 * padding - dilation * (kernel_rows - 1) - 1) / stride) + 1)

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

/* convert v into 2d array (Mat) of dims (rows,column) */
template <typename T> Mat<T> v2mat(std::vector<T> &v, int rows, int columns) {
  Mat<T> m;
  for (int i = 0; i < rows; ++i) {
    std::vector<T> vv;
    for (int j = 0; j < columns; ++j) {
      vv.push_back(v.at(i * columns + j));
    }
    m.push_back(vv);
  }
  return m;
}

template <typename T>
std::vector<T> mat2v(Mat<T> const &m, int rows, int columns) {
  std::vector<T> v;
  for (int i = 0; i < m.size(); i++) {
    for (int j = 0; j < m.at(0).size(); j++) {
      v.push_back(m.at(i, j));
    }
  }
  return v;
}

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

class Relu : public Chainblock {
  int clip_val;
  bool no_clip;

public:
  Relu(int clip_val);
  Relu();
  int exec(int x) override;
};

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

class Bias : public Chainblock {
  int bias;

public:
  int exec(int x) override;
};

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
  if (mp.pad[0] != 0) {
    input = Padder(input, mp.pad[0]);
  }
  return movement(input, mp, max_pooler_action);
}

template <typename T>
Mat<T> Pooler<T>::average_pooler(Mat<T> &input, Op::MaxpoolParams const &mp) {
  if (mp.pad[0] != 0) {
    input = Padder<T>(input, mp.pad[0]);
  }
  return movement(input, mp, average_pooler_action);
}

template <typename T>
Mat<T> Pooler<T>::global_average_pooler(Mat<T> &input,
                                        Op::MaxpoolParams const &mp) {
  if (mp.pad[0] != 0) {
    input = Padder<T>(input, mp.pad[0]);
  }
  Op::MaxpoolParams mp_copy = mp;
  mp_copy.stride[0] = mp.stride[0] + mp.imap[0] + 2 * mp.pad[0];
  mp_copy.stride[1] = mp.stride[1] + mp.imap[1] + 2 * mp.pad[3];
  mp_copy.k[0] = mp.imap[0] + 2 * mp.pad[0];
  mp_copy.k[1] = mp.imap[1] + 2 * mp.pad[1];
  return movement(input, mp_copy, average_pooler_action);
}

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

struct SaDims {
  int rows;
  int cols;
  int num;
};

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
  void _propagate(PE_Graph::Vertex<inputT, outputT> &v, Chain &chain);
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
  void propagate(Mat<inputT> const &input_mat, Chain &chain);
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
void SA<inputT, outputT>::propagate(Mat<inputT> const &input_mat,
                                    Chain &chain) {
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
        _propagate(v, chain);
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
      _propagate(v, chain);
    }
    //print_array();
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
void SA<inputT, outputT>::_propagate(PE_Graph::Vertex<inputT, outputT> &v,
                                     Chain &chain) {
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
    // t1 = chain.pass_through(t1);
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
      std::cout << get_pe_from_vertex(vertarray[i * columns + j]).get_weight() << '\t';
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


class Executor {
  template <typename inputT, typename outputT, typename intr_inputT, typename intr_outputT>
  void execute(const Op::Parser &parser,
                               const std::string &abs_img_path) {
    std::vector<inputT> ifmap(224*224*3, 10);
  }

public:
  Executor(const Op::Parser &parser, const std::string &img_path) {
    onnx::TensorProto_DataType weight_type = parser.get_model_weight_type();
    onnx::TensorProto_DataType input_type = parser.get_model_input_type();
    onnx::TensorProto_DataType output_type = parser.get_model_output_type();

    if (weight_type == onnx::TensorProto_DataType_INT8) {
      execute<float, float, int8_t, int32_t>(parser, img_path);
    } else if (weight_type == onnx::TensorProto_DataType_FLOAT) {
      execute<float, float, float, float>(parser, img_path);
    }
  }
};
