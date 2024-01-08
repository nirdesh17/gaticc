#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <utility>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>

#include "sim.h"
#include "utils.h"

#define DEBUG 1
#define SA_output_dimension                                                    \
  ((ip_rows + 2 * padding - dilation * (kernel_rows - 1) - 1) / stride) + 1

PE::PE(int id, weight_t w, reg_t r, reg_t input_buffer)
    : id{id}, weight{w}, reg{r}, input_buffer{input_buffer}, aux_buffer{0},
      ps_buffer{0} {};

int PE::get_id() { return this->id; }

weight_t PE::get_weight() { return this->weight; }
reg_t PE::get_reg() { return this->reg; }
reg_t PE::get_input_buffer() { return this->input_buffer; }
reg_t PE::get_aux_buffer() { return this->aux_buffer; }
reg_t PE::get_ps_buffer() { return this->ps_buffer; }
void PE::set_input_buffer(reg_t r) { this->input_buffer = r; }
void PE::set_weight(weight_t w) { this->weight = w; }
void PE::set_reg(reg_t r) { this->reg = r; }
void PE::set_aux_buffer(reg_t r) { this->aux_buffer = r; }
void PE::set_ps_buffer(reg_t r) { this->ps_buffer = r; }
reg_t PE::mac() {
  reg_t ps = this->get_input_buffer() * this->get_weight();
  return ps;
}

void PE::print_pe() {
  printf("Id: %d\tWeight: %d\tReg: %d\tInput: %d\tAux: %d\tPs: %d\n", id,
         weight, reg, input_buffer, aux_buffer, ps_buffer);
}

/* auxillary functions */
Point get_cartesian_cord(int index, int r, int c) {
  int row_n = std::floor((float)index / r);
  int col_n = index % c;
  return Point(row_n, col_n);
}

SA::SA(int r, int c) : SA(r, c, false) {}

SA::SA(int r, int c, bool profile_enabled)
    : rows{r}, columns{c}, profile_enabled{profile_enabled},
      output_array{Mat(columns)},
      output_array_counts{std::vector<int>(columns)} {
  /* add null vertices to the graph
   * do not connect any vertices with edges
   */
  for (int i = 0; i < r * c; ++i) {
    /* zero initialize weights and partial sums */
    PE_Graph::Vertex v = boost::add_vertex(PE(i, 0, 0, 0), g);
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

int SA::get_index_from_vertex(PE_Graph::Vertex &v) {
  int index;
  for (index = 0; index < vertarray.size(); ++index) {
    if (vertarray.at(index) == v) {
      break;
    }
  }
  return index;
}
PE &SA::get_pe_from_vertex(PE_Graph::Vertex &v) { return g[v]; }

PE &SA::get_pe_from_adjacency_iterator(PE_Graph::Adjacency_iterator &itr) {
  return g[*itr];
}

PE_Graph::Neighbours SA::get_neighbouring_vertices(PE_Graph::Vertex &v) {
  return boost::adjacent_vertices(v, g);
}

PE_Graph::Vertex &
SA::get_vertex_from_adjacency_iterator(PE_Graph::Adjacency_iterator &itr) {
  for (PE_Graph::Vertex &v : vertarray) {
    if (v == *itr) {
      return v;
    }
  }
  assert(1 == 0 && "adjacency iterator didn't match up to any vertex");
}

int SA::total_vertices() {
  assert(vertarray.size() == boost::num_vertices(g) &&
         "vertarray and total vertices in 'g' mismatch");
  return boost::num_vertices(g);
}

int SA::total_edges() { return boost::num_edges(g); }

int SA::get_rows() { return rows; }

int SA::get_cols() { return columns; }

void SA::load_weights(std::vector<int> &weights) {
  for (int i = 0; i < rows * columns; ++i) {
    get_pe_from_vertex(vertarray[i]).set_weight(weights[i]);
  }
}

bool SA::should_pass_input(PE_Graph::Vertex &vi) {
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

bool SA::should_pass_partial_sum(PE_Graph::Vertex &vi) {
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

PE_Graph::Neighbours SA::assign_vertices(const PE_Graph::Neighbours &neighbours,
                                         PE_Graph::Vertex &v) {
  PE_Graph::Adjacency_iterator right{}, down{};
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
  return PE_Graph::Neighbours(right, down);
}

/* carry out mac on one PE pointed by Vertex v, and pass inputs/partial sums
 * to the neighbouring PEs
 */
void SA::_propagate(PE_Graph::Vertex &v, Chain &chain) {
  /* expects ps and input_buffer to be filled */
  reg_t res = get_pe_from_vertex(v).mac();

  PE_Graph::Adjacency_iterator right{}, down{};
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
    reg_t t1 = get_pe_from_adjacency_iterator(right).get_input_buffer();
    get_pe_from_adjacency_iterator(right).set_aux_buffer(t1);

    reg_t t2 = get_pe_from_vertex(v).get_aux_buffer();
    get_pe_from_adjacency_iterator(right).set_input_buffer(t2);
  }

  if (should_pass_partial_sum(v)) {
    // pass partial sum
    reg_t t1 = get_pe_from_adjacency_iterator(down).get_ps_buffer();
    get_pe_from_adjacency_iterator(down).set_reg(t1);

    reg_t t2 = get_pe_from_vertex(v).get_reg() + res;
    get_pe_from_adjacency_iterator(down).set_ps_buffer(t2);
  }

  if (is_output_vertex(v)) {
    reg_t t1 = get_pe_from_vertex(v).get_reg() + res;
    t1 = chain.pass_through(t1);
    int h = output_array_hash(get_index_from_vertex(v));
    push_to_output_array(h, t1);
  }
}

void SA::push_to_output_array(int h, reg_t t1) {
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

Mat SA::get_output() { return output_array; }

/* true if v is a bottom row PE (PEs responsible for outputs) */
bool SA::is_output_vertex(PE_Graph::Vertex &v) {
  int index = get_index_from_vertex(v);
  if (index >= (vertarray.size() - columns) &&
      index <= (vertarray.size() - 1)) {
    return true;
  }
  return false;
}

int SA::output_array_hash(int n) {
  assert(n >= vertarray.size() - columns);
  return n - (vertarray.size() - columns);
}

void SA::generate_profile_report() {
  printf("Total Cycles: %d\n", profiler.get_cycles());
}

void SA::print_array() {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < columns; ++j) {
      printf("%d\t",
             get_pe_from_vertex(vertarray[i * columns + j]).get_weight());
#if 0
            printf("%d\t%d\t|\t", get_pe_from_vertex(vertarray[i*columns+j]).get_reg(),
                    get_pe_from_vertex(vertarray[i*columns+j]).get_ps_buffer());
            printf("%d,\t%d,\t%d\t|\t", 
                    get_pe_from_vertex(vertarray[i*columns+j]).get_weight(), 
                    get_pe_from_vertex(vertarray[i*columns+j]).get_input_buffer(),
                    get_pe_from_vertex(vertarray[i*columns+j]).get_reg());
#endif
    }
    printf("\n");
  }
  printf("\n");
}

/* load inputs at first n PEs where n = 0 3 6 9 ... for a Mx3 systolic array  */
void SA::load_inputs(std::vector<int> &inputs) {
  for (int i = 0, vi = 0; i < rows; ++i) {
    get_pe_from_vertex(vertarray[vi]).set_input_buffer(inputs.at(i));
    get_pe_from_vertex(vertarray[vi]).set_aux_buffer(inputs.at(i));
    vi += columns;
  }
}

template <typename T>
void exchange_queues(std::queue<T> &dest, std::queue<T> &src) {
  assert(dest.empty() == true);
  std::set<PE_Graph::Vertex> s;
  while (!src.empty()) {
    s.insert(src.front());
    src.pop();
  }
  for (auto &i : s) {
    dest.push(i);
  }
}

void SA::prepare_queue(std::queue<PE_Graph::Vertex> &exec_queue) {
  std::vector<int> order(rows * columns);
  std::iota(order.begin(), order.end(), 0);
  Mat order2d = v2mat<int, int>(order, rows, columns);
  Tree t(order2d, rows, columns);
  std::vector<int> new_order = t.breadth_first_order();
  for (auto i : new_order) {
    exec_queue.push(vertarray[i]);
  }
}

// iterate over all PEs in systolic manner and call _propagate on them
void SA::propagate(Mat &input_mat, Chain &chain) {
  std::queue<PE_Graph::Vertex> exec_queue;
  std::queue<PE_Graph::Vertex> alt_queue;

  prepare_queue(exec_queue);
  for (int i = 0; i != input_mat.size(); ++i) {
    load_inputs(input_mat.at(i));
    while (!exec_queue.empty()) {
      PE_Graph::Vertex v = exec_queue.front();
      alt_queue.push(v);
      exec_queue.pop();
      _propagate(v, chain);
    }
    exchange_queues<PE_Graph::Vertex>(exec_queue, alt_queue);
    if (profile_enabled) {
      profiler.incr_cycles(1);
    }
  }
}

void Tree::generate_btree(Mat const &v, std::pair<int, int> xy) {
  int x = xy.first;
  int y = xy.second;
  if (x >= rows || y >= columns) {
    return;
  }
  if (vertex_map.find(xy) == vertex_map.end()) {
    Int_Graph::Vertex vertex = boost::add_vertex(Mat_at<int>(v, x, y), g);
    vertex_map.insert({xy, vertex});
  }
  Tree::generate_btree(v, std::pair<int, int>(x + 1, y));
  Tree::generate_btree(v, std::pair<int, int>(x, y + 1));
}

bool Tree::is_node_allowed(std::pair<int, int> node) {
  return (node.first >= rows || node.second >= columns) ? false : true;
}

void Tree::connect_btree(std::pair<int, int> xy) {
  int x = xy.first;
  int y = xy.second;
  std::pair<int, int> left_node(x + 1, y);
  std::pair<int, int> right_node(x, y + 1);

  if (is_node_allowed(left_node)) {
    if ((std::find_if(edge_memo.begin(), edge_memo.end(),
                      [&xy, &left_node](auto p) {
                        return xy == p.first && left_node == p.second;
                      })) == edge_memo.end()) {
      if ((std::find_if(child_memo.begin(), child_memo.end(),
                        [&left_node](auto p) { return p == left_node; })) ==
          child_memo.end()) {
        boost::add_edge(vertex_map.at(xy), vertex_map.at(left_node), g);
        edge_memo.push_back(
            std::pair<std::pair<int, int>, std::pair<int, int>>(xy, left_node));
        child_memo.push_back(left_node);
        connect_btree(left_node);
      }
    }
  }
  if (is_node_allowed(right_node)) {
    if ((std::find_if(edge_memo.begin(), edge_memo.end(),
                      [&xy, &right_node](auto p) {
                        return xy == p.first && right_node == p.second;
                      })) == edge_memo.end()) {
      if ((std::find_if(child_memo.begin(), child_memo.end(),
                        [&right_node](auto p) { return p == right_node; })) ==
          child_memo.end()) {
        boost::add_edge(vertex_map.at(xy), vertex_map.at(right_node), g);
        edge_memo.push_back(std::pair<std::pair<int, int>, std::pair<int, int>>(
            xy, right_node));
        child_memo.push_back(right_node);
        connect_btree(right_node);
      }
    }
  }
  return;
}

Tree::Tree(Mat const &v, int rows, int columns) : rows{rows}, columns{columns} {
  Tree::generate_btree(v, std::pair<int, int>(0, 0));
  Tree::connect_btree(std::pair<int, int>(0, 0));
}

std::vector<int> Tree::breadth_first_order() {
  Int_Graph::Vertex root = get_root();
  std::queue<Int_Graph::Vertex> q;
  std::vector<int> sorted_order;

  q.push(root);
  while (!q.empty()) {
    Int_Graph::Vertex v = q.front();
    q.pop();
    sorted_order.push_back(g[v]);
    auto ai = boost::adjacent_vertices(v, g);
    std::for_each(ai.first, ai.second, [&q](auto const &v) { q.push(v); });
  }
  return sorted_order;
}

Int_Graph::Vertex Tree::get_root() {
  auto vi = boost::vertices(g);
  Int_Graph::Vertex root = *(vi.first);
  return root;
}

int Tree::total_vertices() { return boost::num_vertices(g); }

int Tree::total_edges() { return boost::num_edges(g); }

void Tree::print_btree() {
  Int_Graph::Vertex_Iterator vi, vi_end;
  std::tie(vi, vi_end) = boost::vertices(g);
  for (; vi != vi_end; ++vi) {
    std::cout << g[*vi] << ":";
    Int_Graph::Adjacency_iterator ai, ai_end;
    std::tie(ai, ai_end) = boost::adjacent_vertices(*vi, g);
    for (; ai != ai_end; ++ai) {
      std::cout << g[*ai] << ' ';
    }
    std::cout << '\n';
  }
}

void Profiler::incr_cycles(int count) { cycles += count; }

uint64_t Profiler::get_cycles() { return cycles; }

int Chainblock::exec(int x) { return x; }

Chainblock::~Chainblock() {}

void Chain::push(Chainblock *b) { arr.push_back(b); }

Chainblock *Chain::at(int i) { return arr.at(i); }

void Chain::replace(int i, Chainblock *b) {
  delete arr[i];
  arr.at(i) = b;
}

Chain::~Chain() {
  for (int i = 0; i < arr.size(); ++i) {
    delete arr[i];
  }
}

int Chain::pass_through(int x) {
  for (int i = 0; i < arr.size(); ++i) {
    x = arr.at(i)->exec(x);
  }
  return x;
}

Relu::Relu(int clip_val) : clip_val{clip_val}, no_clip{false} {}

Relu::Relu() : no_clip{true} {}

int Relu::exec(int x) {
  if (no_clip) {
    return (x < 0) ? 0 : x;
  } else {
    return (x < 0) ? 0 : ((x > clip_val) ? clip_val : x);
  }
}

Quantize::Quantize(int scale, int shift) : scale{scale}, shift{shift} {}

int Quantize::exec(int x) { return clipper(x); }

int Quantize::clipper(int x) { // return type uint8 is not accepting bc exec
                               // return type in parent is uint32
  if (x * scale <= -127) {
    return (-127);
  } else if (x * scale >= 127) {
    return 127;
  } else
    return (x * scale);
}

BatchNorm::BatchNorm(int mean, int sd, int gamma, int beta)
    : mean{mean}, sd{sd}, gamma{gamma}, beta{beta} {}

int BatchNorm::exec(int x) { return (gamma * ((x - mean) / sd)) + beta; }

int Bias::exec(int x) {
  return x + bias; // element wise addition
}

/*this max pooler pools out the max val when a kernel window is slid over the
output matrix after convolution
* this max pooler has a kernel window and a stride whos value can be set
manually, and this process overall reduces the
* size of the input image.

* IF dimension of the input matrix is odd ,
    in valid padding we crop out the last column
        in same padding we add a column of zeroes
*/

/*
 * feeding the same new_ mat to the transformer again and repeat the process
 */

/*
* the process is divided into two parts: 1. Movement   2. Action

    1. Movement : moves the kernel window over the input matrix and call a
generic action on it which will be decided by the caller.
*/
auto Pooler::movement(Mat &input, int ip_rows, int ip_columns, int stride,
                      int padding, int dilation, int kernel_rows,
                      int kernel_cols, action func) {
  std::vector<int> ret;
  std::vector<float> temp_matrix;
  fMat output_matrix;
  // relation bw stride and dilation stride +dilation < columns
  for (int i = 0; i + (dilation * (kernel_rows - 1)) < ip_rows + 2 * padding;
       i += stride) {
    for (int j = 0;
         j + (dilation * (kernel_cols - 1)) < ip_columns + 2 * padding;
         j += stride) {
      ret.clear();
      //    output_matrix.clear();
      for (int k = 0; k < (kernel_rows); k++) {
        for (int l = 0; l < kernel_cols; l++) {
          ret.push_back(
              Mat_at<int>(input, i + (k * dilation), j + (l * dilation)));
        }
      }

      temp_matrix.push_back(func(ret, ip_rows, ip_columns, stride, padding,
                                 dilation, kernel_rows, kernel_cols));
    }
  }
  output_matrix = v2mat<float, float>(temp_matrix, SA_output_dimension,
                                      SA_output_dimension);
  return output_matrix;
}

float Pooler::max_pooler_action(std::vector<int> &input, int ip_rows,
                                int ip_columns, int stride, int padding,
                                int dilation, int kernel_rows,
                                int kernel_cols) {

  auto max = max_element(input.begin(), input.end());

  return ((float)*max);
}
float Pooler::average_pooler_action(std::vector<int> &input, int ip_rows,
                                    int ip_columns, int stride, int padding,
                                    int dilation, int kernel_rows,
                                    int kernel_cols) {

  float avg = std::accumulate(input.begin(), input.end(), 0) /
              ((float)(kernel_rows * kernel_cols));
  return avg;
}

fMat Pooler ::max_pooler(Mat &input, int ip_rows, int ip_columns, int stride,
                         int padding, int dilation, int kernel_rows,
                         int kernel_cols) {

  fMat out;
  if (padding != 0) {
    input = Padder(input, padding);
  }
  out = movement(input, ip_rows, ip_columns, stride, padding, dilation,
                 kernel_rows, kernel_cols, max_pooler_action);
  return out;
}

fMat Pooler ::average_pooler(Mat &input, int ip_rows, int ip_columns,
                             int stride, int padding, int dilation,
                             int kernel_rows, int kernel_cols) {
  fMat out;
  if (padding != 0) {
    input = Padder(input, padding);
  }
  out = movement(input, ip_rows, ip_columns, stride, padding, dilation,
                 kernel_rows, kernel_cols, average_pooler_action);
  return out;
}

//  global average pooler is using average pooler inside it with kernel dims ==
//  input_dims  + padding .
fMat Pooler ::global_average_pooler(Mat &input, int ip_rows, int ip_columns,
                                    int stride, int padding, int dilation,
                                    int kernel_rows, int kernel_cols) {

  Mat out;
  if (padding != 0) {
    input = Padder(input, padding);
  }
  int temp_stride = stride + ip_rows + 2 * padding;
  int temp_kernel_rows = ip_rows + 2 * padding;
  int temp_kernel_cols = ip_columns + 2 * padding;
  fMat out_matrix;

  out_matrix =
      movement(input, ip_rows, ip_columns, temp_stride, padding, dilation,
               temp_kernel_rows, temp_kernel_cols, average_pooler_action);

  return out_matrix;
}

Mat Padder(Mat input, int padding) {
  Mat new_mat;
  std::vector<int> store;
  for (int i = 0; i < input.size() + 2 * padding; i++) {
    for (int j = 0; j < input.at(0).size() + 2 * padding; j++) {
      if (((i < padding) || (i >= input.size() + padding)) ||
          ((j < padding) || (j >= input.at(0).size() + padding))) {
        store.push_back(0);
      } else {
        store.push_back(Mat_at<int>(input, i - padding, j - padding));
      }
    }
  }
  new_mat = v2mat<int, int>(store, input.size() + 2 * padding,
                            input.at(0).size() + 2 * padding);
  print_vec_vec("inside padder matrix before returning", new_mat);
  return new_mat;
}
