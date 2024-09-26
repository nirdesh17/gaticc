#include "pch.h"
// #include <algorithm>
// #include <cmath>
// #include <cstdint>
// #include <functional>
// #include <iostream>
// #include <iterator>
// #include <numeric>
// #include <utility>

// #include <boost/graph/adjacency_list.hpp>
// #include <boost/graph/graph_traits.hpp>

#include "sim.h"
#include "utils.h"

/* auxillary functions */
Point get_cartesian_cord(int index, int r, int c) {
  int row_n = std::floor((float)index / r);
  int col_n = index % c;
  return Point(row_n, col_n);
}

#if 0
void Tree::generate_btree(Mat<int> const &v, std::pair<int, int> xy) {
  int x = xy.first;
  int y = xy.second;
  if (x >= rows || y >= columns) {
    return;
  }
  if (vertex_map.find(xy) == vertex_map.end()) {
    Int_Graph::Vertex vertex = boost::add_vertex(v.at(x, y), g);
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

Tree::Tree(Mat<int> const &v, int rows, int columns) : rows{rows}, columns{columns} {
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

#endif

std::vector<float> compute_output_scale(const std::vector<float>& x_scale,
    const std::vector<float>& w_scale, const std::vector<float>& y_scale) {
  auto new_x_scale = broadcast_vec(x_scale, w_scale.size());
  auto new_y_scale = broadcast_vec(y_scale, w_scale.size());
  std::vector<float> ret(w_scale.size());
  for (int i = 0; i < w_scale.size(); ++i) {
    ret[i] = new_y_scale[i] / (new_x_scale[i] * w_scale[i] );
  }
  return ret;
}
