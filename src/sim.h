#pragma once

#include <iostream>
#include <iterator>
#include <utility>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <queue>
#include <vector>
#include <functional>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>

using input_t = std::int8_t;
using weight_t = std::int8_t;
/* for partial sums */
using reg_t = std::int32_t;

using Point = std::pair<int, int>;
using Mat = std::vector<std::vector<int>>;
using fMat = std::vector<std::vector<float>>;
using action = std::function<float(std::vector<int>, int, int,int  , int  , int , int,int )> ;

class PE {
    private:
        int id;
        weight_t weight;    
        /* for partial sums */
        reg_t reg;
        /* this is written by other PEs */
        reg_t input_buffer;

        reg_t aux_buffer;
        reg_t ps_buffer;
    public:
        PE(int id, weight_t w, reg_t r, reg_t input_buffer);
        int get_id();
        weight_t get_weight();
        reg_t get_reg();
        reg_t get_input_buffer();
        reg_t get_ps_buffer();
        void set_input_buffer(reg_t r);
        void set_weight(weight_t w);
        void set_reg(reg_t r);
        void set_aux_buffer(reg_t r);
        void set_ps_buffer(reg_t r);
        reg_t get_aux_buffer();
        reg_t mac();
        void print_pe();
};

class Profiler {
    uint64_t cycles;
    public:
        void incr_cycles(int count);
        uint64_t get_cycles();
};

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
    std::vector<Chainblock*> arr;
    public:
        void push(Chainblock *b);
        Chainblock *at(int i);
        void replace(int i, Chainblock *b);
        ~Chain();
        int pass_through(int x);
};

class Relu: public Chainblock {
    int clip_val;
    bool no_clip;
    public:
        Relu(int clip_val);
        Relu();
        int exec(int x) override;
};

/*Quantizer is used to re-encode information.It is used to reduce the size and bandwidth required
 * by 4 times.
 *In our case we are reducing a 32bit-Int to a 8bit-Int.
 *the original value is multiplied to a scale value and then clipped into the range of (-127,127)
 */
class Quantize: public Chainblock {
    int scale;
    int shift;
    public:
        Quantize(int scale, int shift);
        int exec(int x) override;
        int clipper(int x);
};

class BatchNorm: public Chainblock {
    int mean;
    int sd;
    int gamma;
    int beta;
    public:
        BatchNorm(int mean, int sd, int gamma, int beta);
        int exec(int x);
};

class Bias : public Chainblock
{
    int bias;

public:
    int exec(int x) override;
};

class Pooler
{
    int stride;
    int kernel_rows;
    int kernel_cols;
    int padding;
    int dilation;

    auto movement(Mat input,  int ip_rows, int ip_columns, action );


public:
    
    Pooler(int stride,int kernel_rows,int kernel_cols ,int padding ,int dilation);
    
    
    fMat max_pooler(Mat input , int ip_rows , int ip_columns  );
    fMat average_pooler(Mat input , int ip_rows , int ip_columns );
    fMat global_average_pooler(Mat input, int ip_rows , int ip_columns );

};



float max_pooler_action(std::vector<int> input, int ip_rows, int ip_columns,int stride , int padding , int dilation, int kernel_rows,int kernel_cols);

float average_pooler_action(std::vector<int> input, int ip_rows, int ip_columns,int stride , int padding , int dilation, int kernel_rows, int kernel_cols);

fMat global_average_pooler_action(std::vector<int> input, int ip_rows, int ip_columns,int stride , int padding , int dilation, int kernel_rows, int kernel_cols);

namespace PE_Graph {
    using Graph  = boost::adjacency_list<boost::vecS, boost::listS, boost::directedS, PE>;
    using Vertex = boost::graph_traits<Graph>::vertex_descriptor;
    using Adjacency_iterator = Graph::adjacency_iterator;
    using Neighbours = std::pair<PE_Graph::Adjacency_iterator, PE_Graph::Adjacency_iterator>;
}

class SA {
    private:
        PE_Graph::Graph g;
        int rows;
        int columns;
        std::vector<PE_Graph::Vertex> vertarray;
        std::vector<std::vector<int>> output_array;
        std::vector<int> output_array_counts;
        Profiler profiler;
        bool profile_enabled;

        int get_index_from_vertex(PE_Graph::Vertex& v);
        PE& get_pe_from_vertex(PE_Graph::Vertex& v);
        PE& get_pe_from_adjacency_iterator(PE_Graph::Adjacency_iterator& itr);
        PE_Graph::Neighbours get_neighbouring_vertices(PE_Graph::Vertex& v);
        PE_Graph::Vertex& get_vertex_from_adjacency_iterator(PE_Graph::Adjacency_iterator& itr);
        void _propagate(PE_Graph::Vertex& v, Chain &chain);
        void load_inputs(std::vector<int>& inputs);
        bool is_output_vertex(PE_Graph::Vertex& v);
        int output_array_hash(int n);
        void push_to_output_array(int h, reg_t t1);
        void prepare_queue(std::queue<PE_Graph::Vertex>& exec_queue);
        bool should_pass_partial_sum(PE_Graph::Vertex& vi);
        bool should_pass_input(PE_Graph::Vertex& vi);
        PE_Graph::Neighbours assign_vertices(const PE_Graph::Neighbours& neighbours, PE_Graph::Vertex& v);
    public:
        SA(int r, int c);
        SA(int r, int c, bool profile_enabled);
        void propagate(Mat& input_mat, Chain &chain);
        Mat get_output();
        void load_weights(std::vector<int>& weights);
        void print_array();
        void generate_profile_report();
        int total_vertices();
        int total_edges();
        int get_rows();
        int get_cols();
};


namespace Int_Graph {
    using Graph  = boost::adjacency_list<boost::vecS, boost::listS, boost::directedS, int>;
    using Vertex = boost::graph_traits<Graph>::vertex_descriptor;
    using Adjacency_iterator = Graph::adjacency_iterator;
    using Vertex_Iterator = Graph::vertex_iterator;
};

class Tree {
    private:
        Int_Graph::Graph g;
        int rows;
        int columns;
        std::map<std::pair<int,int>, Int_Graph::Vertex> vertex_map;
        /* TODO: turn these into maps 
         * since they are used only internally during construction by 
         * connect_btree, hide them
         */
        std::vector<std::pair<std::pair<int,int>, std::pair<int,int>>> edge_memo;
        std::vector<std::pair<int,int>> child_memo;

        void generate_btree(Mat const &v, std::pair<int,int> xy);
        bool is_node_allowed(std::pair<int,int> node);
        void connect_btree(std::pair<int,int> xy);
    public:
        Tree(Mat const &v, int rows, int columns);
        Int_Graph::Vertex get_root();
        int total_vertices();
        int total_edges();
        void print_btree();
        std::vector<int> breadth_first_order();
};

/* return element at (x,y) in v */
template<typename T1>
T1 Mat_at(std::vector<std::vector<T1>> const &v, int x, int y) {
    assert(x < v.size());
    assert(y < v[0].size());
    return v.at(x).at(y);
}

/* convert v into 2d array (Mat) of dims (rows,column) */
template<typename T1, typename T2>
std::vector<std::vector<T1>> v2mat(std::vector<T2> &v, int rows, int columns) {
    std::vector<std::vector<T1>> m;
    for (int i = 0; i < rows; ++i) {
        std::vector<T1> vv;
        for (int j = 0; j < columns; ++j) {
            vv.push_back(v.at(i*columns + j));
        }
        m.push_back(vv);
    }
    return m;
}

template<typename T1, typename T2>
std::vector<T1> mat2v(std::vector<std::vector<T2>> m,int rows, int columns){
        std::vector<T1> v;
        for(int i = 0 ; i <m.size();i++){
            for( int j = 0 ; j<m.at(0).size();j++){
                v.push_back(Mat_at<T2>(m ,i,j));
            }
        }
        return v;

}
Mat Padder(Mat input, int padding);

